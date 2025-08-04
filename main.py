import os

# -------------------------------------------------
#  Performance optimisation flags
# -------------------------------------------------
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

import sys
import asyncio
import logging
import time
from typing import List, Dict, Any, Tuple
from contextlib import asynccontextmanager
import uuid
import json
import tempfile
import hashlib
import re
from datetime import datetime
from functools import lru_cache
from urllib.parse import urlparse
import traceback

# -------------------------------------------------
#  Core libraries
# -------------------------------------------------
import torch
import numpy as np

# FastAPI and web
from fastapi import (
    FastAPI,
    File,
    UploadFile,
    Form,
    HTTPException,
    Request,
    status,
    Depends,
)
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

# LangChain
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    Docx2txtLoader,
    TextLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever

# FAISS
try:
    import faiss

    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    faiss = None

# Sentence-Transformers / OpenAI
from sentence_transformers import CrossEncoder, SentenceTransformer
from openai import AsyncOpenAI

# Token counting
import tiktoken

# -------------------------------------------------
#  Logging
# -------------------------------------------------
logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s]: %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("main")

# -------------------------------------------------
#  SIMPLE CONFIG (Fix 1 applied – lower threshold)
# -------------------------------------------------
SIMPLE_CONFIG = {
    "chunk_size": 1_000,
    "chunk_overlap": 200,
    "retrieval_k": 20,
    "rerank_k": 10,
    "final_k": 5,
    "confidence_threshold": 0.15,  # ↓ lowered from 0.30
    "max_tokens": 4_000,
}

# -------------------------------------------------
#  Globals / env
# -------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
HACKRX_TOKEN = "9a1163c13e8927960b857a674794a62c57baf588998981151b0753a4d6d17905"

EMBEDDING_CACHE: Dict[str, np.ndarray] = {}

base_sentence_model = None
embedding_model = None
reranker = None
openai_client: AsyncOpenAI | None = None

# -------------------------------------------------
#  5 CORE CLASSES
# -------------------------------------------------
class DocumentProcessor:
    """Load & split documents"""

    def __init__(self) -> None:
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=SIMPLE_CONFIG["chunk_size"],
            chunk_overlap=SIMPLE_CONFIG["chunk_overlap"],
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", " ", ""],
        )

    async def load_document(self, file_path: str) -> List[Document]:
        try:
            ext = os.path.splitext(file_path)[1].lower()
            if ext == ".pdf":
                loader = PyMuPDFLoader(file_path)
            elif ext in {".docx", ".doc"}:
                loader = Docx2txtLoader(file_path)
            else:
                loader = TextLoader(file_path, encoding="utf-8")
            return await asyncio.to_thread(loader.load)
        except Exception as e:
            logger.error(f"❌ Failed to load {file_path}: {e}")
            raise

    def split_documents(self, docs: List[Document]) -> List[Document]:
        try:
            chunks = self.text_splitter.split_documents(docs)
            return [c for c in chunks if len(c.page_content.strip()) >= 50]
        except Exception as e:
            logger.error(f"❌ Split error: {e}")
            return docs


class VectorRetriever:
    """FAISS + BM25 (+ optional rerank)"""

    def __init__(self) -> None:
        self.faiss_index = None
        self.documents: List[Document] = []
        self.bm25_retriever = None
        self.dimension = 384

    async def initialize(self, documents: List[Document]) -> None:
        self.documents = documents

        # FAISS
        if HAS_FAISS and documents:
            try:
                self.faiss_index = faiss.IndexFlatIP(self.dimension)
                embs = await self._get_embeddings_batch([d.page_content for d in documents])
                arr = np.array(embs, dtype=np.float32)
                faiss.normalize_L2(arr)
                self.faiss_index.add(arr)
                logger.info(f"✅ FAISS initialized with {len(documents)} docs")
            except Exception as e:
                logger.warning(f"⚠️ FAISS setup failed: {e}")

        # BM25
        try:
            self.bm25_retriever = await asyncio.to_thread(
                BM25Retriever.from_documents, documents
            )
            self.bm25_retriever.k = SIMPLE_CONFIG["retrieval_k"]
            logger.info("✅ BM25 retriever initialized")
        except Exception as e:
            logger.warning(f"⚠️ BM25 setup failed: {e}")

    # Fix 4 – better FAISS score handling
    async def retrieve(self, query: str) -> Tuple[List[Document], List[float]]:
        try:
            all_docs, all_scores, seen = [], [], set()

            # Vector search
            if self.faiss_index and base_sentence_model:
                try:
                    q_emb = await self._get_query_embedding(query)
                    q_arr = np.array(q_emb, dtype=np.float32).reshape(1, -1)
                    faiss.normalize_L2(q_arr)
                    k = min(SIMPLE_CONFIG["retrieval_k"], len(self.documents))
                    scores, idxs = self.faiss_index.search(q_arr, k)

                    for score, idx in zip(scores[0], idxs[0]):
                        if 0 <= idx < len(self.documents):
                            doc = self.documents[idx]
                            h = hashlib.md5(doc.page_content.encode()).hexdigest()
                            if h not in seen:
                                seen.add(h)
                                all_docs.append(doc)
                                # normalise inner-product score
                                norm_score = min(1.0, max(0.0, float(score)))
                                all_scores.append(norm_score)
                except Exception as e:
                    logger.warning(f"⚠️ Vector search error: {e}")

            # BM25 search
            if self.bm25_retriever:
                try:
                    bm25_docs = await asyncio.to_thread(self.bm25_retriever.invoke, query)
                    for doc in bm25_docs:
                        h = hashlib.md5(doc.page_content.encode()).hexdigest()
                        if h not in seen:
                            seen.add(h)
                            all_docs.append(doc)
                            all_scores.append(0.6)  # default BM25 score
                except Exception as e:
                    logger.warning(f"⚠️ BM25 error: {e}")

            # Rerank
            if reranker and len(all_docs) > 1:
                return await self._rerank_documents(query, all_docs, all_scores)

            # Top-k
            ranked = sorted(zip(all_docs, all_scores), key=lambda x: x[1], reverse=True)
            docs, scores = zip(*ranked[: SIMPLE_CONFIG["final_k"]]) if ranked else ([], [])
            return list(docs), list(scores)
        except Exception as e:
            logger.error(f"❌ Retrieval error: {e}")
            return self.documents[: SIMPLE_CONFIG["final_k"]], [0.5] * min(
                len(self.documents), SIMPLE_CONFIG["final_k"]
            )

    async def _rerank_documents(
        self, query: str, docs: List[Document], scores: List[float]
    ) -> Tuple[List[Document], List[float]]:
        try:
            pairs = [[query, d.page_content[:512]] for d in docs[: SIMPLE_CONFIG["rerank_k"]]]
            rerank_scores = await asyncio.to_thread(reranker.predict, pairs)
            rerank_norm = [(s + 1) / 2 for s in rerank_scores]

            comb = [
                0.7 * r + 0.3 * o for r, o in zip(rerank_norm, scores[: len(rerank_norm)])
            ]
            ranked = sorted(
                zip(docs[: len(comb)], comb), key=lambda x: x[1], reverse=True
            )
            docs_final, scores_final = zip(*ranked[: SIMPLE_CONFIG["final_k"]])
            return list(docs_final), list(scores_final)
        except Exception as e:
            logger.error(f"❌ Rerank error: {e}")
            return docs[: SIMPLE_CONFIG["final_k"]], scores[: SIMPLE_CONFIG["final_k"]]

    async def _get_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        results = []
        for text in texts:
            h = hashlib.md5(text.encode()).hexdigest()
            if h in EMBEDDING_CACHE:
                results.append(EMBEDDING_CACHE[h])
            else:
                if base_sentence_model:
                    emb = await asyncio.to_thread(
                        base_sentence_model.encode, text, convert_to_numpy=True
                    )
                    EMBEDDING_CACHE[h] = emb
                    results.append(emb)
                else:
                    results.append(np.zeros(self.dimension))
        return results

    async def _get_query_embedding(self, query: str) -> np.ndarray:
        h = hashlib.md5(query.encode()).hexdigest()
        if h in EMBEDDING_CACHE:
            return EMBEDDING_CACHE[h]
        if base_sentence_model:
            emb = await asyncio.to_thread(
                base_sentence_model.encode, query, convert_to_numpy=True
            )
            EMBEDDING_CACHE[h] = emb
            return emb
        return np.zeros(self.dimension)


class ResponseGenerator:
    """Build context & call OpenAI"""

    def __init__(self) -> None:
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.tokenizer = None

    def estimate_tokens(self, text: str) -> int:
        if not text:
            return 0
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception:
                pass
        return max(1, len(text) // 4)

    def build_context(self, docs: List[Document], query: str) -> str:
        if not docs:
            return ""
        parts, budget = [], SIMPLE_CONFIG["max_tokens"] - 500
        for doc in docs:
            dtok = self.estimate_tokens(doc.page_content)
            if dtok <= budget:
                parts.append(doc.page_content)
                budget -= dtok
            elif budget > 200:
                max_chars = budget * 4
                parts.append(doc.page_content[:max_chars] + " …")
                break
        return "\n\n".join(parts)

    async def generate_response(self, query: str, context: str) -> str:
        try:
            if not openai_client:
                return "System is still initializing. Please try again."

            msgs = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful document analyst. Answer using only the "
                        "information provided in the context."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {query}",
                },
            ]
            resp = await openai_client.chat.completions.create(
                messages=msgs, model="gpt-4o", temperature=0.1, max_tokens=1_000
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"❌ OpenAI error: {e}")
            return f"I encountered an error while generating the answer: {e}"


class ConfidenceScorer:
    """Compute answer-worthiness (Fix 2 applied)"""

    def calculate_confidence(
        self, query: str, similarity_scores: List[float], docs: List[Document]
    ) -> float:
        if not similarity_scores:
            return 0.0
        try:
            arr = np.array(similarity_scores, dtype=np.float32)

            # normalise FAISS inner-product (>1.0)
            if np.max(arr) > 1.0:
                arr /= np.max(arr)

            arr = np.clip(arr, 0.0, 1.0)

            max_s, avg_s = float(np.max(arr)), float(np.mean(arr))

            # query-doc term overlap
            q_terms = set(query.lower().split())
            overlaps = []
            for doc in docs[:3]:
                d_terms = set(doc.page_content.lower().split())
                overlaps.append(len(q_terms & d_terms) / max(len(q_terms), 1))
            q_match = float(np.mean(overlaps)) if overlaps else 0.0

            confidence = 0.5 * max_s + 0.3 * avg_s + 0.2 * q_match

            # boost for any 3+-char term appearing verbatim
            if docs:
                if any(
                    any(t in doc.page_content.lower() for t in q_terms if len(t) > 3)
                    for doc in docs[:2]
                ):
                    confidence += 0.1

            if max_s > 0.1 and q_match > 0.1:
                confidence = max(confidence, 0.2)

            return min(1.0, max(0.0, confidence))
        except Exception as e:
            logger.warning(f"⚠️ Confidence calc error: {e}")
            return 0.2


class APIService:
    """Compose the 4 helper classes"""

    def __init__(self) -> None:
        self.doc_proc = DocumentProcessor()
        self.retriever = VectorRetriever()
        self.resp_gen = ResponseGenerator()
        self.scorer = ConfidenceScorer()


# -------------------------------------------------
#  Auth
# -------------------------------------------------
security = HTTPBearer(auto_error=False)


async def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    if not credentials or credentials.credentials != HACKRX_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials


# -------------------------------------------------
#  Model init (unchanged, kept for completeness)
# -------------------------------------------------
async def ensure_models_ready() -> None:
    global base_sentence_model, embedding_model, reranker, openai_client

    if base_sentence_model is None:
        try:
            base_sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("✅ SentenceTransformer loaded (cached)")
        except Exception as e:
            logger.warning(f"⚠️ SentenceTransformer load failed: {e}")

    if embedding_model is None:
        try:
            embedding_model = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"}
            )
            logger.info("✅ Embedding model loaded (cached)")
        except Exception as e:
            logger.warning(f"⚠️ Embedding model load failed: {e}")

    if reranker is None:
        try:
            reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            logger.info("✅ CrossEncoder loaded (cached)")
        except Exception as e:
            logger.warning(f"⚠️ CrossEncoder load failed: {e}")

    if openai_client is None and OPENAI_API_KEY:
        try:
            openai_client = AsyncOpenAI(
                api_key=OPENAI_API_KEY,
                timeout=httpx.Timeout(connect=10, read=60, write=30, pool=5),
                max_retries=3,
            )
            # quick ping
            await openai_client.chat.completions.create(
                messages=[{"role": "user", "content": "ping"}],
                model="gpt-4o",
                max_tokens=2,
            )
            logger.info("✅ OpenAI client ready")
        except Exception as e:
            logger.error(f"❌ OpenAI init failed: {e}")


# -------------------------------------------------
#  FastAPI app
# -------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    await ensure_models_ready()
    yield
    EMBEDDING_CACHE.clear()


app = FastAPI(
    title="Simplified RAG System",
    description="Lean RAG with FAISS + BM25",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
#  Pydantic models
# -------------------------------------------------
class HackRxRequest(BaseModel):
    documents: str
    questions: List[str]


class HackRxResponse(BaseModel):
    answers: List[str]


# -------------------------------------------------
#  Endpoints
# -------------------------------------------------
@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "Simplified RAG System",
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/health")
async def health():
    try:
        await ensure_models_ready()
    except Exception:
        pass
    return {
        "status": "healthy",
        "models_loaded": {
            "sentence_transformer": base_sentence_model is not None,
            "embedding_model": embedding_model is not None,
            "reranker": reranker is not None,
            "openai_client": openai_client is not None,
            "faiss": HAS_FAISS,
        },
        "cache_size": len(EMBEDDING_CACHE),
        "config": SIMPLE_CONFIG,
    }


# Fix 3 – debug logging & new threshold logic
@app.post("/hackrx/run", response_model=HackRxResponse)
async def hackrx_run_endpoint(request: HackRxRequest):
    """Stateless document QA for HackRx"""
    try:
        await ensure_models_ready()
        api = APIService()

        # download doc
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.get(request.documents)
            resp.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(resp.content)
            tmp_path = f.name

        try:
            docs = await api.doc_proc.load_document(tmp_path)
            chunks = api.doc_proc.split_documents(docs)
            if not chunks:
                raise HTTPException(status_code=400, detail="No content extracted")

            await api.retriever.initialize(chunks)

            answers = []
            for q in request.questions:
                try:
                    retrieved, scores = await api.retriever.retrieve(q)
                    logger.info(f"📊 Question: {q}")
                    logger.info(f"📊 Retrieved {len(retrieved)} docs, scores: {scores[:3]}")

                    conf = api.scorer.calculate_confidence(q, scores, retrieved)
                    logger.info(
                        f"📊 Confidence: {conf:.3f} (threshold {SIMPLE_CONFIG['confidence_threshold']})"
                    )

                    if conf >= SIMPLE_CONFIG["confidence_threshold"]:
                        ctx = api.resp_gen.build_context(retrieved, q)
                        logger.info(f"📊 Context length: {len(ctx)} chars")
                        ans = await api.resp_gen.generate_response(q, ctx)
                    else:
                        logger.warning(f"⚠️ Low confidence {conf:.3f}")
                        ans = (
                            "I don't have enough relevant information to answer this "
                            "question accurately."
                        )
                    answers.append(ans)
                except Exception as e:
                    logger.error(f"❌ Error for question '{q}': {e}")
                    answers.append(f"Error processing question: {e}")

            return HackRxResponse(answers=answers)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    except Exception as e:
        logger.error(f"❌ /hackrx/run error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}") from e


# -------------------------------------------------
#  Error handlers
# -------------------------------------------------
@app.exception_handler(HTTPException)
async def http_exc_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "detail": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat(),
        },
    )


@app.exception_handler(Exception)
async def generic_exc_handler(request: Request, exc: Exception):
    logger.error(f"❌ Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "detail": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.now().isoformat(),
        },
    )


# -------------------------------------------------
#  Entrypoint
# -------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    logger.info(f"🚀 Starting server on port {port}")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False, log_level="info")
