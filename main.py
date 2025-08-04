import os

# Performance optimization
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

import sys
import asyncio
import logging
import time
from typing import List, Dict, Any, Tuple, Optional
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

# Core libraries
import torch
import numpy as np

# FastAPI and web
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, HttpUrl
import httpx

# Document processing
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever

# FAISS for vector storage
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    faiss = None

# AI and embeddings
from sentence_transformers import CrossEncoder, SentenceTransformer
import openai
from openai import AsyncOpenAI

# Token counting
import tiktoken

# Configure logging
logging.basicConfig(
    format='%(asctime)s %(levelname)s [%(name)s]: %(message)s',
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ================================
# SIMPLIFIED CONFIGURATION
# ================================

SIMPLE_CONFIG = {
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "retrieval_k": 20,
    "rerank_k": 10,
    "final_k": 5,
    "confidence_threshold": 0.3,
    "max_tokens": 4000,
}

# Environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
HACKRX_TOKEN = "9a1163c13e8927960b857a674794a62c57baf588998981151b0753a4d6d17905"

# Simple in-memory cache for embeddings only
EMBEDDING_CACHE = {}

# Global variables
embedding_model = None
base_sentence_model = None
reranker = None
openai_client = None

# ================================
# CORE CLASSES (SIMPLIFIED TO 5)
# ================================

class DocumentProcessor:
    """Simple document loading and splitting"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=SIMPLE_CONFIG["chunk_size"],
            chunk_overlap=SIMPLE_CONFIG["chunk_overlap"],
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", " ", ""]
        )
    
    async def load_document(self, file_path: str) -> List[Document]:
        """Load document with fallback handling"""
        try:
            file_extension = os.path.splitext(file_path)[1].lower()
            
            if file_extension == '.pdf':
                loader = PyMuPDFLoader(file_path)
            elif file_extension in ['.docx', '.doc']:
                loader = Docx2txtLoader(file_path)
            else:
                loader = TextLoader(file_path, encoding='utf-8')
            
            docs = await asyncio.to_thread(loader.load)
            return docs
            
        except Exception as e:
            logger.error(f"❌ Failed to load {file_path}: {e}")
            raise
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        try:
            chunks = self.text_splitter.split_documents(documents)
            # Filter very short chunks
            return [chunk for chunk in chunks if len(chunk.page_content.strip()) >= 50]
        except Exception as e:
            logger.error(f"❌ Error splitting documents: {e}")
            return documents

class VectorRetriever:
    """FAISS + BM25 + reranking retrieval"""
    
    def __init__(self):
        self.faiss_index = None
        self.documents = []
        self.bm25_retriever = None
        self.dimension = 384
    
    async def initialize(self, documents: List[Document]):
        """Initialize vector store and BM25"""
        self.documents = documents
        
        # Setup FAISS
        if HAS_FAISS and documents:
            try:
                self.faiss_index = faiss.IndexFlatIP(self.dimension)
                
                # Get embeddings
                doc_texts = [doc.page_content for doc in documents]
                embeddings = await self._get_embeddings_batch(doc_texts)
                
                # Add to FAISS
                embeddings_array = np.array(embeddings, dtype=np.float32)
                faiss.normalize_L2(embeddings_array)
                self.faiss_index.add(embeddings_array)
                
                logger.info(f"✅ FAISS initialized with {len(documents)} documents")
            except Exception as e:
                logger.warning(f"⚠️ FAISS setup failed: {e}")
        
        # Setup BM25
        try:
            self.bm25_retriever = await asyncio.to_thread(
                BM25Retriever.from_documents, documents
            )
            self.bm25_retriever.k = SIMPLE_CONFIG["retrieval_k"]
            logger.info("✅ BM25 retriever initialized")
        except Exception as e:
            logger.warning(f"⚠️ BM25 setup failed: {e}")
    
    async def retrieve(self, query: str) -> Tuple[List[Document], List[float]]:
        """Retrieve documents using hybrid approach"""
        try:
            all_docs = []
            all_scores = []
            seen_content = set()
            
            # Vector search
            if self.faiss_index and base_sentence_model:
                try:
                    query_embedding = await self._get_query_embedding(query)
                    query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
                    faiss.normalize_L2(query_embedding)
                    
                    k = min(SIMPLE_CONFIG["retrieval_k"], len(self.documents))
                    scores, indices = self.faiss_index.search(query_embedding, k)
                    
                    for score, idx in zip(scores[0], indices[0]):
                        if idx >= 0 and idx < len(self.documents):
                            doc = self.documents[idx]
                            content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
                            if content_hash not in seen_content:
                                all_docs.append(doc)
                                all_scores.append(float(score))
                                seen_content.add(content_hash)
                except Exception as e:
                    logger.warning(f"⚠️ Vector search error: {e}")
            
            # BM25 search
            if self.bm25_retriever:
                try:
                    bm25_docs = await asyncio.to_thread(self.bm25_retriever.invoke, query)
                    for doc in bm25_docs:
                        content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
                        if content_hash not in seen_content:
                            all_docs.append(doc)
                            all_scores.append(0.6)  # Default BM25 score
                            seen_content.add(content_hash)
                except Exception as e:
                    logger.warning(f"⚠️ BM25 search error: {e}")
            
            # Rerank if available
            if reranker and len(all_docs) > 1:
                return await self._rerank_documents(query, all_docs, all_scores)
            
            # Sort by score and return top-k
            scored_docs = list(zip(all_docs, all_scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            final_docs = [doc for doc, _ in scored_docs[:SIMPLE_CONFIG["final_k"]]]
            final_scores = [score for _, score in scored_docs[:SIMPLE_CONFIG["final_k"]]]
            
            return final_docs, final_scores
            
        except Exception as e:
            logger.error(f"❌ Retrieval error: {e}")
            return self.documents[:SIMPLE_CONFIG["final_k"]], [0.5] * min(len(self.documents), SIMPLE_CONFIG["final_k"])
    
    async def _rerank_documents(self, query: str, documents: List[Document], scores: List[float]) -> Tuple[List[Document], List[float]]:
        """Rerank documents using cross-encoder"""
        try:
            pairs = [[query, doc.page_content[:512]] for doc in documents[:SIMPLE_CONFIG["rerank_k"]]]
            rerank_scores = await asyncio.to_thread(reranker.predict, pairs)
            
            # Normalize scores
            normalized_scores = [(score + 1) / 2 for score in rerank_scores]
            
            # Combine with original scores
            combined_scores = []
            for i, (orig_score, rerank_score) in enumerate(zip(scores[:len(normalized_scores)], normalized_scores)):
                combined = 0.7 * rerank_score + 0.3 * orig_score
                combined_scores.append(combined)
            
            # Sort and return top-k
            scored_docs = list(zip(documents[:len(combined_scores)], combined_scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            final_docs = [doc for doc, _ in scored_docs[:SIMPLE_CONFIG["final_k"]]]
            final_scores = [score for _, score in scored_docs[:SIMPLE_CONFIG["final_k"]]]
            
            return final_docs, final_scores
            
        except Exception as e:
            logger.error(f"❌ Reranking error: {e}")
            return documents[:SIMPLE_CONFIG["final_k"]], scores[:SIMPLE_CONFIG["final_k"]]
    
    async def _get_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Get embeddings for batch of texts"""
        results = []
        for text in texts:
            text_hash = hashlib.md5(text.encode()).hexdigest()
            if text_hash in EMBEDDING_CACHE:
                results.append(EMBEDDING_CACHE[text_hash])
            else:
                if base_sentence_model:
                    embedding = await asyncio.to_thread(
                        base_sentence_model.encode, text, convert_to_numpy=True
                    )
                    EMBEDDING_CACHE[text_hash] = embedding
                    results.append(embedding)
                else:
                    results.append(np.zeros(self.dimension))
        return results
    
    async def _get_query_embedding(self, query: str) -> np.ndarray:
        """Get embedding for query"""
        query_hash = hashlib.md5(query.encode()).hexdigest()
        if query_hash in EMBEDDING_CACHE:
            return EMBEDDING_CACHE[query_hash]
        
        if base_sentence_model:
            embedding = await asyncio.to_thread(
                base_sentence_model.encode, query, convert_to_numpy=True
            )
            EMBEDDING_CACHE[query_hash] = embedding
            return embedding
        
        return np.zeros(self.dimension)

class ResponseGenerator:
    """Context building and OpenAI response generation"""
    
    def __init__(self):
        self.tokenizer = None
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            pass
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count"""
        if not text:
            return 0
        
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception:
                pass
        
        # Fallback estimation
        return len(text) // 4
    
    def build_context(self, documents: List[Document], query: str) -> str:
        """Build optimized context from documents"""
        if not documents:
            return ""
        
        context_parts = []
        token_budget = SIMPLE_CONFIG["max_tokens"] - 500  # Reserve for prompt
        
        for doc in documents:
            doc_tokens = self.estimate_tokens(doc.page_content)
            if doc_tokens <= token_budget:
                context_parts.append(doc.page_content)
                token_budget -= doc_tokens
            elif token_budget > 200:
                # Truncate document
                max_chars = token_budget * 4
                truncated = doc.page_content[:max_chars] + "..."
                context_parts.append(truncated)
                break
        
        return "\n\n".join(context_parts)
    
    async def generate_response(self, query: str, context: str) -> str:
        """Generate response using OpenAI"""
        try:
            if not openai_client:
                return "System is still initializing. Please try again."
            
            system_prompt = """You are a helpful document analyst. Provide accurate, comprehensive answers based on the provided context.

INSTRUCTIONS:
1. Answer questions directly based on the context provided
2. If information is not available in the context, clearly state this
3. Be concise but comprehensive
4. Cite specific details from the context when relevant
5. Maintain accuracy and avoid speculation"""
            
            user_message = f"""Context:
{context}

Question: {query}

Please provide a comprehensive answer based on the context above."""
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
            
            response = await openai_client.chat.completions.create(
                messages=messages,
                model="gpt-4o",
                temperature=0.1,
                max_tokens=1000
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"❌ Response generation error: {e}")
            return f"I apologize, but I encountered an error while processing your query: {str(e)}"

class ConfidenceScorer:
    """Simple confidence scoring"""
    
    def calculate_confidence(self, query: str, similarity_scores: List[float], documents: List[Document]) -> float:
        """Calculate confidence score"""
        if not similarity_scores:
            return 0.0
        
        try:
            scores_array = np.array(similarity_scores)
            scores_array = np.clip(scores_array, 0.0, 1.0)
            
            max_score = np.max(scores_array)
            avg_score = np.mean(scores_array)
            
            # Query-document match
            query_terms = set(query.lower().split())
            doc_matches = []
            
            for doc in documents[:3]:
                doc_terms = set(doc.page_content.lower().split())
                overlap = len(query_terms.intersection(doc_terms)) / max(len(query_terms), 1)
                doc_matches.append(overlap)
            
            query_match = np.mean(doc_matches) if doc_matches else 0.0
            
            # Combined confidence
            confidence = 0.4 * max_score + 0.3 * avg_score + 0.3 * query_match
            
            # Boost for exact matches
            query_lower = query.lower()
            for doc in documents[:2]:
                if query_lower in doc.page_content.lower():
                    confidence += 0.1
                    break
            
            return min(1.0, max(0.0, confidence))
            
        except Exception as e:
            logger.warning(f"⚠️ Confidence calculation error: {e}")
            return 0.3

class APIService:
    """FastAPI endpoints and request handling"""
    
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.vector_retriever = VectorRetriever()
        self.response_generator = ResponseGenerator()
        self.confidence_scorer = ConfidenceScorer()

# ================================
# AUTHENTICATION
# ================================

security = HTTPBearer(auto_error=False)

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify Bearer token"""
    if not credentials or credentials.credentials != HACKRX_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

# ================================
# MODEL INITIALIZATION
# ================================

# ================================
# MODEL INITIALIZATION (Updated for Docker pre-loading compatibility)
# ================================

async def ensure_models_ready():
    """Load pre-downloaded models quickly"""
    global base_sentence_model, embedding_model, reranker, openai_client
    
    if base_sentence_model is None:
        try:
            # Models are pre-downloaded in Docker, this should be fast
            base_sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ Sentence transformer loaded from cache")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load sentence transformer: {e}")
    
    if embedding_model is None:
        try:
            embedding_model = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            logger.info("✅ Embedding model loaded from cache")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load embedding model: {e}")
    
    if reranker is None:
        try:
            reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            logger.info("✅ Reranker loaded from cache")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load reranker: {e}")
    
    if openai_client is None and OPENAI_API_KEY:
        try:
            openai_client = AsyncOpenAI(
                api_key=OPENAI_API_KEY,
                timeout=httpx.Timeout(
                    connect=10.0,
                    read=60.0,
                    write=30.0,
                    pool=5.0
                ),
                max_retries=3
            )
            # Quick connection test
            await openai_client.chat.completions.create(
                messages=[{"role": "user", "content": "Hello"}],
                model="gpt-4o",
                max_tokens=5
            )
            logger.info("✅ OpenAI client initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize OpenAI: {e}")


# ================================
# FASTAPI APPLICATION
# ================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan"""
    await ensure_models_ready()
    yield
    # Cleanup
    EMBEDDING_CACHE.clear()

app = FastAPI(
    title="Simplified RAG System",
    description="Streamlined Retrieval-Augmented Generation System",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================================
# PYDANTIC MODELS
# ================================

class HackRxRequest(BaseModel):
    documents: str
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]

# ================================
# API ENDPOINTS
# ================================

@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "online",
        "service": "Simplified RAG System",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    # Try to ensure models are loaded
    try:
        await ensure_models_ready()
    except Exception:
        pass  # Don't fail health check if models aren't ready yet
    
    return {
        "status": "healthy",
        "models_loaded": {
            "sentence_transformer": base_sentence_model is not None,
            "embedding_model": embedding_model is not None,
            "reranker": reranker is not None,
            "openai_client": openai_client is not None,
            "faiss_available": HAS_FAISS
        },
        "cache_size": len(EMBEDDING_CACHE),
        "config": SIMPLE_CONFIG
    }


@app.post("/hackrx/run", response_model=HackRxResponse)
async def hackrx_run_endpoint(request: HackRxRequest):
    """HackRx endpoint - stateless document processing"""
    try:
        # Ensure models are ready
        await ensure_models_ready()
        
        # Initialize API service
        api_service = APIService()
        
        # Download document
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(request.documents)
            response.raise_for_status()
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(response.content)
            temp_file_path = tmp_file.name
        
        try:
            # Load and process document
            documents = await api_service.document_processor.load_document(temp_file_path)
            chunks = api_service.document_processor.split_documents(documents)
            
            if not chunks:
                raise HTTPException(status_code=400, detail="No content could be extracted")
            
            # Initialize retriever
            await api_service.vector_retriever.initialize(chunks)
            
            # Process all questions
            answers = []
            for question in request.questions:
                try:
                    # Retrieve relevant documents
                    retrieved_docs, similarity_scores = await api_service.vector_retriever.retrieve(question)
                    
                    # Calculate confidence
                    confidence = api_service.confidence_scorer.calculate_confidence(
                        question, similarity_scores, retrieved_docs
                    )
                    
                    # Generate response only if confidence is sufficient
                    if confidence >= SIMPLE_CONFIG["confidence_threshold"]:
                        context = api_service.response_generator.build_context(retrieved_docs, question)
                        answer = await api_service.response_generator.generate_response(question, context)
                    else:
                        answer = "I don't have enough relevant information to answer this question accurately."
                    
                    answers.append(answer)
                    
                except Exception as e:
                    logger.error(f"❌ Error processing question '{question}': {e}")
                    answers.append(f"Error processing question: {str(e)}")
            
            return HackRxResponse(answers=answers)
            
        finally:
            # Cleanup temp file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                
    except Exception as e:
        logger.error(f"❌ HackRx endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

# ================================
# ERROR HANDLERS
# ================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "detail": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"❌ Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "detail": "Internal server error occurred",
            "status_code": 500,
            "timestamp": datetime.now().isoformat()
        }
    )

# ================================
# MAIN ENTRY POINT
# ================================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    logger.info(f"🚀 Starting Simplified RAG System on port {port}...")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )
