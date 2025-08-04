import os
import sys
import asyncio
import logging
import time
import json
import tempfile
import hashlib
import re
import threading
from typing import List, Dict, Any, Tuple, Optional
from contextlib import asynccontextmanager
from datetime import datetime
from functools import lru_cache
from collections import defaultdict
from urllib.parse import urlparse

# Performance optimization
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

# Core libraries
import torch
import numpy as np

# FastAPI and web
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

# Document processing
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
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
from openai import AsyncOpenAI

# Token counting
import tiktoken

# Optional imports with fallbacks
try:
    import magic
    HAS_MAGIC = True
except ImportError:
    HAS_MAGIC = False
    magic = None

try:
    import cachetools
    HAS_CACHETOOLS = True
except ImportError:
    HAS_CACHETOOLS = False
    cachetools = None

# Configure logging
logging.basicConfig(
    format='%(asctime)s %(levelname)s [%(name)s]: %(message)s',
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ================================
# CONFIGURATION
# ================================
HACKRX_TOKEN = "9a1163c13e8927960b857a674794a62c57baf588998981151b0753a4d6d17905"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
SEMANTIC_SEARCH_K = 24
CONTEXT_DOCS = 18
CONFIDENCE_THRESHOLD = 0.15
RERANK_TOP_K = 35
MMR_LAMBDA = 0.75
MAX_FILE_SIZE_MB = 100
QUESTION_TIMEOUT = 25.0
OPTIMAL_BATCH_SIZE = 16
MAX_CONCURRENT_BATCHES = 4
PARALLEL_EMBEDDING_ENABLED = True
SUPPORTED_EXTENSIONS = ['.pdf', '.docx', '.doc', '.txt', '.md', '.csv']
SUPPORTED_URL_SCHEMES = ['http', 'https', 'blob', 'drive', 'dropbox']

DOMAIN_KEYWORDS = {
    "insurance": [
        'policy', 'premium', 'claim', 'coverage', 'benefit', 'exclusion', 'deductible',
        'co-payment', 'policyholder', 'insured', 'underwriting', 'actuary', 'risk assessment'
    ],
    "legal": [
        'contract', 'agreement', 'clause', 'statute', 'regulation', 'compliance', 'litigation',
        'jurisdiction', 'liability', 'court', 'legal', 'law', 'attorney', 'counsel'
    ],
    "medical": [
        'patient', 'diagnosis', 'treatment', 'clinical', 'medical', 'healthcare', 'physician',
        'hospital', 'therapy', 'medication', 'symptoms', 'disease', 'procedure'
    ],
    "financial": [
        'investment', 'portfolio', 'revenue', 'profit', 'financial', 'accounting', 'audit',
        'balance', 'asset', 'liability', 'equity', 'cash flow', 'budget'
    ],
    "technical": [
        'system', 'software', 'hardware', 'network', 'database', 'API', 'configuration',
        'deployment', 'architecture', 'infrastructure', 'technical', 'specification'
    ],
    "academic": [
        'research', 'study', 'analysis', 'methodology', 'hypothesis', 'experiment',
        'data', 'results', 'conclusion', 'literature', 'citation', 'peer review'
    ],
    "business": [
        'strategy', 'management', 'operations', 'marketing', 'sales', 'customer',
        'business', 'corporate', 'organization', 'project', 'team', 'leadership'
    ]
}

base_sentence_model = None
reranker = None
openai_client = None

# ================================
# ENHANCED CACHING SYSTEM
# ================================
class SmartCacheManager:
    """Smart cache manager with TTL/LRU primary and dict fallback"""
    def __init__(self):
        try:
            if HAS_CACHETOOLS:
                self.embedding_cache = cachetools.TTLCache(maxsize=10000, ttl=86400)
                self.document_chunk_cache = cachetools.LRUCache(maxsize=500)
                self.domain_cache = cachetools.LRUCache(maxsize=1000)
                self.primary_available = True
                logger.info("✅ Advanced caching with TTL/LRU enabled")
            else:
                raise ImportError("cachetools not available")
        except ImportError:
            self.embedding_cache = {}
            self.document_chunk_cache = {}
            self.domain_cache = {}
            self.primary_available = False
            logger.info("📦 Using dict fallback caching (cachetools not available)")
        self._lock = threading.RLock()

    def clear_all_caches(self):
        """Clear ALL caches when new document is uploaded"""
        with self._lock:
            self.embedding_cache.clear()
            self.document_chunk_cache.clear()
            self.domain_cache.clear()
            logger.info("🧹 All caches cleared for new document upload")

    def get_embedding(self, text_hash: str) -> Optional[Any]:
        with self._lock:
            return self.embedding_cache.get(text_hash)

    def set_embedding(self, text_hash: str, embedding: Any):
        with self._lock:
            self.embedding_cache[text_hash] = embedding

    def get_document_chunks(self, cache_key: str) -> Optional[Any]:
        with self._lock:
            return self.document_chunk_cache.get(cache_key)

    def set_document_chunks(self, cache_key: str, chunks: Any):
        with self._lock:
            self.document_chunk_cache[cache_key] = chunks

    def get_domain_result(self, cache_key: str) -> Optional[Any]:
        with self._lock:
            return self.domain_cache.get(cache_key)

    def set_domain_result(self, cache_key: str, result: Any):
        with self._lock:
            self.domain_cache[cache_key] = result

    def cleanup_if_needed(self):
        """Only needed for dict fallback"""
        if not self.primary_available:
            with self._lock:
                if len(self.embedding_cache) > 10000:
                    items = list(self.embedding_cache.items())[-5000:]
                    self.embedding_cache.clear()
                    self.embedding_cache.update(items)
                if len(self.document_chunk_cache) > 500:
                    items = list(self.document_chunk_cache.items())[-250:]
                    self.document_chunk_cache.clear()
                    self.document_chunk_cache.update(items)
                if len(self.domain_cache) > 1000:
                    items = list(self.domain_cache.items())[-500:]
                    self.domain_cache.clear()
                    self.domain_cache.update(items)

    def get_cache_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "embedding_cache_size": len(self.embedding_cache),
                "document_chunk_cache_size": len(self.document_chunk_cache),
                "domain_cache_size": len(self.domain_cache),
                "primary_cache_available": self.primary_available,
                "cache_type": "TTLCache/LRUCache" if self.primary_available else "dict_fallback"
            }

CACHE_MANAGER = SmartCacheManager()

# ================================
# SIMPLE AUTHENTICATION
# ================================
def simple_auth_check(request: Request) -> bool:
    """Simple authentication check"""
    auth = request.headers.get("Authorization", "")
    expected = f"Bearer {HACKRX_TOKEN}"
    return auth == expected

# ================================
# UTILITY FUNCTIONS
# ================================
def sanitize_pii(text: str) -> str:
    """Remove PII patterns from text"""
    patterns = [
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        r'\b\d{3}-\d{3}-\d{4}\b',
        r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'
    ]
    sanitized = text
    for pattern in patterns:
        sanitized = re.sub(pattern, '[REDACTED]', sanitized)
    return sanitized

def validate_url_scheme(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme.lower() in SUPPORTED_URL_SCHEMES

def validate_file_extension(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext in SUPPORTED_EXTENSIONS

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if hasattr(obj, 'dtype'):
        if obj.dtype == bool:
            return bool(obj)
        elif obj.dtype in ['int32', 'int64']:
            return int(obj)
        elif obj.dtype in ['float32', 'float64']:
            return float(obj)
    return obj

def sanitize_for_json(data):
    """Recursively sanitize data for JSON serialization"""
    if isinstance(data, dict):
        return {k: sanitize_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_for_json(v) for v in data]
    elif isinstance(data, (np.integer, np.floating, np.bool_)):
        return convert_numpy_types(data)
    elif hasattr(data, 'item'):
        return data.item()
    elif isinstance(data, (np.ndarray,)):
        return data.tolist()
    return data

# ================================
# ENHANCED UNIFIED LOADER
# ================================
class UnifiedLoader:
    """Unified document loader with URL support"""
    def __init__(self):
        self.mime_detector = magic if HAS_MAGIC else None
        self.drive_patterns = [
            r'drive\.google\.com/file/d/([a-zA-Z0-9-_]+)',
            r'docs\.google\.com'
        ]
        self.dropbox_patterns = [
            r'dropbox\.com/s/([a-zA-Z0-9]+)',
            r'dropbox\.com/sh/([a-zA-Z0-9]+)'
        ]

    async def load_document(self, source: str) -> List[Document]:
        """Universal document loader"""
        try:
            if self._is_url(source):
                docs = await self._load_from_url(source)
            else:
                docs = await self._load_from_file(source)
            for doc in docs:
                doc.metadata.update({
                    'source': source,
                    'load_time': time.time(),
                    'loader_version': '2.0'
                })
            logger.info(f"✅ Loaded {len(docs)} documents from {sanitize_pii(source)}")
            return docs
        except Exception as e:
            logger.error(f"❌ Failed to load {sanitize_pii(source)}: {e}")
            raise

    def _is_url(self, source: str) -> bool:
        return source.startswith(('http://', 'https://', 'blob:', 'drive:', 'dropbox:'))

    async def _load_from_url(self, url: str) -> List[Document]:
        if not validate_url_scheme(url):
            raise ValueError(f"Unsupported URL scheme: {urlparse(url).scheme}")
        download_url = self._transform_special_url(url)
        timeout = 30.0
        if any(pattern in url for pattern in ['drive.google.com', 'dropbox.com']):
            timeout = 60.0
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(download_url)
            response.raise_for_status()
            content = response.content
        file_ext = self._get_extension_from_url(url) or self._detect_extension_from_content(content)
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            tmp_file.write(content)
            temp_path = tmp_file.name
        try:
            return await self._load_from_file(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def _transform_special_url(self, url: str) -> str:
        for pattern in self.drive_patterns:
            match = re.search(pattern, url)
            if match:
                file_id = match.group(1) if match.groups() else None
                if file_id:
                    return f"https://drive.google.com/uc?export=download&id={file_id}"
        for pattern in self.dropbox_patterns:
            if re.search(pattern, url):
                return url.replace('dropbox.com', 'dl.dropboxusercontent.com').replace('?dl=0', '')
        return url

    def _get_extension_from_url(self, url: str) -> Optional[str]:
        parsed = urlparse(url)
        path = parsed.path
        return os.path.splitext(path)[1] if path else None

    def _detect_extension_from_content(self, content: bytes) -> str:
        if self.mime_detector:
            try:
                mime_type = magic.from_buffer(content, mime=True)
                mime_to_ext = {
                    'application/pdf': '.pdf',
                    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
                    'application/msword': '.doc',
                    'text/plain': '.txt',
                    'text/csv': '.csv'
                }
                return mime_to_ext.get(mime_type, '.txt')
            except Exception:
                pass
        if content.startswith(b'%PDF'):
            return '.pdf'
        elif b'PK' in content[:10]:
            return '.docx'
        return '.txt'

    async def _load_from_file(self, file_path: str) -> List[Document]:
        file_extension = os.path.splitext(file_path)[1].lower()
        file_size = os.path.getsize(file_path)
        if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
            raise ValueError(f"File too large: {file_size / (1024*1024):.1f}MB (max: {MAX_FILE_SIZE_MB}MB)")
        if not validate_file_extension(file_path):
            raise ValueError(f"Unsupported file extension: {file_extension}")
        logger.info(f"📄 Loading {file_extension} file ({file_size} bytes): {sanitize_pii(file_path)}")
        mime_type = None
        if self.mime_detector:
            try:
                mime_type = magic.from_file(file_path, mime=True)
            except Exception as e:
                logger.warning(f"⚠️ MIME detection failed: {e}")
        docs = None
        loader_used = None
        if mime_type == 'application/pdf' or file_extension == '.pdf':
            try:
                loader = PyMuPDFLoader(file_path)
                docs = await asyncio.to_thread(loader.load)
                loader_used = "PyMuPDFLoader"
            except Exception as e:
                logger.warning(f"⚠️ PyMuPDF failed: {e}")
        elif ('word' in (mime_type or '') or
              'officedocument' in (mime_type or '') or
              file_extension in ['.docx', '.doc']):
            try:
                loader = Docx2txtLoader(file_path)
                docs = await asyncio.to_thread(loader.load)
                loader_used = "Docx2txtLoader"
            except Exception as e:
                logger.warning(f"⚠️ DOCX loader failed: {e}")
        elif ('text' in (mime_type or '') or
              file_extension in ['.txt', '.md', '.csv', '.log']):
            for encoding in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']:
                try:
                    loader = TextLoader(file_path, encoding=encoding)
                    docs = await asyncio.to_thread(loader.load)
                    loader_used = f"TextLoader ({encoding})"
                    break
                except (UnicodeDecodeError, Exception) as e:
                    logger.warning(f"⚠️ Text loader failed with {encoding}: {e}")
                    continue
        if not docs:
            try:
                loader = TextLoader(file_path, encoding='utf-8')
                docs = await asyncio.to_thread(loader.load)
                loader_used = "TextLoader (fallback)"
            except Exception as e:
                logger.error(f"❌ All loaders failed: {e}")
                raise ValueError(f"Could not load file {file_path}: {str(e)}")
        if not docs:
            raise ValueError(f"No content extracted from {file_path}")
        for doc in docs:
            doc.metadata.update({
                'file_size': file_size,
                'file_extension': file_extension,
                'mime_type': mime_type,
                'loader_used': loader_used
            })
        logger.info(f"✅ Loaded {len(docs)} documents using {loader_used}")
        return docs

# ================================
# ADAPTIVE TEXT SPLITTER
# ================================
class AdaptiveTextSplitter:
    """Adaptive text splitter with smart caching"""
    def __init__(self):
        self.separators = [
            "\n\n## ", "\n\n# ", "\n\n### ", "\n\n#### ",
            "\n\n", "\n", ". ", "! ", "? ", "; ", " ", ""
        ]

    def split_documents(self, documents: List[Document], detected_domain: str = "general") -> List[Document]:
        if not documents:
            return []
        content_hash = self._calculate_content_hash(documents)
        cache_key = f"chunks_{content_hash}_{detected_domain}"
        cached_chunks = CACHE_MANAGER.get_document_chunks(cache_key)
        if cached_chunks is not None:
            logger.info(f"📄 Using cached chunks: {len(cached_chunks)} chunks")
            return cached_chunks
        chunk_size, chunk_overlap = self._adapt_for_content(detected_domain)
        all_chunks = []
        for doc in documents:
            try:
                chunks = self._split_document(doc, chunk_size, chunk_overlap)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.warning(f"⚠️ Error splitting document: {e}")
                chunks = self._simple_split(doc, chunk_size, chunk_overlap)
                all_chunks.extend(chunks)
        all_chunks = [chunk for chunk in all_chunks if len(chunk.page_content.strip()) >= 50]
        CACHE_MANAGER.set_document_chunks(cache_key, all_chunks)
        logger.info(f"📄 Created {len(all_chunks)} adaptive chunks")
        return all_chunks

    def _calculate_content_hash(self, documents: List[Document]) -> str:
        content_sample = "".join([doc.page_content[:100] for doc in documents[:5]])
        return hashlib.sha256(content_sample.encode()).hexdigest()[:16]

    def _adapt_for_content(self, detected_domain: str) -> Tuple[int, int]:
        domain_multipliers = {
            "legal": 1.25, "medical": 1.0, "insurance": 0.85, "financial": 1.0,
            "technical": 1.0, "academic": 1.1, "business": 1.0, "general": 1.0
        }
        multiplier = domain_multipliers.get(detected_domain, 1.0)
        adapted_size = int(CHUNK_SIZE * multiplier)
        adapted_overlap = min(adapted_size // 4, int(CHUNK_OVERLAP * 1.2))
        adapted_size = max(600, min(2000, adapted_size))
        return adapted_size, adapted_overlap

    def _split_document(self, document: Document, chunk_size: int, chunk_overlap: int) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=self.separators
        )
        chunks = splitter.split_documents([document])
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "chunk_index": i, "total_chunks": len(chunks), "chunk_type": "adaptive_split"
            })
        return chunks

    def _simple_split(self, document: Document, chunk_size: int, chunk_overlap: int) -> List[Document]:
        try:
            splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            return splitter.split_documents([document])
        except Exception as e:
            logger.error(f"❌ Even simple splitting failed: {e}")
            return [document]

# ================================
# FAISS VECTOR STORE
# ================================
class FAISSVectorStore:
    """FAISS-based vector store implementation"""
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = None
        self.documents = []
        self.is_trained = False

    def initialize(self):
        if not HAS_FAISS:
            raise ImportError("FAISS not available")
        try:
            self.index = faiss.IndexFlatIP(self.dimension)
            self.is_trained = True
            logger.info("✅ FAISS vector store initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize FAISS: {e}")
            raise

    async def add_documents(self, documents: List[Document], embeddings: List[np.ndarray]):
        try:
            if not self.is_trained:
                self.initialize()
            if len(documents) != len(embeddings):
                raise ValueError("Number of documents must match number of embeddings")
            embeddings_array = np.array(embeddings, dtype=np.float32)
            faiss.normalize_L2(embeddings_array)
            self.index.add(embeddings_array)
            self.documents.extend(documents)
            logger.info(f"✅ Added {len(documents)} documents to FAISS index (total: {len(self.documents)})")
        except Exception as e:
            logger.error(f"❌ Error adding documents to FAISS: {e}")
            raise

    async def similarity_search_with_score(self, query_embedding: np.ndarray, k: int = 10) -> List[Tuple[Document, float]]:
        try:
            if not self.is_trained or len(self.documents) == 0:
                return []
            query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
            faiss.normalize_L2(query_embedding)
            k = min(k, len(self.documents))
            scores, indices = self.index.search(query_embedding, k)
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if 0 <= idx < len(self.documents):
                    doc = self.documents[idx]
                    normalized_score = min(1.0, max(0.0, float(score)))
                    results.append((doc, normalized_score))
            return results
        except Exception as e:
            logger.error(f"❌ Error in FAISS similarity search: {e}")
            return []

    def clear(self):
        self.documents.clear()
        if self.index:
            self.index.reset()

# ================================
# DOMAIN DETECTOR
# ================================
class DomainDetector:
    """Universal domain detector with smart caching"""
    def detect_domain(self, documents: List[Document], confidence_threshold: float = 0.3) -> Tuple[str, float]:
        if not documents:
            return "general", 0.5
        combined_text = ' '.join([doc.page_content[:200] for doc in documents[:5]]).lower()
        cache_key = hashlib.md5(combined_text.encode()).hexdigest()[:16]
        cached_result = CACHE_MANAGER.get_domain_result(cache_key)
        if cached_result is not None:
            logger.info(f"🔍 Using cached domain: {cached_result[0]} (confidence: {cached_result[1]:.2f})")
            return cached_result
        try:
            domain_scores = self._keyword_based_detection(combined_text)
            if domain_scores:
                best_domain = max(domain_scores, key=domain_scores.get)
                best_score = domain_scores[best_domain]
                if best_score < confidence_threshold:
                    best_domain = "general"
                    best_score = confidence_threshold
                result = (best_domain, best_score)
                CACHE_MANAGER.set_domain_result(cache_key, result)
                logger.info(f"🔍 Domain detected: {best_domain} (confidence: {best_score:.2f})")
                return result
            return "general", confidence_threshold
        except Exception as e:
            logger.warning(f"⚠️ Domain detection error: {e}")
            return "general", confidence_threshold

    def _keyword_based_detection(self, combined_text: str) -> Dict[str, float]:
        domain_scores = {}
        text_length = max(len(combined_text), 1)
        for domain, keywords in DOMAIN_KEYWORDS.items():
            matches = sum(combined_text.count(keyword.lower()) for keyword in keywords)
            normalized_score = matches / (len(keywords) * text_length / 1000)
            domain_scores[domain] = min(1.0, normalized_score)
        return domain_scores

# ================================
# TOKEN PROCESSOR
# ================================
class TokenProcessor:
    """Token optimization processor"""
    def __init__(self):
        self.max_context_tokens = 4000
        self.tokenizer = None
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load tiktoken tokenizer: {e}")

    @lru_cache(maxsize=2000)
    def estimate_tokens(self, text: str) -> int:
        if not text:
            return 0
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception:
                pass
        avg_chars_per_token = 3.0 if any(term in text.lower() for term in ['api', 'json', 'xml', 'code', 'function']) else 3.5
        return max(1, int(len(text) / avg_chars_per_token * 1.1))

    def optimize_context(self, documents: List[Document], query: str, max_tokens: int = None) -> str:
        if not documents:
            return ""
        max_tokens = max_tokens or self.max_context_tokens
        doc_scores = []
        for doc in documents:
            relevance = self._calculate_relevance(doc, query)
            tokens = self.estimate_tokens(doc.page_content)
            efficiency = relevance / max(tokens, 1)
            doc_scores.append((doc, relevance, tokens, efficiency))
        doc_scores.sort(key=lambda x: x[3], reverse=True)
        context_parts = []
        token_budget = max_tokens - 200
        for doc, relevance, tokens, efficiency in doc_scores:
            if tokens <= token_budget:
                context_parts.append(doc.page_content)
                token_budget -= tokens
            elif token_budget > 200 and relevance > 0.6:
                partial_content = self._truncate_content(doc.page_content, token_budget)
                context_parts.append(partial_content)
                break
        context = "\n\n".join(context_parts)
        estimated_tokens = self.estimate_tokens(context)
        logger.info(f"🔍 Context optimized: {len(context_parts)} documents, ~{estimated_tokens} tokens")
        return context

    def _calculate_relevance(self, doc: Document, query: str) -> float:
        try:
            query_terms = set(query.lower().split())
            doc_terms = set(doc.page_content.lower().split())
            keyword_overlap = len(query_terms.intersection(doc_terms)) / max(len(query_terms), 1)
            position_boost = 1.0
            chunk_index = doc.metadata.get('chunk_index', 0)
            total_chunks = doc.metadata.get('total_chunks', 1)
            if total_chunks > 1:
                position_boost = 1.15 - (chunk_index / total_chunks) * 0.3
            length_penalty = 0.8 if len(doc.page_content) < 100 else 1.0
            return min(1.0, keyword_overlap * position_boost * length_penalty)
        except Exception as e:
            logger.warning(f"⚠️ Error calculating relevance: {e}")
            return 0.5

    def _truncate_content(self, content: str, max_tokens: int) -> str:
        max_chars = max_tokens * 4
        if len(content) <= max_chars:
            return content
        keep_chars = max_chars - 100
        first_part = content[:keep_chars // 2]
        last_part = content[-keep_chars // 2:]
        return f"{first_part}\n\n[... content truncated ...]\n\n{last_part}"

# ================================
# RAG SYSTEM
# ================================
class RAGSystem:
    """Enhanced RAG system with smart caching and cleanup"""
    def __init__(self):
        self.documents = []
        self.vector_store = None
        self.bm25_retriever = None
        self.domain = "general"
        self.loader = UnifiedLoader()
        self.text_splitter = AdaptiveTextSplitter()
        self.token_processor = TokenProcessor()

    async def cleanup(self):
        self.documents.clear()
        if self.vector_store:
            self.vector_store.clear()
        logger.info("🧹 RAGSystem cleaned up")

    async def process_documents(self, sources: List[str]) -> Dict[str, Any]:
        start_time = time.time()
        CACHE_MANAGER.clear_all_caches()
        try:
            logger.info(f"📄 Processing {len(sources)} documents")
            raw_documents = []
            for source in sources:
                try:
                    docs = await self.loader.load_document(source)
                    raw_documents.extend(docs)
                except Exception as e:
                    logger.warning(f"⚠️ Error loading {sanitize_pii(source)}: {e}")
                    continue
            if not raw_documents:
                raise ValueError("No documents could be loaded")
            domain, domain_confidence = DOMAIN_DETECTOR.detect_domain(raw_documents)
            self.domain = domain
            logger.info(f"🔍 Detected domain: {domain} (confidence: {domain_confidence:.2f})")
            self.documents = self.text_splitter.split_documents(raw_documents, domain)
            await self._setup_retrievers()
            processing_time = time.time() - start_time
            result = {
                'domain': domain, 'domain_confidence': float(domain_confidence),
                'total_chunks': len(self.documents), 'processing_time': processing_time
            }
            logger.info(f"✅ Processing complete in {processing_time:.2f}s")
            return sanitize_for_json(result)
        except Exception as e:
            logger.error(f"❌ Document processing error: {e}")
            raise

    async def _setup_retrievers(self):
        try:
            logger.info("🔧 Setting up retrievers...")
            if HAS_FAISS and self.documents:
                try:
                    await ensure_models_ready()
                    self.vector_store = FAISSVectorStore(dimension=384)
                    self.vector_store.initialize()
                    doc_texts = [doc.page_content for doc in self.documents]
                    embeddings = await get_embeddings_batch(doc_texts)
                    await self.vector_store.add_documents(self.documents, embeddings)
                    logger.info("✅ FAISS vector store setup complete")
                except Exception as e:
                    logger.warning(f"⚠️ FAISS setup failed: {e}")
                    self.vector_store = None
            if self.documents:
                try:
                    self.bm25_retriever = await asyncio.to_thread(BM25Retriever.from_documents, self.documents)
                    self.bm25_retriever.k = min(RERANK_TOP_K, len(self.documents))
                    logger.info(f"✅ BM25 retriever setup complete (k={self.bm25_retriever.k})")
                except Exception as e:
                    logger.warning(f"⚠️ BM25 retriever setup failed: {e}")
        except Exception as e:
            logger.error(f"❌ Retriever setup error: {e}")

    async def retrieve_and_rerank(self, query: str, top_k: int = None) -> Tuple[List[Document], List[float]]:
        top_k = top_k or CONTEXT_DOCS
        try:
            if not self.documents:
                return [], []
            query_embedding = await get_query_embedding(query)
            retrieval_results = []
            if self.vector_store:
                try:
                    vector_results = await self.vector_store.similarity_search_with_score(query_embedding, k=SEMANTIC_SEARCH_K)
                    if vector_results:
                        retrieval_results.append(vector_results)
                except Exception as e:
                    logger.warning(f"⚠️ Vector search failed: {e}")
            if self.bm25_retriever:
                try:
                    bm25_docs = await asyncio.to_thread(self.bm25_retriever.invoke, query) or []
                    bm25_results = [(doc, max(0.1, 1.0 - (i * 0.1))) for i, doc in enumerate(bm25_docs)]
                    if bm25_results:
                        retrieval_results.append(bm25_results)
                except Exception as e:
                    logger.warning(f"⚠️ BM25 search failed: {e}")
            if len(retrieval_results) > 1:
                fused_results = reciprocal_rank_fusion(retrieval_results)
                final_docs = [doc for doc, score in fused_results[:RERANK_TOP_K]]
                final_scores = [score for doc, score in fused_results[:RERANK_TOP_K]]
            elif len(retrieval_results) == 1:
                results = retrieval_results[0]
                final_docs = [doc for doc, score in results[:RERANK_TOP_K]]
                final_scores = [score for doc, score in results[:RERANK_TOP_K]]
            else:
                final_docs, final_scores = await self._fallback_search(query, SEMANTIC_SEARCH_K)
            if reranker and len(final_docs) > 1:
                try:
                    return await self._semantic_rerank(query, final_docs, final_scores, top_k)
                except Exception as e:
                    logger.warning(f"⚠️ Reranking failed: {e}")
            return final_docs[:top_k], final_scores[:top_k]
        except Exception as e:
            logger.error(f"❌ Retrieval error: {e}")
            return self.documents[:top_k], [0.5] * min(len(self.documents), top_k)

    async def _fallback_search(self, query: str, k: int) -> Tuple[List[Document], List[float]]:
        try:
            query_terms = set(query.lower().split())
            doc_scores = []
            for doc in self.documents:
                doc_terms = set(doc.page_content.lower().split())
                overlap = len(query_terms.intersection(doc_terms))
                score = overlap / max(len(query_terms), 1)
                if query.lower() in doc.page_content.lower():
                    score += 0.3
                doc_scores.append((doc, score))
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            return [d for d, s in doc_scores[:k]], [s for d, s in doc_scores[:k]]
        except Exception as e:
            logger.error(f"❌ Fallback search error: {e}")
            return self.documents[:k], [0.5] * min(len(self.documents), k)

    async def _semantic_rerank(self, query: str, docs: List[Document], scores: List[float], top_k: int) -> Tuple[List[Document], List[float]]:
        try:
            if len(docs) <= 2:
                return docs[:top_k], scores[:top_k]
            await ensure_models_ready()
            if not reranker:
                return docs[:top_k], scores[:top_k]
            pairs = [[query, doc.page_content[:512]] for doc in docs]
            rerank_scores = reranker.predict(pairs)
            normalized_rerank = [(score + 1) / 2 for score in rerank_scores]
            combined_scores = [min(1.0, 0.7 * r + 0.3 * o) for r, o in zip(normalized_rerank, scores)]
            scored_docs = sorted(zip(docs, combined_scores), key=lambda x: x[1], reverse=True)
            return [d for d, s in scored_docs[:top_k]], [s for d, s in scored_docs[:top_k]]
        except Exception as e:
            logger.error(f"❌ Semantic reranking error: {e}")
            return docs[:top_k], scores[:top_k]

    async def query(self, query: str) -> Dict[str, Any]:
        start_time = time.time()
        try:
            CACHE_MANAGER.cleanup_if_needed()
            retrieved_docs, similarity_scores = await self.retrieve_and_rerank(query, CONTEXT_DOCS)
            if not retrieved_docs:
                return {"query": query, "answer": "No relevant documents found.", "confidence": 0.0,
                        "domain": self.domain, "processing_time": time.time() - start_time}
            confidence = self._calculate_confidence(query, similarity_scores, retrieved_docs)
            if confidence < CONFIDENCE_THRESHOLD:
                return {"query": query, "answer": "I don't have enough information to answer accurately.",
                        "confidence": float(confidence), "domain": self.domain,
                        "retrieved_chunks": len(retrieved_docs), "processing_time": time.time() - start_time}
            context = self.token_processor.optimize_context(retrieved_docs, query)
            answer = await self._generate_response(query, context, self.domain, confidence)
            return sanitize_for_json({
                "query": query, "answer": answer, "confidence": float(confidence),
                "domain": self.domain, "retrieved_chunks": len(retrieved_docs),
                "processing_time": time.time() - start_time
            })
        except Exception as e:
            logger.error(f"❌ Query processing error: {e}")
            return {"query": query, "answer": f"An error occurred: {sanitize_pii(str(e))}", "confidence": 0.0,
                    "domain": self.domain, "processing_time": time.time() - start_time}

    def _calculate_confidence(self, query: str, scores: List[float], docs: List[Document]) -> float:
        if not scores:
            return 0.0
        try:
            scores_array = np.clip(np.array(scores), 0.0, 1.0)
            max_score, avg_score = np.max(scores_array), np.mean(scores_array)
            score_std = np.std(scores_array) if len(scores_array) > 1 else 0.0
            score_consistency = max(0.0, 1.0 - score_std)
            query_match = self._calculate_query_match(query, docs)
            confidence = 0.35 * max_score + 0.25 * avg_score + 0.25 * query_match + 0.15 * score_consistency
            if any(query.lower() in doc.page_content.lower() for doc in docs[:3]):
                confidence += 0.1
            return min(1.0, max(0.0, confidence))
        except Exception as e:
            logger.warning(f"⚠️ Confidence calculation error: {e}")
            return 0.3

    def _calculate_query_match(self, query: str, docs: List[Document]) -> float:
        query_terms = set(query.lower().split())
        if not query_terms:
            return 0.5
        match_scores = []
        for doc in docs[:5]:
            doc_terms = set(doc.page_content.lower().split())
            match_score = len(query_terms.intersection(doc_terms)) / len(query_terms)
            if query.lower() in doc.page_content.lower():
                match_score += 0.2
            match_scores.append(match_score)
        return np.mean(match_scores) if match_scores else 0.5

    async def _generate_response(self, query: str, context: str, domain: str, confidence: float) -> str:
        try:
            await ensure_openai_ready()
            if not openai_client:
                return "System is initializing. Please try again."
            system_prompt = f"""You are an expert document analyst in {domain}. Answer based on the context. If not available, state it. Be concise. Confidence: {confidence:.1%}"""
            user_message = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer based on the context:"
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}]
            response = await asyncio.wait_for(
                openai_client.chat.completions.create(
                    messages=messages, model="gpt-4o", temperature=0.1, max_tokens=1000
                ), timeout=QUESTION_TIMEOUT
            )
            return response.choices[0].message.content.strip()
        except asyncio.TimeoutError:
            logger.error(f"❌ Response generation timeout after {QUESTION_TIMEOUT}s")
            return "Response generation took too long. Please try a simpler question."
        except Exception as e:
            logger.error(f"❌ Response generation error: {e}")
            return f"I encountered an error processing your query: {str(e)}"

# ================================
# UTILITY FUNCTIONS
# ================================
def reciprocal_rank_fusion(results_list: List[List[Tuple[Document, float]]], k_value: int = 60) -> List[Tuple[Document, float]]:
    if not results_list:
        return []
    doc_scores, seen_docs = defaultdict(float), {}
    weights = {"semantic": 0.6, "bm25": 0.4}
    for i, results in enumerate(results_list):
        weight = weights.get("semantic" if i == 0 else "bm25", 1.0 / len(results_list))
        for rank, (doc, score) in enumerate(results):
            doc_key = hashlib.md5(doc.page_content.encode()).hexdigest()
            doc_scores[doc_key] += weight / (k_value + rank + 1)
            if doc_key not in seen_docs:
                seen_docs[doc_key] = doc
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    max_score = sorted_docs[0][1] if sorted_docs else 1.0
    return [(seen_docs[key], score / max_score) for key, score in sorted_docs]

async def get_embeddings_batch(texts: List[str]) -> List[np.ndarray]:
    if not texts:
        return []
    results, uncached_texts, uncached_indices = [], [], []
    for i, text in enumerate(texts):
        text_hash = hashlib.md5(text.encode()).hexdigest()
        cached_embedding = CACHE_MANAGER.get_embedding(text_hash)
        if cached_embedding is not None:
            results.append((i, cached_embedding))
        else:
            uncached_texts.append(text)
            uncached_indices.append((i, text))
    if uncached_texts:
        try:
            await ensure_models_ready()
            if base_sentence_model:
                async def process_single_batch(batch_data):
                    batch_texts = [item[1] for item in batch_data]
                    batch_indices = [item[0] for item in batch_data]
                    batch_embeddings = await asyncio.to_thread(
                        base_sentence_model.encode, batch_texts, show_progress_bar=False, convert_to_numpy=True
                    )
                    for text, embedding in zip(batch_texts, batch_embeddings):
                        CACHE_MANAGER.set_embedding(hashlib.md5(text.encode()).hexdigest(), embedding)
                    return list(zip(batch_indices, batch_embeddings))
                batches = [uncached_indices[i:i + OPTIMAL_BATCH_SIZE] for i in range(0, len(uncached_indices), OPTIMAL_BATCH_SIZE)]
                semaphore = asyncio.Semaphore(MAX_CONCURRENT_BATCHES)
                async def limited_process_batch(batch):
                    async with semaphore: return await process_single_batch(batch)
                logger.info(f"🚀 Processing {len(uncached_texts)} embeddings in {len(batches)} parallel batches")
                batch_results = await asyncio.gather(*[limited_process_batch(b) for b in batches])
                for batch_result in batch_results:
                    results.extend(batch_result)
                logger.info("✅ Completed parallel embedding processing")
            else:
                logger.warning("⚠️ No embedding model available, using zero vectors")
                results.extend([(i, np.zeros(384)) for i, _ in uncached_indices])
        except Exception as e:
            logger.error(f"❌ Parallel embedding error: {e}")
            results.extend([(i, np.zeros(384)) for i, _ in uncached_indices])
    results.sort(key=lambda x: x[0])
    return [embedding for _, embedding in results]

async def get_query_embedding(query: str) -> np.ndarray:
    if not query.strip():
        return np.zeros(384)
    query_hash = hashlib.md5(query.encode()).hexdigest()
    cached_embedding = CACHE_MANAGER.get_embedding(query_hash)
    if cached_embedding is not None:
        return cached_embedding
    try:
        await ensure_models_ready()
        if base_sentence_model:
            embedding = await asyncio.to_thread(base_sentence_model.encode, query, convert_to_numpy=True)
            CACHE_MANAGER.set_embedding(query_hash, embedding)
            return embedding
        logger.warning("⚠️ No embedding model available for query")
        return np.zeros(384)
    except Exception as e:
        logger.error(f"❌ Query embedding error: {e}")
        return np.zeros(384)

async def ensure_openai_ready():
    global openai_client
    if openai_client is None:
        if not OPENAI_API_KEY:
            raise HTTPException(status_code=503, detail="OpenAI API key not configured")
        try:
            openai_client = AsyncOpenAI(
                api_key=OPENAI_API_KEY, timeout=httpx.Timeout(connect=10.0, read=60.0, write=30.0, pool=5.0), max_retries=3
            )
            logger.info("✅ OpenAI client initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize OpenAI client: {e}")
            raise HTTPException(status_code=503, detail="OpenAI client not available")

async def ensure_models_ready():
    global base_sentence_model, reranker
    if base_sentence_model is None:
        try:
            base_sentence_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            logger.info("✅ Sentence transformer loaded")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load sentence transformer: {e}")
    if reranker is None:
        try:
            reranker = CrossEncoder(RERANKER_MODEL_NAME)
            logger.info("✅ Reranker loaded")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load reranker: {e}")

DOMAIN_DETECTOR = DomainDetector()

# ================================
# FASTAPI APPLICATION
# ================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("🚀 Starting RAG System...")
    await ensure_openai_ready()
    await ensure_models_ready()
    logger.info("✅ System initialized")
    yield
    logger.info("🔄 Shutting down system...")
    CACHE_MANAGER.clear_all_caches()
    if openai_client and hasattr(openai_client, 'close'):
        try:
            await openai_client.close()
        except Exception:
            pass
    logger.info("✅ System shutdown complete")

app = FastAPI(title="HackRx RAG System", version="2.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

# ================================
# API ENDPOINTS
# ================================
@app.get("/")
async def root():
    return {"status": "online", "service": "HackRx RAG System", "version": "2.0.0", "timestamp": datetime.now().isoformat()}

@app.get("/cache-stats")
async def get_cache_stats_endpoint():
    return CACHE_MANAGER.get_cache_stats()

@app.post("/hackrx/run")
async def hackrx_run_endpoint(request: Request):
    """HackRx specific endpoint"""
    if not simple_auth_check(request):
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    try:
        data = await request.json()
        documents_url = data.get("documents")
        questions = data.get("questions", [])
        if not questions:
            raise HTTPException(status_code=400, detail="No questions provided")
        if not documents_url:
            raise HTTPException(status_code=400, detail="No documents URL provided")
        if not validate_url_scheme(documents_url):
            raise HTTPException(status_code=400, detail="Invalid or unsupported URL scheme")
        rag_system = RAGSystem()
        try:
            await rag_system.process_documents([documents_url])
            answers = []
            for question in questions:
                try:
                    result = await rag_system.query(question)
                    answers.append(result["answer"])
                except Exception as e:
                    logger.error(f"❌ Error processing question '{question}': {e}")
                    answers.append(f"Error: {sanitize_pii(str(e))}")
            return {"answers": answers}
        finally:
            await rag_system.cleanup()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ HackRx endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {sanitize_pii(str(e))}")

# ================================
# ERROR HANDLERS
# ================================
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"answers": [f"Error: {exc.detail}"], "error": True, "detail": exc.detail,
                   "status_code": exc.status_code, "timestamp": datetime.now().isoformat()}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"❌ Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"answers": ["Internal server error"], "error": True, "detail": "Internal server error",
                   "status_code": 500, "timestamp": datetime.now().isoformat()}
    )

# ================================
# MAIN ENTRY POINT
# ================================
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    logger.info(f"🚀 Starting HackRx RAG System on port {port}")
    uvicorn.run("hackrx7:app", host="0.0.0.0", port=port, reload=False, log_level="info")
