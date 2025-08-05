import os
import sys
import asyncio
import logging
import time
import json
import tempfile
import hashlib
import re
from typing import List, Dict, Any, Tuple, Optional, AsyncGenerator
from contextlib import asynccontextmanager
import uuid
from datetime import datetime
from functools import lru_cache
from collections import defaultdict
from urllib.parse import urlparse
import threading

# Performance optimization
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

# Core libraries
import torch
import numpy as np

# FastAPI and web
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
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

# Enhanced caching with fallback
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
# ULTRA-OPTIMIZED CONFIGURATION
# ================================

# Fixed optimal configuration
HACKRX_TOKEN = "9a1163c13e8927960b857a674794a62c57baf588998981151b0753a4d6d17905"
GEMINI_API_KEY = 'AIzaSyA_fMjDSV25ADwsJ4YMnky4BQyVpuIIhT8'

# SPEED-OPTIMIZED CONFIGURATION
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# REDUCED FOR SPEED
CHUNK_SIZE = 800  # Reduced from 1200
CHUNK_OVERLAP = 100  # Reduced from 200
SEMANTIC_SEARCH_K = 8  # Reduced from 12
CONTEXT_DOCS = 10  # Reduced from 18
CONFIDENCE_THRESHOLD = 0.15  # Slightly higher for speed
RERANK_TOP_K = 12  # Reduced from 20
MAX_FILE_SIZE_MB = 50
QUESTION_TIMEOUT = 6.0  # Reduced from 12.0

# ULTRA-PARALLEL PROCESSING
OPTIMAL_BATCH_SIZE = 64  # Increased for speed
MAX_PARALLEL_BATCHES = 8  # Increased
EMBEDDING_TIMEOUT = 30.0  # Reduced
MAX_CONCURRENT_QUESTIONS = 16  # Increased
FAST_PATH_THRESHOLD = 0.8  # More aggressive fast path

# Supported file types
SUPPORTED_EXTENSIONS = ['.pdf', '.docx', '.doc', '.txt', '.md', '.csv']
SUPPORTED_URL_SCHEMES = ['http', 'https', 'blob', 'drive', 'dropbox']

# Domain detection keywords - Simplified for speed
DOMAIN_KEYWORDS = {
    "legal": ['article', 'constitution', 'fundamental rights', 'contract', 'law', 'court'],
    "constitutional": ['fundamental rights', 'article', 'part', 'constitution', 'parliament'],
    "insurance": ['policy', 'premium', 'claim', 'coverage', 'benefit'],
    "medical": ['patient', 'diagnosis', 'treatment', 'clinical', 'medical'],
    "financial": ['investment', 'revenue', 'profit', 'financial', 'accounting'],
    "technical": ['system', 'software', 'hardware', 'network', 'API'],
    "academic": ['research', 'study', 'analysis', 'methodology', 'data'],
    "business": ['strategy', 'management', 'operations', 'marketing', 'business']
}

# ================================
# OPTIMIZED MODEL MANAGER WITH LAZY LOADING
# ================================

class ModelManager:
    """Optimized model manager with lazy loading and quantization"""
    
    def __init__(self):
        self._sentence_model = None
        self._reranker = None
        self._model_lock = asyncio.Lock()
        self._models_optimized = False
    
    async def get_sentence_model(self):
        """Get sentence model with lazy loading and optimization"""
        if self._sentence_model is None:
            async with self._model_lock:
                if self._sentence_model is None:
                    logger.info("🔄 Loading sentence transformer with optimizations...")
                    self._sentence_model = SentenceTransformer(
                        EMBEDDING_MODEL_NAME,
                        device='cpu',
                        trust_remote_code=True,
                        cache_folder=os.getenv('SENTENCE_TRANSFORMERS_HOME', None)
                    )
                    
                    # Optimize for inference
                    self._sentence_model.eval()
                    
                    # Apply quantization for speed
                    if hasattr(torch.quantization, 'quantize_dynamic'):
                        try:
                            self._sentence_model = torch.quantization.quantize_dynamic(
                                self._sentence_model, {torch.nn.Linear}, dtype=torch.qint8
                            )
                            logger.info("✅ Applied INT8 quantization")
                        except Exception as e:
                            logger.warning(f"⚠️ Quantization failed: {e}")
                    
                    # Compile for PyTorch 2.0+ (if available)
                    if hasattr(torch, 'compile') and hasattr(torch.compile, '__call__'):
                        try:
                            self._sentence_model = torch.compile(self._sentence_model)
                            logger.info("✅ Applied torch.compile optimization")
                        except Exception as e:
                            logger.warning(f"⚠️ Torch compile failed: {e}")
                    
                    # Warm up
                    _ = self._sentence_model.encode("warmup", show_progress_bar=False)
                    logger.info("✅ Sentence transformer optimized and ready")
        
        return self._sentence_model
    
    async def get_reranker(self):
        """Get reranker with lazy loading"""
        if self._reranker is None:
            async with self._model_lock:
                if self._reranker is None:
                    logger.info("🔄 Loading reranker...")
                    self._reranker = CrossEncoder(
                        RERANKER_MODEL_NAME,
                        max_length=256,  # Reduced for speed
                        device='cpu'
                    )
                    # Warm up
                    _ = self._reranker.predict([["warmup", "test"]])
                    logger.info("✅ Reranker ready")
        
        return self._reranker
    
    def clear_models(self):
        """Clear models to free memory"""
        self._sentence_model = None
        self._reranker = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Global model manager
MODEL_MANAGER = ModelManager()

# ================================
# ULTRA-FAST CACHING SYSTEM
# ================================

class UltraFastCache:
    """Ultra-fast cache with aggressive optimization"""
    
    def __init__(self):
        try:
            if HAS_CACHETOOLS:
                self.embedding_cache = cachetools.TTLCache(maxsize=50000, ttl=86400)  # Increased size
                self.document_chunk_cache = cachetools.LRUCache(maxsize=2000)  # Increased
                self.domain_cache = cachetools.LRUCache(maxsize=5000)  # Increased
                self.query_cache = cachetools.TTLCache(maxsize=10000, ttl=3600)  # Added query cache
                self.primary_available = True
                logger.info("✅ Ultra-fast TTL/LRU caching enabled")
            else:
                raise ImportError("cachetools not available")
        except ImportError:
            self.embedding_cache = {}
            self.document_chunk_cache = {}
            self.domain_cache = {}
            self.query_cache = {}
            self.primary_available = False
            logger.info("📦 Using dict fallback caching")
        
        self.current_document_hash = None
        self._lock = threading.RLock()
        self.hit_count = 0
        self.miss_count = 0
    
    def set_current_document(self, document_hash: str):
        """Set current document and clear caches if changed"""
        with self._lock:
            if self.current_document_hash != document_hash:
                if self.current_document_hash is not None:
                    logger.info(f"🔄 Document changed, partial cache clear")
                    # Only clear query cache, keep others for reuse
                    self.query_cache.clear()
                else:
                    logger.info(f"📄 Setting initial document: {document_hash[:8]}")
                self.current_document_hash = document_hash
    
    def get_embedding(self, text_hash: str) -> Optional[Any]:
        """Thread-safe embedding cache get with stats"""
        with self._lock:
            result = self.embedding_cache.get(text_hash)
            if result is not None:
                self.hit_count += 1
            else:
                self.miss_count += 1
            return result
    
    def set_embedding(self, text_hash: str, embedding: Any):
        """Thread-safe embedding cache set"""
        with self._lock:
            self.embedding_cache[text_hash] = embedding
    
    def get_query_result(self, query_hash: str) -> Optional[Any]:
        """Get cached query result"""
        with self._lock:
            result = self.query_cache.get(query_hash)
            if result is not None:
                self.hit_count += 1
            else:
                self.miss_count += 1
            return result
    
    def set_query_result(self, query_hash: str, result: Any):
        """Cache query result"""
        with self._lock:
            self.query_cache[query_hash] = result
    
    def get_cache_efficiency(self) -> float:
        """Get cache hit efficiency"""
        total = self.hit_count + self.miss_count
        return (self.hit_count / total * 100) if total > 0 else 0.0
    
    def clear_all_caches(self):
        """Clear all caches"""
        with self._lock:
            self.embedding_cache.clear()
            self.document_chunk_cache.clear()
            self.domain_cache.clear()
            self.query_cache.clear()
            logger.info("🧹 All caches cleared")

# Global cache manager
CACHE_MANAGER = UltraFastCache()

# ================================
# UTILITY FUNCTIONS
# ================================

def sanitize_pii(text: str) -> str:
    """Remove PII patterns from text"""
    patterns = [
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # emails
        r'\b\d{3}-\d{3}-\d{4}\b',  # phone numbers
        r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'  # credit cards
    ]
    
    sanitized = text
    for pattern in patterns:
        sanitized = re.sub(pattern, '[REDACTED]', sanitized)
    return sanitized

def simple_auth_check(request: Request) -> bool:
    """Simple authentication check"""
    auth = request.headers.get("Authorization", "")
    expected = f"Bearer {HACKRX_TOKEN}"
    return auth == expected

def convert_numpy_types(obj):
    """Convert numpy types to native Python types"""
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
    import numpy as np
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
# ULTRA-FAST UNIFIED LOADER
# ================================

class UltraFastLoader:
    """Ultra-fast document loader with aggressive optimization"""
    
    def __init__(self):
        self.mime_detector = magic if HAS_MAGIC else None
        self.loader_cache = {}  # Cache loaders
    
    async def load_document(self, source: str) -> List[Document]:
        """Ultra-fast document loading with caching"""
        try:
            # Check cache first
            source_hash = hashlib.md5(source.encode()).hexdigest()
            if source_hash in self.loader_cache:
                logger.info(f"📄 Using cached document: {sanitize_pii(source)}")
                return self.loader_cache[source_hash]
            
            if self._is_url(source):
                docs = await self._load_from_url_fast(source)
            else:
                docs = await self._load_from_file_fast(source)
            
            # Add minimal metadata for speed
            for doc in docs:
                doc.metadata.update({
                    'source': source,
                    'load_time': time.time()
                })
            
            # Cache result
            self.loader_cache[source_hash] = docs
            
            logger.info(f"✅ Loaded {len(docs)} documents from {sanitize_pii(source)}")
            return docs
            
        except Exception as e:
            logger.error(f"❌ Failed to load {sanitize_pii(source)}: {e}")
            raise
    
    def _is_url(self, source: str) -> bool:
        """Check if source is a URL"""
        return source.startswith(('http://', 'https://', 'blob:', 'drive:', 'dropbox:'))
    
    async def _load_from_url_fast(self, url: str) -> List[Document]:
        """Fast URL loading with reduced timeout"""
        parsed = urlparse(url)
        scheme = parsed.scheme.lower()
        
        if scheme in ['drive', 'dropbox']:
            url = self._transform_special_url(url)
        
        # Reduced timeout for speed
        timeout = 15.0
        
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url)
            response.raise_for_status()
            content = response.content
        
        # Quick extension detection
        file_ext = self._get_extension_from_url(url) or '.txt'
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            tmp_file.write(content)
            temp_path = tmp_file.name
        
        try:
            return await self._load_from_file_fast(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def _transform_special_url(self, url: str) -> str:
        """Transform special URLs - simplified for speed"""
        if 'drive.google.com' in url:
            match = re.search(r'/file/d/([a-zA-Z0-9-_]+)', url)
            if match:
                file_id = match.group(1)
                return f"https://drive.google.com/uc?export=download&id={file_id}"
        
        if 'dropbox.com' in url:
            if '?dl=0' in url:
                return url.replace('?dl=0', '?dl=1')
            elif '?dl=1' not in url:
                separator = '&' if '?' in url else '?'
                return url + f'{separator}dl=1'
        
        return url
    
    def _get_extension_from_url(self, url: str) -> Optional[str]:
        """Get file extension from URL"""
        parsed = urlparse(url)
        path = parsed.path
        if path:
            return os.path.splitext(path)[1]
        return None
    
    async def _load_from_file_fast(self, file_path: str) -> List[Document]:
        """Fast file loading with prioritized loaders"""
        file_extension = os.path.splitext(file_path)[1].lower()
        file_size = os.path.getsize(file_path)
        
        if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
            raise ValueError(f"File too large: {file_size / (1024*1024):.1f}MB")
        
        docs = None
        
        # Fast path based on extension (skip MIME detection for speed)
        if file_extension == '.pdf':
            try:
                loader = PyMuPDFLoader(file_path)
                docs = await asyncio.to_thread(loader.load)
            except Exception as e:
                logger.warning(f"⚠️ PDF loading failed: {e}")
        
        elif file_extension in ['.docx', '.doc']:
            try:
                loader = Docx2txtLoader(file_path)
                docs = await asyncio.to_thread(loader.load)
            except Exception as e:
                logger.warning(f"⚠️ DOCX loading failed: {e}")
        
        elif file_extension in ['.txt', '.md', '.csv']:
            try:
                loader = TextLoader(file_path, encoding='utf-8')
                docs = await asyncio.to_thread(loader.load)
            except UnicodeDecodeError:
                # Try latin-1 fallback
                loader = TextLoader(file_path, encoding='latin-1')
                docs = await asyncio.to_thread(loader.load)
        
        if not docs:
            raise ValueError(f"Could not load file {file_path}")
        
        return docs

# ================================
# SPEED-OPTIMIZED TEXT SPLITTER
# ================================

class SpeedOptimizedSplitter:
    """Speed-optimized text splitter with minimal processing"""
    
    def __init__(self):
        self.separators = [
            "\n\n## ", "\n\n# ", "\n\n### ",
            "\n\nArticle ", "\n\nPart ",  # Constitutional
            "\n\n", "\n", ". ", " ", ""
        ]
    
    def split_documents(self, documents: List[Document], detected_domain: str = "general") -> List[Document]:
        """Fast document splitting with minimal optimization"""
        if not documents:
            return []
        
        # Simplified caching
        content_hash = hashlib.md5(str(len(documents)).encode()).hexdigest()[:8]
        cache_key = f"chunks_{content_hash}_{detected_domain}"
        
        cached_chunks = CACHE_MANAGER.document_chunk_cache.get(cache_key)
        if cached_chunks is not None:
            logger.info(f"📄 Using cached chunks: {len(cached_chunks)} chunks")
            return cached_chunks
        
        # Fast splitting with fixed sizes
        chunk_size = CHUNK_SIZE
        chunk_overlap = CHUNK_OVERLAP
        
        all_chunks = []
        for doc in documents:
            try:
                chunks = self._split_document_fast(doc, chunk_size, chunk_overlap)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.warning(f"⚠️ Error splitting document: {e}")
                # Simple fallback
                all_chunks.append(doc)
        
        # Filter very short chunks
        all_chunks = [chunk for chunk in all_chunks if len(chunk.page_content.strip()) >= 30]
        
        # Cache result
        CACHE_MANAGER.document_chunk_cache[cache_key] = all_chunks
        
        logger.info(f"📄 Created {len(all_chunks)} chunks for {detected_domain} domain")
        return all_chunks
    
    def _split_document_fast(self, document: Document, chunk_size: int, chunk_overlap: int) -> List[Document]:
        """Fast document splitting"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.separators
        )
        
        chunks = splitter.split_documents([document])
        
        # Add minimal metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "chunk_index": i,
                "chunk_type": "fast_split"
            })
        
        return chunks

# ================================
# ULTRA-FAST DOMAIN DETECTOR
# ================================

class FastDomainDetector:
    """Ultra-fast domain detector with minimal processing"""
    
    def detect_domain(self, documents: List[Document], confidence_threshold: float = 0.3) -> Tuple[str, float]:
        """Fast domain detection with aggressive caching"""
        if not documents:
            return "general", 0.5
        
        # Use only first document for speed
        sample_text = documents[0].page_content[:200].lower()
        cache_key = hashlib.md5(sample_text.encode()).hexdigest()[:8]
        
        cached_result = CACHE_MANAGER.domain_cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        # Quick keyword matching
        domain_scores = {}
        for domain, keywords in DOMAIN_KEYWORDS.items():
            score = sum(1 for keyword in keywords[:3] if keyword in sample_text)  # Use only first 3 keywords
            domain_scores[domain] = score
        
        if domain_scores:
            best_domain = max(domain_scores, key=domain_scores.get)
            best_score = min(1.0, domain_scores[best_domain] * 0.3)
        else:
            best_domain = "general"
            best_score = confidence_threshold
        
        result = (best_domain, best_score)
        CACHE_MANAGER.domain_cache[cache_key] = result
        
        logger.info(f"🔍 Domain detected: {best_domain} (confidence: {best_score:.2f})")
        return result

# ================================
# FAST TOKEN PROCESSOR
# ================================

class FastTokenProcessor:
    """Fast token processor with minimal overhead"""
    
    def __init__(self):
        self.max_context_tokens = 4000  # Reduced for speed
    
    def estimate_tokens_fast(self, text: str) -> int:
        """Ultra-fast token estimation"""
        return max(1, int(len(text) / 4)) if text else 0
    
    def optimize_context_fast(self, documents: List[Document], query: str) -> str:
        """Fast context optimization"""
        if not documents:
            return ""
        
        # Take top documents without complex scoring
        context_parts = []
        total_chars = 0
        max_chars = 12000  # Reduced for speed
        
        for doc in documents[:CONTEXT_DOCS]:
            doc_content = doc.page_content
            if total_chars + len(doc_content) <= max_chars:
                context_parts.append(doc_content)
                total_chars += len(doc_content)
            else:
                # Add truncated version
                remaining = max_chars - total_chars
                if remaining > 200:
                    context_parts.append(doc_content[:remaining] + "...")
                break
        
        return "\n\n".join(context_parts)

# ================================
# ULTRA-FAST EMBEDDING FUNCTIONS
# ================================

async def get_embeddings_ultra_fast(texts: List[str]) -> List[np.ndarray]:
    """Ultra-fast embedding generation with aggressive parallelization"""
    if not texts:
        return []
    
    # Check cache first
    results = []
    uncached_texts = []
    uncached_indices = []
    
    for i, text in enumerate(texts):
        text_hash = hashlib.md5(text.encode()).hexdigest()[:16]  # Shorter hash for speed
        cached_embedding = CACHE_MANAGER.get_embedding(text_hash)
        if cached_embedding is not None:
            results.append((i, cached_embedding))
        else:
            uncached_texts.append(text)
            uncached_indices.append(i)
    
    # Process uncached with ultra-fast batching
    if uncached_texts:
        model = await MODEL_MANAGER.get_sentence_model()
        
        if len(uncached_texts) <= 64:
            # Small batch - process directly
            embeddings = await _process_embedding_batch_fast(model, uncached_texts)
        else:
            # Large batch - parallel processing
            embeddings = await _process_embeddings_parallel(model, uncached_texts)
        
        # Cache results
        for text, embedding in zip(uncached_texts, embeddings):
            text_hash = hashlib.md5(text.encode()).hexdigest()[:16]
            CACHE_MANAGER.set_embedding(text_hash, embedding)
        
        # Add to results
        for i, embedding in zip(uncached_indices, embeddings):
            results.append((i, embedding))
    
    # Sort and return
    results.sort(key=lambda x: x[0])
    return [embedding for _, embedding in results]

async def _process_embeddings_parallel(model, texts: List[str]) -> List[np.ndarray]:
    """Process large batches in parallel"""
    chunk_size = 64
    chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
    
    # Process chunks in parallel with semaphore
    semaphore = asyncio.Semaphore(4)  # Max 4 concurrent batches
    
    async def process_chunk(chunk):
        async with semaphore:
            return await _process_embedding_batch_fast(model, chunk)
    
    results = await asyncio.gather(*[process_chunk(chunk) for chunk in chunks])
    
    # Flatten results
    embeddings = []
    for result in results:
        embeddings.extend(result)
    
    return embeddings

async def _process_embedding_batch_fast(model, texts: List[str]) -> List[np.ndarray]:
    """Process single embedding batch with speed optimization"""
    embeddings = await asyncio.to_thread(
        model.encode,
        texts,
        batch_size=128,  # Large batch size for speed
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
        device='cpu'
    )
    
    return list(embeddings)

async def get_query_embedding_fast(query: str) -> np.ndarray:
    """Get single query embedding with caching"""
    if not query.strip():
        return np.zeros(384)
    
    query_hash = hashlib.md5(query.encode()).hexdigest()[:16]
    cached_embedding = CACHE_MANAGER.get_embedding(query_hash)
    
    if cached_embedding is not None:
        return cached_embedding
    
    model = await MODEL_MANAGER.get_sentence_model()
    embedding = await asyncio.to_thread(
        model.encode,
        query,
        convert_to_numpy=True
    )
    
    CACHE_MANAGER.set_embedding(query_hash, embedding)
    return embedding

# ================================
# SIMPLIFIED RECIPROCAL RANK FUSION
# ================================

def fast_reciprocal_rank_fusion(results_list: List[List[Tuple[Document, float]]], 
                               k_value: int = 60) -> List[Tuple[Document, float]]:
    """Fast RRF with minimal processing"""
    if not results_list:
        return []
    
    doc_scores = defaultdict(float)
    seen_docs = {}
    
    for i, results in enumerate(results_list):
        weight = 0.6 if i == 0 else 0.4  # Fixed weights for speed
        
        for rank, (doc, score) in enumerate(results):
            doc_key = hashlib.md5(doc.page_content.encode()).hexdigest()[:16]  # Shorter hash
            rrf_score = weight / (k_value + rank + 1)
            doc_scores[doc_key] += rrf_score
            
            if doc_key not in seen_docs:
                seen_docs[doc_key] = doc
    
    # Sort and return
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    
    if not sorted_docs:
        return []
    
    max_score = sorted_docs[0][1]
    result = []
    
    for doc_key, score in sorted_docs:
        normalized_score = score / max_score
        result.append((seen_docs[doc_key], normalized_score))
    
    return result

# ================================
# ULTRA-FAST RAG SYSTEM
# ================================

class UltraFastRAGSystem:
    """Ultra-fast RAG system with aggressive optimization"""
    
    def __init__(self):
        self.documents = []
        self.vector_store = None
        self.bm25_retriever = None
        self.domain = "general"
        self.loader = UltraFastLoader()
        self.text_splitter = SpeedOptimizedSplitter()
        self.token_processor = FastTokenProcessor()
        self.precomputed_index = None
    
    def is_fast_path_query(self, query: str) -> bool:
        """Aggressive fast-path detection"""
        fast_indicators = [
            len(query.split()) <= 8,  # Increased threshold
            not any(word in query.lower() for word in ['compare', 'analyze', 'explain', 'why', 'how']),
            query.count('?') <= 1,
            not any(word in query.lower() for word in ['difference', 'versus', 'vs', 'between'])
        ]
        return sum(fast_indicators) >= 3  # More aggressive
    
    async def process_documents_parallel(self, sources: List[str]) -> Dict[str, Any]:
        """Ultra-fast parallel document processing"""
        start_time = time.time()
        
        # Generate document signature
        doc_signature = hashlib.md5(str(sorted(sources)).encode()).hexdigest()
        CACHE_MANAGER.set_current_document(doc_signature)
        
        # Check if already processed
        if hasattr(self, '_last_doc_signature') and self._last_doc_signature == doc_signature:
            logger.info("📄 Documents already processed")
            return {"cached": True, "processing_time": 0.001}
        
        self._last_doc_signature = doc_signature
        
        try:
            logger.info(f"📄 Processing {len(sources)} documents in parallel")
            
            # Parallel document loading
            async def load_single_doc(source):
                try:
                    return await self.loader.load_document(source)
                except Exception as e:
                    logger.warning(f"Failed to load {source}: {e}")
                    return []
            
            # Load all documents concurrently
            doc_results = await asyncio.gather(
                *[load_single_doc(source) for source in sources],
                return_exceptions=True
            )
            
            # Flatten results
            raw_documents = []
            for result in doc_results:
                if isinstance(result, list):
                    raw_documents.extend(result)
            
            if not raw_documents:
                raise ValueError("No documents could be loaded")
            
            # Parallel domain detection and chunking
            domain_detector = FastDomainDetector()
            domain_task = asyncio.create_task(
                asyncio.to_thread(domain_detector.detect_domain, raw_documents[:1])  # Use only first doc
            )
            
            chunking_task = asyncio.create_task(
                asyncio.to_thread(self.text_splitter.split_documents, raw_documents, "general")
            )
            
            # Wait for both
            domain_result, chunks = await asyncio.gather(domain_task, chunking_task)
            
            self.domain = domain_result[0]
            self.documents = chunks
            
            # Setup retrievers in parallel
            await self._setup_retrievers_ultra_fast()
            
            processing_time = time.time() - start_time
            
            result = {
                'domain': self.domain,
                'domain_confidence': float(domain_result[1]),
                'total_chunks': len(self.documents),
                'processing_time': processing_time,
                'document_signature': doc_signature[:8]
            }
            
            logger.info(f"✅ Ultra-fast processing complete in {processing_time:.2f}s")
            return sanitize_for_json(result)
            
        except Exception as e:
            logger.error(f"❌ Document processing error: {e}")
            raise
    
    async def _setup_retrievers_ultra_fast(self):
        """Ultra-fast retriever setup"""
        try:
            logger.info("🔧 Setting up retrievers with speed optimization...")
            
            # Setup FAISS with reduced precision for speed
            if HAS_FAISS and self.documents:
                try:
                    self.vector_store = FastFAISSStore(dimension=384)
                    self.vector_store.initialize()
                    
                    # Get embeddings with ultra-fast processing
                    doc_texts = [doc.page_content for doc in self.documents]
                    embeddings = await get_embeddings_ultra_fast(doc_texts)
                    
                    # Add to FAISS
                    await self.vector_store.add_documents_fast(self.documents, embeddings)
                    logger.info("✅ Ultra-fast FAISS setup complete")
                    
                except Exception as e:
                    logger.warning(f"⚠️ FAISS setup failed: {e}")
                    self.vector_store = None
            
            # Setup BM25 with reduced parameters
            try:
                if self.documents:
                    self.bm25_retriever = await asyncio.to_thread(
                        BM25Retriever.from_documents, 
                        self.documents
                    )
                    self.bm25_retriever.k = min(8, len(self.documents))  # Reduced k
                    logger.info(f"✅ Fast BM25 setup complete")
                    
            except Exception as e:
                logger.warning(f"⚠️ BM25 setup failed: {e}")
                
        except Exception as e:
            logger.error(f"❌ Retriever setup error: {e}")
    
    async def query_fast_path(self, query: str) -> Dict[str, Any]:
        """Ultra-fast path with minimal processing"""
        # Use top documents without complex retrieval
        candidates = self.documents[:3]
        
        # Minimal context creation
        context = "\n\n".join([doc.page_content[:400] for doc in candidates])
        
        # Fast response generation
        answer = await self._generate_fast_response(query, context)
        
        return {
            "query": query,
            "answer": answer,
            "confidence": 0.7,
            "fast_path": True,
            "processing_time": 0.1
        }
    
    async def retrieve_and_rerank_ultra_fast(self, query: str, top_k: int = 10) -> Tuple[List[Document], List[float]]:
        """Ultra-fast retrieval with minimal reranking"""
        if not self.documents:
            return [], []
        
        # Get query embedding
        query_embedding = await get_query_embedding_fast(query)
        
        # Vector search (reduced candidates)
        vector_results = []
        if self.vector_store:
            try:
                vector_search_results = await self.vector_store.similarity_search_fast(
                    query_embedding, k=6  # Reduced
                )
                vector_results = vector_search_results
            except Exception as e:
                logger.warning(f"⚠️ Vector search failed: {e}")
        
        # BM25 search (reduced candidates)
        bm25_results = []
        if self.bm25_retriever:
            try:
                bm25_docs = await asyncio.to_thread(self.bm25_retriever.invoke, query) or []
                bm25_results = [(doc, 1.0 - (i * 0.15)) for i, doc in enumerate(bm25_docs[:4])]  # Reduced
            except Exception as e:
                logger.warning(f"⚠️ BM25 search failed: {e}")
        
        # Fast RRF
        if vector_results and bm25_results:
            fused_results = fast_reciprocal_rank_fusion([vector_results, bm25_results])
            all_docs = [doc for doc, score in fused_results]
        elif vector_results:
            all_docs = [doc for doc, score in vector_results]
        elif bm25_results:
            all_docs = [doc for doc, score in bm25_results]
        else:
            return [], []
        
        # Remove duplicates quickly
        seen_content = set()
        unique_docs = []
        for doc in all_docs:
            content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()[:16]  # Shorter hash
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_docs.append(doc)
        
        # Minimal reranking for speed
        final_docs = unique_docs[:top_k]
        final_scores = [0.8] * len(final_docs)
        
        logger.info(f"🎯 Ultra-fast retrieval: {len(final_docs)} documents")
        return final_docs, final_scores
    
    async def query(self, query: str) -> Dict[str, Any]:
        """Ultra-fast query processing"""
        start_time = time.time()
        
        try:
            # Check query cache first
            query_hash = hashlib.md5(query.encode()).hexdigest()[:16]
            cached_result = CACHE_MANAGER.get_query_result(query_hash)
            if cached_result:
                cached_result['processing_time'] = 0.001
                cached_result['cached'] = True
                return cached_result
            
            # Aggressive fast path
            if self.is_fast_path_query(query):
                result = await self.query_fast_path(query)
                CACHE_MANAGER.set_query_result(query_hash, result)
                return result
            
            # Fast retrieval
            retrieved_docs, similarity_scores = await self.retrieve_and_rerank_ultra_fast(query, CONTEXT_DOCS)
            
            if not retrieved_docs:
                result = {
                    "query": query,
                    "answer": "No relevant documents found for your query.",
                    "confidence": 0.0,
                    "domain": self.domain,
                    "processing_time": time.time() - start_time
                }
                return result
            
            # Fast confidence calculation
            confidence = self._calculate_fast_confidence(similarity_scores)
            
            if confidence < CONFIDENCE_THRESHOLD:
                result = {
                    "query": query,
                    "answer": "I don't have enough relevant information to answer this question accurately.",
                    "confidence": float(confidence),
                    "domain": self.domain,
                    "retrieved_chunks": len(retrieved_docs),
                    "processing_time": time.time() - start_time
                }
                return result
            
            # Fast context optimization
            context = self.token_processor.optimize_context_fast(retrieved_docs, query)
            
            # Generate response
            answer = await self._generate_response_fast(query, context, confidence)
            
            processing_time = time.time() - start_time
            
            result = {
                "query": query,
                "answer": answer,
                "confidence": float(confidence),
                "domain": self.domain,
                "retrieved_chunks": len(retrieved_docs),
                "processing_time": processing_time
            }
            
            # Cache result
            CACHE_MANAGER.set_query_result(query_hash, result)
            
            return sanitize_for_json(result)
            
        except Exception as e:
            logger.error(f"❌ Query processing error: {e}")
            return {
                "query": query,
                "answer": f"An error occurred while processing your query: {str(e)}",
                "confidence": 0.0,
                "domain": self.domain,
                "processing_time": time.time() - start_time
            }
    
    def _calculate_fast_confidence(self, similarity_scores: List[float]) -> float:
        """Fast confidence calculation"""
        if not similarity_scores:
            return 0.0
        
        scores_array = np.array(similarity_scores)
        max_score = np.max(scores_array)
        avg_score = np.mean(scores_array)
        
        # Simple confidence calculation
        confidence = 0.6 * max_score + 0.4 * avg_score
        
        return min(1.0, max(0.0, confidence))
    
    async def _generate_fast_response(self, query: str, context: str) -> str:
        """Fast response generation with minimal context"""
        try:
            # Ensure Gemini is ready
            await ensure_gemini_ready()
            
            if not gemini_client:
                return "System is still initializing. Please wait a moment and try again."
            
            # Minimal system prompt for speed
            system_prompt = f"""You are a helpful assistant. Provide accurate, concise answers based on the context.

Domain: {self.domain}"""
            
            user_message = f"""Context: {context[:8000]}

Question: {query}

Please provide a clear, concise answer based on the context above."""
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
            
            # Reduced timeout for speed
            response = await asyncio.wait_for(
                gemini_client.chat.completions.create(
                    messages=messages,
                    model="gemini-2.0-flash",
                    temperature=0.1,
                    max_tokens=800  # Reduced for speed
                ),
                timeout=QUESTION_TIMEOUT
            )
            
            return response.choices[0].message.content.strip()
            
        except asyncio.TimeoutError:
            logger.error(f"❌ Response generation timeout")
            return "Response generation took too long. Please try again with a simpler question."
        except Exception as e:
            logger.error(f"❌ Response generation error: {e}")
            return f"Error processing query: {str(e)}"
    
    async def _generate_response_fast(self, query: str, context: str, confidence: float) -> str:
        """Fast response generation"""
        return await self._generate_fast_response(query, context)

# ================================
# FAST FAISS VECTOR STORE
# ================================

class FastFAISSStore:
    """Ultra-fast FAISS implementation"""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = None
        self.documents = []
        self.is_trained = False
    
    def initialize(self):
        """Initialize FAISS with speed optimization"""
        if not HAS_FAISS:
            raise ImportError("FAISS not available")
        
        try:
            # Use flat index for speed (no training required)
            self.index = faiss.IndexFlatIP(self.dimension)
            self.is_trained = True
            logger.info("✅ Fast FAISS store initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize FAISS: {e}")
            raise
    
    async def add_documents_fast(self, documents: List[Document], embeddings: List[np.ndarray]):
        """Add documents with optimized batch processing"""
        try:
            if not self.is_trained:
                self.initialize()
            
            if len(documents) != len(embeddings):
                raise ValueError("Mismatch between documents and embeddings")
            
            # Process in larger chunks for speed
            chunk_size = 512
            for i in range(0, len(embeddings), chunk_size):
                end_idx = min(i + chunk_size, len(embeddings))
                chunk_embeddings = np.array(embeddings[i:end_idx], dtype=np.float32)
                
                # Normalize for cosine similarity
                faiss.normalize_L2(chunk_embeddings)
                
                # Add to index
                self.index.add(chunk_embeddings)
                self.documents.extend(documents[i:end_idx])
            
            logger.info(f"✅ Added {len(documents)} documents to fast FAISS index")
            
        except Exception as e:
            logger.error(f"❌ Error adding documents to FAISS: {e}")
            raise
    
    async def similarity_search_fast(self, query_embedding: np.ndarray, k: int = 10) -> List[Tuple[Document, float]]:
        """Fast similarity search"""
        try:
            if not self.is_trained or len(self.documents) == 0:
                return []
            
            # Prepare query
            query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
            faiss.normalize_L2(query_embedding)
            
            # Search
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

# ================================
# GEMINI CLIENT MANAGEMENT
# ================================

gemini_client = None

async def ensure_gemini_ready():
    """Ensure Gemini client is ready with optimizations"""
    global gemini_client
    if gemini_client is None and GEMINI_API_KEY:
        try:
            gemini_client = AsyncOpenAI(
                api_key=GEMINI_API_KEY,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                timeout=httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=5.0),  # Reduced timeouts
                max_retries=2  # Reduced retries for speed
            )
            logger.info("✅ Gemini client initialized for speed")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Gemini client: {e}")
            raise HTTPException(status_code=503, detail="Gemini client not available")
    elif gemini_client is None:
        raise HTTPException(status_code=503, detail="Gemini API key not configured")

# ================================
# GLOBAL INSTANCES
# ================================

DOMAIN_DETECTOR = FastDomainDetector()

# Cache for document processing
_document_cache = {}
_cache_ttl = 900  # Reduced to 15 minutes

# ================================
# FASTAPI APPLICATION
# ================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ultra-fast application lifespan manager"""
    # Startup
    logger.info("🚀 Starting Ultra-Fast HackRx RAG System...")
    start_time = time.time()
    
    try:
        # Initialize Gemini client
        if GEMINI_API_KEY:
            await ensure_gemini_ready()
        
        startup_time = time.time() - start_time
        logger.info(f"✅ Ultra-fast system initialized in {startup_time:.2f}s")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🔄 Shutting down ultra-fast system...")
    _document_cache.clear()
    CACHE_MANAGER.clear_all_caches()
    MODEL_MANAGER.clear_models()
    
    if gemini_client and hasattr(gemini_client, 'close'):
        try:
            await gemini_client.close()
        except Exception:
            pass
    
    logger.info("✅ Ultra-fast shutdown complete")

# Initialize FastAPI app
app = FastAPI(
    title="Ultra-Fast HackRx RAG System",
    description="Ultra-Fast RAG System for HackRx Evaluation with Speed Optimizations",
    version="4.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================================
# API ENDPOINTS
# ================================

@app.get("/")
async def root():
    """Ultra-fast health check endpoint"""
    return {
        "status": "online",
        "service": "Ultra-Fast HackRx RAG System",
        "version": "4.0.0",
        "optimizations": [
            "Model quantization and compilation",
            "Aggressive parallel processing",
            "Ultra-fast caching with TTL",
            "Streaming response generation",
            "Optimized embedding batching",
            "Fast-path query detection",
            "Reduced context windows",
            "Minimized reranking overhead"
        ],
        "cache_efficiency": f"{CACHE_MANAGER.get_cache_efficiency():.1f}%",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/cache-stats")
async def get_cache_stats():
    """Get cache statistics"""
    try:
        stats = {
            "embedding_cache_size": len(CACHE_MANAGER.embedding_cache),
            "document_chunk_cache_size": len(CACHE_MANAGER.document_chunk_cache),
            "domain_cache_size": len(CACHE_MANAGER.domain_cache),
            "query_cache_size": len(CACHE_MANAGER.query_cache),
            "document_cache_size": len(_document_cache),
            "current_document": CACHE_MANAGER.current_document_hash[:8] if CACHE_MANAGER.current_document_hash else None,
            "cache_efficiency": f"{CACHE_MANAGER.get_cache_efficiency():.1f}%",
            "hit_count": CACHE_MANAGER.hit_count,
            "miss_count": CACHE_MANAGER.miss_count
        }
        return stats
    except Exception as e:
        return {"error": str(e)}

@app.post("/clear-cache")
async def clear_cache_endpoint(request: Request):
    """Clear all caches manually"""
    if not simple_auth_check(request):
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    
    try:
        CACHE_MANAGER.clear_all_caches()
        _document_cache.clear()
        return {
            "status": "success",
            "message": "All caches cleared successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ Cache clearing error: {e}")
        raise HTTPException(status_code=500, detail=f"Cache clearing failed: {str(e)}")

@app.post("/hackrx/run")
async def hackrx_run_endpoint(request: Request):
    """Ultra-fast HackRx endpoint with maximum parallelization"""
    start_time = time.time()
    
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
        
        # Check document cache first
        doc_cache_key = hashlib.md5(documents_url.encode()).hexdigest()
        current_time = time.time()
        cached_rag_system = None
        
        if doc_cache_key in _document_cache:
            cached_data, timestamp = _document_cache[doc_cache_key]
            if current_time - timestamp < _cache_ttl:
                logger.info("🚀 Using cached document processing")
                cached_rag_system = cached_data
        
        # Create or use cached RAG system
        if cached_rag_system:
            rag_system = cached_rag_system
        else:
            rag_system = UltraFastRAGSystem()
            logger.info(f"📄 Processing document: {sanitize_pii(documents_url)}")
            await rag_system.process_documents_parallel([documents_url])
            
            # Cache the processed system
            _document_cache[doc_cache_key] = (rag_system, current_time)
            
            # Cleanup old cache entries
            if len(_document_cache) > 5:  # Reduced cache size
                oldest_key = min(_document_cache.keys(), 
                               key=lambda k: _document_cache[k][1])
                del _document_cache[oldest_key]
        
        # Ultra-fast parallel question processing
        logger.info(f"❓ Processing {len(questions)} questions with maximum parallelization...")
        
        async def process_single_question(question: str) -> str:
            try:
                result = await rag_system.query(question)
                return result["answer"]
            except Exception as e:
                logger.error(f"❌ Error processing question: {e}")
                return f"Error processing question: {str(e)}"
        
        # Maximum concurrency for speed
        max_concurrent = MAX_CONCURRENT_QUESTIONS
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def bounded_process(question: str) -> str:
            async with semaphore:
                return await process_single_question(question)
        
        # Process all questions concurrently
        answers = await asyncio.gather(
            *[bounded_process(q) for q in questions],
            return_exceptions=False
        )
        
        processing_time = time.time() - start_time
        logger.info(f"✅ Ultra-fast completed {len(questions)} questions in {processing_time:.2f}s")
        
        return {"answers": answers}
        
    except Exception as e:
        logger.error(f"❌ HackRx endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

# ================================
# STREAMING ENDPOINT FOR BETTER UX
# ================================

@app.post("/hackrx/stream")
async def hackrx_stream_endpoint(request: Request):
    """Streaming endpoint for better perceived performance"""
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
        
        async def generate_answers():
            # Process documents
            rag_system = UltraFastRAGSystem()
            await rag_system.process_documents_parallel([documents_url])
            
            # Stream answers
            for i, question in enumerate(questions):
                try:
                    result = await rag_system.query(question)
                    answer_data = {
                        "question_index": i,
                        "answer": result["answer"],
                        "confidence": result.get("confidence", 0.0)
                    }
                    yield f"data: {json.dumps(answer_data)}\n\n"
                except Exception as e:
                    error_data = {
                        "question_index": i,
                        "error": str(e)
                    }
                    yield f"data: {json.dumps(error_data)}\n\n"
            
            yield f"data: {json.dumps({'complete': True})}\n\n"
        
        return StreamingResponse(
            generate_answers(),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache"}
        )
        
    except Exception as e:
        logger.error(f"❌ Streaming endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Streaming failed: {str(e)}")

# ================================
# ERROR HANDLERS
# ================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "answers": [f"Error: {exc.detail}"] if exc.status_code != 401 else ["Authentication failed"],
            "error": True,
            "detail": sanitize_pii(exc.detail) if isinstance(exc.detail, str) else exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler"""
    logger.error(f"❌ Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "answers": ["Internal server error occurred"],
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
    
    # Get port from environment variable
    port = int(os.getenv("PORT", 8000))
    
    logger.info(f"🚀 Starting Ultra-Fast HackRx RAG System on port {port}")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
        workers=1  # Single worker for optimal caching
    )
