import os
import sys
import asyncio
import logging
import time
import json
import tempfile
import hashlib
import re
from typing import List, Dict, Any, Tuple, Optional
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

# Enhanced caching with fallback
try:
    import cachetools
    HAS_CACHETOOLS = True
except ImportError:
    HAS_CACHETOOLS = False
    cachetools = None

# Memory management
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    psutil = None

# Configure logging
logging.basicConfig(
    format='%(asctime)s %(levelname)s [%(name)s]: %(message)s',
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

# ================================
# OPTIMIZED CONFIGURATION
# ================================

# Fixed optimal configuration
HACKRX_TOKEN = "9a1163c13e8927960b857a674794a62c57baf588998981151b0753a4d6d17905"
GEMINI_API_KEY = 'AIzaSyA_fMjDSV25ADwsJ4YMnky4BQyVpuIIhT8'

# SPEED-OPTIMIZED CONFIGURATION
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 75
SEMANTIC_SEARCH_K = 8
CONTEXT_DOCS = 6
CONFIDENCE_THRESHOLD = 0.15
RERANK_TOP_K = 12
MAX_FILE_SIZE_MB = 50
QUESTION_TIMEOUT = 8.0

# PARALLEL PROCESSING - OPTIMIZED
OPTIMAL_BATCH_SIZE = 128  # Increased from 32
MAX_PARALLEL_BATCHES = 4
EMBEDDING_TIMEOUT = 60.0

# Supported file types
SUPPORTED_EXTENSIONS = ['.pdf', '.docx', '.doc', '.txt', '.md', '.csv']
SUPPORTED_URL_SCHEMES = ['http', 'https', 'blob', 'drive', 'dropbox']

# Domain detection keywords
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

# GLOBAL MODEL STATE MANAGEMENT
_models_loaded = False
_model_lock = asyncio.Lock()
_startup_complete = False

# Cache for document processing - FIXED
_document_cache = {}
_cache_ttl = 1800  # 30 minutes

# Global models
base_sentence_model = None
reranker = None
gemini_client = None

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
        """Clear ALL caches when new document is uploaded - prevents stale answers"""
        with self._lock:
            self.embedding_cache.clear()
            self.document_chunk_cache.clear()
            self.domain_cache.clear()
            logger.info("🧹 All caches cleared for new document upload")

    def get_embedding(self, text_hash: str) -> Optional[Any]:
        """Thread-safe embedding cache get"""
        with self._lock:
            return self.embedding_cache.get(text_hash)

    def set_embedding(self, text_hash: str, embedding: Any):
        """Thread-safe embedding cache set"""
        with self._lock:
            self.embedding_cache[text_hash] = embedding

    def get_document_chunks(self, cache_key: str) -> Optional[Any]:
        """Thread-safe document chunk cache get"""
        with self._lock:
            return self.document_chunk_cache.get(cache_key)

    def set_document_chunks(self, cache_key: str, chunks: Any):
        """Thread-safe document chunk cache set"""
        with self._lock:
            self.document_chunk_cache[cache_key] = chunks

    def get_domain_result(self, cache_key: str) -> Optional[Any]:
        """Thread-safe domain cache get"""
        with self._lock:
            return self.domain_cache.get(cache_key)

    def set_domain_result(self, cache_key: str, result: Any):
        """Thread-safe domain cache set"""
        with self._lock:
            self.domain_cache[cache_key] = result

    def cleanup_if_needed(self):
        """Only needed for dict fallback - TTL/LRU auto-manage"""
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
        """Get cache statistics"""
        with self._lock:
            return {
                "embedding_cache_size": len(self.embedding_cache),
                "document_chunk_cache_size": len(self.document_chunk_cache),
                "domain_cache_size": len(self.domain_cache),
                "primary_cache_available": self.primary_available,
                "cache_type": "TTLCache/LRUCache" if self.primary_available else "dict_fallback"
            }

# Query Result Cache
class QueryResultCache:
    def __init__(self):
        self.cache = {}
        self.max_size = 1000
        self._lock = threading.RLock()

    def get_cached_answer(self, query: str, doc_hash: str) -> Optional[str]:
        with self._lock:
            cache_key = f"{hashlib.md5(query.encode()).hexdigest()[:8]}_{doc_hash[:8]}"
            return self.cache.get(cache_key)

    def cache_answer(self, query: str, doc_hash: str, answer: str):
        with self._lock:
            cache_key = f"{hashlib.md5(query.encode()).hexdigest()[:8]}_{doc_hash[:8]}"
            if len(self.cache) >= self.max_size:
                old_keys = list(self.cache.keys())[:200]
                for key in old_keys:
                    del self.cache[key]
            self.cache[cache_key] = answer

    def clear(self):
        """Clear query cache"""
        with self._lock:
            self.cache.clear()

# Document State Manager
class DocumentStateManager:
    def __init__(self):
        self.current_doc_hash = None
        self.current_doc_timestamp = None

    def generate_doc_signature(self, sources: List[str]) -> str:
        signature_data = {
            'sources': sorted(sources),
            'timestamp': time.time(),
            'system_version': '2.0'
        }
        return hashlib.sha256(json.dumps(signature_data, sort_keys=True).encode()).hexdigest()

    def should_invalidate_cache(self, new_doc_hash: str) -> bool:
        if self.current_doc_hash is None:
            return True
        return self.current_doc_hash != new_doc_hash

    def invalidate_all_caches(self):
        """FIXED: Clear both cache managers and document cache"""
        CACHE_MANAGER.clear_all_caches()
        QUERY_CACHE.clear()
        # Clear document processing cache
        global _document_cache
        _document_cache.clear()
        logger.info("🧹 All system caches cleared including document processing cache")

# Memory Manager
class MemoryManager:
    def __init__(self):
        self.memory_threshold = 0.85

    def should_cleanup(self) -> bool:
        if HAS_PSUTIL:
            memory_percent = psutil.virtual_memory().percent / 100
            return memory_percent > self.memory_threshold
        return False

    def cleanup_if_needed(self):
        if self.should_cleanup():
            import gc
            gc.collect()
            CACHE_MANAGER.cleanup_if_needed()
            logger.info("🧹 Memory cleanup performed")

# Performance Monitor
class PerformanceMonitor:
    def __init__(self):
        self.metrics = defaultdict(list)

    def record_timing(self, operation: str, duration: float):
        self.metrics[operation].append(duration)
        if len(self.metrics[operation]) > 100:
            self.metrics[operation] = self.metrics[operation][-50:]

    def get_average_timing(self, operation: str) -> float:
        return np.mean(self.metrics.get(operation, [0]))

# Query Analyzer
class QueryAnalyzer:
    def __init__(self):
        self.analytical_keywords = [
            'analyze', 'compare', 'contrast', 'evaluate', 'assess', 'why',
            'how does', 'what causes', 'relationship', 'impact', 'effect',
            'trends', 'patterns', 'implications', 'significance'
        ]

    def classify_query(self, query: str) -> Dict[str, Any]:
        query_lower = query.lower()
        is_analytical = any(keyword in query_lower for keyword in self.analytical_keywords)

        complexity_score = (
            len(query.split()) * 0.1 +
            query.count('?') * 0.2 +
            (1.0 if is_analytical else 0.3)
        )

        return {
            'type': 'analytical' if is_analytical else 'factual',
            'complexity': min(1.0, complexity_score),
            'requires_multi_context': is_analytical
        }

# Global cache manager instances
CACHE_MANAGER = SmartCacheManager()
QUERY_CACHE = QueryResultCache()
DOC_STATE_MANAGER = DocumentStateManager()
MEMORY_MANAGER = MemoryManager()
PERFORMANCE_MONITOR = PerformanceMonitor()
QUERY_ANALYZER = QueryAnalyzer()

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
    """Validate URL scheme against whitelist"""
    parsed = urlparse(url)
    return parsed.scheme.lower() in SUPPORTED_URL_SCHEMES

def validate_file_extension(filename: str) -> bool:
    """Validate file extension against whitelist"""
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
# ENHANCED UNIFIED LOADER
# ================================

class UnifiedLoader:
    """Unified document loader with enhanced URL support"""

    def __init__(self):
        self.mime_detector = magic if HAS_MAGIC else None
        # Enhanced patterns for Google Drive and Dropbox
        self.google_patterns = [
            r'drive\.google\.com/file/d/([a-zA-Z0-9-_]+)',
            r'docs\.google\.com/document/d/([a-zA-Z0-9-_]+)',
            r'docs\.google\.com/spreadsheets/d/([a-zA-Z0-9-_]+)',
            r'docs\.google\.com/presentation/d/([a-zA-Z0-9-_]+)',
        ]
        
        self.dropbox_patterns = [
            r'dropbox\.com/s/([a-zA-Z0-9]+)',
            r'dropbox\.com/sh/([a-zA-Z0-9]+)',
            r'dropbox\.com/scl/fi/([a-zA-Z0-9-_]+)',
            r'dropbox\.com/scl/fo/([a-zA-Z0-9-_]+)',
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
        """Check if source is a URL"""
        return source.startswith(('http://', 'https://', 'blob:', 'drive:', 'dropbox:'))

    async def _load_from_url(self, url: str) -> List[Document]:
        """Enhanced URL loading with retry logic and better Google Drive/Dropbox support"""
        parsed = urlparse(url)
        scheme = parsed.scheme.lower()

        # Normalize custom schemes (drive:, dropbox:)
        if scheme in ["drive", "dropbox"]:
            if scheme == "drive":
                url = url.replace("drive:", "https://")
            elif scheme == "dropbox":
                url = url.replace("dropbox:", "https://")

        # Validate scheme after normalization
        if not validate_url_scheme(url):
            raise ValueError(f"Unsupported URL scheme: {scheme}")

        download_url = self._transform_special_url(url)
        
        # Enhanced headers for better compatibility
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                timeout = httpx.Timeout(
                    timeout=120.0,
                    connect=15.0,
                    read=120.0,
                    write=30.0,
                    pool=5.0
                )

                async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
                    response = await client.get(download_url, follow_redirects=True)
                    response.raise_for_status()
                    content = response.content

                    # Determine extension
                    file_ext = (
                        self._get_extension_from_url(url)
                        or self._detect_extension_from_content(content)
                    )

                    # Write content to a temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                        tmp_file.write(content)
                        temp_path = tmp_file.name

                    try:
                        return await self._load_from_file(temp_path)
                    finally:
                        if os.path.exists(temp_path):
                            os.unlink(temp_path)

            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                # Exponential back-off: 1s, 2s, 4s
                await asyncio.sleep(2 ** attempt)
                logger.warning(f"⚠️ Retry {attempt + 1}/{max_retries} for URL loading: {e}")

    def _transform_special_url(self, url: str) -> str:
        """Enhanced URL transformation for Google Drive and Dropbox"""
        # Enhanced Google Drive transformation
        for pattern in self.google_patterns:
            match = re.search(pattern, url)
            if match:
                file_id = match.group(1)
                # For documents, try export as PDF first, then Word
                if 'docs.google.com/document' in url:
                    return f"https://docs.google.com/document/d/{file_id}/export?format=pdf"
                elif 'docs.google.com/spreadsheets' in url:
                    return f"https://docs.google.com/spreadsheets/d/{file_id}/export?format=pdf"
                elif 'docs.google.com/presentation' in url:
                    return f"https://docs.google.com/presentation/d/{file_id}/export/pdf"
                else:
                    # General file download
                    return f"https://drive.google.com/uc?export=download&id={file_id}"

        # Enhanced Dropbox transformation
        for pattern in self.dropbox_patterns:
            if re.search(pattern, url):
                if '?dl=0' in url:
                    return url.replace('?dl=0', '?dl=1')
                elif '?dl=1' not in url:
                    separator = '&' if '?' in url else '?'
                    return f"{url}{separator}dl=1"

        return url

    def _get_extension_from_url(self, url: str) -> Optional[str]:
        """Get file extension from URL"""
        parsed = urlparse(url)
        path = parsed.path
        if path:
            return os.path.splitext(path)[1]
        return None

    def _detect_extension_from_content(self, content: bytes) -> str:
        """Detect file extension from content"""
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

        # Simple content-based detection
        if content.startswith(b'%PDF'):
            return '.pdf'
        elif b'PK' in content[:10]:  # ZIP-based formats (DOCX, etc.)
            return '.docx'
        
        return '.txt'

    async def _load_from_file(self, file_path: str) -> List[Document]:
        """Load document from file"""
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

        # PDF files
        if mime_type == 'application/pdf' or file_extension == '.pdf':
            try:
                loader = PyMuPDFLoader(file_path)
                docs = await asyncio.to_thread(loader.load)
                loader_used = "PyMuPDFLoader"
            except Exception as e:
                logger.warning(f"⚠️ PyMuPDF failed: {e}")

        # Word documents
        elif ('word' in (mime_type or '') or
              'officedocument' in (mime_type or '') or
              file_extension in ['.docx', '.doc']):
            try:
                loader = Docx2txtLoader(file_path)
                docs = await asyncio.to_thread(loader.load)
                loader_used = "Docx2txtLoader"
            except Exception as e:
                logger.warning(f"⚠️ DOCX loader failed: {e}")

        # Text files
        elif ('text' in (mime_type or '') or
              file_extension in ['.txt', '.md', '.csv', '.log']):
            for encoding in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']:
                try:
                    loader = TextLoader(file_path, encoding=encoding)
                    docs = await asyncio.to_thread(loader.load)
                    loader_used = f"TextLoader ({encoding})"
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    logger.warning(f"⚠️ Text loader failed with {encoding}: {e}")

        # Fallback loader
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
# HIERARCHICAL TEXT SPLITTER
# ================================

class HierarchicalChunker:
    def __init__(self):
        self.chunk_sizes = [800, 1200, 400]
        self.overlap_ratio = 0.2

    def create_hierarchical_chunks(self, documents: List[Document]) -> List[Document]:
        all_chunks = []
        for doc in documents:
            for chunk_size in self.chunk_sizes:
                overlap = int(chunk_size * self.overlap_ratio)
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=overlap
                )
                chunks = splitter.split_documents([doc])

                for chunk in chunks:
                    chunk.metadata.update({
                        'chunk_size_category': chunk_size,
                        'hierarchy_level': 'primary' if chunk_size == 800 else 'context'
                    })

                all_chunks.extend(chunks)

        return all_chunks

class AdaptiveTextSplitter:
    """Adaptive text splitter with smart caching and hierarchical support"""

    def __init__(self):
        self.separators = [
            "\n\n## ", "\n\n# ", "\n\n### ", "\n\n#### ",
            "\n\n", "\n", ". ", "! ", "? ", "; ", " ", ""
        ]
        self.hierarchical_chunker = HierarchicalChunker()

    def split_documents(self, documents: List[Document], detected_domain: str = "general") -> List[Document]:
        """Split documents with smart caching"""
        if not documents:
            return []

        content_hash = self._calculate_content_hash(documents)
        cache_key = f"chunks_{content_hash}_{detected_domain}"

        cached_chunks = CACHE_MANAGER.get_document_chunks(cache_key)
        if cached_chunks is not None:
            logger.info(f"📄 Using cached chunks: {len(cached_chunks)} chunks")
            return cached_chunks

        chunk_size, chunk_overlap = self._adapt_for_content(documents, detected_domain)

        all_chunks = []
        for doc in documents:
            try:
                chunks = self._split_document(doc, chunk_size, chunk_overlap)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.warning(f"⚠️ Error splitting document: {e}")
                chunks = self._simple_split(doc, chunk_size, chunk_overlap)
                all_chunks.extend(chunks)

        # Filter very short chunks
        all_chunks = [chunk for chunk in all_chunks if len(chunk.page_content.strip()) >= 50]

        CACHE_MANAGER.set_document_chunks(cache_key, all_chunks)
        logger.info(f"📄 Created {len(all_chunks)} adaptive chunks")

        return all_chunks

    def _calculate_content_hash(self, documents: List[Document]) -> str:
        """Calculate hash for content caching"""
        content_sample = "".join([doc.page_content[:100] for doc in documents[:5]])
        return hashlib.sha256(content_sample.encode()).hexdigest()[:16]

    def _adapt_for_content(self, documents: List[Document], detected_domain: str) -> Tuple[int, int]:
        """Adapt chunk size based on content and domain"""
        domain_multipliers = {
            "legal": 1.25,
            "medical": 1.0,
            "insurance": 0.85,
            "financial": 1.0,
            "technical": 1.0,
            "academic": 1.1,
            "business": 1.0,
            "general": 1.0
        }

        multiplier = domain_multipliers.get(detected_domain, 1.0)
        adapted_size = int(CHUNK_SIZE * multiplier)
        adapted_overlap = min(adapted_size // 4, int(CHUNK_OVERLAP * 1.2))

        adapted_size = max(600, min(2000, adapted_size))
        return adapted_size, adapted_overlap

    def _split_document(self, document: Document, chunk_size: int, chunk_overlap: int) -> List[Document]:
        """Split single document"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.separators
        )

        chunks = splitter.split_documents([document])

        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "chunk_index": i,
                "total_chunks": len(chunks),
                "chunk_type": "adaptive_split"
            })

        return chunks

    def _simple_split(self, document: Document, chunk_size: int, chunk_overlap: int) -> List[Document]:
        """Simple fallback splitting"""
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            return splitter.split_documents([document])
        except Exception as e:
            logger.error(f"❌ Even simple splitting failed: {e}")
            return [document]

# ================================
# OPTIMIZED FAISS VECTOR STORE
# ================================

class FAISSVectorStore:
    """FAISS-based vector store implementation with optimizations"""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = None
        self.documents = []
        self.is_trained = False

    def initialize(self):
        """Initialize FAISS index"""
        if not HAS_FAISS:
            raise ImportError("FAISS not available")

        try:
            self.index = faiss.IndexFlatIP(self.dimension)
            self.is_trained = True
            logger.info("✅ FAISS vector store initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize FAISS: {e}")
            raise

    async def add_documents_optimized(self, documents: List[Document], embeddings: List[np.ndarray]):
        """Optimized bulk FAISS operations"""
        try:
            if not self.is_trained:
                self.initialize()

            if len(documents) != len(embeddings):
                raise ValueError("Number of documents must match number of embeddings")

            # Convert all embeddings at once for better performance
            all_embeddings = np.array(embeddings, dtype=np.float32)
            faiss.normalize_L2(all_embeddings)

            # For large datasets, use IVF index for better performance
            if len(embeddings) > 1000 and not hasattr(self.index, 'nlist'):
                logger.info("🔧 Upgrading to IVF index for large dataset")
                nlist = min(int(np.sqrt(len(embeddings))), 256)
                quantizer = faiss.IndexFlatIP(self.dimension)
                new_index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
                new_index.train(all_embeddings)
                new_index.add(all_embeddings)
                self.index = new_index
            else:
                # Add all embeddings in single operation
                self.index.add(all_embeddings)

            self.documents.extend(documents)
            logger.info(f"✅ Added {len(documents)} documents to optimized FAISS index (total: {len(self.documents)})")

        except Exception as e:
            logger.error(f"❌ Error adding documents to FAISS: {e}")
            raise

    async def add_documents(self, documents: List[Document], embeddings: List[np.ndarray]):
        """Legacy method that calls optimized version"""
        return await self.add_documents_optimized(documents, embeddings)

    async def similarity_search_with_score(self, query_embedding: np.ndarray, k: int = 10) -> List[Tuple[Document, float]]:
        """Search for similar documents with scores"""
        try:
            if not self.is_trained or len(self.documents) == 0:
                return []

            query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
            faiss.normalize_L2(query_embedding)

            k = min(k, len(self.documents))
            scores, indices = self.index.search(query_embedding, k)

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx < len(self.documents):
                    doc = self.documents[idx]
                    normalized_score = min(1.0, max(0.0, float(score)))
                    results.append((doc, normalized_score))

            return results

        except Exception as e:
            logger.error(f"❌ Error in FAISS similarity search: {e}")
            return []

    def clear(self):
        """Clear the vector store"""
        self.documents.clear()
        if self.index:
            self.index.reset()

# ================================
# DOMAIN DETECTOR - WITH SMART CACHING
# ================================

class DomainDetector:
    """Universal domain detector with smart caching"""

    def detect_domain(self, documents: List[Document], confidence_threshold: float = 0.3) -> Tuple[str, float]:
        """Universal domain detection with smart caching"""
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
        """Keyword-based domain detection"""
        domain_scores = {}

        for domain, keywords in DOMAIN_KEYWORDS.items():
            matches = 0
            for keyword in keywords:
                matches += combined_text.count(keyword.lower())

            text_length = max(len(combined_text), 1)
            normalized_score = matches / (len(keywords) * text_length / 1000)
            domain_scores[domain] = min(1.0, normalized_score)

        return domain_scores

# ================================
# INTELLIGENT CONTEXT BUILDER
# ================================

class IntelligentContextBuilder:
    """Intelligent context builder with token optimization"""

    def __init__(self):
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.tokenizer = None
        self.max_context_tokens = 3500

    def build_optimal_context(self, query: str, documents: List[Document]) -> str:
        """Build context with intelligent prioritization"""
        if not documents:
            return ""

        # Analyze query for key terms
        query_terms = self._extract_key_terms(query)

        # Score documents by relevance to query
        scored_docs = []
        for doc in documents:
            relevance_score = self._calculate_relevance(doc, query_terms)
            token_count = self._estimate_tokens(doc.page_content)
            efficiency_score = relevance_score / max(token_count, 1)

            scored_docs.append((doc, relevance_score, efficiency_score, token_count))

        # Sort by efficiency (relevance per token)
        scored_docs.sort(key=lambda x: x[2], reverse=True)

        # Build context within token limit
        context_parts = []
        total_tokens = 0

        for doc, rel_score, eff_score, token_count in scored_docs:
            if total_tokens + token_count <= self.max_context_tokens:
                context_parts.append(doc.page_content)
                total_tokens += token_count
            elif total_tokens < self.max_context_tokens * 0.8:
                # Try to fit partial content
                remaining_tokens = self.max_context_tokens - total_tokens
                if remaining_tokens > 100:  # Minimum useful chunk
                    partial_content = self._truncate_intelligently(
                        doc.page_content, remaining_tokens, query_terms
                    )
                    context_parts.append(partial_content)
                    break
            else:
                break

        return "\n\n---\n\n".join(context_parts)

    def _extract_key_terms(self, query: str) -> List[str]:
        """Extract key terms from query"""
        # Simple extraction - split and filter
        terms = query.lower().split()
        # Filter out common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        return [term for term in terms if term not in stop_words and len(term) > 2]

    def _calculate_relevance(self, doc: Document, query_terms: List[str]) -> float:
        """Calculate document relevance to query terms"""
        content_lower = doc.page_content.lower()
        score = 0.0

        for term in query_terms:
            count = content_lower.count(term)
            score += count * (1.0 / len(doc.page_content))  # Normalized by document length

        return score

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count"""
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception:
                pass
        # Fallback estimation
        return max(1, int(len(text) / 3.8))

    def _truncate_intelligently(self, content: str, max_tokens: int, query_terms: List[str]) -> str:
        """Truncate content while preserving query-relevant parts"""
        sentences = content.split('. ')

        # Score sentences by query relevance
        sentence_scores = []
        for sentence in sentences:
            score = sum(1 for term in query_terms if term.lower() in sentence.lower())
            sentence_scores.append((sentence, score))

        # Sort by relevance and fit within token limit
        sentence_scores.sort(key=lambda x: x[1], reverse=True)

        selected_sentences = []
        current_tokens = 0

        for sentence, score in sentence_scores:
            sentence_tokens = self._estimate_tokens(sentence)
            if current_tokens + sentence_tokens <= max_tokens:
                selected_sentences.append(sentence)
                current_tokens += sentence_tokens

        return '. '.join(selected_sentences) + "..."

# ================================
# OPTIMIZED TOKEN PROCESSOR
# ================================

class TokenProcessor:
    """Token optimization processor with intelligent context building"""

    def __init__(self):
        self.max_context_tokens = 4000
        self.context_builder = IntelligentContextBuilder()
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load tiktoken tokenizer: {e}")
            self.tokenizer = None

    def estimate_tokens_fast(self, text: str) -> int:
        """Fast token estimation"""
        if not text:
            return 0
        return max(1, int(len(text) / 3.8))

    def optimize_context_intelligent(self, documents: List[Document], query: str) -> str:
        """Intelligent context optimization using the context builder"""
        return self.context_builder.build_optimal_context(query, documents)

    def optimize_context_fast(self, documents: List[Document], query: str) -> str:
        """Fast context optimization - fallback method"""
        if not documents:
            return ""

        # Use intelligent context builder if available, otherwise use fast method
        try:
            return self.optimize_context_intelligent(documents, query)
        except Exception:
            # Fallback to fast method
            context_parts = []
            total_chars = 0
            max_chars = 12000

            for doc in documents[:6]:
                if total_chars + len(doc.page_content) <= max_chars:
                    context_parts.append(doc.page_content)
                    total_chars += len(doc.page_content)
                else:
                    remaining = max_chars - total_chars
                    if remaining > 200:
                        context_parts.append(doc.page_content[:remaining] + "...")
                        break

            return "\n\n".join(context_parts)

    @lru_cache(maxsize=2000)
    def estimate_tokens(self, text: str) -> int:
        """Accurate token estimation with fallback"""
        return self.estimate_tokens_fast(text)

    def optimize_context(self, documents: List[Document], query: str, max_tokens: int = None) -> str:
        """Optimize context for token limit"""
        return self.optimize_context_fast(documents, query)

# ================================
# LARGE DOCUMENT PROCESSOR
# ================================

class LargeDocumentProcessor:
    def __init__(self):
        self.max_chunk_batch_size = 100
        self.embedding_batch_size = OPTIMAL_BATCH_SIZE  # Use optimized batch size

    async def process_large_document(self, documents: List[Document]) -> Dict[str, Any]:
        total_chunks = len(documents)
        if total_chunks > 1000:
            return await self._process_in_streaming_mode(documents)
        else:
            return await self._process_normally(documents)

    async def _process_in_streaming_mode(self, documents: List[Document]) -> Dict[str, Any]:
        processed_chunks = 0
        batch_size = self.max_chunk_batch_size

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_texts = [doc.page_content for doc in batch]

            embeddings = await self._get_embeddings_streaming(batch_texts)
            processed_chunks += len(batch)

            logger.info(f"📊 Processed {processed_chunks}/{len(documents)} chunks")

        return {"streaming_mode": True, "total_chunks": len(documents)}

    async def _process_normally(self, documents: List[Document]) -> Dict[str, Any]:
        return {"streaming_mode": False, "total_chunks": len(documents)}

    async def _get_embeddings_streaming(self, texts: List[str]) -> List[np.ndarray]:
        """Process embeddings with memory-efficient streaming"""
        all_embeddings = []
        batch_size = self.embedding_batch_size

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            cached_embeddings = []
            uncached_texts = []

            for text in batch_texts:
                text_hash = hashlib.md5(text.encode()).hexdigest()
                cached = CACHE_MANAGER.get_embedding(text_hash)
                if cached is not None:
                    cached_embeddings.append(cached)
                else:
                    uncached_texts.append((text, text_hash))

            if uncached_texts:
                texts_to_embed = [text for text, _ in uncached_texts]

                new_embeddings = await asyncio.to_thread(
                    base_sentence_model.encode,
                    texts_to_embed,
                    batch_size=len(texts_to_embed),  # Process entire batch at once
                    show_progress_bar=False,
                    normalize_embeddings=True
                )

                for (text, text_hash), embedding in zip(uncached_texts, new_embeddings):
                    CACHE_MANAGER.set_embedding(text_hash, embedding)

                all_embeddings.extend(new_embeddings)
            all_embeddings.extend(cached_embeddings)

        return all_embeddings

# ================================
# RAG SYSTEM - ENHANCED
# ================================

class RAGSystem:
    """Enhanced RAG system with optimizations"""

    def __init__(self):
        self.documents = []
        self.vector_store = None
        self.bm25_retriever = None
        self.domain = "general"
        self.loader = UnifiedLoader()
        self.text_splitter = AdaptiveTextSplitter()
        self.token_processor = TokenProcessor()
        self.doc_state_manager = DocumentStateManager()
        self.large_doc_processor = LargeDocumentProcessor()

    async def cleanup(self):
        """RAGSystem cleanup method"""
        self.documents.clear()
        if self.vector_store:
            self.vector_store.clear()
        logger.info("🧹 RAGSystem cleaned up")

    def is_simple_query(self, query: str) -> bool:
        """Detect simple queries that can use fast path"""
        simple_indicators = [
            len(query.split()) <= 8,
            not any(word in query.lower() for word in ['compare', 'analyze', 'explain why', 'how does', 'what are the differences']),
            query.count('?') <= 1
        ]
        return all(simple_indicators)

    async def query_fast_path(self, query: str) -> Dict[str, Any]:
        """Fast path for simple queries"""
        retrieved_docs = self.documents[:3]
        context = "\n\n".join([doc.page_content[:800] for doc in retrieved_docs])
        answer = await self._generate_response(query, context, self.domain, 0.8)

        return {
            "query": query,
            "answer": answer,
            "confidence": 0.8,
            "domain": self.domain,
            "fast_path": True,
            "processing_time": 0.5
        }

    async def retrieve_for_analytical(self, query: str) -> List[Document]:
        """Enhanced retrieval for analytical queries"""
        vector_docs, scores = await self.retrieve_and_rerank_optimized(query, top_k=10)

        query_entities = self._extract_entities(query)
        related_docs = []
        
        for entity in query_entities:
            entity_docs = await self._find_entity_related_docs(entity, exclude_ids=set())
            related_docs.extend(entity_docs[:2])

        all_docs = vector_docs + related_docs
        return self._deduplicate_documents(all_docs)[:8]

    def _extract_entities(self, query: str) -> List[str]:
        """Simple entity extraction"""
        entities = []
        capitalized = re.findall(r'\b[A-Z][a-z]+\b', query)
        quoted = re.findall(r'"([^"]*)"', query)
        return entities + capitalized + quoted

    async def _find_entity_related_docs(self, entity: str, exclude_ids: set) -> List[Document]:
        """Find documents related to specific entities"""
        related_docs = []
        for doc in self.documents:
            if entity.lower() in doc.page_content.lower():
                doc_id = id(doc)
                if doc_id not in exclude_ids:
                    related_docs.append(doc)
                    exclude_ids.add(doc_id)

        return related_docs[:5]

    def _deduplicate_documents(self, docs: List[Document]) -> List[Document]:
        """Remove duplicate documents"""
        seen_content = set()
        unique_docs = []

        for doc in docs:
            content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_docs.append(doc)

        return unique_docs

    async def process_documents(self, sources: List[str]) -> Dict[str, Any]:
        """Process documents with enhanced caching and state management"""
        start_time = time.time()

        doc_signature = self.doc_state_manager.generate_doc_signature(sources)

        if hasattr(self, '_last_doc_signature') and self._last_doc_signature == doc_signature:
            logger.info("📄 Documents already processed, skipping...")
            return {"cached": True, "processing_time": 0.001}

        if self.doc_state_manager.should_invalidate_cache(doc_signature):
            logger.info("🧹 New documents detected - invalidating all caches")
            self.doc_state_manager.invalidate_all_caches()

        self.doc_state_manager.current_doc_hash = doc_signature
        self.doc_state_manager.current_doc_timestamp = time.time()
        self._last_doc_signature = doc_signature

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

            self.documents = self.text_splitter.split_documents(raw_documents, domain)

            await self._setup_retrievers_optimized()

            processing_time = time.time() - start_time

            result = {
                'domain': domain,
                'domain_confidence': float(domain_confidence),
                'total_chunks': len(self.documents),
                'processing_time': processing_time
            }

            logger.info(f"✅ Processing complete in {processing_time:.2f}s")
            return sanitize_for_json(result)

        except Exception as e:
            logger.error(f"❌ Document processing error: {e}")
            raise

    async def _setup_retrievers_optimized(self):
        """Optimized retriever setup with parallel processing"""
        try:
            logger.info("🔧 Setting up optimized retrievers...")

            # Prepare tasks for parallel execution
            tasks = []

            if HAS_FAISS and self.documents:
                await ensure_models_ready()
                self.vector_store = FAISSVectorStore(dimension=384)
                self.vector_store.initialize()

                doc_texts = [doc.page_content for doc in self.documents]
                
                # Use optimized embedding generation
                embeddings = await get_embeddings_batch_optimized(doc_texts)
                
                # Use optimized FAISS document addition
                await self.vector_store.add_documents_optimized(self.documents, embeddings)
                logger.info("✅ Optimized FAISS vector store setup complete")

            # Setup BM25 retriever in parallel
            if self.documents:
                self.bm25_retriever = await asyncio.to_thread(
                    BM25Retriever.from_documents, self.documents
                )
                self.bm25_retriever.k = min(RERANK_TOP_K, len(self.documents))
                logger.info(f"✅ BM25 retriever setup complete (k={self.bm25_retriever.k})")

        except Exception as e:
            logger.error(f"❌ Optimized retriever setup error: {e}")

    async def retrieve_and_rerank_optimized(self, query: str, top_k: int = 6) -> Tuple[List[Document], List[float]]:
        """Streamlined retrieval with optimized reranking"""
        if not self.documents:
            return [], []

        query_embedding = await get_query_embedding(query)

        vector_docs = []
        if self.vector_store:
            try:
                vector_results = await self.vector_store.similarity_search_with_score(
                    query_embedding, k=8
                )
                vector_docs = [doc for doc, score in vector_results]
            except Exception as e:
                logger.warning(f"⚠️ Vector search failed: {e}")

        bm25_docs = []
        if self.bm25_retriever and len(vector_docs) < 8:
            try:
                bm25_results = await asyncio.to_thread(self.bm25_retriever.invoke, query) or []
                bm25_docs = bm25_results[:4]
            except Exception as e:
                logger.warning(f"⚠️ BM25 search failed: {e}")

        all_docs = vector_docs[:6] + bm25_docs[:2]

        # Deduplicate
        seen_content = set()
        unique_docs = []
        for doc in all_docs:
            content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_docs.append(doc)

        # Optimized reranking with batch processing
        if reranker and len(unique_docs) > 3:
            try:
                pairs = [[query, doc.page_content[:256]] for doc in unique_docs[:12]]
                
                # Process reranking in batches for better performance
                batch_size = 32
                all_scores = []
                
                for i in range(0, len(pairs), batch_size):
                    batch_pairs = pairs[i:i + batch_size]
                    batch_scores = reranker.predict(batch_pairs)
                    all_scores.extend(batch_scores)

                scored_docs = list(zip(unique_docs[:len(all_scores)], all_scores))
                scored_docs.sort(key=lambda x: x[1], reverse=True)

                final_docs = [doc for doc, _ in scored_docs[:top_k]]
                final_scores = [score for _, score in scored_docs[:top_k]]

                return final_docs, final_scores

            except Exception as e:
                logger.warning(f"⚠️ Reranking failed: {e}")

        return unique_docs[:top_k], [0.8] * min(len(unique_docs), top_k)

    async def query_large_document_optimized(self, query: str) -> Dict[str, Any]:
        """Optimized querying for large documents"""
        candidate_chunks = await self._pre_filter_chunks(query)

        if len(candidate_chunks) > 50:
            stage1_docs = await self._fast_retrieval_stage1(query, candidate_chunks, k=20)
            final_docs = await self._precision_retrieval_stage2(query, stage1_docs, k=6)
        else:
            final_docs, scores = await self.retrieve_and_rerank_optimized(query, top_k=6)

        return await self._generate_response_optimized(query, final_docs)

    async def _pre_filter_chunks(self, query: str) -> List[Document]:
        """Fast pre-filtering using BM25 or keyword matching"""
        if self.bm25_retriever:
            candidates = await asyncio.to_thread(
                self.bm25_retriever.invoke,
                query
            )
            return candidates[:100]
        return self.documents[:100]

    async def _fast_retrieval_stage1(self, query: str, candidates: List[Document], k: int) -> List[Document]:
        """Fast first stage retrieval"""
        return candidates[:k]

    async def _precision_retrieval_stage2(self, query: str, docs: List[Document], k: int) -> List[Document]:
        """Precision second stage retrieval"""
        return docs[:k]

    async def _generate_response_optimized(self, query: str, docs: List[Document]) -> Dict[str, Any]:
        """Generate optimized response"""
        context = self.token_processor.optimize_context_fast(docs, query)
        answer = await self._generate_response(query, context, self.domain, 0.8)

        return {
            "query": query,
            "answer": answer,
            "confidence": 0.8,
            "domain": self.domain
        }

    async def query(self, query: str) -> Dict[str, Any]:
        """Process query with caching and return response"""
        start_time = time.time()

        try:
            doc_hash = getattr(self, '_last_doc_signature', 'unknown')
            cached_answer = QUERY_CACHE.get_cached_answer(query, doc_hash)

            if cached_answer:
                return {
                    "query": query,
                    "answer": cached_answer,
                    "confidence": 0.9,
                    "domain": self.domain,
                    "processing_time": time.time() - start_time,
                    "cached": True
                }

            CACHE_MANAGER.cleanup_if_needed()
            MEMORY_MANAGER.cleanup_if_needed()

            # Enhanced query classification
            query_analysis = QUERY_ANALYZER.classify_query(query)

            if self.is_simple_query(query) and query_analysis['type'] == 'factual':
                result = await self.query_fast_path(query)
                QUERY_CACHE.cache_answer(query, doc_hash, result['answer'])
                return result

            # Enhanced retrieval for analytical queries
            if query_analysis['requires_multi_context']:
                retrieved_docs = await self.retrieve_for_analytical(query)
                similarity_scores = [0.8] * len(retrieved_docs)
            else:
                retrieved_docs, similarity_scores = await self.retrieve_and_rerank_optimized(query, CONTEXT_DOCS)

            if not retrieved_docs:
                return {
                    "query": query,
                    "answer": "No relevant documents found for your query.",
                    "confidence": 0.0,
                    "domain": self.domain,
                    "processing_time": time.time() - start_time
                }

            confidence = self._enhanced_confidence_calculation(query, retrieved_docs, similarity_scores)

            if confidence < CONFIDENCE_THRESHOLD:
                return {
                    "query": query,
                    "answer": "I don't have enough relevant information to answer this question accurately.",
                    "confidence": float(confidence),
                    "domain": self.domain,
                    "retrieved_chunks": len(retrieved_docs),
                    "processing_time": time.time() - start_time
                }

            context = self.token_processor.optimize_context_fast(retrieved_docs, query)
            answer = await self._generate_response(query, context, self.domain, confidence)

            processing_time = time.time() - start_time
            PERFORMANCE_MONITOR.record_timing("query_processing", processing_time)

            result = {
                "query": query,
                "answer": answer,
                "confidence": float(confidence),
                "domain": self.domain,
                "retrieved_chunks": len(retrieved_docs),
                "processing_time": processing_time
            }

            QUERY_CACHE.cache_answer(query, doc_hash, answer)
            return sanitize_for_json(result)

        except Exception as e:
            logger.error(f"❌ Query processing error: {e}")
            return {
                "query": query,
                "answer": f"An error occurred while processing your query: {sanitize_pii(str(e))}",
                "confidence": 0.0,
                "domain": self.domain,
                "processing_time": time.time() - start_time
            }

    def _enhanced_confidence_calculation(self, query: str, docs: List[Document], scores: List[float]) -> float:
        """Enhanced confidence calculation with multiple factors"""
        if not scores:
            return 0.0

        try:
            scores_array = np.array(scores)
            if np.max(scores_array) > 1.0:
                scores_array = scores_array / np.max(scores_array)
            scores_array = np.clip(scores_array, 0.0, 1.0)

            factors = {
                'retrieval_score': np.mean(scores_array[:3]),
                'query_coverage': self._calculate_query_coverage(query, docs),
                'answer_consistency': self._check_answer_consistency(docs),
                'source_diversity': self._calculate_source_diversity(docs),
                'entity_match': self._calculate_entity_match(query, docs)
            }

            weights = {
                'retrieval_score': 0.25,
                'query_coverage': 0.25,
                'answer_consistency': 0.20,
                'source_diversity': 0.15,
                'entity_match': 0.15
            }

            confidence = sum(factors[key] * weights[key] for key in factors)

            # Bonus for exact query matches
            query_lower = query.lower()
            for doc in docs[:3]:
                if query_lower in doc.page_content.lower():
                    confidence += 0.1
                    break

            return min(1.0, max(0.0, confidence))

        except Exception as e:
            logger.warning(f"⚠️ Confidence calculation error: {e}")
            return 0.3

    def _calculate_query_coverage(self, query: str, docs: List[Document]) -> float:
        """Calculate query-document match quality"""
        query_terms = set(query.lower().split())
        covered_terms = set()

        for doc in docs[:5]:
            doc_terms = set(doc.page_content.lower().split())
            covered_terms.update(query_terms.intersection(doc_terms))

        return len(covered_terms) / max(len(query_terms), 1)

    def _check_answer_consistency(self, docs: List[Document]) -> float:
        """Check consistency across documents"""
        if len(docs) < 2:
            return 1.0

        # Simple consistency check based on overlapping terms
        all_terms = []
        for doc in docs[:5]:
            terms = set(doc.page_content.lower().split())
            all_terms.append(terms)

        if not all_terms:
            return 0.5

        base_terms = all_terms[0]
        consistency_scores = []

        for terms in all_terms[1:]:
            overlap = len(base_terms.intersection(terms))
            total = len(base_terms.union(terms))
            if total > 0:
                consistency_scores.append(overlap / total)

        return np.mean(consistency_scores) if consistency_scores else 0.5

    def _calculate_source_diversity(self, docs: List[Document]) -> float:
        """Calculate diversity of sources"""
        if not docs:
            return 0.0

        sources = set()
        for doc in docs:
            source = doc.metadata.get('source', 'unknown')
            sources.add(source)

        # Normalize by number of documents
        diversity = len(sources) / len(docs)
        return min(1.0, diversity * 2)  # Scale to give reasonable scores

    def _calculate_entity_match(self, query: str, docs: List[Document]) -> float:
        """Calculate entity matching between query and documents"""
        query_entities = self._extract_entities(query)

        if not query_entities:
            return 0.5

        entity_matches = 0
        for entity in query_entities:
            for doc in docs[:5]:
                if entity.lower() in doc.page_content.lower():
                    entity_matches += 1
                    break

        return entity_matches / len(query_entities)

    def _calculate_confidence(self, query: str, similarity_scores: List[float], retrieved_docs: List[Document]) -> float:
        """Enhanced confidence calculation"""
        if not similarity_scores:
            return 0.0

        try:
            scores_array = np.array(similarity_scores)
            if np.max(scores_array) > 1.0:
                scores_array = scores_array / np.max(scores_array)
            scores_array = np.clip(scores_array, 0.0, 1.0)

            max_score = np.max(scores_array)
            avg_score = np.mean(scores_array)
            score_std = np.std(scores_array) if len(scores_array) > 1 else 0.0
            score_consistency = max(0.0, 1.0 - score_std)

            query_match = self._calculate_query_match(query, retrieved_docs)

            confidence = (
                0.35 * max_score +
                0.25 * avg_score +
                0.25 * query_match +
                0.15 * score_consistency
            )

            # Bonus for exact query matches
            query_lower = query.lower()
            for doc in retrieved_docs[:3]:
                if query_lower in doc.page_content.lower():
                    confidence += 0.1
                    break

            return min(1.0, max(0.0, confidence))

        except Exception as e:
            logger.warning(f"⚠️ Confidence calculation error: {e}")
            return 0.3

    def _calculate_query_match(self, query: str, docs: List[Document]) -> float:
        """Calculate query-document match quality"""
        query_terms = set(query.lower().split())
        if not query_terms:
            return 0.5

        match_scores = []
        for doc in docs[:5]:
            doc_terms = set(doc.page_content.lower().split())
            overlap = len(query_terms.intersection(doc_terms))
            match_score = overlap / len(query_terms)

            if query.lower() in doc.page_content.lower():
                match_score += 0.2

            match_scores.append(match_score)

        return np.mean(match_scores) if match_scores else 0.5

    async def _generate_response(self, query: str, context: str, domain: str, confidence: float) -> str:
        """Generate response using Google Gemini"""
        try:
            await ensure_gemini_ready()

            if not gemini_client:
                return "System is still initializing. Please wait a moment and try again."

            system_prompt = f"""You are an expert document analyst specializing in {domain} content. Provide accurate, helpful responses based on the provided context.

INSTRUCTIONS:
1. Answer questions directly based on the context provided
2. If information is not available in the context, clearly state this
3. Be concise but comprehensive in your responses
4. Cite specific details from the context when relevant
5. Maintain accuracy and avoid speculation beyond the provided information

Confidence Level: {confidence:.1%}"""

            user_message = f"""Context:
{context}

Question: {query}

Please provide a comprehensive answer based on the context above."""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]

            response = await asyncio.wait_for(
                gemini_client.chat.completions.create(
                    messages=messages,
                    model="gemini-2.0-flash",
                    temperature=0.1,
                    max_tokens=1000
                ),
                timeout=QUESTION_TIMEOUT
            )

            return response.choices[0].message.content.strip()

        except asyncio.TimeoutError:
            logger.error(f"❌ Response generation timeout after {QUESTION_TIMEOUT}s")
            return "I apologize, but the response generation took too long. Please try again with a simpler question."

        except Exception as e:
            logger.error(f"❌ Response generation error: {e}")
            return f"I apologize, but I encountered an error while processing your query: {str(e)}"

# ================================
# UTILITY FUNCTIONS WITH SMART CACHING
# ================================

def reciprocal_rank_fusion(results_list: List[List[Tuple[Document, float]]], k_value: int = 60) -> List[Tuple[Document, float]]:
    """Implement Reciprocal Rank Fusion for combining multiple result sets"""
    if not results_list:
        return []

    doc_scores = defaultdict(float)
    seen_docs = {}
    weights = {"semantic": 0.6, "bm25": 0.4}

    for i, results in enumerate(results_list):
        weight = weights.get("semantic" if i == 0 else "bm25", 1.0 / len(results_list))

        for rank, (doc, score) in enumerate(results):
            doc_key = hashlib.md5(doc.page_content.encode()).hexdigest()
            rrf_score = weight / (k_value + rank + 1)
            doc_scores[doc_key] += rrf_score

            if doc_key not in seen_docs:
                seen_docs[doc_key] = doc

    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    max_score = sorted_docs[0][1] if sorted_docs else 1.0

    result = []
    for doc_key, score in sorted_docs:
        normalized_score = score / max_score
        result.append((seen_docs[doc_key], normalized_score))

    return result

async def get_embeddings_batch_optimized(texts: List[str]) -> List[np.ndarray]:
    """Optimized embedding generation with larger batches and parallel processing"""
    if not texts:
        return []

    results = []
    uncached_texts = []
    uncached_indices = []

    # Check cache first
    for i, text in enumerate(texts):
        text_hash = hashlib.md5(text.encode()).hexdigest()
        cached_embedding = CACHE_MANAGER.get_embedding(text_hash)
        if cached_embedding is not None:
            results.append((i, cached_embedding))
        else:
            uncached_texts.append(text)
            uncached_indices.append(i)

    # Process uncached texts in optimized batches
    if uncached_texts:
        await ensure_models_ready()
        if base_sentence_model:
            # Use larger batch size for better performance
            batch_size = OPTIMAL_BATCH_SIZE
            new_embeddings = []
            
            for i in range(0, len(uncached_texts), batch_size):
                batch_texts = uncached_texts[i:i + batch_size]
                
                batch_embeddings = await asyncio.to_thread(
                    base_sentence_model.encode,
                    batch_texts,
                    batch_size=len(batch_texts),  # Process entire batch at once
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
                
                new_embeddings.extend(batch_embeddings)

            # Cache the new embeddings
            for text, embedding in zip(uncached_texts, new_embeddings):
                text_hash = hashlib.md5(text.encode()).hexdigest()
                CACHE_MANAGER.set_embedding(text_hash, embedding)

            # Add to results
            for i, embedding in zip(uncached_indices, new_embeddings):
                results.append((i, embedding))

    # Sort results by original order and return embeddings
    results.sort(key=lambda x: x[0])
    return [embedding for _, embedding in results]

async def get_query_embedding(query: str) -> np.ndarray:
    """Get single query embedding with smart caching"""
    if not query.strip():
        return np.zeros(384)

    query_hash = hashlib.md5(query.encode()).hexdigest()
    cached_embedding = CACHE_MANAGER.get_embedding(query_hash)

    if cached_embedding is not None:
        return cached_embedding

    try:
        await ensure_models_ready()
        if base_sentence_model:
            embedding = await asyncio.to_thread(
                base_sentence_model.encode,
                query,
                convert_to_numpy=True
            )
            CACHE_MANAGER.set_embedding(query_hash, embedding)
            return embedding
        else:
            logger.warning("⚠️ No embedding model available for query")
            return np.zeros(384)

    except Exception as e:
        logger.error(f"❌ Query embedding error: {e}")
        return np.zeros(384)

async def ensure_gemini_ready():
    """Ensure Gemini client is ready"""
    global gemini_client
    
    if gemini_client is None and GEMINI_API_KEY:
        try:
            gemini_client = AsyncOpenAI(
                api_key=GEMINI_API_KEY,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                timeout=httpx.Timeout(connect=10.0, read=60.0, write=30.0, pool=5.0),
                max_retries=3
            )
            logger.info("✅ Gemini client initialized successfully")

        except Exception as e:
            logger.error(f"❌ Failed to initialize Gemini client: {e}")
            raise HTTPException(status_code=503, detail="Gemini client not available")

    elif gemini_client is None:
        raise HTTPException(status_code=503, detail="Gemini API key not configured")

async def ensure_models_ready():
    """Load models only once per container lifecycle - OPTIMIZED"""
    global base_sentence_model, reranker, _models_loaded, _startup_complete

    if _models_loaded and _startup_complete:
        return

    async with _model_lock:
        if _models_loaded and _startup_complete:
            return

        logger.info("🔄 Loading pre-downloaded models...")
        start_time = time.time()

        try:
            if base_sentence_model is None:
                base_sentence_model = SentenceTransformer(
                    EMBEDDING_MODEL_NAME,
                    device='cpu',
                    cache_folder=os.getenv('SENTENCE_TRANSFORMERS_HOME', None)
                )
                base_sentence_model.eval()
                _ = base_sentence_model.encode("warmup", show_progress_bar=False)
                logger.info("✅ Sentence transformer loaded and warmed up")

            if reranker is None:
                reranker = CrossEncoder(
                    RERANKER_MODEL_NAME,
                    max_length=128,
                    device='cpu'
                )
                _ = reranker.predict([["warmup", "test"]])
                logger.info("✅ Reranker loaded and warmed up")

            _models_loaded = True
            _startup_complete = True

            load_time = time.time() - start_time
            logger.info(f"✅ All models ready in {load_time:.2f}s")

        except Exception as e:
            logger.error(f"❌ Failed to load models: {e}")
            raise

# ================================
# GLOBAL INSTANCES
# ================================

DOMAIN_DETECTOR = DomainDetector()

# ================================
# FASTAPI APPLICATION
# ================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager - OPTIMIZED"""
    logger.info("🚀 Starting HackRx RAG System...")
    start_time = time.time()

    try:
        await ensure_models_ready()
        if GEMINI_API_KEY:
            await ensure_gemini_ready()

        startup_time = time.time() - start_time
        logger.info(f"✅ System fully initialized in {startup_time:.2f}s")

    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise

    yield

    logger.info("🔄 Shutting down system...")
    _document_cache.clear()
    CACHE_MANAGER.clear_all_caches()

    if gemini_client and hasattr(gemini_client, 'close'):
        try:
            await gemini_client.close()
        except Exception:
            pass

    logger.info("✅ System shutdown complete")

app = FastAPI(
    title="HackRx RAG System",
    description="Enhanced RAG System for HackRx Evaluation with Google Gemini",
    version="2.0.0",
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
# API ENDPOINTS
# ================================

@app.get("/")
async def root():
    """Basic health check endpoint"""
    return {
        "status": "online",
        "service": "HackRx RAG System with Google Gemini",
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/cache-stats")
async def get_cache_stats():
    """Cache statistics endpoint"""
    return CACHE_MANAGER.get_cache_stats()

@app.post("/hackrx/run")
async def hackrx_run_endpoint(request: Request):
    """HackRx endpoint with OPTIMIZED PROCESSING and FIXED CACHING"""
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

        # FIXED: Better cache key generation and validation
        doc_cache_key = hashlib.md5(documents_url.encode()).hexdigest()
        current_time = time.time()
        cached_rag_system = None

        # Check if we have cached data that's still valid
        if doc_cache_key in _document_cache:
            cached_data, timestamp = _document_cache[doc_cache_key]
            if current_time - timestamp < _cache_ttl:
                logger.info("🚀 Using cached document processing")
                cached_rag_system = cached_data

        if cached_rag_system:
            rag_system = cached_rag_system
        else:
            # Create new RAG system and process documents
            rag_system = RAGSystem()
            logger.info(f"📄 Processing document: {sanitize_pii(documents_url)}")
            await rag_system.process_documents([documents_url])

            # Cache the processed system
            _document_cache[doc_cache_key] = (rag_system, current_time)

            # Clean up old cache entries
            if len(_document_cache) > 10:
                oldest_key = min(_document_cache.keys(),
                               key=lambda k: _document_cache[k][1])
                del _document_cache[oldest_key]

        logger.info(f"❓ Processing {len(questions)} questions in parallel...")

        async def process_single_question(question: str) -> str:
            try:
                result = await rag_system.query(question)
                return result["answer"]
            except Exception as e:
                logger.error(f"❌ Error processing question: {e}")
                return f"Error processing question: {str(e)}"

        # Use semaphore to control concurrency
        semaphore = asyncio.Semaphore(10)

        async def bounded_process(question: str) -> str:
            async with semaphore:
                return await process_single_question(question)

        # Process all questions in parallel
        answers = await asyncio.gather(
            *[bounded_process(q) for q in questions],
            return_exceptions=False
        )

        processing_time = time.time() - start_time
        logger.info(f"✅ Completed {len(questions)} questions in {processing_time:.2f}s")

        return {"answers": answers}

    except Exception as e:
        logger.error(f"❌ HackRx endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

# ================================
# ERROR HANDLERS - STANDARDIZED
# ================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP exception handler - STANDARDIZED FORMAT"""
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
    """General exception handler - STANDARDIZED FORMAT"""
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
    
    port = int(os.getenv("PORT", 8000))
    logger.info(f"🚀 Starting HackRx RAG System with Google Gemini on port {port}")
    
    uvicorn.run(
        "fastx:app",  # Updated to use correct module name
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )
