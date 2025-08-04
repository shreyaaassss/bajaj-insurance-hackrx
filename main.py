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

# NEW IMPORTS FOR PARALLEL PROCESSING
import concurrent.futures
from functools import partial

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
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# OPTIMIZED CONFIGURATION FOR SPEED
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CHUNK_SIZE = 800  # Reduced from 1200
CHUNK_OVERLAP = 100  # Reduced from 200
SEMANTIC_SEARCH_K = 12  # Reduced from 24
CONTEXT_DOCS = 8  # Reduced from 18
CONFIDENCE_THRESHOLD = 0.15
RERANK_TOP_K = 20  # Reduced from 35
MAX_FILE_SIZE_MB = 50  # Reduced from 100
QUESTION_TIMEOUT = 15.0  # Reduced from 25.0

# PARALLEL PROCESSING - OPTIMIZED
OPTIMAL_BATCH_SIZE = 16  # Reduced from 32
MAX_PARALLEL_BATCHES = 2  # Reduced from 4
EMBEDDING_TIMEOUT = 60.0  # Reduced from 120.0
PARALLEL_PROCESSING_THRESHOLD = 32  # Reduced from 64

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

# Cache for document processing
_document_cache = {}
_cache_ttl = 1800  # 30 minutes

# Global models
base_sentence_model = None
reranker = None
openai_client = None

# ================================
# ENHANCED CACHING SYSTEM
# ================================

class SmartCacheManager:
    """Smart cache manager with TTL/LRU primary and dict fallback"""
    
    def __init__(self):
        # Primary caches (TTL for embeddings, LRU for document chunks)
        try:
            if HAS_CACHETOOLS:
                self.embedding_cache = cachetools.TTLCache(maxsize=10000, ttl=86400)  # 24h TTL
                self.document_chunk_cache = cachetools.LRUCache(maxsize=500)
                self.domain_cache = cachetools.LRUCache(maxsize=1000)
                self.primary_available = True
                logger.info("✅ Advanced caching with TTL/LRU enabled")
            else:
                raise ImportError("cachetools not available")
        except ImportError:
            # Fallback to simple dicts if cachetools not available
            self.embedding_cache = {}
            self.document_chunk_cache = {}
            self.domain_cache = {}
            self.primary_available = False
            logger.info("📦 Using dict fallback caching (cachetools not available)")
        
        # Thread-safe access
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
                # Fallback cleanup for simple dicts
                if len(self.embedding_cache) > 10000:
                    # Keep most recent 5000
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

# Global cache manager instance
CACHE_MANAGER = SmartCacheManager()

# ================================
# PARALLEL PROGRESS TRACKER
# ================================

class ParallelProgressTracker:
    def __init__(self, total_batches: int):
        self.total_batches = total_batches
        self.completed_batches = 0
        self.start_time = time.time()
        self._lock = threading.Lock()
    
    def update_progress(self, batch_idx: int):
        with self._lock:
            self.completed_batches += 1
            elapsed = time.time() - self.start_time
            rate = self.completed_batches / elapsed if elapsed > 0 else 0
            eta = (self.total_batches - self.completed_batches) / rate if rate > 0 else 0
            logger.info(f"🔄 Parallel batch {batch_idx + 1}/{self.total_batches} "
                       f"({self.completed_batches}/{self.total_batches}) "
                       f"Rate: {rate:.1f}/s ETA: {eta:.0f}s")

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
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # emails
        r'\b\d{3}-\d{3}-\d{4}\b',  # phone numbers
        r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'  # credit cards
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
    elif hasattr(data, 'item'):  # numpy scalar
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
            
            # Add metadata
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
        """Load document from URL - FIXED URL PROCESSING"""
        # FIXED: Improved URL scheme validation
        parsed = urlparse(url)
        scheme = parsed.scheme.lower()
        
        # Handle special schemes
        if scheme in ['drive', 'dropbox']:
            # Convert to https scheme for processing
            if scheme == 'drive':
                url = url.replace('drive:', 'https://')
            elif scheme == 'dropbox':
                url = url.replace('dropbox:', 'https://')
        
        if not validate_url_scheme(url):
            raise ValueError(f"Unsupported URL scheme: {scheme}")
        
        # Transform special URLs
        download_url = self._transform_special_url(url)
        
        timeout = 30.0
        if any(pattern in url for pattern in ['drive.google.com', 'dropbox.com']):
            timeout = 60.0
        
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(download_url)
            response.raise_for_status()
            content = response.content
        
        # Determine file extension
        file_ext = self._get_extension_from_url(url) or self._detect_extension_from_content(content)
        
        # Save to temporary file and process
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            tmp_file.write(content)
            temp_path = tmp_file.name
        
        try:
            return await self._load_from_file(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def _transform_special_url(self, url: str) -> str:
        """Transform special URLs to direct download links - ENHANCED"""
        # FIXED: Enhanced Google Drive transformation
        for pattern in self.drive_patterns:
            match = re.search(pattern, url)
            if match:
                file_id = match.group(1) if match.groups() else None
                if file_id:
                    return f"https://drive.google.com/uc?export=download&id={file_id}"
        
        # FIXED: Enhanced Dropbox transformation
        for pattern in self.dropbox_patterns:
            if re.search(pattern, url):
                # Handle both sharing formats
                if '?dl=0' in url:
                    url = url.replace('?dl=0', '?dl=1')
                elif '?dl=1' not in url:
                    url += '?dl=1' if '?' not in url else '&dl=1'
                return url.replace('dropbox.com', 'dl.dropboxusercontent.com')
        
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
        
        # Basic content-based detection
        if content.startswith(b'%PDF'):
            return '.pdf'
        elif b'PK' in content[:10]:  # ZIP-based formats
            return '.docx'
        return '.txt'
    
    async def _load_from_file(self, file_path: str) -> List[Document]:
        """Load document from file"""
        file_extension = os.path.splitext(file_path)[1].lower()
        file_size = os.path.getsize(file_path)
        
        # Validate file size
        if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
            raise ValueError(f"File too large: {file_size / (1024*1024):.1f}MB (max: {MAX_FILE_SIZE_MB}MB)")
        
        if not validate_file_extension(file_path):
            raise ValueError(f"Unsupported file extension: {file_extension}")
        
        logger.info(f"📄 Loading {file_extension} file ({file_size} bytes): {sanitize_pii(file_path)}")
        
        # Enhanced MIME detection
        mime_type = None
        if self.mime_detector:
            try:
                mime_type = magic.from_file(file_path, mime=True)
            except Exception as e:
                logger.warning(f"⚠️ MIME detection failed: {e}")
        
        docs = None
        loader_used = None
        
        # PDF handling
        if mime_type == 'application/pdf' or file_extension == '.pdf':
            try:
                loader = PyMuPDFLoader(file_path)
                docs = await asyncio.to_thread(loader.load)
                loader_used = "PyMuPDFLoader"
            except Exception as e:
                logger.warning(f"⚠️ PyMuPDF failed: {e}")
        
        # Word document handling
        elif ('word' in (mime_type or '') or 
              'officedocument' in (mime_type or '') or 
              file_extension in ['.docx', '.doc']):
            try:
                loader = Docx2txtLoader(file_path)
                docs = await asyncio.to_thread(loader.load)
                loader_used = "Docx2txtLoader"
            except Exception as e:
                logger.warning(f"⚠️ DOCX loader failed: {e}")
        
        # Text file handling
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
        
        # Fallback to text loader
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
        
        # Add enhanced metadata
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
# ADAPTIVE TEXT SPLITTER - WITH SMART CACHING
# ================================

class AdaptiveTextSplitter:
    """Adaptive text splitter with smart caching"""
    
    def __init__(self):
        self.separators = [
            "\n\n## ", "\n\n# ", "\n\n### ", "\n\n#### ",
            "\n\n", "\n", ". ", "! ", "? ", "; ", " ", ""
        ]
    
    def split_documents(self, documents: List[Document], detected_domain: str = "general") -> List[Document]:
        """Split documents with smart caching"""
        if not documents:
            return []
        
        # Check cache using cache manager
        content_hash = self._calculate_content_hash(documents)
        cache_key = f"chunks_{content_hash}_{detected_domain}"
        
        cached_chunks = CACHE_MANAGER.get_document_chunks(cache_key)
        if cached_chunks is not None:
            logger.info(f"📄 Using cached chunks: {len(cached_chunks)} chunks")
            return cached_chunks
        
        # Adapt chunk size for content
        chunk_size, chunk_overlap = self._adapt_for_content(documents, detected_domain)
        
        all_chunks = []
        for doc in documents:
            try:
                chunks = self._split_document(doc, chunk_size, chunk_overlap)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.warning(f"⚠️ Error splitting document: {e}")
                # Fallback to simple splitting
                chunks = self._simple_split(doc, chunk_size, chunk_overlap)
                all_chunks.extend(chunks)
        
        # Filter very short chunks
        all_chunks = [chunk for chunk in all_chunks if len(chunk.page_content.strip()) >= 50]
        
        # Cache the result using cache manager
        CACHE_MANAGER.set_document_chunks(cache_key, all_chunks)
        
        logger.info(f"📄 Created {len(all_chunks)} adaptive chunks")
        return all_chunks
    
    def _calculate_content_hash(self, documents: List[Document]) -> str:
        """Calculate hash for content caching"""
        content_sample = "".join([doc.page_content[:100] for doc in documents[:5]])
        return hashlib.sha256(content_sample.encode()).hexdigest()[:16]
    
    def _adapt_for_content(self, documents: List[Document], detected_domain: str) -> Tuple[int, int]:
        """Adapt chunk size based on content and domain"""
        # Domain-specific adjustments
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
        
        # Ensure reasonable bounds
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
    
    async def add_documents(self, documents: List[Document], embeddings: List[np.ndarray]):
        """Add documents with optimized batch processing"""
        try:
            if not self.is_trained:
                self.initialize()
            
            if len(documents) != len(embeddings):
                raise ValueError("Number of documents must match number of embeddings")
            
            # Convert embeddings in chunks to avoid memory spikes
            chunk_size = 256  # Process 256 embeddings at a time
            
            for i in range(0, len(embeddings), chunk_size):
                end_idx = min(i + chunk_size, len(embeddings))
                chunk_embeddings = np.array(embeddings[i:end_idx], dtype=np.float32)
                
                # Normalize embeddings for cosine similarity
                faiss.normalize_L2(chunk_embeddings)
                
                # Add to index
                self.index.add(chunk_embeddings)
                
                # Add documents
                self.documents.extend(documents[i:end_idx])
                
                logger.info(f"✅ Added chunk {i//chunk_size + 1}/{(len(embeddings)-1)//chunk_size + 1}")
            
            logger.info(f"✅ Added {len(documents)} documents to FAISS index (total: {len(self.documents)})")
            
        except Exception as e:
            logger.error(f"❌ Error adding documents to FAISS: {e}")
            raise
    
    async def similarity_search_with_score(self, query_embedding: np.ndarray, k: int = 10) -> List[Tuple[Document, float]]:
        """Search for similar documents with scores"""
        try:
            if not self.is_trained or len(self.documents) == 0:
                return []
            
            # Ensure query embedding is the right shape and type
            query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
            
            # Normalize for cosine similarity
            faiss.normalize_L2(query_embedding)
            
            # Search
            k = min(k, len(self.documents))
            scores, indices = self.index.search(query_embedding, k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx < len(self.documents):
                    doc = self.documents[idx]
                    # FIXED: Add score normalization for FAISS inner product scores
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
            # Keyword-based detection with optimized text
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
        """Keyword-based domain detection - FIXED division by zero"""
        domain_scores = {}
        
        for domain, keywords in DOMAIN_KEYWORDS.items():
            matches = 0
            for keyword in keywords:
                matches += combined_text.count(keyword.lower())
            
            # FIXED: Guard against division by zero
            text_length = max(len(combined_text), 1)
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
        """Accurate token estimation with fallback"""
        if not text:
            return 0
        
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception:
                pass
        
        # Fallback to heuristic
        avg_chars_per_token = 3.5
        if any(term in text.lower() for term in ['api', 'json', 'xml', 'code', 'function']):
            avg_chars_per_token = 3.0
        
        estimated = len(text) / avg_chars_per_token
        return max(1, int(estimated * 1.1))
    
    def optimize_context(self, documents: List[Document], query: str, max_tokens: int = None) -> str:
        """Optimize context for token limit"""
        if not documents:
            return ""
        
        if max_tokens is None:
            max_tokens = self.max_context_tokens
        
        # Calculate relevance scores
        doc_scores = []
        for doc in documents:
            relevance = self._calculate_relevance(doc, query)
            tokens = self.estimate_tokens(doc.page_content)
            efficiency = relevance / max(tokens, 1)
            doc_scores.append((doc, relevance, tokens, efficiency))
        
        # Sort by efficiency
        doc_scores.sort(key=lambda x: x[3], reverse=True)
        
        context_parts = []
        token_budget = max_tokens - 200  # Buffer
        
        for doc, relevance, tokens, efficiency in doc_scores:
            if tokens <= token_budget:
                context_parts.append(doc.page_content)
                token_budget -= tokens
            elif token_budget > 200 and relevance > 0.6:
                # Include highly relevant documents with truncation
                partial_content = self._truncate_content(doc.page_content, token_budget)
                context_parts.append(partial_content)
                break
        
        context = "\n\n".join(context_parts)
        estimated_tokens = self.estimate_tokens(context)
        
        logger.info(f"🔍 Context optimized: {len(context_parts)} documents, ~{estimated_tokens} tokens")
        return context
    
    def _calculate_relevance(self, doc: Document, query: str) -> float:
        """Calculate document relevance"""
        try:
            query_terms = set(query.lower().split())
            doc_terms = set(doc.page_content.lower().split())
            
            # Keyword overlap
            keyword_overlap = len(query_terms.intersection(doc_terms)) / max(len(query_terms), 1)
            
            # Position boost (earlier chunks slightly preferred)
            position_boost = 1.0
            chunk_index = doc.metadata.get('chunk_index', 0)
            total_chunks = doc.metadata.get('total_chunks', 1)
            if total_chunks > 1:
                position_boost = 1.15 - (chunk_index / total_chunks) * 0.3
            
            # Length penalty for very short chunks
            length_penalty = 1.0
            if len(doc.page_content) < 100:
                length_penalty = 0.8
            
            final_score = keyword_overlap * position_boost * length_penalty
            return min(1.0, final_score)
            
        except Exception as e:
            logger.warning(f"⚠️ Error calculating relevance: {e}")
            return 0.5
    
    def _truncate_content(self, content: str, max_tokens: int) -> str:
        """Smart content truncation"""
        max_chars = max_tokens * 4
        
        if len(content) <= max_chars:
            return content
        
        keep_chars = max_chars - 100
        first_part = content[:keep_chars//2]
        last_part = content[-keep_chars//2:]
        
        return f"{first_part}\n\n[... content truncated ...]\n\n{last_part}"

# ================================
# RAG SYSTEM - ENHANCED
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
        """RAGSystem cleanup method"""
        self.documents.clear()
        if self.vector_store:
            self.vector_store.clear()
        logger.info("🧹 RAGSystem cleaned up")
    
    async def process_documents(self, sources: List[str]) -> Dict[str, Any]:
        """Process documents with smart caching - OPTIMIZED"""
        start_time = time.time()
        
        # Generate document signature for caching
        doc_signature = hashlib.md5(str(sorted(sources)).encode()).hexdigest()
        
        # Check if we already processed these exact documents
        if hasattr(self, '_last_doc_signature') and self._last_doc_signature == doc_signature:
            logger.info("📄 Documents already processed, skipping...")
            return {"cached": True, "processing_time": 0.001}
        
        # Only clear caches if documents actually changed
        if not hasattr(self, '_last_doc_signature') or self._last_doc_signature != doc_signature:
            CACHE_MANAGER.clear_all_caches()
            self._last_doc_signature = doc_signature
            logger.info("🧹 Caches cleared for new documents")
        
        try:
            logger.info(f"📄 Processing {len(sources)} documents")
            
            # Load documents
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
            
            # Detect domain (fast)
            domain, domain_confidence = DOMAIN_DETECTOR.detect_domain(raw_documents)
            self.domain = domain
            
            # Split documents (optimized)
            self.documents = self.text_splitter.split_documents(raw_documents, domain)
            
            # Setup retrievers (parallel where possible)
            await self._setup_retrievers()
            
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
    
    async def _setup_retrievers(self):
        """Setup retrievers with FAISS and BM25"""
        try:
            logger.info("🔧 Setting up retrievers...")
            
            # Setup FAISS vector store
            if HAS_FAISS and self.documents:
                try:
                    await ensure_models_ready()
                    self.vector_store = FAISSVectorStore(dimension=384)
                    self.vector_store.initialize()
                    
                    # Get embeddings
                    doc_texts = [doc.page_content for doc in self.documents]
                    embeddings = await get_embeddings_batch(doc_texts)
                    
                    # Add to FAISS
                    await self.vector_store.add_documents(self.documents, embeddings)
                    logger.info("✅ FAISS vector store setup complete")
                    
                except Exception as e:
                    logger.warning(f"⚠️ FAISS setup failed: {e}")
                    self.vector_store = None
            
            # Setup BM25 retriever
            try:
                if self.documents:
                    self.bm25_retriever = await asyncio.to_thread(
                        BM25Retriever.from_documents, self.documents
                    )
                    self.bm25_retriever.k = min(RERANK_TOP_K, len(self.documents))
                    logger.info(f"✅ BM25 retriever setup complete (k={self.bm25_retriever.k})")
                    
            except Exception as e:
                logger.warning(f"⚠️ BM25 retriever setup failed: {e}")
                
        except Exception as e:
            logger.error(f"❌ Retriever setup error: {e}")
    
    async def retrieve_and_rerank(self, query: str, top_k: int = None) -> Tuple[List[Document], List[float]]:
        """Retrieve and rerank documents"""
        if top_k is None:
            top_k = CONTEXT_DOCS
        
        try:
            if not self.documents:
                return [], []
            
            # Get query embedding
            query_embedding = await get_query_embedding(query)
            
            # Retrieve from multiple sources
            retrieval_results = []
            
            # Vector search
            if self.vector_store:
                try:
                    vector_results = await self.vector_store.similarity_search_with_score(
                        query_embedding, k=SEMANTIC_SEARCH_K
                    )
                    if vector_results:
                        retrieval_results.append(vector_results)
                except Exception as e:
                    logger.warning(f"⚠️ Vector search failed: {e}")
            
            # BM25 search with null check
            if self.bm25_retriever:
                try:
                    # FIXED: BM25 null check
                    bm25_docs = await asyncio.to_thread(self.bm25_retriever.invoke, query) or []
                    bm25_results = []
                    for i, doc in enumerate(bm25_docs):
                        score = 1.0 - (i * 0.1)  # Decreasing scores
                        score = max(0.1, score)
                        bm25_results.append((doc, score))
                    
                    if bm25_results:
                        retrieval_results.append(bm25_results)
                        
                except Exception as e:
                    logger.warning(f"⚠️ BM25 search failed: {e}")
            
            # Apply Reciprocal Rank Fusion if multiple result sets
            if len(retrieval_results) > 1:
                fused_results = reciprocal_rank_fusion(retrieval_results)
                final_docs = [doc for doc, score in fused_results[:RERANK_TOP_K]]
                final_scores = [score for doc, score in fused_results[:RERANK_TOP_K]]
            elif len(retrieval_results) == 1:
                results = retrieval_results[0]
                final_docs = [doc for doc, score in results[:RERANK_TOP_K]]
                final_scores = [score for doc, score in results[:RERANK_TOP_K]]
            else:
                # Fallback search
                final_docs, final_scores = await self._fallback_search(query, SEMANTIC_SEARCH_K)
            
            # Apply reranking if available
            if reranker and len(final_docs) > 1:
                try:
                    final_docs, final_scores = await self._semantic_rerank(
                        query, final_docs, final_scores, top_k
                    )
                except Exception as e:
                    logger.warning(f"⚠️ Reranking failed: {e}")
                    final_docs = final_docs[:top_k]
                    final_scores = final_scores[:top_k]
            else:
                final_docs = final_docs[:top_k]
                final_scores = final_scores[:top_k]
            
            return final_docs, final_scores
            
        except Exception as e:
            logger.error(f"❌ Retrieval error: {e}")
            return self.documents[:top_k], [0.5] * min(len(self.documents), top_k)
    
    async def _fallback_search(self, query: str, k: int) -> Tuple[List[Document], List[float]]:
        """Fallback search using keyword matching"""
        try:
            query_terms = set(query.lower().split())
            doc_scores = []
            
            for doc in self.documents:
                doc_terms = set(doc.page_content.lower().split())
                overlap = len(query_terms.intersection(doc_terms))
                score = overlap / max(len(query_terms), 1)
                
                # Boost for exact phrase matches
                if query.lower() in doc.page_content.lower():
                    score += 0.3
                
                doc_scores.append((doc, score))
            
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            docs = [doc for doc, score in doc_scores[:k]]
            scores = [score for doc, score in doc_scores[:k]]
            
            return docs, scores
            
        except Exception as e:
            logger.error(f"❌ Fallback search error: {e}")
            return self.documents[:k], [0.5] * min(len(self.documents), k)
    
    async def _semantic_rerank(self, query: str, documents: List[Document], 
                             scores: List[float], top_k: int) -> Tuple[List[Document], List[float]]:
        """Semantic reranking with cross-encoder"""
        try:
            if len(documents) <= 2:
                return documents[:top_k], scores[:top_k]
            
            await ensure_models_ready()
            
            if not reranker:
                return documents[:top_k], scores[:top_k]
            
            # Create pairs for reranking
            pairs = [[query, doc.page_content[:512]] for doc in documents]
            
            # Synchronous call - no async needed
            rerank_scores = reranker.predict(pairs)
            
            # Normalize rerank scores
            normalized_rerank = [(score + 1) / 2 for score in rerank_scores[:len(documents)]]
            
            # Combine scores
            combined_scores = []
            for i, (orig_score, rerank_score) in enumerate(zip(scores[:len(normalized_rerank)], normalized_rerank)):
                combined = 0.7 * rerank_score + 0.3 * orig_score
                combined_scores.append(min(1.0, combined))
            
            # Sort by combined scores
            scored_docs = list(zip(documents, combined_scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            final_docs = [doc for doc, _ in scored_docs[:top_k]]
            final_scores = [score for _, score in scored_docs[:top_k]]
            
            return final_docs, final_scores
            
        except Exception as e:
            logger.error(f"❌ Semantic reranking error: {e}")
            return documents[:top_k], scores[:top_k]
    
    async def query(self, query: str) -> Dict[str, Any]:
        """Process query and return response"""
        start_time = time.time()
        
        try:
            # Simple cache cleanup for non-TTL caches
            CACHE_MANAGER.cleanup_if_needed()
            
            # Retrieve documents
            retrieved_docs, similarity_scores = await self.retrieve_and_rerank(query, CONTEXT_DOCS)
            
            if not retrieved_docs:
                return {
                    "query": query,
                    "answer": "No relevant documents found for your query.",
                    "confidence": 0.0,
                    "domain": self.domain,
                    "processing_time": time.time() - start_time
                }
            
            # Calculate confidence
            confidence = self._calculate_confidence(query, similarity_scores, retrieved_docs)
            
            # FIXED: Add confidence threshold check
            if confidence < CONFIDENCE_THRESHOLD:
                return {
                    "query": query,
                    "answer": "I don't have enough relevant information to answer this question accurately.",
                    "confidence": float(confidence),
                    "domain": self.domain,
                    "retrieved_chunks": len(retrieved_docs),
                    "processing_time": time.time() - start_time
                }
            
            # Optimize context
            context = self.token_processor.optimize_context(retrieved_docs, query)
            
            # Generate response
            answer = await self._generate_response(query, context, self.domain, confidence)
            
            processing_time = time.time() - start_time
            result = {
                "query": query,
                "answer": answer,
                "confidence": float(confidence),
                "domain": self.domain,
                "retrieved_chunks": len(retrieved_docs),
                "processing_time": processing_time
            }
            
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
    
    def _calculate_confidence(self, query: str, similarity_scores: List[float], 
                            retrieved_docs: List[Document]) -> float:
        """Enhanced confidence calculation"""
        if not similarity_scores:
            return 0.0
        
        try:
            scores_array = np.array(similarity_scores)
            
            # FIXED: Handle FAISS inner product scores > 1.0
            if np.max(scores_array) > 1.0:
                scores_array = scores_array / np.max(scores_array)
            scores_array = np.clip(scores_array, 0.0, 1.0)
            
            max_score = np.max(scores_array)
            avg_score = np.mean(scores_array)
            score_std = np.std(scores_array) if len(scores_array) > 1 else 0.0
            
            # Score consistency
            score_consistency = max(0.0, 1.0 - score_std)
            
            # Query-document match
            query_match = self._calculate_query_match(query, retrieved_docs)
            
            # Combined confidence
            confidence = (
                0.35 * max_score +
                0.25 * avg_score +
                0.25 * query_match +
                0.15 * score_consistency
            )
            
            # Boost for exact phrase matches
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
            
            # Boost for phrase matches
            if query.lower() in doc.page_content.lower():
                match_score += 0.2
            
            match_scores.append(match_score)
        
        return np.mean(match_scores) if match_scores else 0.5
    
    async def _generate_response(self, query: str, context: str, domain: str, confidence: float) -> str:
        """Generate response using OpenAI"""
        try:
            await ensure_openai_ready()
            
            if not openai_client:
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
            
            # FIXED: Apply timeout configuration
            response = await asyncio.wait_for(
                openai_client.chat.completions.create(
                    messages=messages,
                    model="gpt-4o",
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

def reciprocal_rank_fusion(results_list: List[List[Tuple[Document, float]]], 
                          k_value: int = 60) -> List[Tuple[Document, float]]:
    """Implement Reciprocal Rank Fusion for combining multiple result sets"""
    if not results_list:
        return []
    
    # Document scores accumulator
    doc_scores = defaultdict(float)
    seen_docs = {}
    
    # Weights for semantic and BM25
    weights = {"semantic": 0.6, "bm25": 0.4}
    
    # Process each result set
    for i, results in enumerate(results_list):
        weight = weights.get("semantic" if i == 0 else "bm25", 1.0 / len(results_list))
        
        # Apply RRF scoring
        for rank, (doc, score) in enumerate(results):
            doc_key = hashlib.md5(doc.page_content.encode()).hexdigest()
            rrf_score = weight / (k_value + rank + 1)
            doc_scores[doc_key] += rrf_score
            
            if doc_key not in seen_docs:
                seen_docs[doc_key] = doc
    
    # Sort by combined scores
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Return documents with normalized scores
    max_score = sorted_docs[0][1] if sorted_docs else 1.0
    result = []
    for doc_key, score in sorted_docs:
        normalized_score = score / max_score
        result.append((seen_docs[doc_key], normalized_score))
    
    return result

async def get_embeddings_batch(texts: List[str]) -> List[np.ndarray]:
    """Process embeddings with parallel batch processing - OPTIMIZED"""
    if not texts:
        return []
    
    results = []
    uncached_texts = []
    uncached_indices = []
    
    # Check cache first (unchanged)
    for i, text in enumerate(texts):
        text_hash = hashlib.md5(text.encode()).hexdigest()
        cached_embedding = CACHE_MANAGER.get_embedding(text_hash)
        if cached_embedding is not None:
            results.append((i, cached_embedding))
        else:
            uncached_texts.append(text)
            uncached_indices.append(i)
    
    # Process uncached texts with PARALLEL batching
    if uncached_texts:
        try:
            await ensure_models_ready()
            
            if base_sentence_model:
                # PARALLEL PROCESSING - KEY OPTIMIZATION
                if len(uncached_texts) > PARALLEL_PROCESSING_THRESHOLD:
                    logger.info(f"🚀 Processing {len(uncached_texts)} embeddings in {(len(uncached_texts)-1)//OPTIMAL_BATCH_SIZE + 1} parallel batches")
                    
                    # Create batches
                    batches = []
                    for i in range(0, len(uncached_texts), OPTIMAL_BATCH_SIZE):
                        batch = uncached_texts[i:i + OPTIMAL_BATCH_SIZE]
                        batches.append(batch)
                    
                    # Initialize progress tracker
                    progress_tracker = ParallelProgressTracker(len(batches))
                    
                    # Process batches in parallel using ThreadPoolExecutor
                    with concurrent.futures.ThreadPoolExecutor(max_workers=min(MAX_PARALLEL_BATCHES, len(batches))) as executor:
                        # Create partial function for thread pool
                        encode_func = partial(
                            base_sentence_model.encode,
                            show_progress_bar=False,
                            convert_to_numpy=True
                        )
                        
                        # Submit all batches for parallel processing
                        future_to_batch = {
                            executor.submit(encode_func, batch): batch_idx
                            for batch_idx, batch in enumerate(batches)
                        }
                        
                        # Collect results maintaining order
                        batch_results = [None] * len(batches)
                        for future in concurrent.futures.as_completed(future_to_batch):
                            batch_idx = future_to_batch[future]
                            try:
                                batch_embeddings = future.result()
                                batch_results[batch_idx] = batch_embeddings
                                progress_tracker.update_progress(batch_idx)
                            except Exception as e:
                                logger.error(f"❌ Batch {batch_idx} failed: {e}")
                                # Fallback to zero vectors
                                batch_size = len(batches[batch_idx])
                                batch_results[batch_idx] = np.zeros((batch_size, 384))
                        
                        # Flatten results
                        embeddings = []
                        for batch_result in batch_results:
                            if batch_result is not None:
                                embeddings.extend(batch_result)
                                
                elif len(uncached_texts) > OPTIMAL_BATCH_SIZE:
                    logger.info(f"🔄 Processing {len(uncached_texts)} embeddings in batches of {OPTIMAL_BATCH_SIZE}")
                    embeddings = []
                    
                    for i in range(0, len(uncached_texts), OPTIMAL_BATCH_SIZE):
                        batch = uncached_texts[i:i + OPTIMAL_BATCH_SIZE]
                        batch_embeddings = await asyncio.to_thread(
                            base_sentence_model.encode,
                            batch,
                            show_progress_bar=False,
                            convert_to_numpy=True
                        )
                        embeddings.extend(batch_embeddings)
                        logger.info(f"✅ Processed batch {i//OPTIMAL_BATCH_SIZE + 1}/{(len(uncached_texts)-1)//OPTIMAL_BATCH_SIZE + 1}")
                        
                else:
                    # Process normally for small batches
                    embeddings = await asyncio.to_thread(
                        base_sentence_model.encode,
                        uncached_texts,
                        show_progress_bar=False,
                        convert_to_numpy=True
                    )
                
                # Cache embeddings (unchanged)
                for text, embedding in zip(uncached_texts, embeddings):
                    text_hash = hashlib.md5(text.encode()).hexdigest()
                    CACHE_MANAGER.set_embedding(text_hash, embedding)
                
                # Add to results
                for i, embedding in zip(uncached_indices, embeddings):
                    results.append((i, embedding))
                    
            else:
                logger.warning("⚠️ No embedding model available, using zero vectors")
                for i in uncached_indices:
                    results.append((i, np.zeros(384)))
                    
        except Exception as e:
            logger.error(f"❌ Embedding error: {e}")
            for i in uncached_indices:
                results.append((i, np.zeros(384)))
    
    # Sort results by original order
    results.sort(key=lambda x: x[0])
    embeddings = [embedding for _, embedding in results]
    
    return embeddings

async def get_query_embedding(query: str) -> np.ndarray:
    """Get single query embedding with smart caching - FIXED blocking call"""
    if not query.strip():
        return np.zeros(384)
    
    query_hash = hashlib.md5(query.encode()).hexdigest()
    cached_embedding = CACHE_MANAGER.get_embedding(query_hash)
    if cached_embedding is not None:
        return cached_embedding
    
    try:
        await ensure_models_ready()
        
        if base_sentence_model:
            # FIXED: Off-load blocking encoder call to thread
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

async def ensure_openai_ready():
    """Ensure OpenAI client is ready"""
    global openai_client
    
    if openai_client is None and OPENAI_API_KEY:
        try:
            openai_client = AsyncOpenAI(
                api_key=OPENAI_API_KEY,
                timeout=httpx.Timeout(connect=10.0, read=60.0, write=30.0, pool=5.0),
                max_retries=3
            )
            logger.info("✅ OpenAI client initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize OpenAI client: {e}")
            raise HTTPException(status_code=503, detail="OpenAI client not available")
    elif openai_client is None:
        raise HTTPException(status_code=503, detail="OpenAI API key not configured")

async def ensure_models_ready():
    """Load models only once per container lifecycle - OPTIMIZED"""
    global base_sentence_model, reranker, _models_loaded, _startup_complete
    
    if _models_loaded and _startup_complete:
        return
    
    async with _model_lock:
        if _models_loaded and _startup_complete:  # Double-check pattern
            return
            
        logger.info("🔄 Loading pre-downloaded models...")
        start_time = time.time()
        
        try:
            # Load sentence transformer with CPU optimization
            if base_sentence_model is None:
                base_sentence_model = SentenceTransformer(
                    EMBEDDING_MODEL_NAME,
                    device='cpu',
                    cache_folder=os.getenv('SENTENCE_TRANSFORMERS_HOME', None)
                )
                # Warm up the model
                _ = base_sentence_model.encode("warmup", show_progress_bar=False)
                logger.info("✅ Sentence transformer loaded and warmed up")
            
            # Load reranker with optimization
            if reranker is None:
                reranker = CrossEncoder(
                    RERANKER_MODEL_NAME,
                    max_length=256,  # Reduced for speed
                    device='cpu'
                )
                # Warm up reranker
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
    # Startup
    logger.info("🚀 Starting HackRx RAG System...")
    start_time = time.time()
    
    try:
        # Load models first (critical path)
        await ensure_models_ready()
        
        # Initialize OpenAI client
        if OPENAI_API_KEY:
            await ensure_openai_ready()
        
        startup_time = time.time() - start_time
        logger.info(f"✅ System fully initialized in {startup_time:.2f}s")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🔄 Shutting down system...")
    _document_cache.clear()
    CACHE_MANAGER.clear_all_caches()
    
    if openai_client and hasattr(openai_client, 'close'):
        try:
            await openai_client.close()
        except Exception:
            pass
    
    logger.info("✅ System shutdown complete")

# Initialize FastAPI app
app = FastAPI(
    title="HackRx RAG System",
    description="Enhanced RAG System for HackRx Evaluation",
    version="2.0.0",
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
    """Basic health check endpoint"""
    return {
        "status": "online",
        "service": "HackRx RAG System",
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/cache-stats")
async def get_cache_stats():
    """Cache statistics endpoint"""
    return CACHE_MANAGER.get_cache_stats()

@app.post("/hackrx/run")
async def hackrx_run_endpoint(request: Request):
    """HackRx endpoint with PARALLEL PROCESSING and CACHING"""
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
            rag_system = RAGSystem()
            logger.info(f"📄 Processing document: {sanitize_pii(documents_url)}")
            await rag_system.process_documents([documents_url])
            
            # Cache the processed system
            _document_cache[doc_cache_key] = (rag_system, current_time)
            
            # Cleanup old cache entries
            if len(_document_cache) > 10:
                oldest_key = min(_document_cache.keys(), 
                               key=lambda k: _document_cache[k][1])
                del _document_cache[oldest_key]
        
        # PARALLEL QUESTION PROCESSING
        logger.info(f"❓ Processing {len(questions)} questions in parallel...")
        
        async def process_single_question(question: str) -> str:
            try:
                result = await rag_system.query(question)
                return result["answer"]
            except Exception as e:
                logger.error(f"❌ Error processing question: {e}")
                return f"Error processing question: {str(e)}"
        
        # Control concurrency to avoid overwhelming the system
        semaphore = asyncio.Semaphore(3)  # Max 3 concurrent questions
        
        async def bounded_process(question: str) -> str:
            async with semaphore:
                return await process_single_question(question)
        
        # Process all questions concurrently
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
    
    # Get port from environment variable (Cloud Run requirement)
    port = int(os.getenv("PORT", 8000))
    
    logger.info(f"🚀 Starting HackRx RAG System on port {port}")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )
