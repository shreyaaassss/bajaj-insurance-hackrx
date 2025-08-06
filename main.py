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
# HYBRID CONFIGURATION
# ================================

# Fixed tokens
HACKRX_TOKEN = "9a1163c13e8927960b857a674794a62c57baf588998981151b0753a4d6d17905"
GEMINI_API_KEY = 'AIzaSyBVq4GQbmzFwivcQ0cPY3qp8PyVyR613NM'

# Model configuration
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# HYBRID CHUNKING STRATEGY (from recommendation #1)
CHUNK_SIZE = 1200  # Larger chunks for better context
CHUNK_OVERLAP = 200  # Better overlap for continuity

# HYBRID CONTEXT WINDOW (from recommendation #2)
CONTEXT_DOCS = 18  # Better recall from 62.py
MAX_CONTEXT_CHARS = 12000  # Latency safeguard from 61.py

# Other optimized settings
SEMANTIC_SEARCH_K = 12
CONFIDENCE_THRESHOLD = 0.15
RERANK_TOP_K = 20
MAX_FILE_SIZE_MB = 50
QUESTION_TIMEOUT = 8.0

# HYBRID PARALLEL PROCESSING (from recommendation #8)
OPTIMAL_BATCH_SIZE = 64  # From 61.py for speed
MAX_PARALLEL_BATCHES = 8  # From 61.py
EMBEDDING_TIMEOUT = 60.0
MAX_CONCURRENT_QUESTIONS = 16

# Supported file types
SUPPORTED_EXTENSIONS = ['.pdf', '.docx', '.doc', '.txt', '.md', '.csv']
SUPPORTED_URL_SCHEMES = ['http', 'https', 'blob', 'drive', 'dropbox']

# Domain detection keywords
DOMAIN_KEYWORDS = {
    "legal": ['article', 'constitution', 'fundamental rights', 'contract', 'law', 'court', 'clause', 'statute', 'regulation'],
    "constitutional": ['fundamental rights', 'article', 'part', 'constitution', 'parliament', 'directive principles'],
    "insurance": ['policy', 'premium', 'claim', 'coverage', 'benefit', 'exclusion', 'deductible', 'policyholder'],
    "medical": ['patient', 'diagnosis', 'treatment', 'clinical', 'medical', 'healthcare', 'physician', 'therapy'],
    "financial": ['investment', 'revenue', 'profit', 'financial', 'accounting', 'audit', 'balance', 'asset'],
    "technical": ['system', 'software', 'hardware', 'network', 'API', 'configuration', 'deployment', 'architecture'],
    "academic": ['research', 'study', 'analysis', 'methodology', 'hypothesis', 'experiment', 'data', 'results'],
    "business": ['strategy', 'management', 'operations', 'marketing', 'business', 'corporate', 'organization']
}

# ================================
# HYBRID CACHING SYSTEM (Recommendation #8)
# ================================

class HybridCacheManager:
    """Hybrid cache combining UltraFastCache speed with document signature checking from 62.py"""
    
    def __init__(self):
        try:
            if HAS_CACHETOOLS:
                # Increased cache sizes from 61.py for better performance
                self.embedding_cache = cachetools.TTLCache(maxsize=50000, ttl=86400)
                self.document_chunk_cache = cachetools.LRUCache(maxsize=2000)
                self.domain_cache = cachetools.LRUCache(maxsize=5000)
                self.query_cache = cachetools.TTLCache(maxsize=10000, ttl=3600)
                self.primary_available = True
                logger.info("✅ Hybrid TTL/LRU caching enabled")
            else:
                raise ImportError("cachetools not available")
        except ImportError:
            self.embedding_cache = {}
            self.document_chunk_cache = {}
            self.domain_cache = {}
            self.query_cache = {}
            self.primary_available = False
            logger.info("📦 Using dict fallback caching")
        
        # Document signature tracking from 62.py
        self.current_document_hash = None
        self._lock = threading.RLock()
        self.hit_count = 0
        self.miss_count = 0

    def set_current_document(self, document_hash: str):
        """Set current document with signature checking from 62.py"""
        with self._lock:
            if self.current_document_hash != document_hash:
                if self.current_document_hash is not None:
                    logger.info(f"🔄 Document changed, clearing query cache")
                    self.query_cache.clear()
                else:
                    logger.info(f"📄 Setting initial document: {document_hash[:8]}")
                self.current_document_hash = document_hash

    def get_embedding(self, text_hash: str) -> Optional[Any]:
        with self._lock:
            result = self.embedding_cache.get(text_hash)
            if result is not None:
                self.hit_count += 1
            else:
                self.miss_count += 1
            return result

    def set_embedding(self, text_hash: str, embedding: Any):
        with self._lock:
            self.embedding_cache[text_hash] = embedding

    def get_query_result(self, query_hash: str) -> Optional[Any]:
        with self._lock:
            result = self.query_cache.get(query_hash)
            if result is not None:
                self.hit_count += 1
            else:
                self.miss_count += 1
            return result

    def set_query_result(self, query_hash: str, result: Any):
        with self._lock:
            self.query_cache[query_hash] = result

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

    def clear_all_caches(self):
        with self._lock:
            self.embedding_cache.clear()
            self.document_chunk_cache.clear()
            self.domain_cache.clear()
            self.query_cache.clear()
            logger.info("🧹 All caches cleared")

    def get_cache_efficiency(self) -> float:
        total = self.hit_count + self.miss_count
        return (self.hit_count / total * 100) if total > 0 else 0.0

# Global cache manager
CACHE_MANAGER = HybridCacheManager()

# ================================
# OPTIMIZED MODEL MANAGER
# ================================

class ModelManager:
    def __init__(self):
        self._sentence_model = None
        self._reranker = None
        self._model_lock = asyncio.Lock()

    async def get_sentence_model(self):
        if self._sentence_model is None:
            async with self._model_lock:
                if self._sentence_model is None:
                    logger.info("🔄 Loading sentence transformer...")
                    self._sentence_model = SentenceTransformer(
                        EMBEDDING_MODEL_NAME,
                        device='cpu',
                        trust_remote_code=True,
                        cache_folder=os.getenv('SENTENCE_TRANSFORMERS_HOME', None)
                    )
                    self._sentence_model.eval()
                    _ = self._sentence_model.encode("warmup", show_progress_bar=False)
                    logger.info("✅ Sentence transformer ready")
        return self._sentence_model

    async def get_reranker(self):
        if self._reranker is None:
            async with self._model_lock:
                if self._reranker is None:
                    logger.info("🔄 Loading reranker...")
                    self._reranker = CrossEncoder(
                        RERANKER_MODEL_NAME,
                        max_length=256,
                        device='cpu'
                    )
                    _ = self._reranker.predict([["warmup", "test"]])
                    logger.info("✅ Reranker ready")
        return self._reranker

MODEL_MANAGER = ModelManager()

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

def simple_auth_check(request: Request) -> bool:
    auth = request.headers.get("Authorization", "")
    expected = f"Bearer {HACKRX_TOKEN}"
    return auth == expected

def sanitize_for_json(data):
    """Recursively sanitize data for JSON serialization"""
    import numpy as np
    if isinstance(data, dict):
        return {k: sanitize_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_for_json(v) for v in data]
    elif isinstance(data, (np.integer, np.floating, np.bool_)):
        if hasattr(data, 'dtype'):
            if data.dtype == bool:
                return bool(data)
            elif data.dtype in ['int32', 'int64']:
                return int(data)
            elif data.dtype in ['float32', 'float64']:
                return float(data)
        return data
    elif hasattr(data, 'item'):
        return data.item()
    elif isinstance(data, (np.ndarray,)):
        return data.tolist()
    return data

# ================================
# ENHANCED UNIFIED LOADER
# ================================

class UnifiedLoader:
    def __init__(self):
        self.mime_detector = magic if HAS_MAGIC else None
        self.loader_cache = {}

    async def load_document(self, source: str) -> List[Document]:
        try:
            source_hash = hashlib.md5(source.encode()).hexdigest()
            if source_hash in self.loader_cache:
                logger.info(f"📄 Using cached document: {sanitize_pii(source)}")
                return self.loader_cache[source_hash]

            if self._is_url(source):
                docs = await self._load_from_url(source)
            else:
                docs = await self._load_from_file(source)

            for doc in docs:
                doc.metadata.update({
                    'source': source,
                    'load_time': time.time(),
                    'loader_version': '3.0'
                })

            self.loader_cache[source_hash] = docs
            logger.info(f"✅ Loaded {len(docs)} documents from {sanitize_pii(source)}")
            return docs

        except Exception as e:
            logger.error(f"❌ Failed to load {sanitize_pii(source)}: {e}")
            raise

    def _is_url(self, source: str) -> bool:
        return source.startswith(('http://', 'https://', 'blob:', 'drive:', 'dropbox:'))

    async def _load_from_url(self, url: str) -> List[Document]:
        parsed = urlparse(url)
        scheme = parsed.scheme.lower()

        if scheme in ["drive", "dropbox"]:
            if scheme == "drive":
                url = url.replace("drive:", "https://")
            elif scheme == "dropbox":
                url = url.replace("dropbox:", "https://")

        download_url = self._transform_special_url(url)

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Connection": "keep-alive",
        }

        timeout = httpx.Timeout(timeout=120.0, connect=15.0, read=120.0)
        
        async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
            response = await client.get(download_url, follow_redirects=True)
            response.raise_for_status()
            content = response.content

            file_ext = (
                self._get_extension_from_url(url)
                or self._detect_extension_from_content(content)
            )

            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                tmp_file.write(content)
                temp_path = tmp_file.name

            try:
                return await self._load_from_file(temp_path)
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

    def _transform_special_url(self, url: str) -> str:
        # Google Drive transformation
        if 'drive.google.com' in url:
            match = re.search(r'/file/d/([a-zA-Z0-9-_]+)', url)
            if match:
                file_id = match.group(1)
                return f"https://drive.google.com/uc?export=download&id={file_id}"
        
        # Dropbox transformation
        if 'dropbox.com' in url:
            if '?dl=0' in url:
                return url.replace('?dl=0', '?dl=1')
            elif '?dl=1' not in url:
                separator = '&' if '?' in url else '?'
                return f"{url}{separator}dl=1"
        
        return url

    def _get_extension_from_url(self, url: str) -> Optional[str]:
        parsed = urlparse(url)
        path = parsed.path
        if path:
            return os.path.splitext(path)[1]
        return None

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
            raise ValueError(f"File too large: {file_size / (1024*1024):.1f}MB")

        if file_extension not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file extension: {file_extension}")

        docs = None
        loader_used = None

        if file_extension == '.pdf':
            try:
                loader = PyMuPDFLoader(file_path)
                docs = await asyncio.to_thread(loader.load)
                loader_used = "PyMuPDFLoader"
            except Exception as e:
                logger.warning(f"⚠️ PDF loading failed: {e}")

        elif file_extension in ['.docx', '.doc']:
            try:
                loader = Docx2txtLoader(file_path)
                docs = await asyncio.to_thread(loader.load)
                loader_used = "Docx2txtLoader"
            except Exception as e:
                logger.warning(f"⚠️ DOCX loading failed: {e}")

        elif file_extension in ['.txt', '.md', '.csv']:
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

        if not docs:
            raise ValueError(f"Could not load file {file_path}")

        for doc in docs:
            doc.metadata.update({
                'file_size': file_size,
                'file_extension': file_extension,
                'loader_used': loader_used
            })

        return docs

# ================================
# HYBRID TEXT SPLITTER (Recommendation #1)
# ================================

class HybridTextSplitter:
    """Hybrid text splitter combining adaptive chunking from 62.py with speed optimizations from 61.py"""
    
    def __init__(self):
        # Enhanced separators for better document structure preservation
        self.separators = [
            "\n\n## ", "\n\n# ", "\n\n### ", "\n\n#### ",
            "\n\nArticle ", "\n\nPart ", "\n\nSection ",  # Constitutional/legal
            "\n\n", "\n", ". ", "! ", "? ", "; ", " ", ""
        ]
        # Domain-specific adaptations from 62.py
        self.domain_multipliers = {
            "legal": 1.25,      # Larger chunks for legal documents
            "constitutional": 1.3,  # Even larger for constitutional texts
            "medical": 1.0,
            "insurance": 0.85,
            "financial": 1.0,
            "technical": 1.0,
            "academic": 1.1,
            "business": 1.0,
            "general": 1.0
        }

    def split_documents(self, documents: List[Document], detected_domain: str = "general") -> List[Document]:
        """Hybrid document splitting with smart caching and domain adaptation"""
        if not documents:
            return []

        # Smart caching from 62.py
        content_hash = self._calculate_content_hash(documents)
        cache_key = f"chunks_{content_hash}_{detected_domain}"
        
        cached_chunks = CACHE_MANAGER.get_document_chunks(cache_key)
        if cached_chunks is not None:
            logger.info(f"📄 Using cached chunks: {len(cached_chunks)} chunks")
            return cached_chunks

        # Hybrid chunking strategy - adapt chunk size based on domain
        chunk_size, chunk_overlap = self._adapt_for_content(documents, detected_domain)
        
        all_chunks = []
        for doc in documents:
            try:
                chunks = self._split_document_hybrid(doc, chunk_size, chunk_overlap)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.warning(f"⚠️ Error splitting document: {e}")
                # Fallback to simple splitting
                chunks = self._simple_split(doc, chunk_size, chunk_overlap)
                all_chunks.extend(chunks)

        # Filter very short chunks but keep threshold lower for speed
        all_chunks = [chunk for chunk in all_chunks if len(chunk.page_content.strip()) >= 50]

        # Cache result
        CACHE_MANAGER.set_document_chunks(cache_key, all_chunks)
        logger.info(f"📄 Created {len(all_chunks)} hybrid chunks for {detected_domain} domain")
        return all_chunks

    def _calculate_content_hash(self, documents: List[Document]) -> str:
        """Calculate hash for content caching"""
        content_sample = "".join([doc.page_content[:100] for doc in documents[:5]])
        return hashlib.sha256(content_sample.encode()).hexdigest()[:16]

    def _adapt_for_content(self, documents: List[Document], detected_domain: str) -> Tuple[int, int]:
        """Adapt chunk size based on content and domain from 62.py"""
        multiplier = self.domain_multipliers.get(detected_domain, 1.0)
        
        # Use hybrid configuration values
        adapted_size = int(CHUNK_SIZE * multiplier)
        adapted_overlap = min(adapted_size // 4, int(CHUNK_OVERLAP * 1.2))
        
        # Ensure reasonable bounds
        adapted_size = max(600, min(2000, adapted_size))
        
        return adapted_size, adapted_overlap

    def _split_document_hybrid(self, document: Document, chunk_size: int, chunk_overlap: int) -> List[Document]:
        """Hybrid document splitting with enhanced metadata"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.separators
        )

        chunks = splitter.split_documents([document])
        
        # Add enhanced metadata from 62.py
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "chunk_index": i,
                "total_chunks": len(chunks),
                "chunk_type": "hybrid_split",
                "chunk_size_used": chunk_size,
                "chunk_overlap_used": chunk_overlap
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
# DOMAIN DETECTOR
# ================================

class DomainDetector:
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
        for domain, keywords in DOMAIN_KEYWORDS.items():
            matches = 0
            for keyword in keywords:
                matches += combined_text.count(keyword.lower())
            
            text_length = max(len(combined_text), 1)
            normalized_score = matches / (len(keywords) * text_length / 1000)
            domain_scores[domain] = min(1.0, normalized_score)
        
        return domain_scores

DOMAIN_DETECTOR = DomainDetector()

# ================================
# HYBRID TOKEN PROCESSOR (Recommendation #6)
# ================================

class HybridTokenProcessor:
    """Hybrid token processor combining intelligent context from 62.py with speed safeguards from 61.py"""
    
    def __init__(self):
        self.max_context_tokens = 4000
        # Character limit safeguard from 61.py
        self.max_context_chars = MAX_CONTEXT_CHARS
        
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load tiktoken tokenizer: {e}")
            self.tokenizer = None

    def optimize_context_hybrid(self, documents: List[Document], query: str) -> str:
        """Hybrid context optimization with sentence-level pruning and character limits"""
        if not documents:
            return ""

        # Extract key terms from query for intelligent ranking (from 62.py)
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

        # Build context within both token and character limits
        context_parts = []
        total_tokens = 0
        total_chars = 0

        for doc, rel_score, eff_score, token_count in scored_docs:
            # Check both token and character limits (hybrid approach)
            if (total_tokens + token_count <= self.max_context_tokens and 
                total_chars + len(doc.page_content) <= self.max_context_chars):
                
                context_parts.append(doc.page_content)
                total_tokens += token_count
                total_chars += len(doc.page_content)
                
            elif (total_tokens < self.max_context_tokens * 0.8 and 
                  total_chars < self.max_context_chars * 0.8):
                
                # Try to fit partial content with intelligent truncation
                remaining_tokens = self.max_context_tokens - total_tokens
                remaining_chars = self.max_context_chars - total_chars
                remaining = min(remaining_tokens * 4, remaining_chars)  # Estimate character limit from tokens
                
                if remaining > 200:  # Minimum useful chunk
                    partial_content = self._truncate_intelligently(
                        doc.page_content, remaining, query_terms
                    )
                    context_parts.append(partial_content)
                    break
            else:
                break

        return "\n\n---\n\n".join(context_parts)

    def _extract_key_terms(self, query: str) -> List[str]:
        """Extract key terms from query"""
        terms = query.lower().split()
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        return [term for term in terms if term not in stop_words and len(term) > 2]

    def _calculate_relevance(self, doc: Document, query_terms: List[str]) -> float:
        """Calculate document relevance to query terms"""
        content_lower = doc.page_content.lower()
        score = 0.0
        for term in query_terms:
            count = content_lower.count(term)
            score += count * (1.0 / len(doc.page_content))
        return score

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count with fallback"""
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception:
                pass
        return max(1, int(len(text) / 3.8))

    def _truncate_intelligently(self, content: str, max_chars: int, query_terms: List[str]) -> str:
        """Intelligent truncation preserving query-relevant parts"""
        sentences = content.split('. ')
        
        sentence_scores = []
        for sentence in sentences:
            score = sum(1 for term in query_terms if term.lower() in sentence.lower())
            sentence_scores.append((sentence, score))

        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        
        selected_sentences = []
        current_chars = 0
        
        for sentence, score in sentence_scores:
            sentence_chars = len(sentence)
            if current_chars + sentence_chars <= max_chars:
                selected_sentences.append(sentence)
                current_chars += sentence_chars
        
        return '. '.join(selected_sentences) + "..."

# ================================
# HYBRID EMBEDDING FUNCTIONS (Recommendation #7)
# ================================

async def get_embeddings_hybrid(texts: List[str]) -> List[np.ndarray]:
    """Hybrid embedding generation with parallel batching from 61.py and caching from 62.py"""
    if not texts:
        return []

    results = []
    uncached_texts = []
    uncached_indices = []

    # Check cache first (from 62.py)
    for i, text in enumerate(texts):
        text_hash = hashlib.md5(text.encode()).hexdigest()[:16]  # Shorter hash for speed
        cached_embedding = CACHE_MANAGER.get_embedding(text_hash)
        if cached_embedding is not None:
            results.append((i, cached_embedding))
        else:
            uncached_texts.append(text)
            uncached_indices.append(i)

    # Process uncached with parallel batching from 61.py
    if uncached_texts:
        model = await MODEL_MANAGER.get_sentence_model()
        
        if len(uncached_texts) <= OPTIMAL_BATCH_SIZE:
            # Small batch - process directly
            embeddings = await _process_embedding_batch_hybrid(model, uncached_texts)
        else:
            # Large batch - parallel processing
            embeddings = await _process_embeddings_parallel_hybrid(model, uncached_texts)

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

async def _process_embeddings_parallel_hybrid(model, texts: List[str]) -> List[np.ndarray]:
    """Process large batches in parallel with optimized batch size"""
    chunk_size = OPTIMAL_BATCH_SIZE
    chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]

    # Process chunks in parallel with semaphore
    semaphore = asyncio.Semaphore(MAX_PARALLEL_BATCHES)

    async def process_chunk(chunk):
        async with semaphore:
            return await _process_embedding_batch_hybrid(model, chunk)

    results = await asyncio.gather(*[process_chunk(chunk) for chunk in chunks])

    # Flatten results
    embeddings = []
    for result in results:
        embeddings.extend(result)
    return embeddings

async def _process_embedding_batch_hybrid(model, texts: List[str]) -> List[np.ndarray]:
    """Process single embedding batch with speed optimization"""
    embeddings = await asyncio.to_thread(
        model.encode,
        texts,
        batch_size=len(texts),  # Process entire batch at once
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
        device='cpu'
    )
    return list(embeddings)

async def get_query_embedding_hybrid(query: str) -> np.ndarray:
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
# HYBRID RECIPROCAL RANK FUSION (Recommendation #3)
# ================================

def hybrid_reciprocal_rank_fusion(results_list: List[List[Tuple[Document, float]]], 
                                 query_length: int, k_value: int = 60) -> List[Tuple[Document, float]]:
    """Hybrid RRF - fast version for short queries, full version for complex queries"""
    if not results_list:
        return []

    # Fast path for short queries (recommendation #3)
    if query_length <= 8:
        return fast_reciprocal_rank_fusion(results_list, k_value)
    else:
        return full_reciprocal_rank_fusion(results_list, k_value)

def fast_reciprocal_rank_fusion(results_list: List[List[Tuple[Document, float]]], 
                               k_value: int = 60) -> List[Tuple[Document, float]]:
    """Fast RRF with minimal processing from 61.py"""
    doc_scores = defaultdict(float)
    seen_docs = {}

    for i, results in enumerate(results_list):
        weight = 0.6 if i == 0 else 0.4  # Fixed weights for speed
        for rank, (doc, score) in enumerate(results):
            doc_key = hashlib.md5(doc.page_content.encode()).hexdigest()[:16]
            rrf_score = weight / (k_value + rank + 1)
            doc_scores[doc_key] += rrf_score
            if doc_key not in seen_docs:
                seen_docs[doc_key] = doc

    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    if not sorted_docs:
        return []

    max_score = sorted_docs[0][1]
    result = []
    for doc_key, score in sorted_docs:
        normalized_score = score / max_score
        result.append((seen_docs[doc_key], normalized_score))
    return result

def full_reciprocal_rank_fusion(results_list: List[List[Tuple[Document, float]]], 
                               k_value: int = 60) -> List[Tuple[Document, float]]:
    """Full RRF with semantic=0.6, BM25=0.4 weights from 62.py"""
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

# ================================
# FAISS VECTOR STORE
# ================================

class HybridFAISSStore:
    """Hybrid FAISS implementation combining speed from 61.py with robustness from 62.py"""
    
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
            logger.info("✅ Hybrid FAISS store initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize FAISS: {e}")
            raise

    async def add_documents_hybrid(self, documents: List[Document], embeddings: List[np.ndarray]):
        """Hybrid document addition with optimized batching"""
        try:
            if not self.is_trained:
                self.initialize()

            if len(documents) != len(embeddings):
                raise ValueError("Mismatch between documents and embeddings")

            # Convert all embeddings at once for better performance
            all_embeddings = np.array(embeddings, dtype=np.float32)
            faiss.normalize_L2(all_embeddings)

            # For large datasets, use IVF index for better performance (from 62.py)
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
            logger.info(f"✅ Added {len(documents)} documents to hybrid FAISS index (total: {len(self.documents)})")

        except Exception as e:
            logger.error(f"❌ Error adding documents to FAISS: {e}")
            raise

    async def similarity_search_hybrid(self, query_embedding: np.ndarray, k: int = 10) -> List[Tuple[Document, float]]:
        """Hybrid similarity search with error handling"""
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
        self.documents.clear()
        if self.index:
            self.index.reset()

# ================================
# GEMINI CLIENT MANAGEMENT
# ================================

gemini_client = None

async def ensure_gemini_ready():
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

# ================================
# HYBRID RAG SYSTEM
# ================================

class HybridRAGSystem:
    """Hybrid RAG system combining the best features from both implementations"""
    
    def __init__(self):
        self.documents = []
        self.vector_store = None
        self.bm25_retriever = None
        self.domain = "general"
        self.loader = UnifiedLoader()
        self.text_splitter = HybridTextSplitter()
        self.token_processor = HybridTokenProcessor()

    def is_fast_path_query(self, query: str) -> bool:
        """Enhanced fast-path detection (recommendation #4)"""
        fast_indicators = [
            len(query.split()) <= 8,  # Short queries
            not any(word in query.lower() for word in ['compare', 'analyze', 'explain why', 'how does', 'what are the differences']),
            query.count('?') <= 1,
            not any(word in query.lower() for word in ['difference', 'versus', 'vs', 'between'])
        ]
        return sum(fast_indicators) >= 3

    async def process_documents(self, sources: List[str]) -> Dict[str, Any]:
        """Process documents with hybrid optimizations"""
        start_time = time.time()
        
        # Document signature checking from 62.py
        doc_signature = hashlib.md5(str(sorted(sources)).encode()).hexdigest()
        CACHE_MANAGER.set_current_document(doc_signature)
        
        if hasattr(self, '_last_doc_signature') and self._last_doc_signature == doc_signature:
            logger.info("📄 Documents already processed, skipping...")
            return {"cached": True, "processing_time": 0.001}
        
        self._last_doc_signature = doc_signature

        try:
            logger.info(f"📄 Processing {len(sources)} documents")
            
            # Parallel document loading
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

            # Domain detection and chunking
            domain, domain_confidence = DOMAIN_DETECTOR.detect_domain(raw_documents)
            self.domain = domain
            
            self.documents = self.text_splitter.split_documents(raw_documents, domain)
            
            # Setup retrievers
            await self._setup_retrievers_hybrid()

            processing_time = time.time() - start_time
            result = {
                'domain': domain,
                'domain_confidence': float(domain_confidence),
                'total_chunks': len(self.documents),
                'processing_time': processing_time,
                'document_signature': doc_signature[:8]
            }

            logger.info(f"✅ Hybrid processing complete in {processing_time:.2f}s")
            return sanitize_for_json(result)

        except Exception as e:
            logger.error(f"❌ Document processing error: {e}")
            raise

    async def _setup_retrievers_hybrid(self):
        """Hybrid retriever setup with parallel initialization"""
        try:
            logger.info("🔧 Setting up hybrid retrievers...")

            # Setup FAISS vector store
            if HAS_FAISS and self.documents:
                self.vector_store = HybridFAISSStore(dimension=384)
                self.vector_store.initialize()

                doc_texts = [doc.page_content for doc in self.documents]
                embeddings = await get_embeddings_hybrid(doc_texts)
                await self.vector_store.add_documents_hybrid(self.documents, embeddings)
                logger.info("✅ Hybrid FAISS vector store setup complete")

            # Setup BM25 retriever
            if self.documents:
                self.bm25_retriever = await asyncio.to_thread(
                    BM25Retriever.from_documents, self.documents
                )
                self.bm25_retriever.k = min(RERANK_TOP_K, len(self.documents))
                logger.info(f"✅ BM25 retriever setup complete (k={self.bm25_retriever.k})")

        except Exception as e:
            logger.error(f"❌ Hybrid retriever setup error: {e}")

    async def retrieve_and_rerank_hybrid(self, query: str, top_k: int = CONTEXT_DOCS) -> Tuple[List[Document], List[float]]:
        """Hybrid retrieval with conditional reranking (recommendation #4)"""
        if not self.documents:
            return [], []

        query_embedding = await get_query_embedding_hybrid(query)
        
        # Vector search
        vector_docs = []
        if self.vector_store:
            try:
                vector_results = await self.vector_store.similarity_search_hybrid(
                    query_embedding, k=SEMANTIC_SEARCH_K
                )
                vector_docs = vector_results
            except Exception as e:
                logger.warning(f"⚠️ Vector search failed: {e}")

        # BM25 search
        bm25_docs = []
        if self.bm25_retriever and len(vector_docs) < SEMANTIC_SEARCH_K:
            try:
                bm25_results = await asyncio.to_thread(self.bm25_retriever.invoke, query) or []
                bm25_docs = [(doc, 1.0 - (i * 0.1)) for i, doc in enumerate(bm25_results[:6])]
            except Exception as e:
                logger.warning(f"⚠️ BM25 search failed: {e}")

        # Hybrid RRF based on query length (recommendation #3)
        query_length = len(query.split())
        if vector_docs and bm25_docs:
            fused_results = hybrid_reciprocal_rank_fusion([vector_docs, bm25_docs], query_length)
            all_docs = [doc for doc, score in fused_results[:top_k]]
        elif vector_docs:
            all_docs = [doc for doc, score in vector_docs[:top_k]]
        elif bm25_docs:
            all_docs = [doc for doc, score in bm25_docs[:top_k]]
        else:
            return [], []

        # Deduplicate
        seen_content = set()
        unique_docs = []
        for doc in all_docs:
            content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()[:16]
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_docs.append(doc)

        # Conditional reranking (recommendation #4)
        skip_reranker = self.is_fast_path_query(query)
        
        if not skip_reranker and len(unique_docs) > 3:
            try:
                reranker = await MODEL_MANAGER.get_reranker()
                pairs = [[query, doc.page_content[:512]] for doc in unique_docs[:12]]
                
                # Batch processing for speed
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
                
                logger.info(f"🎯 Hybrid retrieval with reranking: {len(final_docs)} documents")
                return final_docs, final_scores
                
            except Exception as e:
                logger.warning(f"⚠️ Reranking failed: {e}")

        # Without reranking
        final_docs = unique_docs[:top_k]
        final_scores = [0.8] * len(final_docs)
        
        logger.info(f"🎯 Hybrid retrieval (fast path): {len(final_docs)} documents")
        return final_docs, final_scores

    async def query(self, query: str) -> Dict[str, Any]:
        """Hybrid query processing with caching and confidence calculation"""
        start_time = time.time()

        try:
            # Check query cache first
            query_hash = hashlib.md5(query.encode()).hexdigest()[:16]
            cached_result = CACHE_MANAGER.get_query_result(query_hash)
            if cached_result:
                cached_result['processing_time'] = 0.001
                cached_result['cached'] = True
                return cached_result

            # Fast path detection
            if self.is_fast_path_query(query):
                result = await self.query_fast_path(query)
                CACHE_MANAGER.set_query_result(query_hash, result)
                return result

            # Full retrieval
            retrieved_docs, similarity_scores = await self.retrieve_and_rerank_hybrid(query, CONTEXT_DOCS)

            if not retrieved_docs:
                result = {
                    "query": query,
                    "answer": "No relevant documents found for your query.",
                    "confidence": 0.0,
                    "domain": self.domain,
                    "processing_time": time.time() - start_time
                }
                return result

            # Hybrid confidence calculation (recommendation #5)
            confidence = self._calculate_hybrid_confidence(query, retrieved_docs, similarity_scores)

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

            # Hybrid context optimization
            context = self.token_processor.optimize_context_hybrid(retrieved_docs, query)

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

            # Cache result
            CACHE_MANAGER.set_query_result(query_hash, result)
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

    def _calculate_hybrid_confidence(self, query: str, docs: List[Document], scores: List[float]) -> float:
        """Hybrid confidence calculation (recommendation #5)"""
        if not scores:
            return 0.0

        try:
            scores_array = np.array(scores)
            if np.max(scores_array) > 1.0:
                scores_array = scores_array / np.max(scores_array)
            scores_array = np.clip(scores_array, 0.0, 1.0)

            # Multi-factor confidence from 62.py
            factors = {
                'max_score': np.max(scores_array),
                'avg_score': np.mean(scores_array),
                'query_coverage': self._calculate_query_coverage(query, docs),
                'score_consistency': max(0.0, 1.0 - np.std(scores_array)) if len(scores_array) > 1 else 1.0
            }

            # Weights from recommendation #5
            weights = {
                'max_score': 0.35,
                'avg_score': 0.25,
                'query_coverage': 0.25,
                'score_consistency': 0.15
            }

            confidence = sum(factors[key] * weights[key] for key in factors)

            # Direct keyword boost from 61.py
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

    async def _generate_response(self, query: str, context: str, domain: str, confidence: float) -> str:
        """Generate response using Gemini"""
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
# GLOBAL CACHE FOR DOCUMENT PROCESSING
# ================================

_document_cache = {}
_cache_ttl = 1800  # 30 minutes

# ================================
# FASTAPI APPLICATION
# ================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("🚀 Starting Hybrid HackRx RAG System...")
    start_time = time.time()
    
    try:
        # Initialize Gemini client
        if GEMINI_API_KEY:
            await ensure_gemini_ready()
        
        startup_time = time.time() - start_time
        logger.info(f"✅ Hybrid system initialized in {startup_time:.2f}s")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise

    yield

    # Shutdown
    logger.info("🔄 Shutting down hybrid system...")
    _document_cache.clear()
    CACHE_MANAGER.clear_all_caches()
    
    if gemini_client and hasattr(gemini_client, 'close'):
        try:
            await gemini_client.close()
        except Exception:
            pass
    
    logger.info("✅ Hybrid shutdown complete")

app = FastAPI(
    title="Hybrid HackRx RAG System",
    description="Hybrid RAG System combining best features from both implementations",
    version="3.0.0",
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
    return {
        "status": "online",
        "service": "Hybrid HackRx RAG System",
        "version": "3.0.0",
        "hybrid_features": [
            "Larger chunks (1200) with overlap (200) for better context",
            "Context window: 18 docs with 12k char safeguard",
            "Conditional reranking based on query complexity",
            "Hybrid RRF: fast for short queries, full for complex",
            "Multi-factor confidence calculation",
            "Intelligent context with sentence-level pruning",
            "Parallel embedding batching with caching",
            "Ultra-fast caching with document signature tracking"
        ],
        "cache_efficiency": f"{CACHE_MANAGER.get_cache_efficiency():.1f}%",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/cache-stats")
async def get_cache_stats():
    try:
        return {
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
    except Exception as e:
        return {"error": str(e)}

@app.post("/hackrx/run")
async def hackrx_run_endpoint(request: Request):
    """Hybrid HackRx endpoint with optimized processing"""
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

        # Check document cache
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
            rag_system = HybridRAGSystem()
            logger.info(f"📄 Processing document: {sanitize_pii(documents_url)}")
            await rag_system.process_documents([documents_url])
            
            # Cache the processed system
            _document_cache[doc_cache_key] = (rag_system, current_time)
            
            # Clean up old cache entries
            if len(_document_cache) > 10:
                oldest_key = min(_document_cache.keys(),
                               key=lambda k: _document_cache[k][1])
                del _document_cache[oldest_key]

        # Process questions in parallel
        logger.info(f"❓ Processing {len(questions)} questions with hybrid optimization...")

        async def process_single_question(question: str) -> str:
            try:
                result = await rag_system.query(question)
                return result["answer"]
            except Exception as e:
                logger.error(f"❌ Error processing question: {e}")
                return f"Error processing question: {str(e)}"

        # Use semaphore to control concurrency
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_QUESTIONS)

        async def bounded_process(question: str) -> str:
            async with semaphore:
                return await process_single_question(question)

        # Process all questions in parallel
        answers = await asyncio.gather(
            *[bounded_process(q) for q in questions],
            return_exceptions=False
        )

        processing_time = time.time() - start_time
        logger.info(f"✅ Hybrid completed {len(questions)} questions in {processing_time:.2f}s")

        return {"answers": answers}

    except Exception as e:
        logger.error(f"❌ HackRx endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

# ================================
# ERROR HANDLERS
# ================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
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
    logger.info(f"🚀 Starting Hybrid HackRx RAG System on port {port}")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )
