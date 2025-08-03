import os

# Performance optimization
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

import sys
import asyncio
import logging
import time
from typing import List, Dict, Any, Tuple, Optional, Union, Generic, TypeVar
from contextlib import asynccontextmanager
import uuid
import json
import tempfile
import hashlib
import re
from datetime import datetime, timedelta
from functools import lru_cache
from collections import defaultdict, deque
from urllib.parse import urlparse, parse_qs, unquote_plus
from cachetools import TTLCache, LRUCache
import traceback
import sqlite3

# Core libraries
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np

# FastAPI and web
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, status, APIRouter, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, HttpUrl
import httpx
import aiohttp

# Document processing
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever

# Vector stores
try:
    import pinecone
    from langchain_community.vectorstores import Pinecone
    HAS_PINECONE = True
except ImportError:
    HAS_PINECONE = False
    pinecone = None
    Pinecone = None

# Redis for caching
try:
    import redis.asyncio as redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False
    redis = None

# AI and embeddings
from sentence_transformers import CrossEncoder, SentenceTransformer, util
import openai
from openai import AsyncOpenAI

# Token counting
import tiktoken

# Optional imports with fallbacks
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    psutil = None

try:
    import magic
    HAS_MAGIC = True
except ImportError:
    HAS_MAGIC = False
    magic = None

# Configure logging
logging.basicConfig(
    format='%(asctime)s %(levelname)s [%(name)s]: %(message)s',
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

# ================================
# GLOBAL VARIABLES & CONFIGURATION
# ================================

# Component ready states
components_ready = {
    "base_sentence_model": False,
    "embedding_model": False,
    "reranker": False,
    "openai_client": False,
    "pinecone": False,
    "redis": False
}

# Global variables with proper initialization
embedding_model = None
query_embedding_model = None
base_sentence_model = None
reranker = None
openai_client = None
redis_client = None
pinecone_index = None

# Configuration with environment variable support
SESSION_TTL = int(os.getenv("SESSION_TTL", 3600))
MAX_CACHE_SIZE = int(os.getenv("MAX_CACHE_SIZE", 1000))
EMBEDDING_CACHE_SIZE = int(os.getenv("EMBEDDING_CACHE_SIZE", 10000))
LOG_VERBOSE = os.getenv("LOG_VERBOSE", "true").lower() == "true"
QUESTION_TIMEOUT = float(os.getenv("QUESTION_TIMEOUT", 25.0))
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", 128))
TOKEN_BUFFER_PERCENT = float(os.getenv("TOKEN_BUFFER_PERCENT", 0.1))

# Environment variables for API keys
PINECONE_API_KEY = "pcsk_711NmG_Tub3XoFEp23axHP4PAdj1FYoQSf9G5oYahsVe2ZhEBe8ktiqcauGEtdQC7eCFWR"
PINECONE_ENVIRONMENT = "us-east1-gcp"
PINECONE_INDEX_NAME = "enhanced-rag-system"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Authentication token
HACKRX_TOKEN = "9a1163c13e8927960b857a674794a62c57baf588998981151b0753a4d6d17905"

# Enhanced caching system
ACTIVE_SESSIONS = {}
EMBEDDING_CACHE = TTLCache(maxsize=EMBEDDING_CACHE_SIZE, ttl=86400)
RESPONSE_CACHE = TTLCache(maxsize=MAX_CACHE_SIZE, ttl=300)

# Document hot-words index for fast domain detection
DOMAIN_HOTWORDS_INDEX = defaultdict(set)

# UNIVERSAL ADAPTIVE CONFIGURATION - REMOVED HARDCODED BIASES
BASE_CONFIG = {
    "chunk_size": 1200,
    "chunk_overlap": 200,
    "semantic_search_k": 24,
    "context_docs": 18,
    "confidence_threshold": 0.15,  # ADAPTIVE - starts very low
    "use_mmr": True,
    "mmr_lambda": 0.75,
    "use_metadata_filtering": True,
    "rerank_top_k": 35,
    "fallback_threshold_multiplier": 0.7,  # For fallback attempts
    "max_fallback_docs": 30
}

# ADAPTIVE DOMAIN CONFIGS - REMOVED SPECIFIC BIASES
ADAPTIVE_DOMAIN_CONFIGS = {
    "insurance": {
        "confidence_adjustment": 1.1,  # Slightly higher confidence needed
        "chunk_size_adjustment": 0.85,  # Smaller chunks
        "semantic_weight": 0.6
    },
    "legal": {
        "confidence_adjustment": 1.15,
        "chunk_size_adjustment": 1.25,  # Larger chunks for legal context
        "semantic_weight": 0.7
    },
    "medical": {
        "confidence_adjustment": 1.2,
        "chunk_size_adjustment": 1.0,
        "semantic_weight": 0.65
    },
    "financial": {
        "confidence_adjustment": 1.1,
        "chunk_size_adjustment": 1.0,
        "semantic_weight": 0.6
    },
    "technical": {
        "confidence_adjustment": 1.0,
        "chunk_size_adjustment": 1.0,
        "semantic_weight": 0.55
    },
    "academic": {
        "confidence_adjustment": 0.9,
        "chunk_size_adjustment": 1.1,
        "semantic_weight": 0.65
    },
    "business": {
        "confidence_adjustment": 1.0,
        "chunk_size_adjustment": 1.0,
        "semantic_weight": 0.6
    },
    "general": {
        "confidence_adjustment": 0.8,  # Most lenient
        "chunk_size_adjustment": 1.0,
        "semantic_weight": 0.5
    }
}

# GENERALIZED DOMAIN KEYWORDS - NO HARDCODED BIASES
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

# ================================
# JSON SERIALIZATION UTILITIES
# ================================

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
# RETRY DECORATOR
# ================================

async def retry_with_backoff(func, retries=3, base_delay=1):
    """Exponential backoff retry decorator"""
    for i in range(retries):
        try:
            return await func()
        except Exception as e:
            if i == retries - 1:
                raise
            delay = base_delay * (2 ** i)
            logger.warning(f"⚠️ Retry {i+1}/{retries} after {delay}s: {str(e)}")
            await asyncio.sleep(delay)
    raise Exception("Max retries exceeded")

# ================================
# AUTHENTICATION MIDDLEWARE
# ================================

security = HTTPBearer(auto_error=False)

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify Bearer token authentication"""
    if not credentials or credentials.credentials != HACKRX_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

# ================================
# LAZY LOADING FOR HEAVY COMPONENTS - FIXED
# ================================

async def ensure_openai_ready():
    """Ensure OpenAI client is ready - critical component"""
    global openai_client
    if openai_client is None and OPENAI_API_KEY:
        try:
            openai_client = OptimizedOpenAIClient()
            await openai_client.initialize(OPENAI_API_KEY)
            components_ready["openai_client"] = True
        except Exception as e:
            logger.error(f"❌ Failed to initialize OpenAI: {e}")
            raise HTTPException(status_code=503, detail="OpenAI client not available")
    elif openai_client is None:
        raise HTTPException(status_code=503, detail="OpenAI API key not configured")

async def ensure_models_ready():
    """Load pre-downloaded models quickly"""
    global base_sentence_model, embedding_model, reranker
    
    if base_sentence_model is None:
        try:
            # Models are pre-downloaded in Docker, this should be fast
            base_sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            components_ready["base_sentence_model"] = True
            logger.info("✅ Sentence transformer loaded from cache")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load sentence transformer: {e}")

    if embedding_model is None:
        try:
            embedding_model = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            components_ready["embedding_model"] = True
            logger.info("✅ Embedding model loaded from cache")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load embedding model: {e}")

    if reranker is None:
        try:
            reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            components_ready["reranker"] = True
            logger.info("✅ Reranker loaded from cache")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load reranker: {e}")

# ================================
# ADAPTIVE TEXT SPLITTER - GENERALIZED
# ================================

class AdaptiveTextSplitter:
    """Adaptive text splitter that works for any document type"""
    
    def __init__(self, base_chunk_size: int = 1200, base_chunk_overlap: int = 200):
        self.base_chunk_size = base_chunk_size
        self.base_chunk_overlap = base_chunk_overlap
        
        # Universal separators - NO domain-specific bias
        self.separators = [
            "\n\n## ", "\n\n# ", "\n\n### ", "\n\n#### ",  # Headers
            "\n\n", "\n", ". ", "! ", "? ", "; ", " ", ""  # General separators
        ]
        
        # Universal section patterns - NO domain-specific bias
        self.section_patterns = [
            r'(?i)^([A-Z][^.:!?]*[.:])\s*$',  # Generic headers ending with . or :
            r'(?i)^(\d+(?:\.\d+)*)\s+(.+)$',  # Numbered sections
            r'(?i)^([IVX]+\.)\s+(.+)$',  # Roman numerals
            r'(?i)^(\([a-zA-Z0-9]\))\s+(.+)$',  # Lettered/numbered lists
            r'(?i)^([A-Z\s]{3,})\s*$',  # ALL CAPS headers
        ]

    def adapt_for_content(self, documents: List[Document], detected_domain: str = "general") -> Tuple[int, int]:
        """Dynamically adapt chunk size based on content analysis"""
        if not documents:
            return self.base_chunk_size, self.base_chunk_overlap

        # Analyze document characteristics
        avg_paragraph_length = self._analyze_paragraph_structure(documents)
        sentence_complexity = self._analyze_sentence_complexity(documents)
        content_density = self._analyze_content_density(documents)
        
        # Get domain adjustments (if any)
        domain_config = ADAPTIVE_DOMAIN_CONFIGS.get(detected_domain, ADAPTIVE_DOMAIN_CONFIGS["general"])
        size_adjustment = domain_config.get("chunk_size_adjustment", 1.0)

        # Calculate adaptive chunk size
        adapted_size = int(self.base_chunk_size * size_adjustment)
        
        # Adjust based on content characteristics
        if avg_paragraph_length > 300:  # Long paragraphs
            adapted_size = int(adapted_size * 1.2)
        elif avg_paragraph_length < 100:  # Short paragraphs
            adapted_size = int(adapted_size * 0.8)

        if sentence_complexity > 20:  # Complex sentences
            adapted_size = int(adapted_size * 1.1)

        # Ensure reasonable bounds
        adapted_size = max(600, min(2000, adapted_size))
        adapted_overlap = min(adapted_size // 4, int(self.base_chunk_overlap * 1.2))
        
        return adapted_size, adapted_overlap

    def split_documents(self, documents: List[Document], detected_domain: str = "general") -> List[Document]:
        """Split documents with adaptive chunking"""
        if not documents:
            return []

        # Adapt chunk size for content
        chunk_size, chunk_overlap = self.adapt_for_content(documents, detected_domain)
        logger.info(f"📄 Adaptive chunking: size={chunk_size}, overlap={chunk_overlap}")

        all_chunks = []
        for doc in documents:
            try:
                # Add basic metadata
                doc.metadata.update(self._extract_basic_metadata(doc.page_content))
                
                # Split with section awareness
                chunks = self._intelligent_split(doc, chunk_size, chunk_overlap)
                all_chunks.extend(chunks)
                
            except Exception as e:
                logger.warning(f"⚠️ Error splitting document: {e}")
                # Fallback to simple splitting
                chunks = self._simple_split(doc, chunk_size, chunk_overlap)
                all_chunks.extend(chunks)

        # Filter very short chunks
        all_chunks = [chunk for chunk in all_chunks if len(chunk.page_content.strip()) >= 50]
        
        logger.info(f"📄 Created {len(all_chunks)} adaptive chunks")
        return all_chunks

    def _analyze_paragraph_structure(self, documents: List[Document]) -> float:
        """Analyze average paragraph length"""
        paragraphs = []
        for doc in documents[:5]:  # Sample first 5 docs
            paras = [p.strip() for p in doc.page_content.split('\n\n') if p.strip()]
            paragraphs.extend(paras)
        
        if not paragraphs:
            return 150  # Default
        
        avg_length = sum(len(p) for p in paragraphs) / len(paragraphs)
        return avg_length

    def _analyze_sentence_complexity(self, documents: List[Document]) -> float:
        """Analyze average sentence length as complexity indicator"""
        sentences = []
        for doc in documents[:3]:
            # Simple sentence splitting
            sents = re.split(r'[.!?]+', doc.page_content[:2000])
            sentences.extend([s.strip() for s in sents if s.strip()])
        
        if not sentences:
            return 15  # Default
        
        avg_length = sum(len(s.split()) for s in sentences) / len(sentences)
        return avg_length

    def _analyze_content_density(self, documents: List[Document]) -> float:
        """Analyze information density (technical terms, numbers, etc.)"""
        total_chars = 0
        technical_indicators = 0
        
        for doc in documents[:3]:
            content = doc.page_content[:1000]
            total_chars += len(content)
            
            # Count technical indicators
            technical_indicators += len(re.findall(r'\d+', content))  # Numbers
            technical_indicators += len(re.findall(r'[A-Z]{2,}', content))  # Acronyms
            technical_indicators += len(re.findall(r'[()[\]{}]', content))  # Brackets
        
        if total_chars == 0:
            return 0.1
        
        density = technical_indicators / total_chars
        return density

    def _extract_basic_metadata(self, text: str) -> Dict[str, Any]:
        """Extract basic, universal metadata"""
        metadata = {}
        
        # Count basic statistics
        metadata['char_count'] = len(text)
        metadata['word_count'] = len(text.split())
        metadata['paragraph_count'] = len([p for p in text.split('\n\n') if p.strip()])
        
        # Find potential dates (universal pattern)
        dates = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}[/-]\d{2}[/-]\d{2}\b', text)
        if dates:
            metadata['contains_dates'] = True
            
        # Find numbers/statistics
        numbers = re.findall(r'\b\d+(?:\.\d+)?%?\b', text)
        if len(numbers) > 5:
            metadata['data_rich'] = True
            
        return metadata

    def _intelligent_split(self, document: Document, chunk_size: int, chunk_overlap: int) -> List[Document]:
        """Intelligent splitting with section detection"""
        text = document.page_content
        
        # Try to identify sections
        sections = self._detect_sections(text)
        
        if len(sections) > 1:
            # Split by sections
            return self._split_by_sections(sections, document.metadata, chunk_size, chunk_overlap)
        else:
            # Use enhanced recursive splitting
            return self._enhanced_recursive_split(document, chunk_size, chunk_overlap)

    def _detect_sections(self, text: str) -> List[Dict[str, Any]]:
        """Universal section detection"""
        lines = text.split('\n')
        sections = []
        current_section = {"header": "", "content": "", "start_line": 0}
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            is_header = self._is_potential_header(line)
            
            if is_header and current_section["content"].strip():
                sections.append(current_section)
                current_section = {"header": line, "content": "", "start_line": i}
            elif is_header and not current_section["content"].strip():
                current_section["header"] = line
                current_section["start_line"] = i
            else:
                current_section["content"] += line + "\n"
        
        if current_section["content"].strip():
            sections.append(current_section)
        
        # Only return sections if we found meaningful divisions
        if len(sections) > 1 and all(len(s["content"]) > 100 for s in sections):
            return sections
        else:
            return [{"header": "Document", "content": text, "start_line": 0}]

    def _is_potential_header(self, line: str) -> bool:
        """Universal header detection without domain bias"""
        if not line or len(line) > 150:  # Too long to be a header
            return False
        
        # Check against universal patterns
        for pattern in self.section_patterns:
            if re.match(pattern, line):
                return True
        
        # Additional universal heuristics
        if (len(line) < 80 and
            (line.isupper() or
             line.endswith(':') or
             re.match(r'^\d+[\.\)]\s', line) or
             (line[0].isupper() and line.count(' ') <= 5))):
            return True
        
        return False

    def _split_by_sections(self, sections: List[Dict], base_metadata: Dict,
                          chunk_size: int, chunk_overlap: int) -> List[Document]:
        """Split content by detected sections"""
        chunks = []
        
        for section in sections:
            header = section["header"]
            content = section["content"]
            
            if len(content) <= chunk_size:
                # Section fits in one chunk
                enhanced_metadata = base_metadata.copy()
                enhanced_metadata.update({
                    "section_header": header,
                    "chunk_type": "complete_section"
                })
                
                full_content = f"{header}\n\n{content}" if header else content
                chunks.append(Document(page_content=full_content, metadata=enhanced_metadata))
                
            else:
                # Section needs to be split
                section_chunks = self._split_large_section(
                    content, header, base_metadata, chunk_size, chunk_overlap
                )
                chunks.extend(section_chunks)
        
        return chunks

    def _split_large_section(self, content: str, header: str, base_metadata: Dict,
                           chunk_size: int, chunk_overlap: int) -> List[Document]:
        """Split large sections into chunks"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.separators
        )
        
        section_chunks = splitter.split_text(content)
        chunks = []
        
        for i, chunk_text in enumerate(section_chunks):
            enhanced_metadata = base_metadata.copy()
            enhanced_metadata.update({
                "section_header": header,
                "chunk_type": "section_part",
                "chunk_index": i,
                "total_chunks_in_section": len(section_chunks)
            })
            
            # Include header in first chunk only
            if i == 0 and header:
                full_content = f"{header}\n\n{chunk_text}"
            else:
                full_content = chunk_text
                
            chunks.append(Document(page_content=full_content, metadata=enhanced_metadata))
        
        return chunks

    def _enhanced_recursive_split(self, document: Document, chunk_size: int, chunk_overlap: int) -> List[Document]:
        """Enhanced recursive splitting as fallback"""
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
                "chunk_type": "recursive_split"
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
# REDIS CACHE IMPLEMENTATION
# ================================

class RedisCache:
    """Production-ready Redis cache with fallback"""
    
    def __init__(self):
        self.redis = None
        self.fallback_cache = TTLCache(maxsize=1000, ttl=300)

    async def initialize(self):
        """Initialize Redis with proper error handling"""
        if not HAS_REDIS:
            logger.warning("⚠️ Redis not available, using fallback cache")
            components_ready["redis"] = False
            return

        try:
            self.redis = redis.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", 6379)),
                db=0,
                socket_connect_timeout=5,
                socket_timeout=5,
                decode_responses=True,
                retry_on_timeout=True,
                health_check_interval=30
            )

            await asyncio.wait_for(self.redis.ping(), timeout=5)
            components_ready["redis"] = True
            logger.info("✅ Redis cache initialized successfully")

        except Exception as e:
            logger.warning(f"⚠️ Redis not available, using fallback cache: {e}")
            self.redis = None
            components_ready["redis"] = False

    async def get_cached_response(self, query_hash: str) -> Optional[Dict]:
        """Get cached response with fallback"""
        if self.redis:
            try:
                cached = await asyncio.wait_for(
                    self.redis.get(f"response:{query_hash}"),
                    timeout=2
                )
                return json.loads(cached) if cached else None
            except Exception as e:
                logger.warning(f"⚠️ Redis get error: {e}")
        
        return self.fallback_cache.get(query_hash)

    async def cache_response(self, query_hash: str, response: dict, ttl: int = 300):
        """Cache response with fallback"""
        self.fallback_cache[query_hash] = response
        
        if self.redis:
            try:
                await asyncio.wait_for(
                    self.redis.setex(f"response:{query_hash}", ttl, json.dumps(response, default=str)),
                    timeout=2
                )
            except Exception as e:
                logger.warning(f"⚠️ Redis cache error: {e}")

# ================================
# OPTIMIZED EMBEDDING SERVICE
# ================================

class OptimizedEmbeddingService:
    """Optimized embedding service with caching"""
    
    def __init__(self):
        self.embedding_cache = EMBEDDING_CACHE
        self.processing_lock = asyncio.Lock()
        self.max_batch_size = MAX_BATCH_SIZE

    async def get_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Process embeddings in batches"""
        if not texts:
            return []

        async with self.processing_lock:
            results = []
            uncached_texts = []
            uncached_indices = []

            for i, text in enumerate(texts):
                text_hash = hashlib.md5(text.encode()).hexdigest()
                if text_hash in self.embedding_cache:
                    results.append((i, self.embedding_cache[text_hash]))
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)

            if uncached_texts:
                try:
                    await ensure_models_ready()
                    if base_sentence_model:
                        batch_size = min(self.max_batch_size, len(uncached_texts))
                        embeddings = await asyncio.to_thread(
                            base_sentence_model.encode,
                            uncached_texts,
                            batch_size=batch_size,
                            show_progress_bar=False,
                            convert_to_numpy=True
                        )

                        for text, embedding in zip(uncached_texts, embeddings):
                            text_hash = hashlib.md5(text.encode()).hexdigest()
                            self.embedding_cache[text_hash] = embedding

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

            results.sort(key=lambda x: x[0])
            return [embedding for _, embedding in results]

    async def get_query_embedding(self, query: str) -> np.ndarray:
        """Get single query embedding"""
        if not query.strip():
            return np.zeros(384)

        query_hash = hashlib.md5(query.encode()).hexdigest()
        if query_hash in self.embedding_cache:
            return self.embedding_cache[query_hash]

        try:
            await ensure_models_ready()
            if base_sentence_model:
                embedding = await asyncio.to_thread(
                    base_sentence_model.encode,
                    query,
                    convert_to_numpy=True
                )
                self.embedding_cache[query_hash] = embedding
                return embedding
            else:
                logger.warning("⚠️ No embedding model available for query")
                return np.zeros(384)
        except Exception as e:
            logger.error(f"❌ Query embedding error: {e}")
            return np.zeros(384)

# ================================
# OPTIMIZED OPENAI CLIENT
# ================================

class OptimizedOpenAIClient:
    """OpenAI client with proper connection pooling"""
    
    def __init__(self):
        self.client = None
        self.prompt_cache = TTLCache(maxsize=1000, ttl=600)
        self.rate_limit_delay = 1.0
        self.request_semaphore = asyncio.Semaphore(10)

    async def initialize(self, api_key: str):
        """Initialize OpenAI client"""
        if not api_key:
            raise ValueError("OpenAI API key is required")

        try:
            self.client = AsyncOpenAI(
                api_key=api_key,
                timeout=httpx.Timeout(
                    connect=10.0,
                    read=60.0,
                    write=30.0,
                    pool=5.0
                ),
                max_retries=3
            )

            await self._test_connection()
            components_ready["openai_client"] = True
            logger.info("✅ OpenAI client initialized successfully")

        except Exception as e:
            logger.error(f"❌ Failed to initialize OpenAI client: {e}")
            components_ready["openai_client"] = False
            raise

    async def _test_connection(self):
        """Test OpenAI connection"""
        try:
            response = await self.client.chat.completions.create(
                messages=[{"role": "user", "content": "Hello"}],
                model="gpt-4o",
                max_tokens=5
            )
            logger.info("✅ OpenAI connection test successful")
        except Exception as e:
            logger.error(f"❌ OpenAI connection test failed: {e}")
            raise

    def _get_prompt_hash(self, messages: List[Dict], **kwargs) -> str:
        """Generate hash for prompt caching"""
        prompt_data = {
            "messages": json.dumps(messages, sort_keys=True),
            "model": kwargs.get("model", "gpt-4o"),
            "temperature": kwargs.get("temperature", 0.1),
            "max_tokens": kwargs.get("max_tokens", 1000)
        }
        return hashlib.md5(json.dumps(prompt_data, sort_keys=True).encode()).hexdigest()

    async def optimized_completion(self, messages: List[Dict], **kwargs) -> str:
        """Optimized completion with caching"""
        if not self.client:
            raise ValueError("OpenAI client not initialized")

        prompt_hash = self._get_prompt_hash(messages, **kwargs)

        # Check local cache first
        cached = self.prompt_cache.get(prompt_hash)
        if cached:
            return cached

        # Check Redis cache
        redis_cached = await REDIS_CACHE.get_cached_response(prompt_hash)
        if redis_cached and 'content' in redis_cached:
            result = redis_cached['content']
            self.prompt_cache[prompt_hash] = result
            return result

        async with self.request_semaphore:
            async def make_request():
                return await self.client.chat.completions.create(
                    messages=messages,
                    model=kwargs.get("model", "gpt-4o"),
                    temperature=kwargs.get("temperature", 0.1),
                    max_tokens=kwargs.get("max_tokens", 1000),
                    timeout=60
                )

            response = await retry_with_backoff(make_request)
            result = response.choices[0].message.content

            self.prompt_cache[prompt_hash] = result
            await REDIS_CACHE.cache_response(prompt_hash, {"content": result})

            return result

# ================================
# UNIVERSAL DOMAIN DETECTOR - GENERALIZED
# ================================

class UniversalDomainDetector:
    """Truly universal domain detector without hardcoded biases"""
    
    def __init__(self):
        self.domain_embeddings = {}
        self.fallback_cache = LRUCache(maxsize=500)
        self._initialize_hotwords_index()
        
        # REMOVED SPECIFIC DOMAIN DESCRIPTIONS - keeping generic
        self.domain_descriptions = {
            domain: " ".join(keywords)
            for domain, keywords in DOMAIN_KEYWORDS.items()
        }

    def _initialize_hotwords_index(self):
        """Build inverted index of domain keywords"""
        for domain, keywords in DOMAIN_KEYWORDS.items():
            for keyword in keywords:
                DOMAIN_HOTWORDS_INDEX[keyword.lower()].add(domain)

    def initialize_embeddings(self):
        """Initialize domain embeddings"""
        if not base_sentence_model:
            logger.warning("⚠️ Base model not loaded, skipping domain embeddings")
            return

        try:
            for domain, description in self.domain_descriptions.items():
                try:
                    self.domain_embeddings[domain] = base_sentence_model.encode(
                        description,
                        convert_to_tensor=False
                    )
                except Exception as e:
                    logger.warning(f"⚠️ Failed to create embedding for domain {domain}: {e}")

            logger.info(f"✅ Initialized embeddings for {len(self.domain_embeddings)} domains")
        except Exception as e:
            logger.error(f"❌ Failed to initialize domain embeddings: {e}")

    def detect_domain(self, documents: List[Document], confidence_threshold: float = 0.3) -> Tuple[str, float]:
        """Universal domain detection - TRULY GENERALIZED"""
        if not documents:
            return "general", 0.5

        combined_text = ' '.join([doc.page_content[:500] for doc in documents[:10]])
        cache_key = hashlib.md5(combined_text.encode()).hexdigest()[:16]

        if cache_key in self.fallback_cache:
            return self.fallback_cache[cache_key]

        try:
            # Multi-strategy detection
            keyword_scores = self._keyword_based_detection(documents)
            hotword_scores = self._hotwords_detection(documents)
            semantic_scores = {}

            if self.domain_embeddings and base_sentence_model:
                semantic_scores = self._semantic_detection(documents)

            # Combine all scores with balanced weighting
            final_scores = {}
            for domain in DOMAIN_KEYWORDS.keys():
                score = 0.0
                weights_sum = 0.0

                if domain in keyword_scores:
                    score += 0.4 * keyword_scores[domain]
                    weights_sum += 0.4

                if domain in hotword_scores:
                    score += 0.4 * hotword_scores[domain]
                    weights_sum += 0.4

                if domain in semantic_scores:
                    score += 0.2 * semantic_scores[domain]
                    weights_sum += 0.2

                # Normalize by actual weights used
                if weights_sum > 0:
                    final_scores[domain] = score / weights_sum
                else:
                    final_scores[domain] = 0.0

            if final_scores:
                best_domain = max(final_scores, key=final_scores.get)
                best_score = final_scores[best_domain]

                # REMOVED HARDCODED DOMAIN PREFERENCES
                # Use adaptive threshold instead
                if best_score < confidence_threshold:
                    best_domain = "general"
                    best_score = confidence_threshold

                result = (best_domain, best_score)
                self.fallback_cache[cache_key] = result

                if LOG_VERBOSE:
                    logger.info(f"🔍 Domain detected: {best_domain} (confidence: {best_score:.2f})")

                return result

            return "general", confidence_threshold

        except Exception as e:
            logger.warning(f"⚠️ Domain detection error: {e}")
            return "general", confidence_threshold

    def _hotwords_detection(self, documents: List[Document]) -> Dict[str, float]:
        """Fast domain detection using hot-words index"""
        combined_text = ' '.join([doc.page_content for doc in documents[:10]])[:5000].lower()
        words = set(combined_text.split())

        domain_hits = defaultdict(int)
        total_hits = 0

        for word in words:
            if word in DOMAIN_HOTWORDS_INDEX:
                for domain in DOMAIN_HOTWORDS_INDEX[word]:
                    domain_hits[domain] += 1
                    total_hits += 1

        domain_scores = {}
        if total_hits > 0:
            for domain, hits in domain_hits.items():
                domain_scores[domain] = hits / total_hits

        return domain_scores

    def _keyword_based_detection(self, documents: List[Document]) -> Dict[str, float]:
        """Keyword-based domain detection - balanced"""
        combined_text = ' '.join([doc.page_content for doc in documents[:15]])[:8000].lower()

        domain_scores = {}
        for domain, keywords in DOMAIN_KEYWORDS.items():
            matches = 0
            for keyword in keywords:
                matches += combined_text.count(keyword.lower())

            # Normalize by keyword count and text length
            normalized_score = matches / (len(keywords) * len(combined_text) / 1000)
            domain_scores[domain] = min(1.0, normalized_score)

        return domain_scores

    def _semantic_detection(self, documents: List[Document]) -> Dict[str, float]:
        """Semantic domain detection"""
        try:
            combined_text = ' '.join([doc.page_content[:1000] for doc in documents[:10]])[:6000]
            content_embedding = base_sentence_model.encode(combined_text, convert_to_tensor=False)

            domain_scores = {}
            for domain, domain_embedding in self.domain_embeddings.items():
                similarity = float(util.cos_sim(content_embedding, domain_embedding)[0][0])
                domain_scores[domain] = similarity

            return domain_scores

        except Exception as e:
            logger.warning(f"⚠️ Semantic detection error: {e}")
            return {}

# ================================
# TOKEN OPTIMIZATION
# ================================

class TokenOptimizedProcessor:
    """Token optimization with tiktoken support"""
    
    def __init__(self):
        self.max_context_tokens = 4000
        self.token_buffer = int(self.max_context_tokens * TOKEN_BUFFER_PERCENT)
        self.tokenizer = None

        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load tiktoken tokenizer: {e}")

    @lru_cache(maxsize=2000)
    def estimate_tokens(self, text: str) -> int:
        """Accurate token estimation"""
        if not text:
            return 0

        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception:
                pass

        # Fallback to heuristic
        words = text.split()
        avg_chars_per_token = 3.5
        if any(term in text.lower() for term in ['api', 'json', 'xml', 'code', 'function']):
            avg_chars_per_token = 3.0

        estimated = len(text) / avg_chars_per_token
        return max(1, int(estimated * 1.1))

    def calculate_relevance_score(self, doc: Document, query: str) -> float:
        """Calculate document relevance - GENERALIZED"""
        try:
            query_terms = set(query.lower().split())
            doc_terms = set(doc.page_content.lower().split())

            # Basic keyword overlap
            keyword_overlap = len(query_terms.intersection(doc_terms)) / max(len(query_terms), 1)

            # Position boost (earlier chunks slightly preferred)
            position_boost = 1.0
            chunk_index = doc.metadata.get('chunk_index', 0)
            total_chunks = doc.metadata.get('total_chunks', 1)
            if total_chunks > 1:
                position_boost = 1.15 - (chunk_index / total_chunks) * 0.3

            # Section completeness boost
            section_boost = 1.0
            if doc.metadata.get('chunk_type') == 'complete_section':
                section_boost = 1.05
            elif doc.metadata.get('section_header'):
                section_boost = 1.02

            # Semantic similarity
            semantic_score = 0.5
            if base_sentence_model:
                try:
                    doc_embedding = self._get_cached_embedding(doc.page_content[:512])
                    query_embedding = self._get_cached_embedding(query)
                    semantic_score = float(util.cos_sim(doc_embedding, query_embedding)[0][0])
                except Exception:
                    pass

            # Length penalty for very short chunks
            length_penalty = 1.0
            if len(doc.page_content) < 100:
                length_penalty = 0.8

            # Balanced final score
            final_score = (
                0.35 * semantic_score +
                0.35 * keyword_overlap +
                0.20 * position_boost +
                0.10 * section_boost
            ) * length_penalty

            return min(1.0, final_score)

        except Exception as e:
            logger.warning(f"⚠️ Error calculating relevance: {e}")
            return 0.5

    @lru_cache(maxsize=3000)
    def _get_cached_embedding(self, text: str) -> np.ndarray:
        """Get cached embedding"""
        if not text.strip():
            return np.zeros(384)

        cache_key = hashlib.md5(text.encode()).hexdigest()
        if cache_key in EMBEDDING_CACHE:
            return EMBEDDING_CACHE[cache_key]

        if base_sentence_model:
            try:
                embedding = base_sentence_model.encode(text, convert_to_tensor=False)
                EMBEDDING_CACHE[cache_key] = embedding
                return embedding
            except Exception:
                pass

        return np.zeros(384)

    def optimize_context_intelligently(self, documents: List[Document], query: str, max_tokens: int = None) -> str:
        """Intelligent context optimization - GENERALIZED"""
        if not documents:
            return ""

        if max_tokens is None:
            max_tokens = self.max_context_tokens

        doc_scores = []
        for doc in documents:
            relevance = self.calculate_relevance_score(doc, query)
            tokens = self.estimate_tokens(doc.page_content)
            efficiency = relevance / max(tokens, 1)
            doc_scores.append((doc, relevance, tokens, efficiency))

        # Sort by efficiency
        doc_scores.sort(key=lambda x: x[3], reverse=True)

        context_parts = []
        token_budget = max_tokens - self.token_buffer

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

        if LOG_VERBOSE:
            estimated_tokens = self.estimate_tokens(context)
            logger.info(f"📝 Context optimized: {len(context_parts)} documents, ~{estimated_tokens} tokens")

        return context

    def _truncate_content(self, content: str, max_tokens: int) -> str:
        """Smart content truncation"""
        max_chars = max_tokens * 4
        if len(content) <= max_chars:
            return content

        keep_chars = max_chars - 100
        first_part = content[:keep_chars//2]
        last_part = content[-keep_chars//2:]

        return f"{first_part}\n\n[... content truncated for token efficiency ...]\n\n{last_part}"

# ================================
# ENHANCED RAG SYSTEM - GENERALIZED
# ================================

class EnhancedRAGSystem:
    """Enhanced RAG system with universal optimization"""
    
    def __init__(self, session_id: str = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.documents = deque()
        self.vector_store = None
        self.bm25_retriever = None
        self.domain = "general"
        self.domain_config = BASE_CONFIG.copy()
        self.document_hash = None
        self.processed_files = []
        self.token_processor = TokenOptimizedProcessor()
        self._processing_lock = asyncio.Lock()
        self.text_splitter = None
        self.document_url = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()

    async def cleanup(self):
        """Proper cleanup"""
        try:
            self.vector_store = None
            self.bm25_retriever = None
            self.documents.clear()
            self.processed_files.clear()
            if LOG_VERBOSE:
                logger.info(f"🧹 Session {self.session_id} cleaned up")
        except Exception as e:
            logger.warning(f"⚠️ Cleanup error: {e}")

    def calculate_document_hash(self, documents: List[Document]) -> str:
        """Calculate unique hash for documents"""
        content_sample = "".join([doc.page_content[:100] for doc in documents[:5]])
        return hashlib.sha256(content_sample.encode()).hexdigest()[:16]

    def _adapt_config_for_domain(self, domain: str) -> Dict[str, Any]:
        """Adapt configuration based on detected domain"""
        adapted_config = BASE_CONFIG.copy()

        if domain in ADAPTIVE_DOMAIN_CONFIGS:
            domain_settings = ADAPTIVE_DOMAIN_CONFIGS[domain]

            # Apply adaptive adjustments
            confidence_adj = domain_settings.get("confidence_adjustment", 1.0)
            adapted_config["confidence_threshold"] = BASE_CONFIG["confidence_threshold"] * confidence_adj

            chunk_adj = domain_settings.get("chunk_size_adjustment", 1.0)
            adapted_config["chunk_size"] = int(BASE_CONFIG["chunk_size"] * chunk_adj)

            semantic_weight = domain_settings.get("semantic_weight", 0.6)
            adapted_config["semantic_weight"] = semantic_weight

        return adapted_config

    async def process_documents(self, file_paths: List[str]) -> Dict[str, Any]:
        """Process documents with universal optimization"""
        async with self._processing_lock:
            start_time = time.time()
            try:
                if not file_paths:
                    raise HTTPException(status_code=400, detail="No file paths provided")

                logger.info(f"📄 Processing {len(file_paths)} documents")

                raw_documents = []
                for file_path in file_paths:
                    try:
                        docs = await self._load_document(file_path)
                        raw_documents.extend(docs)
                    except Exception as e:
                        logger.warning(f"⚠️ Error loading {file_path}: {e}")
                        continue

                if not raw_documents:
                    raise HTTPException(status_code=400, detail="No documents could be loaded")

                # Universal domain detection
                domain, domain_confidence = DOMAIN_DETECTOR.detect_domain(raw_documents)
                self.domain = domain
                self.domain_config = self._adapt_config_for_domain(domain)

                logger.info(f"🔍 Detected domain: {domain} (confidence: {domain_confidence:.2f})")

                # Initialize adaptive text splitter
                self.text_splitter = AdaptiveTextSplitter(
                    base_chunk_size=self.domain_config["chunk_size"],
                    base_chunk_overlap=self.domain_config["chunk_overlap"]
                )

                # Enhanced metadata
                for doc in raw_documents:
                    doc.metadata.update({
                        'detected_domain': domain,
                        'domain_confidence': domain_confidence,
                        'session_id': self.session_id,
                        'processing_timestamp': datetime.now().isoformat(),
                        'file_type': self._get_file_type(doc.metadata.get('source', ''))
                    })

                # Adaptive document chunking
                logger.info("🔄 Starting adaptive document chunking...")
                documents_list = self.text_splitter.split_documents(raw_documents, domain)

                # Filter and convert to deque
                documents_list = [doc for doc in documents_list if len(doc.page_content.strip()) >= 50]
                self.documents = deque(documents_list)

                self.document_hash = self.calculate_document_hash(list(self.documents))
                self.processed_files = [os.path.basename(fp) for fp in file_paths]

                # Setup retrievers
                await self._setup_retrievers()

                processing_time = time.time() - start_time

                result = {
                    'session_id': self.session_id,
                    'document_hash': self.document_hash,
                    'domain': domain,
                    'domain_confidence': float(domain_confidence),
                    'total_chunks': len(self.documents),
                    'processed_files': self.processed_files,
                    'chunk_size': self.domain_config["chunk_size"],
                    'chunk_overlap': self.domain_config["chunk_overlap"],
                    'confidence_threshold': self.domain_config["confidence_threshold"],
                    'processing_time': processing_time,
                    'enhanced_features': {
                        'adaptive_chunking': True,
                        'universal_domain_detection': True,
                        'dynamic_configuration': True,
                        'mmr_enabled': self.domain_config.get("use_mmr", True),
                        'reranking_enabled': True,
                        'context_optimization': True
                    }
                }

                if LOG_VERBOSE:
                    logger.info(f"✅ Processing complete in {processing_time:.2f}s: {result}")

                return result

            except Exception as e:
                logger.error(f"❌ Document processing error: {e}")
                raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

    async def _load_document(self, file_path: str) -> List[Document]:
        """Enhanced universal document loader with better format support"""
        try:
            file_extension = os.path.splitext(file_path)[1].lower()
            file_size = os.path.getsize(file_path)
            
            logger.info(f"📄 Loading {file_extension} file ({file_size} bytes): {file_path}")
            
            # Enhanced MIME detection
            mime_type = None
            if HAS_MAGIC and magic:
                try:
                    mime_type = magic.from_file(file_path, mime=True)
                    logger.info(f"📄 Detected MIME type: {mime_type}")
                except Exception as e:
                    logger.warning(f"⚠️ MIME detection failed: {e}")
            
            # Route based on MIME type or extension with fallbacks
            docs = None
            
            if mime_type == 'application/pdf' or file_extension == '.pdf':
                try:
                    loader = PyMuPDFLoader(file_path)
                    docs = await asyncio.to_thread(loader.load)
                except Exception as e:
                    logger.warning(f"⚠️ PyMuPDF failed: {e}, trying fallback")
                    # Add fallback PDF loader if needed
                    
            elif ('word' in (mime_type or '') or 
                  'officedocument' in (mime_type or '') or 
                  file_extension in ['.docx', '.doc']):
                try:
                    loader = Docx2txtLoader(file_path)
                    docs = await asyncio.to_thread(loader.load)
                except Exception as e:
                    logger.warning(f"⚠️ DOCX loader failed: {e}")
                    
            elif ('text' in (mime_type or '') or 
                  file_extension in ['.txt', '.md', '.csv', '.log']):
                try:
                    # Try UTF-8 first, then fallback encodings
                    for encoding in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']:
                        try:
                            loader = TextLoader(file_path, encoding=encoding)
                            docs = await asyncio.to_thread(loader.load)
                            logger.info(f"✅ Text file loaded with {encoding} encoding")
                            break
                        except UnicodeDecodeError:
                            continue
                except Exception as e:
                    logger.warning(f"⚠️ Text loader failed: {e}")
            
            # Fallback to text loader for unknown types
            if not docs:
                logger.info("📄 Unknown type, attempting text fallback...")
                try:
                    loader = TextLoader(file_path, encoding='utf-8')
                    docs = await asyncio.to_thread(loader.load)
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
                    'loader_used': type(loader).__name__
                })
            
            logger.info(f"✅ Loaded {len(docs)} documents from {file_path}")
            return docs
            
        except Exception as e:
            logger.error(f"❌ Failed to load {file_path}: {e}")
            raise

    def _get_file_type(self, file_path: str) -> str:
        """Get simplified file type for metadata"""
        ext = os.path.splitext(file_path)[1].lower()
        type_mapping = {
            '.pdf': 'PDF',
            '.docx': 'Word Document',
            '.doc': 'Word Document',
            '.txt': 'Text File',
            '.md': 'Markdown',
            '.csv': 'CSV'
        }
        return type_mapping.get(ext, 'Unknown')

    async def _setup_retrievers(self):
        """Setup retrievers with proper Pinecone fallback"""
        try:
            logger.info("🔧 Setting up retrievers...")
            
            # Setup vector store if available - with better error handling
            if HAS_PINECONE and pinecone and embedding_model and PINECONE_API_KEY:
                try:
                    await ensure_models_ready()
                    namespace = f"{self.domain}_{self.document_hash}"
                    
                    # Check Pinecone limits before initialization
                    try:
                        import pinecone
                        pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
                        
                        # Check if index exists and create if needed
                        existing_indexes = pinecone.list_indexes()
                        if PINECONE_INDEX_NAME not in existing_indexes:
                            # Check if we can create index (pod limits)
                            try:
                                pinecone.create_index(
                                    name=PINECONE_INDEX_NAME,
                                    dimension=384,
                                    metric="cosine",
                                    pods=1,  # Start with minimum pods
                                    replicas=1
                                )
                                logger.info("✅ Pinecone index created successfully")
                            except Exception as create_error:
                                if "max pods" in str(create_error).lower():
                                    logger.warning("⚠️ Pinecone pod limit reached, using fallback storage")
                                    components_ready["pinecone"] = False
                                    raise create_error
                        
                        global pinecone_index
                        pinecone_index = pinecone.Index(PINECONE_INDEX_NAME)
                        components_ready["pinecone"] = True
                        
                        # Setup vector store
                        self.vector_store = Pinecone(
                            index=pinecone_index,
                            embedding=embedding_model,
                            text_key="text",
                            namespace=namespace
                        )
                        
                        # Add documents to vector store
                        if self.documents:
                            document_texts = [doc.page_content for doc in self.documents]
                            document_metadatas = [doc.metadata for doc in self.documents]
                            
                            await asyncio.to_thread(
                                self.vector_store.add_texts,
                                texts=document_texts,
                                metadatas=document_metadatas
                            )
                        
                        logger.info(f"✅ Pinecone vector store setup complete (namespace: {namespace})")
                        
                    except Exception as pinecone_error:
                        logger.warning(f"⚠️ Pinecone setup failed: {pinecone_error}")
                        components_ready["pinecone"] = False
                        # Continue without Pinecone
                        
                except Exception as e:
                    logger.warning(f"⚠️ Vector store setup failed, continuing with BM25 only: {e}")
                    components_ready["pinecone"] = False
            
            # Always setup BM25 as fallback
            try:
                if self.documents:
                    self.bm25_retriever = await asyncio.to_thread(
                        BM25Retriever.from_documents,
                        list(self.documents)
                    )
                    self.bm25_retriever.k = min(self.domain_config["rerank_top_k"], len(self.documents))
                    logger.info(f"✅ BM25 retriever setup complete (k={self.bm25_retriever.k})")
            except Exception as e:
                logger.error(f"❌ BM25 retriever setup failed: {e}")
                
        except Exception as e:
            logger.error(f"❌ Retriever setup error: {e}")

    def calculate_confidence_score(self, query: str, similarity_scores: List[float],
                                 retrieved_docs: List[Document], domain_confidence: float = 1.0) -> float:
        """Improved confidence calculation with better scoring"""
        if not similarity_scores:
            return 0.0

        try:
            scores_array = np.array(similarity_scores)
            
            # Handle negative similarity scores from cosine similarity
            scores_array = np.clip(scores_array, 0.0, 1.0)
            
            # For distance-based scores (where lower is better), convert to similarity
            if np.mean(scores_array) > 1.0:  # Likely distance scores
                max_score = np.max(scores_array)
                if max_score > 0:
                    scores_array = 1.0 - (scores_array / max_score)
                scores_array = np.clip(scores_array, 0.0, 1.0)
            
            max_score = np.max(scores_array)
            avg_score = np.mean(scores_array)
            score_std = np.std(scores_array) if len(scores_array) > 1 else 0.0
            
            # Calculate percentage of good documents (adjusted threshold)
            decent_docs = np.sum(scores_array > 0.2) / len(scores_array)
            
            # Score consistency (penalize high variance less severely)
            score_consistency = max(0.0, 1.0 - (score_std * 1.0))
            
            # Enhanced query-document match
            query_match = self._calculate_query_match(query, retrieved_docs)
            
            # Improved confidence calculation
            confidence = (
                0.35 * max_score +           # Best match
                0.25 * avg_score +           # Overall quality
                0.25 * query_match +         # Query relevance
                0.10 * score_consistency +   # Consistency
                0.05 * decent_docs          # Coverage
            )
            
            # Domain confidence boost (small but meaningful)
            confidence += 0.05 * domain_confidence
            
            # Boost for exact phrase matches
            query_lower = query.lower()
            exact_match_boost = 0.0
            for doc in retrieved_docs[:5]:
                doc_content_lower = doc.page_content.lower()
                for phrase in query_lower.split():
                    if len(phrase) > 3 and phrase in doc_content_lower:
                        exact_match_boost += 0.03
            
            confidence += min(0.15, exact_match_boost)
            
            # Ensure reasonable minimum confidence for decent matches
            if max_score > 0.3 and query_match > 0.2:
                confidence = max(confidence, 0.25)
            
            confidence = min(1.0, max(0.0, confidence))
            
            return confidence
            
        except Exception as e:
            logger.warning(f"⚠️ Confidence calculation error: {e}")
            return 0.3  # More reasonable fallback

    def _calculate_query_match(self, query: str, docs: List[Document]) -> float:
        """Calculate how well documents match the query"""
        if not docs or not query.strip():
            return 0.0
        
        query_words = set(query.lower().split())
        total_match = 0.0
        
        for doc in docs[:5]:  # Check top 5 docs
            doc_words = set(doc.page_content.lower().split())
            overlap = len(query_words.intersection(doc_words))
            match_ratio = overlap / max(len(query_words), 1)
            total_match += match_ratio
        
        return min(1.0, total_match / min(len(docs), 5))

    async def hybrid_retrieve(self, query: str, k: int = None) -> Tuple[List[Document], List[float]]:
        """Enhanced hybrid retrieval with improved scoring"""
        if k is None:
            k = self.domain_config["context_docs"]

        if not self.documents:
            return [], []

        logger.info(f"🔍 Retrieving: semantic_k={self.domain_config['semantic_search_k']}, rerank_k={self.domain_config['rerank_top_k']}, final_k={k}")

        all_docs = []
        all_scores = []

        # Vector retrieval (if available)
        if self.vector_store and components_ready.get("pinecone", False):
            try:
                vector_docs = await asyncio.to_thread(
                    self.vector_store.similarity_search_with_score,
                    query,
                    k=self.domain_config["semantic_search_k"]
                )
                
                for doc, score in vector_docs:
                    all_docs.append(doc)
                    all_scores.append(float(score))
                    
            except Exception as e:
                logger.warning(f"⚠️ Vector retrieval failed: {e}")

        # BM25 retrieval (always available)
        if self.bm25_retriever:
            try:
                bm25_docs = await asyncio.to_thread(
                    self.bm25_retriever.get_relevant_documents,
                    query
                )
                
                # Calculate BM25 scores
                for doc in bm25_docs:
                    if doc not in all_docs:  # Avoid duplicates
                        all_docs.append(doc)
                        # Simplified BM25 scoring
                        bm25_score = self._calculate_bm25_score(query, doc)
                        all_scores.append(bm25_score)
                        
            except Exception as e:
                logger.warning(f"⚠️ BM25 retrieval failed: {e}")

        if not all_docs:
            logger.warning("⚠️ No documents retrieved")
            return [], []

        # Reranking
        if reranker and components_ready.get("reranker", False):
            try:
                pairs = [[query, doc.page_content] for doc in all_docs]
                rerank_scores = await asyncio.to_thread(reranker.predict, pairs)
                
                # Combine with existing scores
                combined_scores = []
                for i, rerank_score in enumerate(rerank_scores):
                    original_score = all_scores[i] if i < len(all_scores) else 0.5
                    combined_score = 0.7 * float(rerank_score) + 0.3 * original_score
                    combined_scores.append(combined_score)
                
                all_scores = combined_scores
                
            except Exception as e:
                logger.warning(f"⚠️ Reranking failed: {e}")

        # Sort by score and take top k
        scored_docs = list(zip(all_docs, all_scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        final_docs = [doc for doc, _ in scored_docs[:k]]
        final_scores = [score for _, score in scored_docs[:k]]

        return final_docs, final_scores

    def _calculate_bm25_score(self, query: str, doc: Document) -> float:
        """Simple BM25-like scoring"""
        query_terms = query.lower().split()
        doc_text = doc.page_content.lower()
        
        score = 0.0
        for term in query_terms:
            tf = doc_text.count(term)
            if tf > 0:
                score += np.log(1 + tf)
        
        return min(1.0, score / max(len(query_terms), 1))

    async def answer_question(self, question: str, timeout: float = QUESTION_TIMEOUT) -> Dict[str, Any]:
        """Answer question with timeout and fallback handling"""
        try:
            return await asyncio.wait_for(
                self._answer_question_internal(question),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.warning(f"⏰ Question timeout after {timeout}s: {question[:100]}")
            return {
                'answer': "I apologize, but the request timed out. Please try asking a shorter question or try again later.",
                'confidence': 0.0,
                'sources_used': 0,
                'processing_time': timeout,
                'timeout': True
            }

    async def _answer_question_internal(self, question: str) -> Dict[str, Any]:
        """Internal question answering logic"""
        start_time = time.time()
        
        if not question.strip():
            raise HTTPException(status_code=400, detail="Empty question provided")

        await ensure_openai_ready()

        # Retrieve relevant documents with fallback
        retrieved_docs, similarity_scores = await self.hybrid_retrieve(
            question,
            k=self.domain_config["context_docs"]
        )

        if not retrieved_docs:
            return {
                'answer': "I apologize, but I couldn't find any relevant information in the documents to answer your question.",
                'confidence': 0.0,
                'sources_used': 0,
                'processing_time': time.time() - start_time
            }

        # Calculate confidence
        confidence = self.calculate_confidence_score(
            question,
            similarity_scores,
            retrieved_docs
        )

        # Check if we need fallback retrieval
        if confidence < self.domain_config["confidence_threshold"]:
            logger.info("🔄 Low confidence ({:.2f}), attempting fallback".format(confidence))
            
            fallback_k = min(
                self.domain_config["max_fallback_docs"],
                len(self.documents)
            )
            
            retrieved_docs, similarity_scores = await self.hybrid_retrieve(
                question,
                k=fallback_k
            )
            
            confidence = self.calculate_confidence_score(
                question,
                similarity_scores,
                retrieved_docs
            )

        # Optimize context
        context = self.token_processor.optimize_context_intelligently(
            retrieved_docs, question
        )

        if not context.strip():
            return {
                'answer': "I couldn't find relevant information to answer your question accurately.",
                'confidence': 0.0,
                'sources_used': 0,
                'processing_time': time.time() - start_time
            }

        # Generate response
        try:
            prompt = self._create_answer_prompt(context, question)
            response = await openai_client.optimized_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1000
            )

            processing_time = time.time() - start_time

            result = {
                'answer': response.strip(),
                'confidence': float(confidence),
                'sources_used': len(retrieved_docs),
                'processing_time': processing_time
            }

            # Add debug info if verbose
            if LOG_VERBOSE:
                result.update({
                    'domain': self.domain,
                    'retrieval_scores': [float(s) for s in similarity_scores[:5]]
                })

            return sanitize_for_json(result)

        except Exception as e:
            logger.error(f"❌ Response generation failed: {e}")
            return {
                'answer': "I encountered an error while generating the response. Please try again.",
                'confidence': 0.0,
                'sources_used': len(retrieved_docs),
                'processing_time': time.time() - start_time,
                'error': str(e)
            }

    def _create_answer_prompt(self, context: str, question: str) -> str:
        """Create optimized prompt for answer generation"""
        prompt = f"""You are an expert assistant that answers questions based on the provided context. Follow these guidelines:

1. Answer directly and concisely based ONLY on the provided context
2. If the context doesn't contain enough information, clearly state this limitation
3. Use specific details from the context when available
4. Maintain a helpful and professional tone
5. Do not make assumptions beyond what's stated in the context

Context:
{context}

Question: {question}

Please provide a comprehensive answer based on the context above:"""
        
        return prompt

# ================================
# GLOBAL INSTANCES
# ================================

# Initialize global components
EMBEDDING_SERVICE = OptimizedEmbeddingService()
REDIS_CACHE = RedisCache()
DOMAIN_DETECTOR = UniversalDomainDetector()

# ================================
# UNIVERSAL URL DOWNLOAD HANDLER
# ================================

async def download_from_url(url: str) -> str:
    """Universal file downloader supporting Google Drive, Azure Blob, and direct URLs"""
    try:
        logger.info(f"📥 Processing URL: {url}")
        
        # Determine URL type and extract appropriate download URL
        if "drive.google.com" in url:
            download_url = await get_google_drive_download_url(url)
        elif "blob.core.windows.net" in url:
            download_url = url  # Azure blob URLs are direct download links
        elif url.startswith(("http://", "https://")):
            download_url = url  # Direct download link
        else:
            raise ValueError("Unsupported URL format")
        
        # Create temporary file with appropriate extension
        parsed_url = urlparse(download_url)
        file_extension = os.path.splitext(parsed_url.path)[1] or ".pdf"
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
        temp_path = temp_file.name
        temp_file.close()
        
        logger.info(f"📥 Downloading from: {download_url}")
        
        # Download with extended timeout for large files
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=600),  # 10 minutes for large files
            headers={'User-Agent': 'Mozilla/5.0 (compatible; DocumentProcessor/1.0)'}
        ) as session:
            async with session.get(download_url) as response:
                if response.status == 200:
                    content = await response.read()
                    
                    # Handle Google Drive virus scan warning
                    if b'Google Drive - Virus scan warning' in content:
                        content = await handle_google_drive_warning(session, content)
                    
                    # Save content
                    with open(temp_path, 'wb') as f:
                        f.write(content)
                    
                    # Verify file
                    if os.path.getsize(temp_path) == 0:
                        os.unlink(temp_path)
                        raise HTTPException(status_code=400, detail="Downloaded file is empty")
                    
                    logger.info(f"✅ File downloaded: {os.path.getsize(temp_path)} bytes")
                    return temp_path
                    
                else:
                    raise HTTPException(
                        status_code=response.status,
                        detail=f"Failed to download: HTTP {response.status}"
                    )
                    
    except HTTPException:
        raise
    except Exception as e:
        if 'temp_path' in locals() and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass
        logger.error(f"❌ Download error: {e}")
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")

async def get_google_drive_download_url(url: str) -> str:
    """Extract Google Drive download URL"""
    file_id = extract_drive_file_id(url)
    if not file_id:
        raise ValueError("Invalid Google Drive URL format")
    return f"https://drive.google.com/uc?export=download&id={file_id}"

def extract_drive_file_id(url: str) -> Optional[str]:
    """Extract file ID from Google Drive URL"""
    patterns = [
        r'/file/d/([a-zA-Z0-9_-]+)',
        r'id=([a-zA-Z0-9_-]+)',
        r'/open\?id=([a-zA-Z0-9_-]+)',
        r'/uc\?id=([a-zA-Z0-9_-]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

async def handle_google_drive_warning(session, content: bytes) -> bytes:
    """Handle Google Drive virus scan warning"""
    warning_content = content.decode('utf-8', errors='ignore')
    import re
    match = re.search(r'href="(/uc\?export=download[^"]+)"', warning_content)
    if match:
        actual_url = "https://drive.google.com" + match.group(1).replace('&amp;', '&')
        async with session.get(actual_url) as actual_response:
            if actual_response.status == 200:
                return await actual_response.read()
    raise HTTPException(status_code=400, detail="Cannot bypass virus scan warning")

# ================================
# PYDANTIC MODELS
# ================================

class ProcessDocumentsRequest(BaseModel):
    documents: HttpUrl
    
class ProcessDocumentsResponse(BaseModel):
    session_id: str
    document_hash: str
    domain: str
    domain_confidence: float
    total_chunks: int
    processed_files: List[str]
    chunk_size: int
    chunk_overlap: int
    confidence_threshold: float
    processing_time: float
    enhanced_features: Dict[str, bool]

class QueryRequest(BaseModel):
    documents: Optional[HttpUrl] = None
    questions: List[str]

class AnswerResponse(BaseModel):
    question: str
    answer: str
    confidence: float
    sources_used: int
    processing_time: float

class QueryResponse(BaseModel):
    answers: List[AnswerResponse]
    session_id: str
    total_processing_time: float

# ================================
# FASTAPI APPLICATION LIFECYCLE
# ================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    logger.info("🚀 Starting Enhanced RAG System...")
    
    # Initialize components
    try:
        await REDIS_CACHE.initialize()
        await ensure_models_ready()
        await ensure_openai_ready()
        
        # Initialize domain detector
        if components_ready.get("base_sentence_model", False):
            DOMAIN_DETECTOR.initialize_embeddings()
        
        logger.info("✅ All components initialized successfully")
        
    except Exception as e:
        logger.warning(f"⚠️ Some components failed to initialize: {e}")
    
    yield
    
    # Cleanup
    logger.info("🛑 Shutting down Enhanced RAG System...")
    try:
        for session in list(ACTIVE_SESSIONS.values()):
            await session.cleanup()
        ACTIVE_SESSIONS.clear()
        logger.info("✅ Cleanup completed")
    except Exception as e:
        logger.warning(f"⚠️ Cleanup error: {e}")

# ================================
# FASTAPI APPLICATION
# ================================

app = FastAPI(
    title="Enhanced RAG System",
    description="Universal Retrieval-Augmented Generation system with adaptive optimization",
    version="2.0.0",
    lifespan=lifespan
)

# CORS configuration
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
    """Root endpoint"""
    return {
        "message": "Enhanced RAG System API",
        "version": "2.0.0",
        "status": "operational",
        "components": components_ready,
        "active_sessions": len(ACTIVE_SESSIONS)
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "components": components_ready,
        "active_sessions": len(ACTIVE_SESSIONS),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/process-documents", response_model=ProcessDocumentsResponse)
async def process_documents_endpoint(
    request: ProcessDocumentsRequest,
    token: str = Depends(verify_token)
):
    """Process documents from any supported URL"""
    session_id = None
    try:
        logger.info(f"📥 Processing documents from: {request.documents}")
        
        # Use universal downloader
        file_path = await download_from_url(str(request.documents))
        
        # Create new session
        session = EnhancedRAGSystem()
        session_id = session.session_id
        ACTIVE_SESSIONS[session_id] = session
        
        # Process documents
        result = await session.process_documents([file_path])
        
        # Cleanup temporary file
        try:
            os.unlink(file_path)
        except Exception as e:
            logger.warning(f"⚠️ Could not delete temp file {file_path}: {e}")
        
        return ProcessDocumentsResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Document processing failed: {e}")
        # Cleanup on error
        if session_id and session_id in ACTIVE_SESSIONS:
            try:
                await ACTIVE_SESSIONS[session_id].cleanup()
                del ACTIVE_SESSIONS[session_id]
            except Exception as cleanup_error:
                logger.warning(f"⚠️ Error during cleanup: {cleanup_error}")
        
        raise HTTPException(
            status_code=500,
            detail=f"Document processing failed: {str(e)}"
        )

@app.post("/hackrx/run")
async def hackrx_run_endpoint(
    request: QueryRequest,
    token: str = Depends(verify_token)
):
    """Process documents and answer questions - HackRx format with universal URL support"""
    try:
        if not request.questions:
            raise HTTPException(status_code=400, detail="No questions provided")

        logger.info(f"❓ Processing {len(request.questions)} questions for HackRx")

        # Use universal URL processing
        session = None
        if request.documents:
            document_processed = False
            for existing_session in ACTIVE_SESSIONS.values():
                if hasattr(existing_session, 'document_url') and existing_session.document_url == str(request.documents):
                    session = existing_session
                    document_processed = True
                    break

            if not document_processed:
                logger.info(f"📥 Processing new document: {request.documents}")
                file_path = await download_from_url(str(request.documents))  # Changed here
                
                session = EnhancedRAGSystem()
                session.document_url = str(request.documents)
                ACTIVE_SESSIONS[session.session_id] = session
                
                await session.process_documents([file_path])
                
                try:
                    os.unlink(file_path)
                except Exception as e:
                    logger.warning(f"⚠️ Could not delete temp file {file_path}: {e}")

        if not session:
            raise HTTPException(status_code=400, detail="No document provided for processing")

        # Process questions
        start_time = time.time()
        answers = []
        
        for i, question in enumerate(request.questions, 1):
            try:
                logger.info(f"❓ Processing question {i}/{len(request.questions)}")
                
                result = await session.answer_question(question)
                
                answers.append(AnswerResponse(
                    question=question,
                    answer=result['answer'],
                    confidence=result['confidence'],
                    sources_used=result['sources_used'],
                    processing_time=result['processing_time']
                ))
                
                logger.info(f"✅ Question {i} processed (confidence: {result['confidence']:.2f})")
                
            except Exception as e:
                logger.error(f"❌ Error processing question {i}: {e}")
                answers.append(AnswerResponse(
                    question=question,
                    answer=f"Error processing question: {str(e)}",
                    confidence=0.0,
                    sources_used=0,
                    processing_time=0.0
                ))

        total_time = time.time() - start_time
        
        return QueryResponse(
            answers=answers,
            session_id=session.session_id,
            total_processing_time=total_time
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ HackRx endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.delete("/sessions/{session_id}")
async def cleanup_session(
    session_id: str,
    token: str = Depends(verify_token)
):
    """Cleanup a specific session"""
    if session_id in ACTIVE_SESSIONS:
        try:
            await ACTIVE_SESSIONS[session_id].cleanup()
            del ACTIVE_SESSIONS[session_id]
            return {"message": f"Session {session_id} cleaned up successfully"}
        except Exception as e:
            logger.warning(f"⚠️ Error cleaning up session {session_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")
    else:
        raise HTTPException(status_code=404, detail="Session not found")

@app.get("/sessions")
async def list_sessions(token: str = Depends(verify_token)):
    """List active sessions"""
    sessions_info = []
    for session_id, session in ACTIVE_SESSIONS.items():
        sessions_info.append({
            "session_id": session_id,
            "domain": getattr(session, 'domain', 'unknown'),
            "document_count": len(getattr(session, 'documents', [])),
            "document_hash": getattr(session, 'document_hash', 'unknown')
        })
    
    return {
        "active_sessions": len(ACTIVE_SESSIONS),
        "sessions": sessions_info
    }

# ================================
# APPLICATION STARTUP
# ================================

if __name__ == "__main__":
    import uvicorn
    import os
    
    # Get port from environment variable (Cloud Run requirement)
    port = int(os.getenv("PORT", 8000))
    
    logger.info(f"🚀 Starting Enhanced RAG System server on port {port}...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,  # Use environment port
        reload=False,
        log_level="info"
    )
