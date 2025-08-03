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
PINECONE_API_KEY = "pcsk_5SJNxg_B3sWxTJSuUBgYi6GDEuyHgNyt337K2Mts2SFY3udWPLdd2MiyETruf7iyV6SRhe"
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
# LAZY LOADING FOR HEAVY COMPONENTS
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
    """Lazy load heavy models on demand"""
    global base_sentence_model, embedding_model, reranker
    
    if base_sentence_model is None:
        try:
            logger.info("🔄 Loading sentence transformer...")
            base_sentence_model = await asyncio.to_thread(
                SentenceTransformer, 'all-MiniLM-L6-v2'
            )
            components_ready["base_sentence_model"] = True
            logger.info("✅ Sentence transformer loaded")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load sentence transformer: {e}")

    if embedding_model is None:
        try:
            logger.info("🔄 Loading embedding model...")
            embedding_model = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            components_ready["embedding_model"] = True
            logger.info("✅ Embedding model loaded")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load embedding model: {e}")

    if reranker is None:
        try:
            logger.info("🔄 Loading reranker...")
            reranker = await asyncio.to_thread(
                CrossEncoder, 'cross-encoder/ms-marco-MiniLM-L-6-v2'
            )
            components_ready["reranker"] = True
            logger.info("✅ Reranker loaded")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load reranker: {e}")
            components_ready["reranker"] = False

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
            "\n\n", "\n", ". ", "! ", "? ", "; ", " ", ""   # General separators
        ]
        
        # Universal section patterns - NO domain-specific bias
        self.section_patterns = [
            r'(?i)^([A-Z][^.:!?]*[.:])\s*$',  # Generic headers ending with . or :
            r'(?i)^(\d+(?:\.\d+)*)\s+(.+)$',  # Numbered sections
            r'(?i)^([IVX]+\.)\s+(.+)$',       # Roman numerals
            r'(?i)^(\([a-zA-Z0-9]\))\s+(.+)$',  # Lettered/numbered lists
            r'(?i)^([A-Z\s]{3,})\s*$',        # ALL CAPS headers
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
        """Universal document loader"""
        try:
            # Try to detect MIME type with fallback
            try:
                if HAS_MAGIC and magic:
                    mime_type = magic.from_file(file_path, mime=True)
                    logger.info(f"📄 Detected MIME type: {mime_type} for {file_path}")
                else:
                    mime_type = None
            except Exception as e:
                logger.warning(f"⚠️ MIME detection failed: {e}")
                mime_type = None

            file_extension = os.path.splitext(file_path)[1].lower()

            # Route based on MIME type or extension
            if mime_type == 'application/pdf' or file_extension == '.pdf':
                loader = PyMuPDFLoader(file_path)
            elif 'word' in (mime_type or '') or file_extension in ['.docx', '.doc']:
                loader = Docx2txtLoader(file_path)
            elif 'text' in (mime_type or '') or file_extension in ['.txt', '.md', '.csv']:
                loader = TextLoader(file_path, encoding='utf-8')
            else:
                logger.info(f"📄 Unknown type {mime_type or file_extension}, trying as text...")
                loader = TextLoader(file_path, encoding='utf-8')

            docs = await asyncio.to_thread(loader.load)

            if not docs:
                raise ValueError(f"No content loaded from {file_path}")

            logger.info(f"✅ Loaded {len(docs)} documents from {file_path}")
            return docs

        except Exception as e:
            logger.error(f"❌ Failed to load {file_path}: {e}")
            raise

    def _get_file_type(self, source_path: str) -> str:
        """Get file type from path"""
        extension = os.path.splitext(source_path)[1].lower()
        type_mapping = {
            '.pdf': 'pdf',
            '.docx': 'docx', '.doc': 'docx',
            '.txt': 'text', '.md': 'markdown',
            '.csv': 'csv',
            '.pptx': 'powerpoint',
            '.xlsx': 'excel'
        }
        return type_mapping.get(extension, 'unknown')

    async def _setup_retrievers(self):
        """Setup retrievers"""
        try:
            logger.info("🔧 Setting up retrievers...")

            # Setup vector store if available
            if HAS_PINECONE and pinecone and embedding_model and PINECONE_API_KEY:
                try:
                    await ensure_models_ready()
                    namespace = f"{self.domain}_{self.document_hash}"

                    # Initialize Pinecone if not done
                    try:
                        pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
                        if PINECONE_INDEX_NAME not in pinecone.list_indexes():
                            pinecone.create_index(
                                name=PINECONE_INDEX_NAME,
                                dimension=384,
                                metric="cosine"
                            )

                        global pinecone_index
                        pinecone_index = pinecone.Index(PINECONE_INDEX_NAME)
                        components_ready["pinecone"] = True
                    except Exception as e:
                        logger.warning(f"⚠️ Pinecone initialization failed: {e}")
                        components_ready["pinecone"] = False

                    if pinecone_index:
                        self.vector_store = Pinecone(
                            index=pinecone_index,
                            embedding=embedding_model,
                            text_key="text",
                            namespace=namespace
                        )

                        stats = pinecone_index.describe_index_stats()
                        current_count = stats.get('namespaces', {}).get(namespace, {}).get('vector_count', 0)

                        if current_count < len(self.documents):
                            logger.info(f"📊 Adding {len(self.documents)} documents to vector store")
                            await self._batch_upsert_to_pinecone(list(self.documents), namespace)

                        logger.info("✅ Vector store setup complete")

                except Exception as e:
                    logger.warning(f"⚠️ Vector store setup failed: {e}")

            # Setup BM25 retriever
            try:
                if self.documents:
                    self.bm25_retriever = await asyncio.to_thread(
                        BM25Retriever.from_documents,
                        list(self.documents)
                    )
                    self.bm25_retriever.k = min(self.domain_config["rerank_top_k"], len(self.documents))
                    logger.info(f"✅ BM25 retriever setup complete (k={self.bm25_retriever.k})")
            except Exception as e:
                logger.warning(f"⚠️ BM25 retriever setup failed: {e}")

        except Exception as e:
            logger.error(f"❌ Retriever setup error: {e}")

    async def _batch_upsert_to_pinecone(self, documents: List[Document], namespace: str):
        """Efficient batch upsert to Pinecone"""
        try:
            batch_size = 50
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]

                # Prepare vectors
                vectors = []
                for j, doc in enumerate(batch):
                    doc_id = f"{namespace}_{i+j}"
                    # Get embedding
                    embedding = await EMBEDDING_SERVICE.get_query_embedding(doc.page_content)
                    vectors.append((
                        doc_id,
                        embedding.tolist(),
                        {
                            "text": doc.page_content,
                            "source": doc.metadata.get("source", ""),
                            "chunk_type": doc.metadata.get("chunk_type", ""),
                            "section_header": doc.metadata.get("section_header", "")
                        }
                    ))

                # Upsert to Pinecone
                await asyncio.to_thread(pinecone_index.upsert, vectors=vectors, namespace=namespace)

                if LOG_VERBOSE:
                    logger.info(f"📊 Upserted batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")

        except Exception as e:
            logger.error(f"❌ Batch upsert error: {e}")

    async def enhanced_retrieve_and_rerank(self, query: str, top_k: int = None) -> Tuple[List[Document], List[float]]:
        """Enhanced retrieval with parallel processing"""
        if top_k is None:
            top_k = self.domain_config["context_docs"]

        try:
            if not self.documents:
                return [], []

            semantic_k = min(self.domain_config["semantic_search_k"], len(self.documents))
            rerank_k = min(self.domain_config["rerank_top_k"], len(self.documents))

            if LOG_VERBOSE:
                logger.info(f"🔍 Retrieving: semantic_k={semantic_k}, rerank_k={rerank_k}, final_k={top_k}")

            # Parallel retrieval
            tasks = []
            if self.vector_store:
                if self.domain_config.get("use_mmr", True):
                    tasks.append(self._mmr_search(query, semantic_k))
                else:
                    tasks.append(self._vector_search(query, semantic_k))

            if self.bm25_retriever:
                tasks.append(self._bm25_search(query))

            if tasks:
                search_results = await asyncio.gather(*tasks, return_exceptions=True)
            else:
                search_results = [await self._fallback_search(query, semantic_k)]

            return await self._merge_and_rerank(query, search_results, top_k, rerank_k)

        except Exception as e:
            logger.error(f"❌ Retrieval error: {e}")
            return list(self.documents)[:top_k], [0.5] * min(len(self.documents), top_k)

    async def _mmr_search(self, query: str, k: int) -> List[Tuple[Document, float]]:
        """MMR search with error handling"""
        try:
            lambda_mult = self.domain_config.get("mmr_lambda", 0.75)
            results = await asyncio.to_thread(
                self.vector_store.max_marginal_relevance_search_with_score,
                query,
                k=k,
                lambda_mult=lambda_mult
            )
            return results
        except Exception as e:
            logger.warning(f"⚠️ MMR search error: {e}")
            return await self._vector_search(query, k)

    async def _vector_search(self, query: str, k: int) -> List[Tuple[Document, float]]:
        """Vector search with error handling"""
        try:
            results = await asyncio.to_thread(
                self.vector_store.similarity_search_with_score,
                query,
                k=k
            )
            return results
        except Exception as e:
            logger.warning(f"⚠️ Vector search error: {e}")
            return []

    async def _bm25_search(self, query: str) -> List[Document]:
        """BM25 search"""
        try:
            return await asyncio.to_thread(self.bm25_retriever.invoke, query)
        except Exception as e:
            logger.warning(f"⚠️ BM25 search error: {e}")
            return []

    async def _fallback_search(self, query: str, k: int) -> List[Tuple[Document, float]]:
        """Fallback search using keyword matching"""
        try:
            query_terms = set(query.lower().split())
            doc_scores = []
            
            for doc in list(self.documents):
                doc_terms = set(doc.page_content.lower().split())
                overlap = len(query_terms.intersection(doc_terms))
                score = overlap / max(len(query_terms), 1)
                
                # Boost score for exact phrase matches
                query_lower = query.lower()
                if query_lower in doc.page_content.lower():
                    score += 0.3
                    
                # Boost for section headers
                if doc.metadata.get('section_header', '').lower() in query_lower:
                    score += 0.2
                    
                doc_scores.append((doc, score))

            doc_scores.sort(key=lambda x: x[1], reverse=True)
            return doc_scores[:k]
            
        except Exception as e:
            logger.error(f"❌ Fallback search error: {e}")
            return [(doc, 0.5) for doc in list(self.documents)[:k]]

    async def _merge_and_rerank(self, query: str, search_results: List, top_k: int, rerank_k: int) -> Tuple[List[Document], List[float]]:
        """Merge and rerank results"""
        all_docs = []
        all_scores = []
        seen_content = set()

        # Process vector search results
        if search_results and not isinstance(search_results[0], Exception):
            vector_results = search_results[0]
            if isinstance(vector_results, list) and vector_results:
                for item in vector_results:
                    if isinstance(item, tuple) and len(item) == 2:
                        doc, score = item
                        content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
                        if content_hash not in seen_content:
                            all_docs.append(doc)
                            # Normalize score if needed
                            normalized_score = max(0.0, min(1.0, (2.0 - score) / 2.0)) if score > 1.0 else score
                            all_scores.append(normalized_score)
                            seen_content.add(content_hash)

        # Process BM25 results
        if len(search_results) > 1 and not isinstance(search_results[1], Exception):
            bm25_results = search_results[1]
            if isinstance(bm25_results, list):
                for doc in bm25_results[:rerank_k]:
                    if hasattr(doc, 'page_content'):
                        content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
                        if content_hash not in seen_content:
                            all_docs.append(doc)
                            all_scores.append(0.6)
                            seen_content.add(content_hash)

        if not all_docs:
            all_docs = list(self.documents)[:rerank_k]
            all_scores = [0.4] * len(all_docs)

        # Rerank if available
        if reranker and len(all_docs) > 1:
            try:
                return await self._semantic_rerank(query, all_docs, all_scores, top_k)
            except Exception as e:
                logger.warning(f"⚠️ Reranking failed: {e}")

        # Fallback: sort by score
        scored_docs = list(zip(all_docs, all_scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        final_docs = [doc for doc, _ in scored_docs[:top_k]]
        final_scores = [score for _, score in scored_docs[:top_k]]
        
        return final_docs, final_scores

    async def _semantic_rerank(self, query: str, documents: List[Document], scores: List[float], top_k: int) -> Tuple[List[Document], List[float]]:
        """Semantic reranking"""
        try:
            if len(documents) <= 2:
                return documents[:top_k], scores[:top_k]

            await ensure_models_ready()
            if not reranker:
                return documents[:top_k], scores[:top_k]

            pairs = [[query, doc.page_content[:512]] for doc in documents[:25]]
            rerank_scores = await asyncio.to_thread(reranker.predict, pairs)

            # Normalize rerank scores
            normalized_rerank = [(score + 1) / 2 for score in rerank_scores]

            combined_scores = []
            for i, (orig_score, rerank_score) in enumerate(zip(scores[:len(normalized_rerank)], normalized_rerank)):
                doc = documents[i]
                boost = 1.0

                # Universal boost for complete sections
                if doc.metadata.get('chunk_type') == 'complete_section':
                    boost = 1.05

                combined = (0.7 * rerank_score + 0.3 * orig_score) * boost
                combined_scores.append(min(1.0, combined))

            # Handle remaining documents
            if len(documents) > len(combined_scores):
                combined_scores.extend(scores[len(combined_scores):])

            scored_docs = list(zip(documents, combined_scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)

            final_docs = [doc for doc, _ in scored_docs[:top_k]]
            final_scores = [score for _, score in scored_docs[:top_k]]

            return final_docs, final_scores

        except Exception as e:
            logger.error(f"❌ Semantic reranking error: {e}")
            return documents[:top_k], scores[:top_k]

# ================================
# UNIVERSAL DECISION ENGINE
# ================================

class UniversalDecisionEngine:
    """Universal decision engine without domain-specific biases"""
    
    def __init__(self):
        self.token_processor = TokenOptimizedProcessor()
        self.confidence_cache = LRUCache(maxsize=2000)
        self.response_cache = LRUCache(maxsize=1000)

    def calculate_confidence_score(self, query: str, similarity_scores: List[float],
                                 retrieved_docs: List[Document], domain_confidence: float = 1.0) -> float:
        """Universal confidence calculation"""
        if not similarity_scores:
            return 0.0

        scores_str = str(sorted(similarity_scores))
        cache_key = hashlib.md5(f"{query}_{scores_str}_{domain_confidence}".encode()).hexdigest()[:12]
        
        if cache_key in self.confidence_cache:
            return self.confidence_cache[cache_key]

        try:
            scores_array = np.array(similarity_scores)
            max_score = np.max(scores_array)
            avg_score = np.mean(scores_array)
            score_std = np.std(scores_array)
            
            # Calculate percentage of decent documents
            decent_docs = np.sum(scores_array > 0.4) / len(scores_array)
            
            # Score consistency metric
            score_consistency = max(0.0, 1.0 - (score_std * 1.5))
            
            # Query-document match quality
            query_match = self._calculate_query_match(query, retrieved_docs)
            
            # Base confidence calculation - BALANCED
            confidence = (
                0.30 * max_score +
                0.25 * avg_score +
                0.20 * query_match +
                0.15 * score_consistency +
                0.10 * decent_docs
            )
            
            # Domain confidence boost (small)
            confidence += 0.05 * domain_confidence
            
            # Additional boost for exact matches
            query_lower = query.lower()
            exact_match_boost = 0.0
            for doc in retrieved_docs[:5]:
                if any(phrase in doc.page_content.lower() for phrase in query_lower.split() if len(phrase) > 3):
                    exact_match_boost += 0.02
                    
            confidence += min(0.1, exact_match_boost)
            
            confidence = min(1.0, max(0.0, confidence))
            
            self.confidence_cache[cache_key] = confidence
            return confidence
            
        except Exception as e:
            logger.warning(f"⚠️ Confidence calculation error: {e}")
            return 0.5

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
                
            # Small boost for header matches
            header = doc.metadata.get('section_header', '')
            if header and any(term in header.lower() for term in query_terms):
                match_score += 0.1
                
            match_scores.append(match_score)

        return np.mean(match_scores) if match_scores else 0.5

    def _classify_query_type(self, query: str) -> str:
        """Classify query type - UNIVERSAL"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['what is', 'who is', 'when is', 'where is', 'define', 'definition']):
            return 'factoid'
        if any(word in query_lower for word in ['how to', 'how do', 'how can', 'steps', 'process', 'procedure']):
            return 'procedural'
        if any(word in query_lower for word in ['compare', 'difference', 'vs', 'versus', 'better', 'contrast']):
            return 'comparison'
        if any(word in query_lower for word in ['analyze', 'explain', 'why', 'because', 'reason', 'analyze']):
            return 'analysis'
        if any(word in query_lower for word in ['list', 'enumerate', 'describe', 'overview']):
            return 'descriptive'
            
        return 'general'

    async def process_query_with_fallback(self, query: str, retrieved_docs: List[Document],
                                        similarity_scores: List[float], domain: str,
                                        domain_confidence: float = 1.0, query_type: str = "general",
                                        rag_system: 'EnhancedRAGSystem' = None) -> Dict[str, Any]:
        """Universal query processing"""
        start_time = time.time()
        
        try:
            if not retrieved_docs:
                return self._empty_response(query, domain)

            # Classify query type
            classified_query_type = self._classify_query_type(query)
            
            # Calculate confidence
            confidence = self.calculate_confidence_score(query, similarity_scores, retrieved_docs, domain_confidence)

            # Get adaptive confidence threshold
            base_threshold = rag_system.domain_config["confidence_threshold"] if rag_system else 0.15
            
            # Adjust threshold based on query type
            if classified_query_type == "factoid":
                threshold = base_threshold * 1.2
            elif classified_query_type in ["procedural", "comparison"]:
                threshold = base_threshold * 1.1
            else:
                threshold = base_threshold

            # Fallback if confidence is low
            if confidence < threshold and rag_system:
                logger.info(f"🔄 Low confidence ({confidence:.2f}), attempting fallback")
                fallback_result = await self._attempt_fallback(
                    query, domain, rag_system, threshold
                )
                if fallback_result:
                    retrieved_docs, similarity_scores, confidence = fallback_result
                    logger.info(f"✅ Fallback improved confidence to {confidence:.2f}")

            # Optimize context
            context = self.token_processor.optimize_context_intelligently(
                retrieved_docs, query, max_tokens=4000
            )

            # Generate response
            response = await self._generate_universal_response(
                query, context, domain, confidence, classified_query_type
            )

            processing_time = time.time() - start_time

            result = {
                "query": query,
                "answer": response,
                "confidence": float(confidence),
                "domain": domain,
                "domain_confidence": float(domain_confidence),
                "query_type": classified_query_type,
                "reasoning_chain": [
                    f"Retrieved {len(retrieved_docs)} documents",
                    f"Confidence: {confidence:.1%} (threshold: {threshold:.1%})",
                    f"Domain: {domain} ({domain_confidence:.1%})",
                    f"Context optimized: {len(context)} chars",
                    f"Query type: {classified_query_type}"
                ],
                "source_documents": list(set([
                    doc.metadata.get('source', 'Unknown') for doc in retrieved_docs
                ])),
                "retrieved_chunks": len(retrieved_docs),
                "processing_time": processing_time,
                "enhanced_features": {
                    "universal_processing": True,
                    "adaptive_thresholding": True,
                    "context_optimization": True,
                    "query_type_classification": True,
                    "fallback_attempted": confidence < threshold
                }
            }

            return result

        except Exception as e:
            logger.error(f"❌ Query processing error: {e}")
            return self._error_response(query, domain, str(e))

    async def _attempt_fallback(self, query: str, domain: str, rag_system: 'EnhancedRAGSystem',
                              threshold: float) -> Optional[Tuple[List[Document], List[float], float]]:
        """Attempt fallback retrieval strategies"""
        try:
            fallback_multiplier = rag_system.domain_config.get("fallback_threshold_multiplier", 0.7)
            max_fallback_docs = rag_system.domain_config.get("max_fallback_docs", 30)
            
            # Strategy 1: Retrieve more documents
            expanded_docs, expanded_scores = await rag_system.enhanced_retrieve_and_rerank(
                query, top_k=min(max_fallback_docs, len(rag_system.documents))
            )
            
            if len(expanded_docs) > len(list(rag_system.documents)[:rag_system.domain_config["context_docs"]]):
                new_confidence = self.calculate_confidence_score(query, expanded_scores, expanded_docs)
                if new_confidence > threshold * fallback_multiplier:
                    return expanded_docs, expanded_scores, new_confidence

            # Strategy 2: Query expansion
            expanded_query = self._expand_query(query)
            if expanded_query != query:
                fallback_docs, fallback_scores = await rag_system.enhanced_retrieve_and_rerank(
                    expanded_query, top_k=rag_system.domain_config["context_docs"]
                )
                
                if fallback_docs:
                    new_confidence = self.calculate_confidence_score(expanded_query, fallback_scores, fallback_docs)
                    if new_confidence > threshold * (fallback_multiplier * 0.9):
                        return fallback_docs, fallback_scores, new_confidence

            # Strategy 3: Broader keyword search
            query_keywords = [word for word in query.split() if len(word) > 3]
            if len(query_keywords) > 1:
                broader_query = " ".join(query_keywords[:3])
                fallback_docs, fallback_scores = await rag_system.enhanced_retrieve_and_rerank(
                    broader_query, top_k=rag_system.domain_config["context_docs"]
                )
                
                if fallback_docs:
                    new_confidence = self.calculate_confidence_score(broader_query, fallback_scores, fallback_docs)
                    if new_confidence > threshold * (fallback_multiplier * 0.8):
                        return fallback_docs, fallback_scores, new_confidence

            return None

        except Exception as e:
            logger.warning(f"⚠️ Fallback error: {e}")
            return None

    def _expand_query(self, query: str) -> str:
        """Universal query expansion"""
        query_lower = query.lower()
        expansions = []

        # Universal question type expansions
        if "what" in query_lower and "is" in query_lower:
            expansions.extend(["definition", "meaning", "explanation"])
        elif "who" in query_lower:
            expansions.extend(["person", "individual", "identity"])
        elif "how" in query_lower:
            expansions.extend(["process", "method", "way"])
        elif "when" in query_lower:
            expansions.extend(["time", "date", "period"])
        elif "why" in query_lower:
            expansions.extend(["reason", "cause", "purpose"])
        elif "where" in query_lower:
            expansions.extend(["location", "place", "position"])

        if expansions:
            return f"{query} {' '.join(expansions[:2])}"
        return query

    async def _generate_universal_response(self, query: str, context: str, domain: str,
                                         confidence: float, query_type: str = "general") -> str:
        """Generate universal response"""
        try:
            await ensure_openai_ready()
            if not openai_client:
                return "System is still initializing. Please wait a moment and try again."

            # Universal system prompt
            system_prompt = f"""You are an expert document analyst specializing in {domain} content. Provide accurate, helpful responses based on the provided context.

INSTRUCTIONS:
1. Answer questions directly based on the context provided
2. If information is not available in the context, clearly state this
3. Be concise but comprehensive in your responses
4. Cite specific details from the context when relevant
5. Maintain accuracy and avoid speculation beyond the provided information
6. For factual questions, be precise and direct
7. For procedural questions, provide clear step-by-step information
8. For analytical questions, support your analysis with evidence from the context"""

            user_message = f"""Context:
{context}

Question: {query}

Please provide a comprehensive answer based on the context above."""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]

            response = await openai_client.optimized_completion(
                messages=messages,
                model="gpt-4o",
                temperature=0.1 if query_type == "factoid" else 0.2,
                max_tokens=1200 if query_type in ["analysis", "comparison", "descriptive"] else 800
            )

            return response.strip()

        except Exception as e:
            logger.error(f"❌ Response generation error: {e}")
            return f"I apologize, but I encountered an error while processing your query: {str(e)}"

    def _empty_response(self, query: str, domain: str) -> Dict[str, Any]:
        """Generate response when no documents retrieved"""
        return {
            "query": query,
            "answer": "No relevant documents found for your query. Please check if documents are properly loaded.",
            "confidence": 0.0,
            "domain": domain,
            "domain_confidence": 0.0,
            "query_type": "unknown",
            "reasoning_chain": ["No documents retrieved"],
            "source_documents": [],
            "retrieved_chunks": 0,
            "processing_time": 0.0,
            "enhanced_features": {"error_response": True}
        }

    def _error_response(self, query: str, domain: str, error: str) -> Dict[str, Any]:
        """Generate error response"""
        return {
            "query": query,
            "answer": f"An error occurred while processing your query: {error}",
            "confidence": 0.0,
            "domain": domain,
            "domain_confidence": 0.0,
            "query_type": "error",
            "reasoning_chain": [f"Error: {error}"],
            "source_documents": [],
            "retrieved_chunks": 0,
            "processing_time": 0.0,
            "enhanced_features": {"error_response": True}
        }

# ================================
# GLOBAL INSTANCES
# ================================

# Initialize global instances
REDIS_CACHE = RedisCache()
EMBEDDING_SERVICE = OptimizedEmbeddingService()
DOMAIN_DETECTOR = UniversalDomainDetector()
DECISION_ENGINE = UniversalDecisionEngine()

# ================================
# ASYNC STARTUP AND SHUTDOWN
# ================================

async def initialize_system():
    """Initialize all system components"""
    logger.info("🚀 Starting system initialization...")
    
    try:
        # Initialize Redis cache
        await REDIS_CACHE.initialize()
        
        # Initialize OpenAI client
        if OPENAI_API_KEY:
            await ensure_openai_ready()
        
        # Initialize models (lazy loading)
        await ensure_models_ready()
        
        # Initialize domain embeddings
        if base_sentence_model:
            DOMAIN_DETECTOR.initialize_embeddings()
        
        logger.info("✅ System initialization complete")
        
        # Log component status
        ready_components = [k for k, v in components_ready.items() if v]
        logger.info(f"📊 Components ready: {', '.join(ready_components)}")
        
    except Exception as e:
        logger.error(f"❌ System initialization failed: {e}")
        raise

async def shutdown_system():
    """Cleanup system resources"""
    logger.info("🔄 Shutting down system...")
    
    try:
        # Clear caches
        EMBEDDING_CACHE.clear()
        RESPONSE_CACHE.clear()
        
        # Clear active sessions
        for session_id in list(ACTIVE_SESSIONS.keys()):
            try:
                session = ACTIVE_SESSIONS.pop(session_id, None)
                if session:
                    await session.cleanup()
            except Exception as e:
                logger.warning(f"⚠️ Error cleaning up session {session_id}: {e}")
        
        # Close Redis connection
        if REDIS_CACHE.redis:
            try:
                await REDIS_CACHE.redis.aclose()
            except Exception as e:
                logger.warning(f"⚠️ Error closing Redis connection: {e}")
        
        # Close OpenAI client
        if openai_client and hasattr(openai_client, 'client') and openai_client.client:
            try:
                await openai_client.client.close()
            except Exception as e:
                logger.warning(f"⚠️ Error closing OpenAI client: {e}")
        
        logger.info("✅ System shutdown complete")
        
    except Exception as e:
        logger.error(f"❌ Shutdown error: {e}")

# ================================
# FASTAPI APPLICATION
# ================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    await initialize_system()
    yield
    # Shutdown
    await shutdown_system()

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Universal RAG System v2.0",
    description="Enhanced Retrieval-Augmented Generation System with Universal Domain Support",
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
# PYDANTIC MODELS
# ================================

class ProcessDocumentsRequest(BaseModel):
    documents: HttpUrl = Field(description="Google Drive download URL for documents")

class QueryRequest(BaseModel):
    questions: List[str] = Field(description="List of questions to process")

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

class QuestionAnswer(BaseModel):
    question: str
    answer: str
    confidence: float
    source_info: Dict[str, Any]

class QueryResponse(BaseModel):
    questions_and_answers: List[QuestionAnswer]
    processing_summary: Dict[str, Any]
    system_info: Dict[str, Any]

class HealthCheckResponse(BaseModel):
    status: str
    version: str
    components: Dict[str, bool]
    system_metrics: Optional[Dict[str, Any]] = None

# ================================
# API ENDPOINTS
# ================================

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Enhanced health check endpoint"""
    try:
        # Basic health metrics
        health_data = {
            "status": "healthy",
            "version": "2.0.0",
            "components": components_ready.copy(),
        }
        
        # Add system metrics if available
        if HAS_PSUTIL and psutil:
            try:
                health_data["system_metrics"] = {
                    "memory_usage_percent": psutil.virtual_memory().percent,
                    "cpu_usage_percent": psutil.cpu_percent(interval=0.1),
                    "active_sessions": len(ACTIVE_SESSIONS),
                    "cache_sizes": {
                        "embedding_cache": len(EMBEDDING_CACHE),
                        "response_cache": len(RESPONSE_CACHE)
                    }
                }
            except Exception as e:
                logger.warning(f"⚠️ Could not get system metrics: {e}")
        
        return HealthCheckResponse(**health_data)
        
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return HealthCheckResponse(
            status="unhealthy",
            version="2.0.0",
            components=components_ready.copy()
        )

@app.post("/process-documents", response_model=ProcessDocumentsResponse)
async def process_documents_endpoint(
    request: ProcessDocumentsRequest,
    token: str = Depends(verify_token)
):
    """Process documents from Google Drive URL"""
    session_id = None
    
    try:
        logger.info(f"📥 Processing documents from: {request.documents}")
        
        # Download file from Google Drive
        file_path = await download_from_google_drive(str(request.documents))
        
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

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(
    request: QueryRequest,
    token: str = Depends(verify_token)
):
    """Process multiple queries with enhanced features"""
    start_time = time.time()
    
    try:
        if not request.questions:
            raise HTTPException(status_code=400, detail="No questions provided")
        
        if len(request.questions) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 questions allowed")
        
        logger.info(f"❓ Processing {len(request.questions)} questions")
        
        # Find the most recent session
        if not ACTIVE_SESSIONS:
            raise HTTPException(
                status_code=400, 
                detail="No active document sessions. Please process documents first."
            )
        
        latest_session = max(ACTIVE_SESSIONS.values(), key=lambda s: s.session_id)
        
        results = []
        successful_answers = 0
        
        # Process each question
        for i, question in enumerate(request.questions, 1):
            question_start = time.time()
            
            try:
                logger.info(f"❓ Processing question {i}/{len(request.questions)}: {question[:50]}...")
                
                # Enhanced retrieval and processing
                retrieved_docs, similarity_scores = await asyncio.wait_for(
                    latest_session.enhanced_retrieve_and_rerank(question),
                    timeout=QUESTION_TIMEOUT
                )
                
                # Process with decision engine
                result = await DECISION_ENGINE.process_query_with_fallback(
                    query=question,
                    retrieved_docs=retrieved_docs,
                    similarity_scores=similarity_scores,
                    domain=latest_session.domain,
                    domain_confidence=1.0,
                    query_type="general",
                    rag_system=latest_session
                )
                
                question_time = time.time() - question_start
                
                # Format response
                question_answer = QuestionAnswer(
                    question=question,
                    answer=result["answer"],
                    confidence=result["confidence"],
                    source_info={
                        "retrieved_chunks": result["retrieved_chunks"],
                        "domain": result["domain"],
                        "processing_time": question_time
                    }
                )
                
                results.append(question_answer)
                successful_answers += 1
                
            except asyncio.TimeoutError:
                logger.warning(f"⏰ Question {i} timed out after {QUESTION_TIMEOUT}s")
                results.append(QuestionAnswer(
                    question=question,
                    answer="Query timed out. Please try a more specific question.",
                    confidence=0.0,
                    source_info={
                        "retrieved_chunks": 0,
                        "domain": latest_session.domain,
                        "processing_time": QUESTION_TIMEOUT
                    }
                ))
            except Exception as e:
                logger.error(f"❌ Error processing question {i}: {e}")
                results.append(QuestionAnswer(
                    question=question,
                    answer=f"Error processing question: {str(e)}",
                    confidence=0.0,
                    source_info={
                        "retrieved_chunks": 0,
                        "domain": latest_session.domain,
                        "processing_time": time.time() - question_start
                    }
                ))
        
        total_time = time.time() - start_time
        
        # Prepare response
        response = QueryResponse(
            questions_and_answers=results,
            processing_summary={
                "total_questions": len(request.questions),
                "successful_answers": successful_answers,
                "total_processing_time": total_time,
                "document_info": {
                    "filename": f"google_drive_{latest_session.document_hash}.pdf",
                    "document_hash": latest_session.document_hash,
                    "total_chunks": len(latest_session.documents),
                    "detected_domain": latest_session.domain
                }
            },
            system_info={
                "version": "2.0.0",
                "session_id": latest_session.session_id,
                "timestamp": datetime.now().isoformat(),
                "components_status": components_ready.copy()
            }
        )
        
        logger.info(f"✅ Query processing completed in {total_time:.2f}s")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Query processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.get("/sessions")
async def list_sessions(token: str = Depends(verify_token)):
    """List active sessions with details"""
    try:
        sessions_info = {}
        for session_id, session in ACTIVE_SESSIONS.items():
            sessions_info[session_id] = {
                "domain": session.domain,
                "document_hash": session.document_hash,
                "total_chunks": len(session.documents),
                "processed_files": session.processed_files,
                "confidence_threshold": session.domain_config.get("confidence_threshold", 0.15)
            }
        
        return {
            "active_sessions": len(ACTIVE_SESSIONS),
            "sessions": sessions_info,
            "cache_stats": {
                "embedding_cache_size": len(EMBEDDING_CACHE),
                "response_cache_size": len(RESPONSE_CACHE)
            }
        }
    except Exception as e:
        logger.error(f"❌ Error listing sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str, token: str = Depends(verify_token)):
    """Delete a specific session"""
    try:
        if session_id not in ACTIVE_SESSIONS:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = ACTIVE_SESSIONS.pop(session_id)
        await session.cleanup()
        
        return {"message": f"Session {session_id} deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error deleting session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/sessions")
async def clear_all_sessions(token: str = Depends(verify_token)):
    """Clear all active sessions"""
    try:
        session_count = len(ACTIVE_SESSIONS)
        
        for session_id in list(ACTIVE_SESSIONS.keys()):
            try:
                session = ACTIVE_SESSIONS.pop(session_id)
                await session.cleanup()
            except Exception as e:
                logger.warning(f"⚠️ Error cleaning up session {session_id}: {e}")
        
        # Clear caches
        EMBEDDING_CACHE.clear()
        RESPONSE_CACHE.clear()
        
        return {
            "message": f"Cleared {session_count} sessions and all caches",
            "cleared_sessions": session_count
        }
    except Exception as e:
        logger.error(f"❌ Error clearing sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/domains")
async def list_supported_domains(token: str = Depends(verify_token)):
    """List all supported domains and their configurations"""
    try:
        domain_info = {}
        for domain, config in ADAPTIVE_DOMAIN_CONFIGS.items():
            domain_info[domain] = {
                "keywords": DOMAIN_KEYWORDS.get(domain, []),
                "confidence_adjustment": config.get("confidence_adjustment", 1.0),
                "chunk_size_adjustment": config.get("chunk_size_adjustment", 1.0),
                "semantic_weight": config.get("semantic_weight", 0.6)
            }
        
        return {
            "supported_domains": list(DOMAIN_KEYWORDS.keys()),
            "domain_configurations": domain_info,
            "base_config": BASE_CONFIG
        }
    except Exception as e:
        logger.error(f"❌ Error listing domains: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/hackrx/run")
async def hackrx_run_endpoint(
    documents: HttpUrl = Field(description="Google Drive download URL for documents"),
    questions: List[str] = Field(description="List of questions to process"),
    token: str = Depends(verify_token)
):
    """HackRx combined endpoint - process documents and answer questions in one call"""
    start_time = time.time()
    session_id = None
    
    try:
        if not questions:
            raise HTTPException(status_code=400, detail="No questions provided")
        
        if len(questions) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 questions allowed")
            
        logger.info(f"🚀 HackRx run: processing documents and {len(questions)} questions")
        
        # Step 1: Process documents
        file_path = await download_from_google_drive(str(documents))
        
        session = EnhancedRAGSystem()
        session_id = session.session_id
        ACTIVE_SESSIONS[session_id] = session
        
        doc_result = await session.process_documents([file_path])
        
        # Cleanup temporary file
        try:
            os.unlink(file_path)
        except Exception as e:
            logger.warning(f"⚠️ Could not delete temp file {file_path}: {e}")
        
        # Step 2: Process questions
        results = []
        successful_answers = 0
        
        for i, question in enumerate(questions, 1):
            question_start = time.time()
            
            try:
                logger.info(f"❓ Processing question {i}/{len(questions)}: {question[:50]}...")
                
                retrieved_docs, similarity_scores = await asyncio.wait_for(
                    session.enhanced_retrieve_and_rerank(question),
                    timeout=QUESTION_TIMEOUT
                )
                
                result = await DECISION_ENGINE.process_query_with_fallback(
                    query=question,
                    retrieved_docs=retrieved_docs,
                    similarity_scores=similarity_scores,
                    domain=session.domain,
                    domain_confidence=1.0,
                    query_type="general",
                    rag_system=session
                )
                
                question_time = time.time() - question_start
                
                question_answer = {
                    "question": question,
                    "answer": result["answer"],
                    "confidence": result["confidence"],
                    "source_info": {
                        "retrieved_chunks": result["retrieved_chunks"],
                        "domain": result["domain"],
                        "processing_time": question_time
                    }
                }
                
                results.append(question_answer)
                successful_answers += 1
                
            except Exception as e:
                logger.error(f"❌ Error processing question {i}: {e}")
                results.append({
                    "question": question,
                    "answer": f"Error processing question: {str(e)}",
                    "confidence": 0.0,
                    "source_info": {
                        "retrieved_chunks": 0,
                        "domain": session.domain,
                        "processing_time": time.time() - question_start
                    }
                })
        
        total_time = time.time() - start_time
        
        # Prepare combined response
        response = {
            "questions_and_answers": results,
            "processing_summary": {
                "total_questions": len(questions),
                "successful_answers": successful_answers,
                "total_processing_time": total_time,
                "document_info": {
                    "filename": f"google_drive_{session.document_hash}.pdf",
                    "document_hash": session.document_hash,
                    "total_chunks": len(session.documents),
                    "detected_domain": session.domain
                }
            },
            "system_info": {
                "version": "2.0.0",
                "session_id": session.session_id,
                "timestamp": datetime.now().isoformat(),
                "components_status": components_ready.copy()
            }
        }
        
        logger.info(f"✅ HackRx run completed in {total_time:.2f}s")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ HackRx run failed: {e}")
        
        # Cleanup on error
        if session_id and session_id in ACTIVE_SESSIONS:
            try:
                await ACTIVE_SESSIONS[session_id].cleanup()
                del ACTIVE_SESSIONS[session_id]
            except Exception as cleanup_error:
                logger.warning(f"⚠️ Error during cleanup: {cleanup_error}")
        
        raise HTTPException(status_code=500, detail=f"HackRx run failed: {str(e)}")

# ================================
# UTILITY FUNCTIONS
# ================================

async def download_from_google_drive(url: str) -> str:
    """Download file from Google Drive with enhanced error handling"""
    try:
        # Extract file ID from various Google Drive URL formats
        file_id = None
        
        if "drive.google.com" in url:
            if "/file/d/" in url:
                file_id = url.split("/file/d/")[1].split("/")[0]
            elif "id=" in url:
                file_id = url.split("id=")[1].split("&")[0]
        elif "docs.google.com" in url and "export?" in url:
            if "id=" in url:
                file_id = url.split("id=")[1].split("&")[0]
        
        if not file_id:
            raise ValueError("Could not extract file ID from Google Drive URL")
        
        # Construct direct download URL
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        
        logger.info(f"📥 Downloading from Google Drive: {file_id}")
        
        # Download with timeout and proper error handling
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=300),
            connector=aiohttp.TCPConnector(limit=10)
        ) as session:
            async with session.get(download_url) as response:
                if response.status != 200:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Failed to download file: HTTP {response.status}"
                    )
                
                # Get content type for file extension
                content_type = response.headers.get('content-type', 'application/octet-stream')
                
                # Determine file extension
                extension = '.bin'
                if 'pdf' in content_type:
                    extension = '.pdf'
                elif 'word' in content_type or 'docx' in content_type:
                    extension = '.docx'
                elif 'text' in content_type:
                    extension = '.txt'
                
                # Create temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as temp_file:
                    temp_path = temp_file.name
                    
                    # Download in chunks to handle large files
                    async for chunk in response.content.iter_chunked(8192):
                        temp_file.write(chunk)
                
                logger.info(f"✅ Downloaded to: {temp_path}")
                return temp_path
                
    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        raise HTTPException(status_code=400, detail=f"Download failed: {str(e)}")

# ================================
# MAIN ENTRY POINT
# ================================

if __name__ == "__main__":
    import uvicorn
    
    # Cloud Run compatible configuration
    port = int(os.getenv("PORT", 8080))  # ✅ Use Cloud Run's PORT
    
    config = {
        "host": "0.0.0.0",
        "port": port,
        "workers": 1,
        "loop": "asyncio",
        "http": "httptools",
        "log_level": "info",
        "access_log": True,
        "reload": False,
        "timeout_keep_alive": 300,  # ✅ Increased for heavy processing
        "timeout_graceful_shutdown": 120,  # ✅ Increased for cleanup
    }
    
    logger.info(f"🚀 Starting Universal RAG System v2.0 on port {port}...")
    logger.info(f"📊 Configuration: {config}")
    
    try:
        # Start the server without blocking model loading
        uvicorn.run(app, **config)
    except KeyboardInterrupt:
        logger.info("👋 Graceful shutdown initiated")
    except Exception as e:
        logger.error(f"❌ Server startup failed: {e}")
        sys.exit(1)
