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
import shutil
import tempfile
import hashlib
import re
from datetime import datetime, timedelta
from functools import lru_cache
from collections import defaultdict
from urllib.parse import urlparse, parse_qs
from cachetools import TTLCache, LRUCache
import traceback
import mmap

# Core libraries
import pandas as pd
import numpy as np
import psutil
from sklearn.metrics.pairwise import cosine_similarity
import torch

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
from langchain.retrievers import EnsembleRetriever

# Vector stores - PRODUCTION READY
import pinecone
from langchain.vectorstores import Pinecone

# Redis for caching
import redis.asyncio as redis

# AI and embeddings
from sentence_transformers import CrossEncoder, SentenceTransformer, util
import openai
from openai import AsyncOpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

# Global variables
embedding_model = None
query_embedding_model = None
base_sentence_model = None
reranker = None
openai_client = None
redis_client = None
pinecone_index = None

# Configuration
SESSION_TTL = int(os.getenv("SESSION_TTL", 3600))  # 1 hour
PERSISTENT_CHROMA_DIR = os.getenv("PERSISTENT_CHROMA_DIR", "/tmp/persistent_chroma")
MAX_CACHE_SIZE = int(os.getenv("MAX_CACHE_SIZE", 1000))
EMBEDDING_CACHE_SIZE = int(os.getenv("EMBEDDING_CACHE_SIZE", 10000))
LOG_VERBOSE = os.getenv("LOG_VERBOSE", "true").lower() == "true"

# HARDCODED PINECONE API KEY
PINECONE_API_KEY = "pcsk_5SJNxg_B3sWxTJSuUBgYi6GDEuyHgNyt337K2Mts2SFY3udWPLdd2MiyETruf7iyV6SRhe"
PINECONE_ENVIRONMENT = "gcp-starter"
PINECONE_INDEX_NAME = "enhanced-rag-system"

# Enhanced caching system
ACTIVE_SESSIONS = {}
EMBEDDING_CACHE = LRUCache(maxsize=EMBEDDING_CACHE_SIZE)
RESPONSE_CACHE = TTLCache(maxsize=MAX_CACHE_SIZE, ttl=300)  # 5-minute cache

# Enhanced domain-adaptive configurations with insurance specialization
DEFAULT_DOMAIN_CONFIG = {
    "chunk_size": 1000,
    "chunk_overlap": 150,
    "semantic_search_k": 6,
    "context_docs": 6,
    "confidence_threshold": 0.7
}

DOMAIN_CONFIGS = {
    "technical": {"chunk_size": 1200, "chunk_overlap": 200, "semantic_search_k": 8, "context_docs": 9, "confidence_threshold": 0.75},
    "legal": {"chunk_size": 1200, "chunk_overlap": 250, "semantic_search_k": 7, "context_docs": 8, "confidence_threshold": 0.72},
    "medical": {"chunk_size": 1100, "chunk_overlap": 220, "semantic_search_k": 7, "context_docs": 8, "confidence_threshold": 0.70},
    "financial": {"chunk_size": 1000, "chunk_overlap": 200, "semantic_search_k": 7, "context_docs": 7, "confidence_threshold": 0.68},
    "insurance": {"chunk_size": 1200, "chunk_overlap": 200, "semantic_search_k": 8, "context_docs": 10, "confidence_threshold": 0.65},
    "general": DEFAULT_DOMAIN_CONFIG
}

# Insurance-specific keywords for enhanced processing
INSURANCE_KEYWORDS = [
    'policy', 'premium', 'claim', 'coverage', 'benefit', 'exclusion', 'waiting period',
    'pre-existing condition', 'maternity', 'critical illness', 'hospitalization',
    'cashless', 'network provider', 'sum insured', 'policyholder', 'deductible',
    'co-payment', 'room rent', 'sub-limit', 'renewal', 'grace period', 'nominee'
]

# ================================
# REDIS CACHE IMPLEMENTATION
# ================================

class RedisCache:
    """Production-ready Redis cache for response caching"""
    
    def __init__(self):
        self.redis = None
    
    async def initialize(self):
        """Initialize Redis connection with connection pooling"""
        try:
            self.redis = redis.Redis(
                host="localhost",
                port=6379,
                db=0,
                connection_pool_class_kwargs={"max_connections": 20},
                decode_responses=True
            )
            # Test connection
            await self.redis.ping()
            logger.info("✅ Redis cache initialized successfully")
        except Exception as e:
            logger.warning(f"⚠️ Redis not available, falling back to memory cache: {e}")
            self.redis = None
    
    async def get_cached_response(self, query_hash: str) -> Optional[Dict]:
        """Get cached response from Redis"""
        if not self.redis:
            return None
        try:
            cached = await self.redis.get(f"response:{query_hash}")
            return json.loads(cached) if cached else None
        except Exception as e:
            logger.warning(f"⚠️ Redis get error: {e}")
            return None
    
    async def cache_response(self, query_hash: str, response: dict, ttl: int = 300):
        """Cache response in Redis"""
        if not self.redis:
            return
        try:
            await self.redis.setex(f"response:{query_hash}", ttl, json.dumps(response, default=str))
        except Exception as e:
            logger.warning(f"⚠️ Redis cache error: {e}")
    
    async def clear_cache(self):
        """Clear all cached responses"""
        if not self.redis:
            return
        try:
            await self.redis.flushdb()
            logger.info("🗑️ Redis cache cleared")
        except Exception as e:
            logger.warning(f"⚠️ Redis clear error: {e}")

# ================================
# OPTIMIZED EMBEDDING SERVICE
# ================================

class OptimizedEmbeddingService:
    """Optimized embedding service with batch processing and caching"""
    
    def __init__(self):
        self.embedding_cache = {}
        self.batch_queue = []
        self.processing_batch = False
    
    async def get_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Process embeddings in optimized batches"""
        # Check cache first
        cached_embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            text_hash = hashlib.md5(text.encode()).hexdigest()
            if text_hash in self.embedding_cache:
                cached_embeddings.append((i, self.embedding_cache[text_hash]))
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Batch process only uncached
        new_embeddings = []
        if uncached_texts and base_sentence_model:
            new_embeddings = await asyncio.to_thread(
                base_sentence_model.encode,
                uncached_texts,
                batch_size=128,  # Larger batch size
                show_progress_bar=False,
                convert_to_numpy=True
            )
        
        # Cache new embeddings
        for text, embedding in zip(uncached_texts, new_embeddings):
            text_hash = hashlib.md5(text.encode()).hexdigest()
            self.embedding_cache[text_hash] = embedding
        
        # Reconstruct full embedding list in original order
        result_embeddings = [None] * len(texts)
        # Place cached embeddings
        for i, embedding in cached_embeddings:
            result_embeddings[i] = embedding
        # Place new embeddings
        for i, embedding in zip(uncached_indices, new_embeddings):
            result_embeddings[i] = embedding
        
        return result_embeddings
    
    async def get_query_embedding(self, query: str) -> np.ndarray:
        """Get single query embedding with caching"""
        if not base_sentence_model:
            return np.zeros(384)  # Return zero vector if model not loaded
            
        query_hash = hashlib.md5(query.encode()).hexdigest()
        
        if query_hash in self.embedding_cache:
            return self.embedding_cache[query_hash]
        
        embedding = await asyncio.to_thread(
            base_sentence_model.encode,
            query,
            convert_to_numpy=True
        )
        
        self.embedding_cache[query_hash] = embedding
        return embedding

# ================================
# OPTIMIZED OPENAI CLIENT
# ================================

class OptimizedOpenAIClient:
    """OpenAI client with advanced caching and connection pooling"""
    
    def __init__(self):
        self.client = None
        self.prompt_cache = TTLCache(maxsize=1000, ttl=600)  # 10-minute cache
        self.rate_limit_delay = 1.0
        self.connection_pool = None
    
    async def initialize(self, api_key: str):
        """Initialize the OpenAI client with connection pooling"""
        try:
            # Connection pooling for better performance
            self.client = AsyncOpenAI(
                api_key=api_key,
                http_client=httpx.AsyncClient(
                    limits=httpx.Limits(max_connections=20, max_keepalive_connections=5),
                    timeout=httpx.Timeout(30.0)
                )
            )
            logger.info("✅ OpenAI client initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize OpenAI client: {e}")
            raise
    
    def _get_prompt_hash(self, messages: List[Dict], **kwargs) -> str:
        """Generate hash for prompt caching"""
        prompt_data = {
            "messages": messages,
            "model": kwargs.get("model", "gpt-4o"),
            "temperature": kwargs.get("temperature", 0.1),
            "max_tokens": kwargs.get("max_tokens", 1000)
        }
        return hashlib.md5(json.dumps(prompt_data, sort_keys=True).encode()).hexdigest()
    
    async def optimized_completion(self, messages: List[Dict], **kwargs) -> str:
        """Optimized completion with caching and retry logic"""
        if not self.client:
            raise ValueError("OpenAI client not initialized")
            
        # Check cache first
        prompt_hash = self._get_prompt_hash(messages, **kwargs)
        if prompt_hash in self.prompt_cache:
            if LOG_VERBOSE:
                logger.info("🔄 Using cached OpenAI response")
            return self.prompt_cache[prompt_hash]
        
        # Check Redis cache
        redis_cached = await REDIS_CACHE.get_cached_response(prompt_hash)
        if redis_cached:
            result = redis_cached.get('content', '')
            self.prompt_cache[prompt_hash] = result
            return result
        
        # Make API call with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = await self.client.chat.completions.create(
                    messages=messages,
                    temperature=kwargs.get("temperature", 0.1),
                    max_tokens=kwargs.get("max_tokens", 1000),
                    model=kwargs.get("model", "gpt-4o")
                )
                
                result = response.choices[0].message.content
                
                # Cache successful response
                self.prompt_cache[prompt_hash] = result
                await REDIS_CACHE.cache_response(prompt_hash, {"content": result})
                
                return result
                
            except openai.RateLimitError as e:
                if attempt < max_retries - 1:
                    delay = self.rate_limit_delay * (2 ** attempt)
                    logger.warning(f"⏰ Rate limit hit, retrying in {delay}s...")
                    await asyncio.sleep(delay)
                    continue
                raise
                
            except Exception as e:
                logger.error(f"❌ OpenAI API error (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                    continue
                raise
        
        raise Exception("Max retries exceeded for OpenAI API call")

# ================================
# STARTUP AND SHUTDOWN - OPTIMIZED FOR CLOUD RUN
# ================================

async def initialize_components():
    """Lightweight initialization for Cloud Run - ESSENTIAL COMPONENTS ONLY"""
    global embedding_model, query_embedding_model, base_sentence_model, reranker, openai_client, pinecone_index
    
    try:
        logger.info("🚀 Quick initialization for Cloud Run...")
        
        # Essential components only - OpenAI client first
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            logger.warning("⚠️ OPENAI_API_KEY not found, some features will be limited")
        else:
            openai_client = OptimizedOpenAIClient()
            await openai_client.initialize(openai_api_key)
        
        # Initialize Pinecone connection (don't wait for index creation)
        try:
            pinecone.init(
                api_key=PINECONE_API_KEY,
                environment=PINECONE_ENVIRONMENT
            )
            logger.info("✅ Pinecone connection initialized")
        except Exception as e:
            logger.warning(f"⚠️ Pinecone initialization failed: {e}")
        
        # Set placeholders to prevent None errors - THIS IS CRITICAL
        embedding_model = "initializing"
        base_sentence_model = None
        reranker = None
        pinecone_index = None
        
        # Start background loading of heavy components
        asyncio.create_task(load_heavy_components())
        
        logger.info("✅ Quick initialization complete - Background loading started")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize: {e}")
        # Don't raise - allow partial functionality for Cloud Run

async def load_heavy_components():
    """Load heavy components in background - RUNS AFTER STARTUP"""
    global embedding_model, base_sentence_model, reranker, pinecone_index
    
    try:
        logger.info("🔄 Loading heavy components in background...")
        
        # Load models in background with error handling
        try:
            logger.info("📊 Loading sentence transformer...")
            base_sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ Sentence transformer loaded")
        except Exception as e:
            logger.error(f"❌ Failed to load sentence transformer: {e}")
        
        try:
            logger.info("📊 Loading embedding model...")
            embedding_model = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            logger.info("✅ Embedding model loaded")
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
        
        try:
            logger.info("📊 Loading reranker...")
            reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-2-v1')
            logger.info("✅ Reranker loaded")
        except Exception as e:
            logger.error(f"❌ Failed to load reranker: {e}")
        
        # Setup Pinecone index
        try:
            if pinecone:
                index_name = PINECONE_INDEX_NAME
                if index_name not in pinecone.list_indexes():
                    logger.info(f"📊 Creating Pinecone index: {index_name}")
                    pinecone.create_index(
                        name=index_name, 
                        dimension=384, 
                        metric="cosine"
                    )
                pinecone_index = pinecone.Index(index_name)
                logger.info("✅ Pinecone index setup complete")
        except Exception as e:
            logger.error(f"❌ Failed to setup Pinecone index: {e}")
        
        # Initialize other components
        try:
            if base_sentence_model:
                DOMAIN_DETECTOR.initialize_embeddings()
                logger.info("✅ Domain detector initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize domain detector: {e}")
        
        try:
            await REDIS_CACHE.initialize()
        except Exception as e:
            logger.error(f"❌ Failed to initialize Redis: {e}")
        
        logger.info("✅ Heavy components loading completed")
        
    except Exception as e:
        logger.error(f"❌ Background loading error: {e}")

# ================================
# ENHANCED TOKEN OPTIMIZATION
# ================================

class TokenOptimizedProcessor:
    """Smart token optimization for better cost efficiency"""
    
    def __init__(self):
        self.max_tokens = {
            'gpt-4o': 128000,
            'context_reserve': 4000,
            'prompt_overhead': 1000
        }
    
    @lru_cache(maxsize=1000)
    def estimate_tokens(self, text: str) -> int:
        """Better token estimation for various content types"""
        if not text:
            return 0
        
        # Account for insurance jargon and technical terms
        words = text.split()
        # Better estimate for insurance docs (more complex vocabulary)
        avg_chars_per_token = 3.8
        return max(1, int(len(text) / avg_chars_per_token))
    
    def calculate_relevance_score(self, doc: Document, query: str) -> float:
        """Calculate document relevance using cached embeddings"""
        try:
            if not base_sentence_model:
                # Fallback to keyword matching if model not loaded
                query_terms = set(query.lower().split())
                doc_terms = set(doc.page_content.lower().split())
                return len(query_terms.intersection(doc_terms)) / max(len(query_terms), 1)
                
            doc_embedding = self._get_cached_embedding(doc.page_content[:512])
            query_embedding = self._get_cached_embedding(query)
            return float(util.cos_sim(doc_embedding, query_embedding)[0][0])
        except Exception as e:
            logger.warning(f"⚠️ Error calculating relevance: {e}")
            # Fallback to keyword matching
            query_terms = set(query.lower().split())
            doc_terms = set(doc.page_content.lower().split())
            return len(query_terms.intersection(doc_terms)) / max(len(query_terms), 1)
    
    @lru_cache(maxsize=5000)
    def _get_cached_embedding(self, text: str) -> np.ndarray:
        """Get cached embedding for text"""
        cache_key = hashlib.md5(text.encode()).hexdigest()
        if cache_key in EMBEDDING_CACHE:
            return EMBEDDING_CACHE[cache_key]
        
        global base_sentence_model
        if base_sentence_model is None:
            # Return zero vector if model not loaded
            return np.zeros(384)
        
        embedding = base_sentence_model.encode(text, convert_to_tensor=False)
        EMBEDDING_CACHE[cache_key] = embedding
        return embedding
    
    def optimize_context_intelligently(self, documents: List[Document], query: str, max_tokens: int = 3000) -> str:
        """Enhanced context optimization with insurance-specific prioritization"""
        if not documents:
            return ""
        
        # Calculate relevance scores and token counts
        doc_scores = []
        for doc in documents:
            relevance = self.calculate_relevance_score(doc, query)
            tokens = self.estimate_tokens(doc.page_content)
            
            # Boost for insurance content
            insurance_boost = 1.0
            if any(keyword in doc.page_content.lower() for keyword in INSURANCE_KEYWORDS):
                insurance_boost = 1.2
            
            efficiency = (relevance * insurance_boost) / max(tokens, 1)
            doc_scores.append((doc, relevance, tokens, efficiency))
        
        # Sort by efficiency (relevance per token)
        doc_scores.sort(key=lambda x: x[3], reverse=True)
        
        # Build context within budget
        context_parts = []
        token_budget = max_tokens
        
        for doc, relevance, tokens, efficiency in doc_scores:
            if tokens <= token_budget:
                context_parts.append(doc.page_content)
                token_budget -= tokens
            elif token_budget > 200:  # Partial inclusion
                partial_content = self._truncate_intelligently(doc.page_content, token_budget)
                context_parts.append(partial_content)
                break
        
        return "\n\n".join(context_parts)
    
    def _truncate_intelligently(self, content: str, max_tokens: int) -> str:
        """Intelligently truncate content preserving important parts"""
        max_chars = max_tokens * 4
        if len(content) <= max_chars:
            return content
        
        # Keep first and last parts, remove middle
        keep_chars = max_chars - 100  # Reserve for truncation message
        first_half = content[:keep_chars//2]
        second_half = content[-keep_chars//2:]
        
        return f"{first_half}\n\n[... content truncated for optimization ...]\n\n{second_half}"

# ================================
# ENHANCED DOMAIN DETECTION
# ================================

class SemanticDomainDetector:
    """Domain detection using enhanced semantic similarity"""
    
    def __init__(self):
        self.domain_embeddings = {}
        
        # Enhanced domain descriptions with more specific insurance terms
        self.domain_descriptions = {
            "technical": "technical documentation engineering software development programming code architecture system design specifications API database network infrastructure",
            "legal": "legal law regulation statute constitution court judicial legislation clause article provision contract agreement litigation compliance regulatory framework",
            "medical": "medical healthcare patient diagnosis treatment clinical therapy medicine hospital physician doctor surgery pharmaceutical health insurance medical coverage",
            "financial": "financial banking investment policy economics business finance accounting audit tax revenue profit loss balance sheet financial planning investment portfolio",
            "insurance": "insurance policy premium claim coverage deductible benefit exclusion waiting period pre-existing condition maternity critical illness hospitalization cashless network provider sum insured policyholder co-payment room rent sub-limit renewal grace period nominee life insurance health insurance motor insurance travel insurance",
            "academic": "academic research study analysis methodology scholarly scientific paper thesis journal publication university education learning curriculum",
            "general": "general information document content text data knowledge base manual guide instructions reference material"
        }
    
    def initialize_embeddings(self):
        """Pre-compute domain embeddings"""
        global base_sentence_model
        if base_sentence_model is None:
            logger.warning("⚠️ Base sentence model not loaded, skipping domain embeddings")
            return
        
        try:
            for domain, description in self.domain_descriptions.items():
                self.domain_embeddings[domain] = base_sentence_model.encode(description, convert_to_tensor=False)
            
            logger.info(f"✅ Initialized embeddings for {len(self.domain_embeddings)} domains")
        except Exception as e:
            logger.error(f"❌ Failed to initialize domain embeddings: {e}")
    
    def detect_domain(self, documents: List[Document], confidence_threshold: float = 0.4) -> Tuple[str, float]:
        """Enhanced domain detection with better content analysis"""
        if not documents:
            return "general", 0.5
            
        if not self.domain_embeddings or not base_sentence_model:
            # Fallback to keyword-based detection
            return self._fallback_domain_detection(documents)
        
        try:
            # Use more content and focus on key sections
            combined_content = []
            for doc in documents[:10]:  # More documents for better context
                content = doc.page_content
                
                # Prioritize content with domain-specific keywords
                if any(term in content.lower() for term in INSURANCE_KEYWORDS):
                    combined_content.append(content[:1000])  # Longer segments for key content
                else:
                    combined_content.append(content[:500])
            
            combined_text = ' '.join(combined_content)[:5000]  # More context
            
            # Get content embedding
            content_embedding = base_sentence_model.encode(combined_text, convert_to_tensor=False)
            
            # Calculate similarities to each domain
            domain_scores = {}
            for domain, domain_embedding in self.domain_embeddings.items():
                similarity = float(util.cos_sim(content_embedding, domain_embedding)[0][0])
                domain_scores[domain] = similarity
            
            # Get best match
            best_domain = max(domain_scores, key=domain_scores.get)
            best_score = domain_scores[best_domain]
            
            # Lower threshold for better domain detection
            if best_score < confidence_threshold:
                # Check for insurance-specific content as fallback
                if any(keyword in combined_text.lower() for keyword in INSURANCE_KEYWORDS[:5]):
                    best_domain = "insurance"
                    best_score = 0.6
                else:
                    best_domain = "general"
                    best_score = confidence_threshold
            
            if LOG_VERBOSE:
                logger.info(f"🔍 Domain detected: {best_domain} (confidence: {best_score:.2f})")
            
            return best_domain, best_score
            
        except Exception as e:
            logger.warning(f"⚠️ Error in domain detection: {e}")
            return self._fallback_domain_detection(documents)
    
    def _fallback_domain_detection(self, documents: List[Document]) -> Tuple[str, float]:
        """Fallback domain detection using keywords"""
        combined_text = ' '.join([doc.page_content for doc in documents[:5]])[:2000]
        
        # Simple keyword-based detection
        if any(keyword in combined_text.lower() for keyword in INSURANCE_KEYWORDS):
            return "insurance", 0.6
        elif any(term in combined_text.lower() for term in ['medical', 'healthcare', 'patient']):
            return "medical", 0.6
        elif any(term in combined_text.lower() for term in ['legal', 'law', 'contract']):
            return "legal", 0.6
        elif any(term in combined_text.lower() for term in ['financial', 'banking', 'investment']):
            return "financial", 0.6
        else:
            return "general", 0.5

# Global domain detector
DOMAIN_DETECTOR = SemanticDomainDetector()

# ================================
# SESSION OBJECT AND RAG SYSTEM
# ================================

T = TypeVar('T')

class SessionObject(Generic[T]):
    """Generic session object with TTL management"""
    
    def __init__(self, session_id: str, data: T, ttl: int = SESSION_TTL):
        self.session_id = session_id
        self.data = data
        self.created_at = time.time()
        self.last_accessed = time.time()
        self.ttl = ttl
    
    def is_expired(self) -> bool:
        """Check if session is expired"""
        return (time.time() - self.last_accessed) > self.ttl
    
    def touch(self):
        """Update last accessed time"""
        self.last_accessed = time.time()
    
    def get_data(self) -> T:
        """Get data and update access time"""
        self.touch()
        return self.data

class EnhancedRAGSystem:
    """Enhanced RAG system with Pinecone and optimizations"""
    
    def __init__(self, session_id: str = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.documents = []
        self.vector_store = None
        self.bm25_retriever = None
        self.domain = "general"
        self.domain_config = DEFAULT_DOMAIN_CONFIG.copy()
        self.document_hash = None
        self.processed_files = []
        self.token_processor = TokenOptimizedProcessor()
        self._processing_lock = asyncio.Lock()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()
    
    async def cleanup(self):
        """Optimized cleanup with proper resource management"""
        try:
            if self.vector_store:
                try:
                    # Pinecone doesn't need explicit cleanup like Chroma
                    pass
                except Exception as e:
                    logger.warning(f"⚠️ Error cleaning vector store: {e}")
                finally:
                    self.vector_store = None
            
            # Clear references
            self.documents.clear()
            self.processed_files.clear()
            
            if LOG_VERBOSE:
                logger.info(f"🧹 Session {self.session_id} cleaned up")
                
        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}")
    
    def calculate_document_hash(self, documents: List[Document]) -> str:
        """Calculate unique hash for documents"""
        content_sample = "".join([doc.page_content[:200] for doc in documents[:10]])
        return hashlib.sha256(content_sample.encode()).hexdigest()[:16]
    
    async def process_documents(self, file_paths: List[str]) -> Dict[str, Any]:
        """Process documents with enhanced domain detection and chunking"""
        async with self._processing_lock:
            try:
                # Load documents
                raw_documents = []
                for file_path in file_paths:
                    try:
                        file_extension = os.path.splitext(file_path)[1].lower()
                        if file_extension == '.pdf':
                            loader = PyMuPDFLoader(file_path)
                        elif file_extension == '.docx':
                            loader = Docx2txtLoader(file_path)
                        elif file_extension == '.txt':
                            loader = TextLoader(file_path, encoding='utf-8')
                        else:
                            continue
                        
                        docs = await asyncio.to_thread(loader.load)
                        raw_documents.extend(docs)
                    except Exception as e:
                        logger.warning(f"⚠️ Error loading {file_path}: {e}")
                        continue
                
                if not raw_documents:
                    raise HTTPException(status_code=400, detail="No documents could be loaded")
                
                # Enhanced domain detection
                self.domain, domain_confidence = DOMAIN_DETECTOR.detect_domain(raw_documents)
                self.domain_config = DOMAIN_CONFIGS.get(self.domain, DEFAULT_DOMAIN_CONFIG).copy()
                
                # Apply insurance-specific optimizations
                if self.domain == "insurance":
                    logger.info("🏥 Applying insurance-specific optimizations")
                    self.domain_config["confidence_threshold"] = 0.6
                    self.domain_config["context_docs"] = 12
                    self.domain_config["semantic_search_k"] = 10
                
                # Update document metadata with domain
                for doc in raw_documents:
                    doc.metadata.update({
                        'detected_domain': self.domain,
                        'domain_confidence': domain_confidence,
                        'session_id': self.session_id
                    })
                
                # Domain-adaptive chunking
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.domain_config["chunk_size"],
                    chunk_overlap=self.domain_config["chunk_overlap"],
                    length_function=len,
                    separators=["\n\n", "\n", ". ", " ", ""]
                )
                
                self.documents = await asyncio.to_thread(text_splitter.split_documents, raw_documents)
                self.documents = [doc for doc in self.documents if len(doc.page_content.strip()) > 50]
                
                self.document_hash = self.calculate_document_hash(self.documents)
                self.processed_files = [os.path.basename(fp) for fp in file_paths]
                
                # Setup retrievers with Pinecone (only if components loaded)
                if base_sentence_model and embedding_model:
                    await self._setup_retrievers()
                else:
                    logger.info("⚠️ Components not fully loaded, skipping retriever setup")
                
                result = {
                    'session_id': self.session_id,
                    'document_hash': self.document_hash,
                    'domain': self.domain,
                    'domain_confidence': domain_confidence,
                    'total_chunks': len(self.documents),
                    'processed_files': self.processed_files,
                    'chunk_size': self.domain_config["chunk_size"],
                    'chunk_overlap': self.domain_config["chunk_overlap"],
                    'insurance_optimizations': self.domain == "insurance"
                }
                
                if LOG_VERBOSE:
                    logger.info(f"✅ Processed documents: {result}")
                
                return result
                
            except Exception as e:
                logger.error(f"❌ Document processing error: {e}")
                raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")
    
    async def _setup_retrievers(self):
        """Setup Pinecone vector store and BM25 retriever"""
        try:
            global embedding_model, pinecone_index
            
            # Check if components are loaded
            if not base_sentence_model or not embedding_model:
                logger.warning("⚠️ Components not fully loaded, skipping retriever setup")
                return
            
            # Use namespace for document isolation
            namespace = f"{self.domain}_{self.document_hash}"
            
            if LOG_VERBOSE:
                logger.info(f"🔧 Setting up retrievers with namespace: {namespace}")
            
            # Create vector store with Pinecone if available
            if pinecone_index and embedding_model != "initializing":
                self.vector_store = Pinecone(
                    index=pinecone_index,
                    embedding=embedding_model,
                    text_key="text",
                    namespace=namespace
                )
                
                # Add documents to Pinecone if not already there
                try:
                    stats = pinecone_index.describe_index_stats()
                    if namespace not in stats.get('namespaces', {}) or stats['namespaces'].get(namespace, {}).get('vector_count', 0) == 0:
                        if LOG_VERBOSE:
                            logger.info(f"📊 Adding {len(self.documents)} documents to Pinecone")
                        
                        # Batch processing for large document sets
                        if len(self.documents) > 50:
                            await self._batch_add_to_pinecone()
                        else:
                            await asyncio.to_thread(
                                self.vector_store.add_documents,
                                self.documents
                            )
                    else:
                        if LOG_VERBOSE:
                            logger.info("📂 Documents already exist in Pinecone, reusing")
                except Exception as e:
                    logger.warning(f"⚠️ Error adding documents to Pinecone: {e}")
            
            # Setup BM25 retriever
            try:
                self.bm25_retriever = await asyncio.to_thread(BM25Retriever.from_documents, self.documents)
                self.bm25_retriever.k = self.domain_config["semantic_search_k"] + 3
                if LOG_VERBOSE:
                    logger.info("✅ BM25 retriever setup complete")
            except Exception as e:
                logger.warning(f"⚠️ Error setting up BM25 retriever: {e}")
                
        except Exception as e:
            logger.error(f"❌ Retriever setup error: {e}")
    
    async def _batch_add_to_pinecone(self):
        """Batch add documents to Pinecone for better performance"""
        batch_size = 32
        total_docs = len(self.documents)
        
        if LOG_VERBOSE:
            logger.info(f"🔄 Batch adding {total_docs} documents to Pinecone in batches of {batch_size}")
        
        for i in range(0, total_docs, batch_size):
            batch = self.documents[i:i + batch_size]
            try:
                await asyncio.to_thread(
                    self.vector_store.add_documents,
                    batch
                )
                if LOG_VERBOSE:
                    logger.info(f"📊 Added batch {i//batch_size + 1}/{(total_docs-1)//batch_size + 1}")
            except Exception as e:
                logger.warning(f"⚠️ Error adding batch {i//batch_size + 1}: {e}")
        
        if LOG_VERBOSE:
            logger.info("✅ Batch addition to Pinecone completed")
    
    async def retrieve_and_rerank_optimized(self, query: str, top_k: int = None) -> Tuple[List[Document], List[float]]:
        """Ultra-fast retrieval with parallel processing"""
        if top_k is None:
            top_k = self.domain_config["context_docs"]
        
        try:
            # Check if components are loaded
            if not base_sentence_model:
                logger.warning("⚠️ Components not loaded, returning basic results")
                # Return first few documents as fallback
                return self.documents[:top_k], [0.5] * min(len(self.documents), top_k)
            
            # Create embedding task immediately
            embedding_task = asyncio.create_task(
                EMBEDDING_SERVICE.get_query_embedding(query)
            )
            
            # Parallel retrieval tasks
            tasks = []
            
            # Vector search task
            if self.vector_store:
                vector_task = asyncio.create_task(
                    self._vector_search_optimized(query, top_k * 2)
                )
                tasks.append(vector_task)
            
            # BM25 search task
            if self.bm25_retriever:
                bm25_task = asyncio.create_task(
                    asyncio.to_thread(self.bm25_retriever.get_relevant_documents, query)
                )
                tasks.append(bm25_task)
            
            # Wait for embedding and searches simultaneously
            embedding = await embedding_task
            search_results = await asyncio.gather(*tasks, return_exceptions=True) if tasks else []
            
            # Fast merge and rerank
            return await self._fast_merge_and_rerank(query, search_results, embedding, top_k)
            
        except Exception as e:
            logger.error(f"❌ Optimized retrieval error: {e}")
            # Fallback to basic document retrieval
            return self.documents[:top_k], [0.5] * min(len(self.documents), top_k)
    
    async def _vector_search_optimized(self, query: str, k: int) -> List[Tuple[Document, float]]:
        """Optimized vector search with Pinecone"""
        try:
            if not self.vector_store:
                return []
                
            results = await asyncio.to_thread(
                self.vector_store.similarity_search_with_score,
                query,
                k=min(k, 20)
            )
            return results
        except Exception as e:
            logger.error(f"❌ Vector search error: {e}")
            return []
    
    async def _fast_merge_and_rerank(self, query: str, search_results: List, embedding: np.ndarray, top_k: int) -> Tuple[List[Document], List[float]]:
        """Fast merge and rerank results"""
        all_docs = []
        all_scores = []
        
        # Process vector search results
        if search_results and not isinstance(search_results[0], Exception):
            vector_results = search_results[0]
            for doc, distance_score in vector_results:
                all_docs.append(doc)
                # Better score normalization for Pinecone
                similarity_score = max(0.0, min(1.0, (2.0 - distance_score) / 2.0))
                all_scores.append(similarity_score)
        
        # Process BM25 results
        if len(search_results) > 1 and not isinstance(search_results[1], Exception):
            bm25_docs = search_results[1][:top_k]
            for doc in bm25_docs:
                if doc not in all_docs:  # Avoid duplicates
                    all_docs.append(doc)
                    all_scores.append(0.7)  # Default BM25 score
        
        # If no search results, use basic document retrieval
        if not all_docs:
            all_docs = self.documents[:top_k]
            all_scores = [0.5] * len(all_docs)
        
        # Enhanced reranking using cross-encoder (if available)
        if all_docs and len(all_docs) > 1 and reranker:
            try:
                reranked_docs, reranked_scores = await self._semantic_rerank(query, all_docs, all_scores)
                return reranked_docs[:top_k], reranked_scores[:top_k]
            except Exception as e:
                logger.warning(f"⚠️ Reranking failed: {e}")
        
        return all_docs[:top_k], all_scores[:top_k]
    
    async def _semantic_rerank(self, query: str, documents: List[Document], initial_scores: List[float]) -> Tuple[List[Document], List[float]]:
        """Enhanced semantic reranking with performance optimization"""
        try:
            global reranker
            if reranker is None:
                return documents, initial_scores
            
            # Skip reranking for small sets to improve performance
            if len(documents) <= 3:
                return documents, initial_scores
            
            # Prepare query-document pairs
            query_doc_pairs = [[query, doc.page_content[:512]] for doc in documents]
            
            # Batch process pairs for better performance
            rerank_scores = await asyncio.to_thread(reranker.predict, query_doc_pairs)
            
            # Combine with initial scores (weighted average)
            combined_scores = []
            for i, (initial_score, rerank_score) in enumerate(zip(initial_scores, rerank_scores)):
                # Normalize rerank score to 0-1 range
                normalized_rerank = float((rerank_score + 1) / 2)  # Convert from [-1,1] to [0,1]
                # Weighted combination: 60% rerank, 40% initial
                combined_score = 0.6 * normalized_rerank + 0.4 * initial_score
                combined_scores.append(combined_score)
            
            # Sort by combined score
            scored_docs = list(zip(documents, combined_scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            reranked_docs, reranked_scores = zip(*scored_docs) if scored_docs else ([], [])
            return list(reranked_docs), list(reranked_scores)
            
        except Exception as e:
            logger.warning(f"⚠️ Reranking error: {e}")
            return documents, initial_scores

# ================================
# DECISION ENGINE
# ================================

class UniversalDecisionEngine:
    """Universal decision engine with enhanced confidence calculation"""
    
    def __init__(self):
        self.token_processor = TokenOptimizedProcessor()
        self.confidence_cache = LRUCache(maxsize=1000)
    
    def calculate_confidence_score(self,
                                 query: str,
                                 similarity_scores: List[float],
                                 query_match_quality: float,
                                 domain_confidence: float = 1.0) -> float:
        """Enhanced confidence calculation with domain awareness"""
        cache_key = f"{hash(tuple(similarity_scores))}_{query_match_quality}_{domain_confidence}"
        if cache_key in self.confidence_cache:
            return self.confidence_cache[cache_key]
        
        if not similarity_scores:
            confidence = 0.0
        else:
            scores_array = np.array(similarity_scores)
            avg_similarity = np.mean(scores_array)
            max_similarity = np.max(scores_array)
            
            # Better score consistency calculation
            score_variance = np.var(scores_array) if len(scores_array) > 1 else 0.0
            score_consistency = max(0.0, 1.0 - (score_variance * 2))  # Penalize high variance
            
            # Improved weighting
            confidence = (
                0.40 * max_similarity +         # Highest weight to best match
                0.25 * avg_similarity +         # Average quality
                0.20 * min(1.0, query_match_quality) +  # Query relevance
                0.10 * score_consistency +      # Consistency bonus
                0.05 * domain_confidence        # Domain confidence
            )
            
            # Apply domain-specific boost for insurance queries
            if domain_confidence > 0.7 and any(term in query.lower() for term in ['policy', 'premium', 'claim', 'coverage']):
                confidence *= 1.1  # 10% boost
        
        confidence = min(1.0, max(0.0, confidence))
        self.confidence_cache[cache_key] = confidence
        return confidence
    
    def _assess_query_match_quality(self, query: str, retrieved_docs: List[Document]) -> float:
        """Enhanced query match quality calculation"""
        query_terms = set(query.lower().split())
        if not query_terms:
            return 0.5
        
        match_scores = []
        for doc in retrieved_docs[:5]:  # Check first 5 documents
            doc_terms = set(doc.page_content.lower().split())
            overlap = len(query_terms.intersection(doc_terms))
            match_score = overlap / len(query_terms)
            match_scores.append(match_score)
        
        return min(1.0, np.mean(match_scores)) if match_scores else 0.0
    
    async def process_query(self,
                          query: str,
                          retrieved_docs: List[Document],
                          similarity_scores: List[float],
                          domain: str,
                          domain_confidence: float = 1.0,
                          query_type: str = "general") -> Dict[str, Any]:
        """Universal query processing with enhanced confidence calculation"""
        if not retrieved_docs:
            return self._empty_response(query, domain)
        
        try:
            # Calculate confidence with enhanced method
            query_match_quality = self._assess_query_match_quality(query, retrieved_docs)
            confidence = self.calculate_confidence_score(
                query, similarity_scores, query_match_quality, domain_confidence
            )
            
            # Optimize context for token efficiency
            context = self.token_processor.optimize_context_intelligently(
                retrieved_docs, query, max_tokens=3000
            )
            
            # Generate response
            response = await self._generate_general_response(query, context, domain)
            
            # Prepare final result
            result = {
                "query": query,
                "answer": response,
                "confidence": confidence,
                "domain": domain,
                "domain_confidence": domain_confidence,
                "query_type": query_type,
                "reasoning_chain": [f"Analyzed {len(retrieved_docs)} documents with {confidence:.1%} confidence"],
                "source_documents": list(set([doc.metadata.get('source', 'Unknown') for doc in retrieved_docs])),
                "retrieved_chunks": len(retrieved_docs),
                "processing_time": time.time(),
                "insurance_optimized": domain == "insurance"
            }
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Query processing error: {e}")
            return self._error_response(query, domain, str(e))
    
    def _empty_response(self, query: str, domain: str) -> Dict[str, Any]:
        """Generate response when no documents are retrieved"""
        return {
            "query": query,
            "answer": "No relevant information found in the provided documents for this query.",
            "confidence": 0.0,
            "domain": domain,
            "query_type": "no_results",
            "reasoning_chain": ["No documents retrieved for analysis"],
            "source_documents": [],
            "retrieved_chunks": 0,
            "processing_time": time.time()
        }
    
    def _error_response(self, query: str, domain: str, error_msg: str) -> Dict[str, Any]:
        """Generate error response"""
        return {
            "query": query,
            "answer": f"Error processing query: {error_msg}",
            "confidence": 0.0,
            "domain": domain,
            "query_type": "error",
            "reasoning_chain": [f"Processing error: {error_msg}"],
            "source_documents": [],
            "retrieved_chunks": 0,
            "processing_time": time.time()
        }
    
    async def _generate_general_response(self, query: str, context: str, domain: str) -> str:
        """Generate general response using optimized LLM call"""
        # Enhanced prompt for insurance domain
        domain_context = ""
        if domain == "insurance":
            domain_context = """
When analyzing insurance documents, pay special attention to:
- Policy terms, conditions, and exclusions
- Coverage limits and waiting periods
- Premium calculations and payment terms
- Claim procedures and documentation requirements
- Pre-existing condition clauses
- Network providers and cashless facilities
"""
        
        prompt = f"""You are an expert document analyst specializing in {domain} content. Provide a clear, accurate answer based solely on the provided context.

{domain_context}

CONTEXT:
{context}

QUESTION: {query}

INSTRUCTIONS:
- Answer directly and comprehensively based only on the context provided
- If information is not in the context, clearly state this
- Be specific and cite relevant details when possible
- Maintain professional tone appropriate for {domain} domain
- If the context contains conflicting information, acknowledge this

ANSWER:"""
        
        try:
            global openai_client
            if not openai_client:
                return "OpenAI client not initialized. Please wait for system to fully load."
                
            response = await openai_client.optimized_completion(
                messages=[
                    {"role": "system", "content": f"You are an expert {domain} document analyst. Provide accurate, context-based responses."},
                    {"role": "user", "content": prompt}
                ],
                model="gpt-4o",
                max_tokens=1500,
                temperature=0.1
            )
            
            return response
            
        except Exception as e:
            logger.error(f"❌ LLM generation error: {e}")
            return f"Error generating response: {str(e)}. The system may still be initializing."

# ================================
# SESSION MANAGEMENT
# ================================

class EnhancedSessionManager:
    """Enhanced session manager with generic support"""
    
    @staticmethod
    async def get_or_create_session(document_hash: str) -> EnhancedRAGSystem:
        """Get existing session or create new one"""
        current_time = time.time()
        
        # Clean expired sessions efficiently
        expired_sessions = []
        for session_id, session_obj in list(ACTIVE_SESSIONS.items()):
            if session_obj.is_expired():
                expired_sessions.append(session_id)
        
        # Batch cleanup
        if expired_sessions:
            cleanup_tasks = []
            for session_id in expired_sessions:
                session_obj = ACTIVE_SESSIONS.pop(session_id)
                cleanup_tasks.append(session_obj.get_data().cleanup())
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            
            if LOG_VERBOSE:
                logger.info(f"🗑️ Cleaned {len(expired_sessions)} expired sessions")
        
        # Get or create session
        if document_hash in ACTIVE_SESSIONS:
            session_obj = ACTIVE_SESSIONS[document_hash]
            session = session_obj.get_data()  # This updates access time
            if LOG_VERBOSE:
                logger.info(f"♻️ Reusing session: {document_hash}")
            return session
        
        # Create new session
        rag_session = EnhancedRAGSystem(session_id=document_hash)
        session_obj = SessionObject(document_hash, rag_session)
        ACTIVE_SESSIONS[document_hash] = session_obj
        
        if LOG_VERBOSE:
            logger.info(f"🆕 Created new session: {document_hash}")
        
        return rag_session

# ================================
# URL DOWNLOADER
# ================================

class UniversalURLDownloader:
    """Universal URL downloader with support for various cloud storage"""
    
    def __init__(self, timeout: float = 60.0):
        self.timeout = timeout
    
    async def download_from_url(self, url: str) -> Tuple[bytes, str]:
        """Download file from URL with enhanced error handling"""
        try:
            download_url, filename = self._prepare_url_and_filename(url)
            
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                follow_redirects=True,
                headers={'User-Agent': 'Mozilla/5.0 (compatible; DocumentProcessor/2.0)'}
            ) as client:
                
                if LOG_VERBOSE:
                    logger.info(f"📥 Downloading from: {download_url}")
                
                response = await client.get(download_url)
                response.raise_for_status()
                
                if not response.content:
                    raise HTTPException(status_code=400, detail="Downloaded file is empty")
                
                if LOG_VERBOSE:
                    logger.info(f"✅ Downloaded {len(response.content)} bytes")
                
                return response.content, filename
                
        except httpx.RequestError as e:
            logger.error(f"❌ Network error downloading from {url}: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Network error: {str(e)}")
        except httpx.HTTPStatusError as e:
            logger.error(f"❌ HTTP {e.response.status_code} error: {str(e)}")
            raise HTTPException(status_code=400, detail=f"HTTP {e.response.status_code}: Download failed")
        except Exception as e:
            logger.error(f"❌ Unexpected download error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")
    
    def _prepare_url_and_filename(self, url: str) -> Tuple[str, str]:
        """Prepare download URL and extract filename"""
        filename = "document.pdf"
        
        try:
            parsed = urlparse(url)
            
            # Google Drive handling
            if 'drive.google.com' in url:
                if '/file/d/' in url:
                    file_id = url.split('/file/d/')[1].split('/')[0]
                    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
                    filename = f"gdrive_{file_id}.pdf"
                else:
                    download_url = url
            
            # Dropbox handling
            elif 'dropbox.com' in url:
                if '?dl=0' in url:
                    download_url = url.replace('?dl=0', '?dl=1')
                else:
                    download_url = url
                path_parts = parsed.path.split('/')
                if path_parts and path_parts[-1]:
                    filename = path_parts[-1]
            
            # Direct URLs
            else:
                download_url = url
                path_parts = parsed.path.split('/')
                if path_parts and path_parts[-1] and '.' in path_parts[-1]:
                    filename = path_parts[-1]
            
            return download_url, filename
            
        except Exception as e:
            logger.warning(f"⚠️ Error preparing URL: {e}")
            return url, filename

# ================================
# PYDANTIC MODELS
# ================================

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="The question to ask about the documents")
    query_type: Optional[str] = Field(default="general", description="Type of query: general, structured_analysis, etc.")
    domain_hint: Optional[str] = Field(default=None, description="Optional domain hint for better processing")

class DocumentResponse(BaseModel):
    query: str
    answer: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    domain: str
    domain_confidence: float = Field(..., ge=0.0, le=1.0)
    query_type: str
    reasoning_chain: List[str]
    source_documents: List[str]
    retrieved_chunks: int
    processing_time_ms: Optional[float] = None
    insurance_optimized: Optional[bool] = Field(default=False, description="Whether insurance optimizations were applied")

class HackRxRequest(BaseModel):
    documents: HttpUrl = Field(..., description="URL to the document(s)")
    questions: List[str] = Field(..., min_items=1, max_items=50, description="List of questions to ask")

class HackRxResponse(BaseModel):
    success: bool = True
    processing_time_seconds: Optional[float] = None
    timestamp: Optional[str] = None
    message: Optional[str] = None
    answers: List[DocumentResponse]
    session_info: Optional[Dict[str, Any]] = None

class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str
    performance_metrics: Dict[str, Any]
    components: Dict[str, str]

# ================================
# AUTHENTICATION
# ================================

security = HTTPBearer()

EXPECTED_BEARER_TOKEN = os.getenv("BEARER_TOKEN", "9a1163c13e8927960b857a674794a62c57baf588998981151b0753a4d6d17905")

def verify_bearer_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify the bearer token"""
    if credentials.credentials != EXPECTED_BEARER_TOKEN:
        logger.warning("❌ Invalid bearer token attempted")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

# ================================
# LIFESPAN FUNCTION - OPTIMIZED FOR CLOUD RUN
# ================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan with fast startup for Cloud Run"""
    logger.info("🚀 Starting Fast Initialization for Cloud Run...")
    
    # Fast initialization - only essential components
    await initialize_components()
    
    # Start cleanup task
    cleanup_task = asyncio.create_task(periodic_cleanup())
    
    try:
        yield
    finally:
        logger.info("🔄 Shutting down...")
        cleanup_task.cancel()
        if ACTIVE_SESSIONS:
            cleanup_tasks = []
            for session_obj in ACTIVE_SESSIONS.values():
                cleanup_tasks.append(session_obj.get_data().cleanup())
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        ACTIVE_SESSIONS.clear()
        logger.info("🧹 Shutdown complete")

async def periodic_cleanup():
    """Periodic cleanup with optimized performance"""
    cleanup_interval = 300  # 5 minutes
    while True:
        try:
            start_time = time.time()
            # Cleanup expired sessions
            expired_count = 0
            for session_id, session_obj in list(ACTIVE_SESSIONS.items()):
                if session_obj.is_expired():
                    try:
                        await session_obj.get_data().cleanup()
                        ACTIVE_SESSIONS.pop(session_id)
                        expired_count += 1
                    except Exception as e:
                        logger.warning(f"⚠️ Error cleaning session {session_id}: {e}")
            
            # Memory monitoring
            memory_info = psutil.Process().memory_info()
            cleanup_time = time.time() - start_time
            
            if expired_count > 0 or LOG_VERBOSE:
                logger.info(
                    f"🧹 Cleanup: {expired_count} sessions, "
                    f"Memory: {memory_info.rss / 1024 / 1024:.1f}MB, "
                    f"Active: {len(ACTIVE_SESSIONS)}, "
                    f"Time: {cleanup_time:.2f}s"
                )
        except Exception as e:
            logger.error(f"❌ Error in periodic cleanup: {e}")
        
        await asyncio.sleep(cleanup_interval)

# Global instances
EMBEDDING_SERVICE = OptimizedEmbeddingService()
REDIS_CACHE = RedisCache()

# ================================
# FASTAPI APPLICATION
# ================================

app = FastAPI(
    title="Enhanced Universal Document Processing API v5.0",
    description="Cloud Run Optimized RAG system with fast startup",
    version="5.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
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

@app.get("/", tags=["Health"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Enhanced Universal Document Processing API v5.0 - Cloud Run Optimized",
        "status": "production_ready",
        "version": "5.0.0",
        "timestamp": datetime.now().isoformat(),
        "cloud_run_optimizations": [
            "Fast startup with background model loading",
            "Graceful degradation during initialization",
            "Immediate health check endpoints",
            "Optimized resource management",
            "Session-based processing"
        ]
    }

@app.get("/health/startup", tags=["Health"])
async def startup_health():
    """Immediate health check for Cloud Run startup"""
    return {
        "status": "ok", 
        "timestamp": datetime.now().isoformat(),
        "components_loaded": {
            "openai_client": openai_client is not None,
            "base_sentence_model": base_sentence_model is not None,
            "embedding_model": embedding_model is not None and embedding_model != "initializing",
            "reranker": reranker is not None,
            "pinecone_index": pinecone_index is not None
        }
    }

@app.post("/upload", tags=["Document Processing"])
async def upload_documents(
    files: List[UploadFile] = File(...),
    query: str = Form(...),
    query_type: str = Form(default="general"),
    domain_hint: Optional[str] = Form(default=None),
    _: str = Depends(verify_bearer_token)
):
    """Enhanced document upload with production optimizations"""
    start_time = time.time()
    temp_files = []
    
    try:
        # Check if components are loaded
        if not base_sentence_model:
            return {
                "message": "System is still initializing. Please try again in 30 seconds.",
                "status": "initializing",
                "retry_after": 30,
                "timestamp": datetime.now().isoformat(),
                "components_status": {
                    "openai_client": openai_client is not None,
                    "base_sentence_model": base_sentence_model is not None,
                    "embedding_model": embedding_model is not None and embedding_model != "initializing",
                    "reranker": reranker is not None
                }
            }
        
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        
        if len(files) > 20:
            raise HTTPException(status_code=400, detail="Maximum 20 files allowed")
        
        # Save uploaded files with optimized handling
        file_paths = []
        for file in files:
            if not file.filename:
                continue
            
            file_ext = os.path.splitext(file.filename)[1].lower()
            if file_ext not in ['.pdf', '.docx', '.txt']:
                logger.warning(f"⚠️ Skipping unsupported file: {file.filename}")
                continue
            
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
            temp_files.append(temp_file.name)
            
            # Stream file content for memory efficiency
            content = await file.read()
            temp_file.write(content)
            temp_file.close()
            file_paths.append(temp_file.name)
        
        if not file_paths:
            raise HTTPException(status_code=400, detail="No supported files provided")
        
        # Calculate document hash for session management
        with open(file_paths[0], 'rb') as f:
            sample_content = f.read(1024)
        doc_hash = hashlib.md5(sample_content).hexdigest()[:12]
        
        # Get or create session with optimizations
        rag_system = await EnhancedSessionManager.get_or_create_session(doc_hash)
        
        # Process documents if not already processed
        if not rag_system.documents or rag_system.document_hash != doc_hash:
            processing_result = await rag_system.process_documents(file_paths)
            logger.info(f"📊 Processed {processing_result['total_chunks']} chunks in domain: {processing_result['domain']}")
        
        # Query processing with optimizations
        query_start = time.time()
        
        # Retrieve and rerank with optimizations
        retrieved_docs, similarity_scores = await rag_system.retrieve_and_rerank_optimized(query)
        
        # Process with decision engine
        decision_engine = UniversalDecisionEngine()
        result = await decision_engine.process_query(
            query=query,
            retrieved_docs=retrieved_docs,
            similarity_scores=similarity_scores,
            domain=rag_system.domain,
            domain_confidence=0.8,
            query_type=query_type
        )
        
        # Add timing information
        result["processing_time_ms"] = (time.time() - start_time) * 1000
        result["query_processing_time_ms"] = (time.time() - query_start) * 1000
        
        logger.info(f"✅ Query processed in {result['processing_time_ms']:.2f}ms")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Upload processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    finally:
        # Cleanup temporary files
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except Exception as e:
                logger.warning(f"⚠️ Failed to cleanup {temp_file}: {e}")

@app.post("/hackrx/run", response_model=HackRxResponse, tags=["Batch Processing"])
async def hackrx_batch_processing(
    request: HackRxRequest,
    _: str = Depends(verify_bearer_token)
):
    """Enhanced batch processing with production optimizations - THE MAIN HACKRX ENDPOINT"""
    start_time = time.time()
    temp_files = []
    
    try:
        logger.info(f"🚀 Starting HackRx batch processing for {len(request.questions)} questions")
        
        # Check if components are loaded
        if not base_sentence_model:
            return HackRxResponse(
                success=False,
                processing_time_seconds=time.time() - start_time,
                timestamp=datetime.now().isoformat(),
                message="System is still initializing. Please try again in 30 seconds.",
                answers=[],
                session_info={
                    "status": "initializing", 
                    "retry_after": 30,
                    "components_status": {
                        "openai_client": openai_client is not None,
                        "base_sentence_model": base_sentence_model is not None,
                        "embedding_model": embedding_model is not None and embedding_model != "initializing",
                        "reranker": reranker is not None
                    }
                }
            )
        
        # Download document with optimizations
        downloader = UniversalURLDownloader(timeout=90.0)
        file_content, filename = await downloader.download_from_url(str(request.documents))
        
        # Save to temporary file
        file_ext = os.path.splitext(filename)[1].lower() or '.pdf'
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
        temp_files.append(temp_file.name)
        temp_file.write(file_content)
        temp_file.close()
        
        # Session management with optimization
        doc_hash = hashlib.md5(file_content[:2048]).hexdigest()[:12]
        rag_system = await EnhancedSessionManager.get_or_create_session(doc_hash)
        
        # Process document if needed
        if not rag_system.documents or rag_system.document_hash != doc_hash:
            processing_result = await rag_system.process_documents([temp_file.name])
            logger.info(f"📊 HackRx processing: {processing_result['total_chunks']} chunks, domain: {processing_result['domain']}")
        
        # Batch process all questions with optimizations
        answers = []
        decision_engine = UniversalDecisionEngine()
        
        # Process questions in parallel batches for maximum performance
        batch_size = 5  # Process 5 questions concurrently
        for i in range(0, len(request.questions), batch_size):
            batch_questions = request.questions[i:i + batch_size]
            batch_tasks = []
            
            for question in batch_questions:
                task = asyncio.create_task(
                    process_single_question_optimized(
                        question, rag_system, decision_engine
                    )
                )
                batch_tasks.append(task)
            
            # Wait for batch completion
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Process results
            for question, result in zip(batch_questions, batch_results):
                if isinstance(result, Exception):
                    logger.error(f"❌ Error processing question '{question}': {result}")
                    # Create error response
                    error_response = DocumentResponse(
                        query=question,
                        answer=f"Error processing question: {str(result)}",
                        confidence=0.0,
                        domain=rag_system.domain,
                        domain_confidence=0.0,
                        query_type="error",
                        reasoning_chain=[f"Processing error: {str(result)}"],
                        source_documents=[],
                        retrieved_chunks=0,
                        insurance_optimized=rag_system.domain == "insurance"
                    )
                    answers.append(error_response)
                else:
                    answers.append(result)
            
            logger.info(f"📊 Completed HackRx batch {i//batch_size + 1}/{(len(request.questions)-1)//batch_size + 1}")
        
        # Prepare response
        total_time = time.time() - start_time
        session_info = {
            "session_id": rag_system.session_id,
            "domain": rag_system.domain,
            "total_chunks": len(rag_system.documents),
            "processing_optimizations": [
                "fast_startup_initialization",
                "background_model_loading",
                "graceful_degradation",
                "batch_processing",
                "session_management"
            ]
        }
        
        if rag_system.domain == "insurance":
            session_info["insurance_optimizations_applied"] = True
        
        response = HackRxResponse(
            success=True,
            processing_time_seconds=total_time,
            timestamp=datetime.now().isoformat(),
            message=f"Successfully processed {len(request.questions)} questions in {total_time:.2f}s with Cloud Run optimizations",
            answers=answers,
            session_info=session_info
        )
        
        logger.info(f"✅ HackRx batch processing completed: {len(answers)} answers in {total_time:.2f}s")
        return response
        
    except Exception as e:
        logger.error(f"❌ HackRx batch processing error: {e}")
        return HackRxResponse(
            success=False,
            processing_time_seconds=time.time() - start_time,
            timestamp=datetime.now().isoformat(),
            message=f"Batch processing failed: {str(e)}",
            answers=[],
            session_info={"error": str(e)}
        )
    finally:
        # Cleanup
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except Exception:
                pass

async def process_single_question_optimized(
    question: str,
    rag_system: EnhancedRAGSystem,
    decision_engine: UniversalDecisionEngine
) -> DocumentResponse:
    """Process single question with full optimization stack"""
    try:
        question_start = time.time()
        
        # Optimized retrieval and reranking
        retrieved_docs, similarity_scores = await rag_system.retrieve_and_rerank_optimized(question)
        
        # Decision engine processing
        result = await decision_engine.process_query(
            query=question,
            retrieved_docs=retrieved_docs,
            similarity_scores=similarity_scores,
            domain=rag_system.domain,
            domain_confidence=0.8,
            query_type="general"
        )
        
        # Add timing
        result["processing_time_ms"] = (time.time() - question_start) * 1000
        
        # Convert to response model
        response = DocumentResponse(
            query=result["query"],
            answer=result["answer"],
            confidence=result["confidence"],
            domain=result["domain"],
            domain_confidence=result.get("domain_confidence", 0.8),
            query_type=result["query_type"],
            reasoning_chain=result["reasoning_chain"],
            source_documents=result["source_documents"],
            retrieved_chunks=result["retrieved_chunks"],
            processing_time_ms=result.get("processing_time_ms"),
            insurance_optimized=result.get("insurance_optimized", False)
        )
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Single question processing error: {e}")
        return DocumentResponse(
            query=question,
            answer=f"Error processing question: {str(e)}",
            confidence=0.0,
            domain=rag_system.domain,
            domain_confidence=0.0,
            query_type="error",
            reasoning_chain=[f"Processing error: {str(e)}"],
            source_documents=[],
            retrieved_chunks=0,
            insurance_optimized=False
        )

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Comprehensive health check with production metrics"""
    try:
        memory_info = psutil.Process().memory_info()
        
        # Component status
        components = {
            "pinecone_index": "connected" if pinecone_index else "disconnected",
            "redis_cache": "connected" if REDIS_CACHE.redis else "disconnected", 
            "embedding_model": "loaded" if embedding_model and embedding_model != "initializing" else "loading",
            "base_sentence_model": "loaded" if base_sentence_model else "loading",
            "reranker": "loaded" if reranker else "loading",
            "openai_client": "ready" if openai_client else "loading",
            "domain_detector": "initialized" if DOMAIN_DETECTOR.domain_embeddings else "loading"
        }
        
        # Performance metrics
        performance_metrics = {
            "memory_usage_mb": round(memory_info.rss / 1024 / 1024, 2),
            "cpu_percent": psutil.cpu_percent(),
            "active_sessions": len(ACTIVE_SESSIONS),
            "embedding_cache_size": len(EMBEDDING_CACHE),
            "response_cache_size": len(RESPONSE_CACHE),
            "startup_optimized": True,
            "background_loading": True
        }
        
        return HealthResponse(
            status="healthy_cloud_run_optimized",
            version="5.0.0",
            timestamp=datetime.now().isoformat(),
            performance_metrics=performance_metrics,
            components=components
        )
        
    except Exception as e:
        logger.error(f"❌ Health check error: {e}")
        return HealthResponse(
            status="error",
            version="5.0.0",
            timestamp=datetime.now().isoformat(),
            performance_metrics={"error": str(e)},
            components={"status": "error"}
        )

# ================================
# DEVELOPMENT SERVER
# ================================

if __name__ == "__main__":
    import uvicorn
    
    # Cloud Run optimized configuration
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        workers=1,  # Single worker for shared memory optimization
        log_level="info",
        access_log=True,
        loop="asyncio"
    )
