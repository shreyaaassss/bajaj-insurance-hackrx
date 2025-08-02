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
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
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

# ENHANCED DOMAIN-ADAPTIVE CONFIGURATIONS
DEFAULT_DOMAIN_CONFIG = {
    "chunk_size": 1000,
    "chunk_overlap": 150,
    "semantic_search_k": 15,  # INCREASED from 6 to 15
    "context_docs": 12,       # INCREASED from 6 to 12
    "confidence_threshold": 0.6,
    "use_mmr": True,          # NEW: Enable MMR diversity
    "mmr_lambda": 0.7,        # NEW: MMR diversity parameter
    "use_metadata_filtering": True,  # NEW: Enable metadata filtering
    "rerank_top_k": 25        # NEW: Rerank more documents
}

DOMAIN_CONFIGS = {
    "technical": {
        "chunk_size": 1200, "chunk_overlap": 200, "semantic_search_k": 20, 
        "context_docs": 15, "confidence_threshold": 0.75, "use_mmr": True,
        "mmr_lambda": 0.6, "use_metadata_filtering": True, "rerank_top_k": 30
    },
    "legal": {
        "chunk_size": 1500, "chunk_overlap": 300, "semantic_search_k": 25,  # ENHANCED for legal docs
        "context_docs": 20, "confidence_threshold": 0.65, "use_mmr": True,
        "mmr_lambda": 0.8, "use_metadata_filtering": True, "rerank_top_k": 35
    },
    "medical": {
        "chunk_size": 1100, "chunk_overlap": 220, "semantic_search_k": 18, 
        "context_docs": 15, "confidence_threshold": 0.70, "use_mmr": True,
        "mmr_lambda": 0.7, "use_metadata_filtering": True, "rerank_top_k": 28
    },
    "financial": {
        "chunk_size": 1000, "chunk_overlap": 200, "semantic_search_k": 15, 
        "context_docs": 12, "confidence_threshold": 0.68, "use_mmr": True,
        "mmr_lambda": 0.7, "use_metadata_filtering": True, "rerank_top_k": 25
    },
    "insurance": {
        "chunk_size": 1200, "chunk_overlap": 200, "semantic_search_k": 22,  # ENHANCED for insurance
        "context_docs": 18, "confidence_threshold": 0.60, "use_mmr": True,
        "mmr_lambda": 0.75, "use_metadata_filtering": True, "rerank_top_k": 32
    },
    "academic": {
        "chunk_size": 1300, "chunk_overlap": 250, "semantic_search_k": 20,
        "context_docs": 15, "confidence_threshold": 0.70, "use_mmr": True,
        "mmr_lambda": 0.6, "use_metadata_filtering": True, "rerank_top_k": 30
    },
    "business": {
        "chunk_size": 1100, "chunk_overlap": 200, "semantic_search_k": 18,
        "context_docs": 14, "confidence_threshold": 0.68, "use_mmr": True,
        "mmr_lambda": 0.7, "use_metadata_filtering": True, "rerank_top_k": 28
    },
    "general": DEFAULT_DOMAIN_CONFIG
}

# Enhanced keywords for better domain detection
INSURANCE_KEYWORDS = [
    'policy', 'premium', 'claim', 'coverage', 'benefit', 'exclusion', 'waiting period',
    'pre-existing condition', 'maternity', 'critical illness', 'hospitalization',
    'cashless', 'network provider', 'sum insured', 'policyholder', 'deductible',
    'co-payment', 'room rent', 'sub-limit', 'renewal', 'grace period', 'nominee',
    'cataract', 'PED', 'clause'  # Added specific terms mentioned in your use case
]

LEGAL_KEYWORDS = [
    'clause', 'section', 'article', 'provision', 'terms', 'conditions', 'agreement',
    'contract', 'liability', 'jurisdiction', 'compliance', 'regulation', 'statute'
]

# ================================
# ENHANCED INTELLIGENT TEXT SPLITTER
# ================================

class IntelligentTextSplitter:
    """Enhanced text splitter with section-aware and semantic chunking"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, domain: str = "general"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.domain = domain
        
        # Enhanced separators based on document structure
        self.separators = [
            "\n\n### ",     # Section headers
            "\n\n## ",      # Sub-section headers
            "\n\n# ",       # Main headers
            "\n\nClause ",  # Legal clauses
            "\n\nSection ", # Sections
            "\n\nArticle ", # Articles
            "\n\n",         # Double newlines
            "\n",           # Single newlines
            ". ",           # Sentences
            " ",            # Words
            ""              # Characters
        ]
        
        # Section markers for metadata extraction
        self.section_patterns = [
            r'(?i)^(clause|section|article|chapter|part)\s+(\d+(?:\.\d+)*)',
            r'(?i)^(\d+(?:\.\d+)*)\s+(clause|section|article)',
            r'(?i)^([A-Z][^.]*:)$',  # Headings ending with colon
            r'(?i)^([IVX]+\.)\s+',   # Roman numerals
            r'(?i)^(\([a-z]\))\s+',  # (a) style numbering
        ]
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents with enhanced section awareness"""
        all_chunks = []
        
        for doc in documents:
            chunks = self._split_single_document(doc)
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def _split_single_document(self, document: Document) -> List[Document]:
        """Split single document with section-aware chunking"""
        text = document.page_content
        
        # First, try to identify sections
        sections = self._identify_sections(text)
        
        if len(sections) > 1:
            # Process section by section
            chunks = []
            for section in sections:
                section_chunks = self._chunk_section(section, document.metadata)
                chunks.extend(section_chunks)
            return chunks
        else:
            # Fallback to enhanced recursive splitting
            return self._enhanced_recursive_split(document)
    
    def _identify_sections(self, text: str) -> List[Dict[str, Any]]:
        """Identify document sections using patterns"""
        sections = []
        lines = text.split('\n')
        current_section = {"header": "", "content": "", "start_line": 0}
        
        for i, line in enumerate(lines):
            is_header = self._is_section_header(line.strip())
            
            if is_header and current_section["content"].strip():
                # Save previous section
                sections.append(current_section)
                current_section = {"header": line.strip(), "content": "", "start_line": i}
            elif is_header and not current_section["content"].strip():
                # Update header if no content yet
                current_section["header"] = line.strip()
                current_section["start_line"] = i
            else:
                current_section["content"] += line + "\n"
        
        # Add the last section
        if current_section["content"].strip():
            sections.append(current_section)
        
        # If no sections found, return the entire text as one section
        if not sections:
            sections = [{"header": "Document", "content": text, "start_line": 0}]
        
        return sections
    
    def _is_section_header(self, line: str) -> bool:
        """Check if a line is likely a section header"""
        if not line.strip():
            return False
        
        for pattern in self.section_patterns:
            if re.match(pattern, line):
                return True
        
        # Additional heuristics
        if (len(line) < 100 and 
            (line.isupper() or 
             line.endswith(':') or 
             re.match(r'^\d+\.', line) or
             re.match(r'^[A-Z][^.]*$', line))):
            return True
        
        return False
    
    def _chunk_section(self, section: Dict[str, Any], base_metadata: Dict) -> List[Document]:
        """Chunk a single section while preserving context"""
        header = section["header"]
        content = section["content"]
        
        if len(content) <= self.chunk_size:
            # Section fits in one chunk
            enhanced_metadata = base_metadata.copy()
            enhanced_metadata.update({
                "section_header": header,
                "section_type": self._classify_section_type(header),
                "chunk_type": "complete_section"
            })
            
            return [Document(
                page_content=f"{header}\n\n{content}" if header else content,
                metadata=enhanced_metadata
            )]
        
        # Split large section into chunks
        chunks = []
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators
        )
        
        section_chunks = splitter.split_text(content)
        
        for i, chunk_text in enumerate(section_chunks):
            enhanced_metadata = base_metadata.copy()
            enhanced_metadata.update({
                "section_header": header,
                "section_type": self._classify_section_type(header),
                "chunk_type": "section_part",
                "chunk_index": i,
                "total_chunks_in_section": len(section_chunks)
            })
            
            # Add section header to first chunk
            if i == 0 and header:
                chunk_content = f"{header}\n\n{chunk_text}"
            else:
                chunk_content = chunk_text
            
            chunks.append(Document(
                page_content=chunk_content,
                metadata=enhanced_metadata
            ))
        
        return chunks
    
    def _classify_section_type(self, header: str) -> str:
        """Classify the type of section based on header"""
        header_lower = header.lower()
        
        if any(word in header_lower for word in ['clause', 'section', 'article']):
            return "clause"
        elif any(word in header_lower for word in ['exclusion', 'exception']):
            return "exclusion"
        elif any(word in header_lower for word in ['coverage', 'benefit']):
            return "coverage"
        elif any(word in header_lower for word in ['definition', 'meaning']):
            return "definition"
        elif any(word in header_lower for word in ['procedure', 'process', 'how to']):
            return "procedure"
        else:
            return "general"
    
    def _enhanced_recursive_split(self, document: Document) -> List[Document]:
        """Enhanced recursive splitting as fallback"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators
        )
        
        chunks = splitter.split_documents([document])
        
        # Enhance metadata for each chunk
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "chunk_index": i,
                "total_chunks": len(chunks),
                "chunk_type": "recursive_split",
                "section_type": "unknown"
            })
        
        return chunks

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
    
    def optimize_context_intelligently(self, documents: List[Document], query: str, max_tokens: int = 4000) -> str:
        """Enhanced context optimization with section-awareness"""
        if not documents:
            return ""
        
        # Calculate relevance scores and token counts
        doc_scores = []
        for doc in documents:
            relevance = self.calculate_relevance_score(doc, query)
            tokens = self.estimate_tokens(doc.page_content)
            
            # Enhanced boosting based on metadata
            section_boost = 1.0
            
            # Boost for specific section types
            section_type = doc.metadata.get('section_type', '')
            if section_type in ['clause', 'coverage', 'definition']:
                section_boost = 1.3
            elif section_type in ['exclusion', 'procedure']:
                section_boost = 1.2
            
            # Boost for complete sections
            if doc.metadata.get('chunk_type') == 'complete_section':
                section_boost *= 1.1
            
            # Boost for domain-specific content
            domain_boost = 1.0
            content_lower = doc.page_content.lower()
            
            if any(keyword in content_lower for keyword in INSURANCE_KEYWORDS):
                domain_boost = 1.2
            elif any(keyword in content_lower for keyword in LEGAL_KEYWORDS):
                domain_boost = 1.15
            
            efficiency = (relevance * section_boost * domain_boost) / max(tokens, 1)
            doc_scores.append((doc, relevance, tokens, efficiency))
        
        # Sort by efficiency (relevance per token)
        doc_scores.sort(key=lambda x: x[3], reverse=True)
        
        # Build context within budget - Enhanced strategy
        context_parts = []
        token_budget = max_tokens
        section_headers_seen = set()
        
        for doc, relevance, tokens, efficiency in doc_scores:
            if tokens <= token_budget:
                # Add section header context if available and not seen
                section_header = doc.metadata.get('section_header', '')
                if section_header and section_header not in section_headers_seen:
                    section_headers_seen.add(section_header)
                
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
        # Enhanced domain descriptions with more specific terms
        self.domain_descriptions = {
            "technical": "technical documentation engineering software development programming code architecture system design specifications API database network infrastructure configuration deployment",
            "legal": "legal law regulation statute constitution court judicial legislation clause article provision contract agreement litigation compliance regulatory framework terms conditions liability jurisdiction",
            "medical": "medical healthcare patient diagnosis treatment clinical therapy medicine hospital physician doctor surgery pharmaceutical health insurance medical coverage clinical trials disease symptoms",
            "financial": "financial banking investment policy economics business finance accounting audit tax revenue profit loss balance sheet financial planning investment portfolio market analysis stock bonds",
            "insurance": "insurance policy premium claim coverage deductible benefit exclusion waiting period pre-existing condition maternity critical illness hospitalization cashless network provider sum insured policyholder co-payment room rent sub-limit renewal grace period nominee life insurance health insurance motor insurance travel insurance cataract PED clause",
            "academic": "academic research study analysis methodology scholarly scientific paper thesis journal publication university education learning curriculum dissertation peer review citation bibliography",
            "business": "business corporate strategy management operations marketing sales human resources organizational development project management leadership team collaboration productivity efficiency",
            "general": "general information document content text data knowledge base manual guide instructions reference material information overview summary"
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
            for doc in documents[:15]:  # More documents for better context
                content = doc.page_content
                
                # Prioritize content with domain-specific keywords
                if any(term in content.lower() for term in INSURANCE_KEYWORDS):
                    combined_content.append(content[:1000])  # Longer segments for key content
                elif any(term in content.lower() for term in LEGAL_KEYWORDS):
                    combined_content.append(content[:1000])
                else:
                    combined_content.append(content[:500])
            
            combined_text = ' '.join(combined_content)[:6000]  # More context
            
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
                # Check for specific content as fallback
                if any(keyword in combined_text.lower() for keyword in INSURANCE_KEYWORDS[:5]):
                    best_domain = "insurance"
                    best_score = 0.6
                elif any(keyword in combined_text.lower() for keyword in LEGAL_KEYWORDS[:5]):
                    best_domain = "legal"
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
        
        # Enhanced keyword-based detection
        if any(keyword in combined_text.lower() for keyword in INSURANCE_KEYWORDS):
            return "insurance", 0.6
        elif any(keyword in combined_text.lower() for keyword in LEGAL_KEYWORDS):
            return "legal", 0.6
        elif any(term in combined_text.lower() for term in ['medical', 'healthcare', 'patient', 'clinical']):
            return "medical", 0.6
        elif any(term in combined_text.lower() for term in ['financial', 'banking', 'investment', 'economic']):
            return "financial", 0.6
        elif any(term in combined_text.lower() for term in ['technical', 'engineering', 'software', 'system']):
            return "technical", 0.6
        elif any(term in combined_text.lower() for term in ['academic', 'research', 'study', 'university']):
            return "academic", 0.6
        elif any(term in combined_text.lower() for term in ['business', 'corporate', 'management', 'strategy']):
            return "business", 0.6
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
    """Enhanced RAG system with all improvements"""
    
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
        self.intelligent_splitter = None  # NEW: Enhanced splitter
    
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
        """Process documents with enhanced chunking and domain detection"""
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
                
                logger.info(f"🔍 Detected domain: {self.domain} (confidence: {domain_confidence:.2f})")
                
                # Initialize intelligent splitter with domain-specific config
                self.intelligent_splitter = IntelligentTextSplitter(
                    chunk_size=self.domain_config["chunk_size"],
                    chunk_overlap=self.domain_config["chunk_overlap"],
                    domain=self.domain
                )
                
                # Update document metadata with domain and enhanced info
                for doc in raw_documents:
                    doc.metadata.update({
                        'detected_domain': self.domain,
                        'domain_confidence': domain_confidence,
                        'session_id': self.session_id,
                        'processing_timestamp': datetime.now().isoformat()
                    })
                
                # ENHANCED CHUNKING - Section-aware splitting
                logger.info("📄 Starting intelligent document chunking...")
                self.documents = self.intelligent_splitter.split_documents(raw_documents)
                
                # Filter out very short chunks
                self.documents = [doc for doc in self.documents if len(doc.page_content.strip()) > 50]
                
                self.document_hash = self.calculate_document_hash(self.documents)
                self.processed_files = [os.path.basename(fp) for fp in file_paths]
                
                # Setup retrievers with enhanced configuration
                if base_sentence_model and embedding_model:
                    await self._setup_enhanced_retrievers()
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
                    'enhanced_features': {
                        'intelligent_chunking': True,
                        'section_awareness': True,
                        'metadata_enrichment': True,
                        'mmr_enabled': self.domain_config.get("use_mmr", False),
                        'reranking_enabled': True
                    }
                }
                
                if LOG_VERBOSE:
                    logger.info(f"✅ Enhanced processing complete: {result}")
                
                return result
                
            except Exception as e:
                logger.error(f"❌ Document processing error: {e}")
                raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")
    
    async def _setup_enhanced_retrievers(self):
        """Setup enhanced retrievers with MMR and metadata support"""
        try:
            global embedding_model, pinecone_index
            
            # Check if components are loaded
            if not base_sentence_model or not embedding_model:
                logger.warning("⚠️ Components not fully loaded, skipping retriever setup")
                return
            
            # Use namespace for document isolation
            namespace = f"{self.domain}_{self.document_hash}"
            
            if LOG_VERBOSE:
                logger.info(f"🔧 Setting up enhanced retrievers with namespace: {namespace}")
            
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
            
            # Setup enhanced BM25 retriever
            try:
                self.bm25_retriever = await asyncio.to_thread(BM25Retriever.from_documents, self.documents)
                # ENHANCED: Increase BM25 retrieval count
                self.bm25_retriever.k = self.domain_config["rerank_top_k"]
                
                if LOG_VERBOSE:
                    logger.info(f"✅ Enhanced BM25 retriever setup complete (k={self.bm25_retriever.k})")
                    
            except Exception as e:
                logger.warning(f"⚠️ Error setting up BM25 retriever: {e}")
                
        except Exception as e:
            logger.error(f"❌ Enhanced retriever setup error: {e}")
    
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
    
    async def enhanced_retrieve_and_rerank(self, query: str, top_k: int = None) -> Tuple[List[Document], List[float]]:
        """ENHANCED retrieval with MMR, metadata filtering, and increased retrieval"""
        if top_k is None:
            top_k = self.domain_config["context_docs"]
        
        try:
            # Check if components are loaded
            if not base_sentence_model:
                logger.warning("⚠️ Components not loaded, returning basic results")
                return self.documents[:top_k], [0.5] * min(len(self.documents), top_k)
            
            # ENHANCEMENT 1: Increased retrieval quantity
            semantic_k = self.domain_config["semantic_search_k"]  # Much higher than original k=5
            rerank_k = self.domain_config["rerank_top_k"]
            
            if LOG_VERBOSE:
                logger.info(f"🔍 Enhanced retrieval: semantic_k={semantic_k}, rerank_k={rerank_k}, final_k={top_k}")
            
            # Create embedding task immediately
            embedding_task = asyncio.create_task(
                EMBEDDING_SERVICE.get_query_embedding(query)
            )
            
            # Parallel retrieval tasks
            tasks = []
            
            # ENHANCEMENT 2: MMR-enabled vector search
            if self.vector_store:
                if self.domain_config.get("use_mmr", True):
                    vector_task = asyncio.create_task(
                        self._mmr_vector_search(query, semantic_k)
                    )
                else:
                    vector_task = asyncio.create_task(
                        self._vector_search_optimized(query, semantic_k)
                    )
                tasks.append(vector_task)
            
            # Enhanced BM25 search task
            if self.bm25_retriever:
                bm25_task = asyncio.create_task(
                    asyncio.to_thread(self.bm25_retriever.get_relevant_documents, query)
                )
                tasks.append(bm25_task)
            
            # Wait for embedding and searches simultaneously
            embedding = await embedding_task
            search_results = await asyncio.gather(*tasks, return_exceptions=True) if tasks else []
            
            # ENHANCEMENT 3: Enhanced merge and rerank with metadata awareness
            return await self._enhanced_merge_and_rerank(query, search_results, embedding, top_k, rerank_k)
            
        except Exception as e:
            logger.error(f"❌ Enhanced retrieval error: {e}")
            # Fallback to basic document retrieval
            return self.documents[:top_k], [0.5] * min(len(self.documents), top_k)
    
    async def _mmr_vector_search(self, query: str, k: int) -> List[Tuple[Document, float]]:
        """MMR-enabled vector search for diversity"""
        try:
            if not self.vector_store:
                return []
            
            # Use MMR search for diversity
            lambda_mult = self.domain_config.get("mmr_lambda", 0.7)
            
            results = await asyncio.to_thread(
                self.vector_store.max_marginal_relevance_search_with_score,
                query,
                k=min(k, 30),  # Limit to prevent too many results
                lambda_mult=lambda_mult
            )
            
            if LOG_VERBOSE:
                logger.info(f"🔄 MMR search returned {len(results)} diverse results (λ={lambda_mult})")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ MMR vector search error: {e}")
            # Fallback to regular similarity search
            return await self._vector_search_optimized(query, k)
    
    async def _vector_search_optimized(self, query: str, k: int) -> List[Tuple[Document, float]]:
        """Optimized vector search with Pinecone"""
        try:
            if not self.vector_store:
                return []
            
            results = await asyncio.to_thread(
                self.vector_store.similarity_search_with_score,
                query,
                k=min(k, 25)
            )
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Vector search error: {e}")
            return []
    
    async def _enhanced_merge_and_rerank(self, query: str, search_results: List, embedding: np.ndarray, 
                                       top_k: int, rerank_k: int) -> Tuple[List[Document], List[float]]:
        """Enhanced merge and rerank with metadata awareness"""
        all_docs = []
        all_scores = []
        seen_content = set()  # Avoid duplicates
        
        # Process vector search results
        if search_results and not isinstance(search_results[0], Exception):
            vector_results = search_results[0]
            for doc, distance_score in vector_results:
                content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
                if content_hash not in seen_content:
                    all_docs.append(doc)
                    # Better score normalization for Pinecone
                    similarity_score = max(0.0, min(1.0, (2.0 - distance_score) / 2.0))
                    all_scores.append(similarity_score)
                    seen_content.add(content_hash)
        
        # Process BM25 results
        if len(search_results) > 1 and not isinstance(search_results[1], Exception):
            bm25_docs = search_results[1][:rerank_k]
            for doc in bm25_docs:
                content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
                if content_hash not in seen_content:  # Avoid duplicates
                    all_docs.append(doc)
                    all_scores.append(0.7)  # Default BM25 score
                    seen_content.add(content_hash)
        
        # If no search results, use more documents from the collection
        if not all_docs:
            all_docs = self.documents[:rerank_k]
            all_scores = [0.5] * len(all_docs)
        
        # ENHANCEMENT 4: Enhanced metadata-aware reranking
        if all_docs and len(all_docs) > 1 and reranker:
            try:
                reranked_docs, reranked_scores = await self._enhanced_semantic_rerank(
                    query, all_docs, all_scores, rerank_k
                )
                return reranked_docs[:top_k], reranked_scores[:top_k]
            except Exception as e:
                logger.warning(f"⚠️ Enhanced reranking failed: {e}")
        
        # ENHANCEMENT 5: Metadata-based final sorting if reranking unavailable
        final_docs, final_scores = self._metadata_aware_sorting(query, all_docs, all_scores)
        
        return final_docs[:top_k], final_scores[:top_k]
    
    async def _enhanced_semantic_rerank(self, query: str, documents: List[Document], 
                                      initial_scores: List[float], rerank_k: int) -> Tuple[List[Document], List[float]]:
        """Enhanced semantic reranking with metadata awareness"""
        try:
            global reranker
            if reranker is None:
                return documents, initial_scores
            
            # Skip reranking for small sets to improve performance
            if len(documents) <= 3:
                return documents, initial_scores
            
            # Take top candidates for reranking to manage cost/performance
            candidates = list(zip(documents, initial_scores))
            candidates.sort(key=lambda x: x[1], reverse=True)
            
            # Limit reranking to top candidates
            rerank_candidates = candidates[:min(rerank_k, len(candidates))]
            rerank_docs = [doc for doc, _ in rerank_candidates]
            
            if LOG_VERBOSE:
                logger.info(f"🔄 Reranking {len(rerank_docs)} documents")
            
            # Prepare query-document pairs
            query_doc_pairs = [[query, doc.page_content[:512]] for doc in rerank_docs]
            
            # Batch process pairs for better performance
            rerank_scores = await asyncio.to_thread(reranker.predict, query_doc_pairs)
            
            # ENHANCEMENT: Metadata-aware score adjustment
            adjusted_scores = []
            for i, (doc, rerank_score) in enumerate(zip(rerank_docs, rerank_scores)):
                # Normalize rerank score to 0-1 range
                normalized_rerank = float((rerank_score + 1) / 2)  # Convert from [-1,1] to [0,1]
                
                # Metadata-based boosting
                metadata_boost = 1.0
                
                # Boost for specific section types
                section_type = doc.metadata.get('section_type', '')
                if section_type == 'clause':
                    metadata_boost = 1.15
                elif section_type in ['coverage', 'definition']:
                    metadata_boost = 1.1
                elif section_type == 'exclusion' and any(word in query.lower() for word in ['exclusion', 'exclude', 'not covered']):
                    metadata_boost = 1.2
                
                # Boost for complete sections
                if doc.metadata.get('chunk_type') == 'complete_section':
                    metadata_boost *= 1.05
                
                # Final score combination: 70% rerank, 30% initial score + metadata boost
                initial_score = initial_scores[candidates.index((doc, initial_scores[i]))] if i < len(initial_scores) else 0.5
                combined_score = (0.7 * normalized_rerank + 0.3 * initial_score) * metadata_boost
                
                adjusted_scores.append(min(1.0, combined_score))
            
            # Sort by adjusted score
            scored_docs = list(zip(rerank_docs, adjusted_scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            # Add remaining documents that weren't reranked
            remaining_docs = [doc for doc, _ in candidates[len(rerank_docs):]]
            remaining_scores = [score for _, score in candidates[len(rerank_docs):]]
            
            final_docs = [doc for doc, _ in scored_docs] + remaining_docs
            final_scores = [score for _, score in scored_docs] + remaining_scores
            
            if LOG_VERBOSE:
                logger.info(f"✅ Enhanced reranking complete: top score = {final_scores[0]:.3f}")
            
            return final_docs, final_scores
            
        except Exception as e:
            logger.warning(f"⚠️ Enhanced reranking error: {e}")
            return documents, initial_scores
    
    def _metadata_aware_sorting(self, query: str, documents: List[Document], 
                              scores: List[float]) -> Tuple[List[Document], List[float]]:
        """Sort documents using metadata when reranking is unavailable"""
        query_lower = query.lower()
        
        # Calculate metadata-enhanced scores
        enhanced_scores = []
        for doc, score in zip(documents, scores):
            enhanced_score = score
            
            # Section type boost
            section_type = doc.metadata.get('section_type', '')
            if section_type == 'clause':
                enhanced_score *= 1.1
            elif section_type in ['coverage', 'definition']:
                enhanced_score *= 1.05
            
            # Query-specific boosts
            if 'exclusion' in query_lower and section_type == 'exclusion':
                enhanced_score *= 1.15
            elif any(term in query_lower for term in ['coverage', 'benefit', 'covered']) and section_type == 'coverage':
                enhanced_score *= 1.15
            
            enhanced_scores.append(min(1.0, enhanced_score))
        
        # Sort by enhanced scores
        scored_docs = list(zip(documents, enhanced_scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        sorted_docs, sorted_scores = zip(*scored_docs) if scored_docs else ([], [])
        return list(sorted_docs), list(sorted_scores)

# ================================
# ENHANCED DECISION ENGINE
# ================================

class UniversalDecisionEngine:
    """Enhanced decision engine with confidence-based fallback"""
    
    def __init__(self):
        self.token_processor = TokenOptimizedProcessor()
        self.confidence_cache = LRUCache(maxsize=1000)
        self.fallback_attempts = {}  # Track fallback attempts per query
    
    def calculate_enhanced_confidence_score(self,
                                          query: str,
                                          similarity_scores: List[float],
                                          query_match_quality: float,
                                          domain_confidence: float = 1.0,
                                          metadata_quality: float = 1.0) -> float:
        """Enhanced confidence calculation with metadata awareness"""
        cache_key = f"{hash(tuple(similarity_scores))}_{query_match_quality}_{domain_confidence}_{metadata_quality}"
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
            
            # Count high-quality matches
            high_quality_matches = np.sum(scores_array > 0.7) / len(scores_array)
            
            # Enhanced weighting with metadata
            confidence = (
                0.35 * max_similarity +              # Highest weight to best match
                0.25 * avg_similarity +              # Average quality
                0.20 * min(1.0, query_match_quality) +  # Query relevance
                0.10 * score_consistency +           # Consistency bonus
                0.05 * domain_confidence +           # Domain confidence
                0.05 * metadata_quality              # Metadata quality
            )
            
            # Boost for multiple high-quality matches
            confidence += 0.1 * high_quality_matches
        
        # Apply domain-specific adjustments
        if domain_confidence > 0.7:
            if any(term in query.lower() for term in ['policy', 'premium', 'claim', 'coverage']):
                confidence *= 1.05  # 5% boost for insurance queries
            elif any(term in query.lower() for term in ['clause', 'section', 'article']):
                confidence *= 1.05  # 5% boost for legal queries
        
        confidence = min(1.0, max(0.0, confidence))
        self.confidence_cache[cache_key] = confidence
        return confidence
    
    def _assess_enhanced_query_match_quality(self, query: str, retrieved_docs: List[Document]) -> Tuple[float, float]:
        """Enhanced query match quality with metadata assessment"""
        query_terms = set(query.lower().split())
        if not query_terms:
            return 0.5, 0.5
        
        match_scores = []
        metadata_scores = []
        
        for doc in retrieved_docs[:8]:  # Check more documents
            doc_terms = set(doc.page_content.lower().split())
            overlap = len(query_terms.intersection(doc_terms))
            match_score = overlap / len(query_terms)
            match_scores.append(match_score)
            
            # Metadata quality assessment
            metadata_score = 0.5  # Base score
            
            # Boost for structured metadata
            if doc.metadata.get('section_header'):
                metadata_score += 0.2
            if doc.metadata.get('section_type') in ['clause', 'coverage', 'definition']:
                metadata_score += 0.2
            if doc.metadata.get('chunk_type') == 'complete_section':
                metadata_score += 0.1
            
            metadata_scores.append(min(1.0, metadata_score))
        
        avg_match = np.mean(match_scores) if match_scores else 0.0
        avg_metadata = np.mean(metadata_scores) if metadata_scores else 0.5
        
        return min(1.0, avg_match), avg_metadata
    
    async def process_query_with_fallback(self,
                                        query: str,
                                        retrieved_docs: List[Document],
                                        similarity_scores: List[float],
                                        domain: str,
                                        domain_confidence: float = 1.0,
                                        query_type: str = "general",
                                        rag_system: 'EnhancedRAGSystem' = None) -> Dict[str, Any]:
        """Enhanced query processing with confidence-based fallback logic"""
        if not retrieved_docs:
            return self._empty_response(query, domain)
        
        try:
            # Enhanced confidence calculation
            query_match_quality, metadata_quality = self._assess_enhanced_query_match_quality(query, retrieved_docs)
            
            confidence = self.calculate_enhanced_confidence_score(
                query, similarity_scores, query_match_quality, domain_confidence, metadata_quality
            )
            
            # ENHANCEMENT 6: Confidence-based fallback logic
            confidence_threshold = DOMAIN_CONFIGS.get(domain, DEFAULT_DOMAIN_CONFIG)["confidence_threshold"]
            
            if confidence < confidence_threshold and rag_system:
                logger.info(f"🔄 Low confidence ({confidence:.2f} < {confidence_threshold:.2f}), attempting fallback")
                
                # Try fallback strategy
                fallback_result = await self._attempt_fallback_retrieval(
                    query, domain, rag_system, confidence_threshold
                )
                
                if fallback_result:
                    retrieved_docs, similarity_scores, confidence = fallback_result
                    query_match_quality, metadata_quality = self._assess_enhanced_query_match_quality(query, retrieved_docs)
                    logger.info(f"✅ Fallback improved confidence to {confidence:.2f}")
            
            # Optimize context for token efficiency
            context = self.token_processor.optimize_context_intelligently(
                retrieved_docs, query, max_tokens=4000  # Increased context budget
            )
            
            # Generate response with enhanced prompting
            response = await self._generate_enhanced_response(query, context, domain, confidence)
            
            # Prepare final result
            result = {
                "query": query,
                "answer": response,
                "confidence": confidence,
                "domain": domain,
                "domain_confidence": domain_confidence,
                "query_type": query_type,
                "reasoning_chain": [
                    f"Enhanced retrieval: {len(retrieved_docs)} documents",
                    f"Confidence: {confidence:.1%} (threshold: {confidence_threshold:.1%})",
                    f"Metadata quality: {metadata_quality:.1%}",
                    f"Query match: {query_match_quality:.1%}"
                ],
                "source_documents": list(set([doc.metadata.get('source', 'Unknown') for doc in retrieved_docs])),
                "retrieved_chunks": len(retrieved_docs),
                "processing_time": time.time(),
                "enhanced_features": {
                    "intelligent_chunking": True,
                    "mmr_diversity": True,
                    "enhanced_reranking": True,
                    "confidence_fallback": confidence < confidence_threshold,
                    "metadata_aware": True
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Enhanced query processing error: {e}")
            return self._error_response(query, domain, str(e))
    
    async def _attempt_fallback_retrieval(self, query: str, domain: str, 
                                        rag_system: 'EnhancedRAGSystem', 
                                        confidence_threshold: float) -> Optional[Tuple[List[Document], List[float], float]]:
        """Attempt fallback retrieval strategies for low confidence queries"""
        try:
            # Strategy 1: Expand query with domain-specific terms
            expanded_query = self._expand_query_for_domain(query, domain)
            
            if expanded_query != query:
                logger.info(f"🔍 Trying expanded query: {expanded_query}")
                
                fallback_docs, fallback_scores = await rag_system.enhanced_retrieve_and_rerank(
                    expanded_query, top_k=rag_system.domain_config["context_docs"] + 5
                )
                
                if fallback_docs:
                    # Recalculate confidence
                    query_match_quality, metadata_quality = self._assess_enhanced_query_match_quality(expanded_query, fallback_docs)
                    new_confidence = self.calculate_enhanced_confidence_score(
                        expanded_query, fallback_scores, query_match_quality, 0.8, metadata_quality
                    )
                    
                    if new_confidence > confidence_threshold * 0.9:  # Slightly lower threshold for fallback
                        return fallback_docs, fallback_scores, new_confidence
            
            # Strategy 2: Retrieve more documents with broader search
            logger.info("🔍 Trying broader retrieval")
            broader_docs, broader_scores = await rag_system.enhanced_retrieve_and_rerank(
                query, top_k=min(25, len(rag_system.documents))  # Get more documents
            )
            
            if len(broader_docs) > len(rag_system.documents[:rag_system.domain_config["context_docs"]]):
                query_match_quality, metadata_quality = self._assess_enhanced_query_match_quality(query, broader_docs)
                new_confidence = self.calculate_enhanced_confidence_score(
                    query, broader_scores, query_match_quality, 0.8, metadata_quality
                )
                
                if new_confidence > confidence_threshold * 0.85:
                    return broader_docs, broader_scores, new_confidence
            
            return None
            
        except Exception as e:
            logger.warning(f"⚠️ Fallback retrieval error: {e}")
            return None
    
    def _expand_query_for_domain(self, query: str, domain: str) -> str:
        """Expand query with domain-specific terms"""
        query_lower = query.lower()
        
        if domain == "insurance":
            expansions = []
            if "claim" in query_lower:
                expansions.extend(["claim process", "claim procedure", "reimbursement"])
            if "coverage" in query_lower:
                expansions.extend(["benefit", "covered", "policy coverage"])
            if "exclusion" in query_lower:
                expansions.extend(["not covered", "excluded", "limitation"])
            
            if expansions:
                return f"{query} {' '.join(expansions[:2])}"  # Add top 2 expansions
        
        elif domain == "legal":
            expansions = []
            if "clause" in query_lower:
                expansions.extend(["section", "article", "provision"])
            if "contract" in query_lower:
                expansions.extend(["agreement", "terms", "conditions"])
            
            if expansions:
                return f"{query} {' '.join(expansions[:2])}"
        
        return query
    
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
    
    async def _generate_enhanced_response(self, query: str, context: str, domain: str, confidence: float) -> str:
        """Generate response with enhanced domain-specific prompting"""
        
        # ENHANCEMENT 7: Enhanced prompt design for extractive behavior
        domain_instructions = self._get_domain_specific_instructions(domain)
        confidence_instruction = self._get_confidence_based_instruction(confidence)
        
        # Enhanced prompt template
        enhanced_prompt = f"""You are an expert {domain} document analyst. Your task is to provide accurate, extractive answers based strictly on the provided context.

{domain_instructions}

CRITICAL INSTRUCTIONS:
1. Answer ONLY based on the information explicitly stated in the context below
2. If the information is not in the context, state "This information is not mentioned in the provided document"
3. Be specific and cite relevant details when available
4. Do not make assumptions or provide general knowledge
5. If multiple relevant sections exist, synthesize them clearly
6. {confidence_instruction}

DOCUMENT CONTEXT:
{context}

QUESTION: {query}

ANALYSIS AND ANSWER:"""

        try:
            global openai_client
            if not openai_client:
                return "OpenAI client not initialized. Please wait for system to fully load."
            
            # Enhanced system message for better behavior
            system_message = f"""You are a precise {domain} document analyst. Provide accurate, context-based responses with high reliability. Extract information directly from the context and be explicit about what is and isn't mentioned in the documents."""
            
            response = await openai_client.optimized_completion(
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": enhanced_prompt}
                ],
                model="gpt-4o",
                max_tokens=1500,
                temperature=0.05  # Lower temperature for more deterministic responses
            )
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Enhanced LLM generation error: {e}")
            return f"Error generating response: {str(e)}. The system may still be initializing."
    
    def _get_domain_specific_instructions(self, domain: str) -> str:
        """Get domain-specific instructions for better prompting"""
        instructions = {
            "insurance": """
When analyzing insurance documents, focus on:
- Policy terms, coverage details, and benefit amounts
- Exclusions, waiting periods, and limitations
- Claim procedures and requirements
- Pre-existing condition clauses
- Premium calculations and payment terms
- Network providers and cashless facilities
Always specify policy clause numbers or section references when available.""",
            
            "legal": """
When analyzing legal documents, focus on:
- Specific clauses, sections, and articles
- Terms, conditions, and provisions
- Rights, obligations, and liabilities
- Jurisdiction and governing law
- Compliance requirements and regulations
Always cite specific clause or section numbers when available.""",
            
            "medical": """
When analyzing medical documents, focus on:
- Diagnosis, treatment, and clinical procedures
- Medical conditions and symptoms
- Healthcare coverage and benefits
- Clinical guidelines and protocols
- Patient care instructions
Always reference specific medical sections or guidelines when available.""",
            
            "financial": """
When analyzing financial documents, focus on:
- Financial terms, calculations, and formulas
- Investment details and portfolio information
- Risk factors and market conditions
- Regulatory compliance and reporting
- Performance metrics and benchmarks
Always cite specific financial sections or data sources when available.""",
            
            "technical": """
When analyzing technical documents, focus on:
- System specifications and requirements
- Implementation details and procedures
- Configuration settings and parameters
- Troubleshooting and maintenance
- Performance metrics and standards
Always reference specific technical sections or specification numbers when available.""",
            
            "academic": """
When analyzing academic documents, focus on:
- Research methodology and findings
- Literature reviews and citations
- Data analysis and conclusions
- Theoretical frameworks and models
- Study limitations and future research
Always cite specific sections, chapters, or research findings when available.""",
            
            "business": """
When analyzing business documents, focus on:
- Strategic objectives and goals
- Operational procedures and processes
- Performance metrics and KPIs
- Organizational structure and roles
- Market analysis and competitive landscape
Always reference specific business sections or data points when available."""
        }
        
        return instructions.get(domain, "Focus on providing accurate, context-based information with specific references when available.")
    
    def _get_confidence_based_instruction(self, confidence: float) -> str:
        """Get confidence-based instruction for response generation"""
        if confidence >= 0.8:
            return "High confidence: Provide a comprehensive and detailed answer."
        elif confidence >= 0.6:
            return "Moderate confidence: Provide a clear answer and note any limitations."
        else:
            return "Low confidence: Be cautious in your response and clearly state if information is incomplete or uncertain."

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
        """Prepare URL and extract filename"""
        parsed_url = urlparse(url)
        
        # Handle Google Drive URLs
        if 'drive.google.com' in parsed_url.netloc:
            if '/file/d/' in url:
                file_id = url.split('/file/d/')[1].split('/')[0]
                download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
                filename = f"google_drive_file_{file_id}.pdf"
            else:
                raise HTTPException(status_code=400, detail="Invalid Google Drive URL format")
        
        # Handle Dropbox URLs
        elif 'dropbox.com' in parsed_url.netloc:
            download_url = url.replace('?dl=0', '?dl=1').replace('?dl=1', '?dl=1')
            if '?dl=1' not in download_url:
                download_url += '?dl=1'
            filename = parsed_url.path.split('/')[-1] or "dropbox_file.pdf"
        
        # Handle OneDrive URLs
        elif 'onedrive.live.com' in parsed_url.netloc or '1drv.ms' in parsed_url.netloc:
            if '1drv.ms' in parsed_url.netloc:
                # Handle shortened OneDrive URLs
                download_url = url + "&download=1"
            else:
                download_url = url.replace('view.aspx', 'download.aspx')
            filename = "onedrive_file.pdf"
        
        # Handle direct URLs
        else:
            download_url = url
            filename = parsed_url.path.split('/')[-1] or "downloaded_file.pdf"
            
            # Ensure filename has extension
            if '.' not in filename:
                filename += '.pdf'
        
        return download_url, filename

# ================================
# GLOBAL INSTANCES
# ================================

REDIS_CACHE = RedisCache()
EMBEDDING_SERVICE = OptimizedEmbeddingService()
DECISION_ENGINE = UniversalDecisionEngine()

# ================================
# FASTAPI APPLICATION SETUP
# ================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Enhanced application lifespan manager"""
    logger.info("🚀 Starting Enhanced RAG System...")
    
    # Initialize components
    await initialize_components()
    
    yield
    
    # Cleanup
    logger.info("🛑 Shutting down Enhanced RAG System...")
    try:
        # Cleanup active sessions
        cleanup_tasks = []
        for session_obj in ACTIVE_SESSIONS.values():
            cleanup_tasks.append(session_obj.get_data().cleanup())
        
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
        # Clear Redis cache
        await REDIS_CACHE.clear_cache()
        
        logger.info("✅ Cleanup completed")
    except Exception as e:
        logger.error(f"❌ Cleanup error: {e}")

# Create FastAPI app
app = FastAPI(
    title="Enhanced RAG System",
    description="Production-ready RAG system with intelligent chunking, MMR diversity, confidence-based fallbacks, and multi-domain support",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS
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

class HackRxRunRequest(BaseModel):
    """Request model for HackRx run endpoint"""
    documents: str = Field(..., description="Document URL to process")
    questions: List[str] = Field(..., description="List of questions to ask")

class HackRxQuestionAnswer(BaseModel):
    """Individual question-answer pair"""
    question: str
    answer: str
    confidence: float
    reasoning_chain: List[str]

class HackRxRunResponse(BaseModel):
    """Response model for HackRx run endpoint"""
    document_url: str
    total_questions: int
    processing_time: float
    domain: str
    domain_confidence: float
    session_id: str
    results: List[HackRxQuestionAnswer]
    status: str
    enhanced_features: Dict[str, Any]

class ProcessDocumentsRequest(BaseModel):
    """Request model for document processing"""
    file_urls: List[HttpUrl] = Field(..., description="List of file URLs to process")
    domain_override: Optional[str] = Field(None, description="Override domain detection")
    session_id: Optional[str] = Field(None, description="Reuse existing session")

class ProcessDocumentsResponse(BaseModel):
    """Response model for document processing"""
    session_id: str
    document_hash: str
    domain: str
    domain_confidence: float
    total_chunks: int
    processed_files: List[str]
    chunk_size: int
    chunk_overlap: int
    enhanced_features: Dict[str, bool]
    processing_time: float
    status: str

class QueryRequest(BaseModel):
    """Request model for queries"""
    query: str = Field(..., description="Question to ask")
    session_id: str = Field(..., description="Session ID from document processing")
    query_type: Optional[str] = Field("general", description="Type of query")

class QueryResponse(BaseModel):
    """Response model for queries"""
    query: str
    answer: str
    confidence: float
    domain: str
    domain_confidence: float
    query_type: str
    reasoning_chain: List[str]
    source_documents: List[str]
    retrieved_chunks: int
    processing_time: float
    enhanced_features: Dict[str, Any]
    status: str

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    components: Dict[str, str]
    memory_usage: Dict[str, str]
    active_sessions: int

# ================================
# API ENDPOINTS
# ================================

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with system information"""
    return {
        "service": "Enhanced RAG System",
        "version": "2.0.0",
        "status": "operational",
        "features": "intelligent-chunking,mmr-diversity,confidence-fallbacks,multi-domain",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Enhanced health check endpoint"""
    try:
        # Check component status
        components = {
            "embedding_model": "loaded" if embedding_model and embedding_model != "initializing" else "loading",
            "openai_client": "ready" if openai_client else "not_initialized",
            "redis_cache": "connected" if REDIS_CACHE.redis else "fallback",
            "pinecone": "connected" if pinecone_index else "not_connected",
            "reranker": "loaded" if reranker else "loading",
            "base_model": "loaded" if base_sentence_model else "loading"
        }
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_usage = {
            "total_gb": f"{memory.total / (1024**3):.1f}",
            "available_gb": f"{memory.available / (1024**3):.1f}",
            "used_percent": f"{memory.percent:.1f}%"
        }
        
        # Overall status
        critical_components = ["openai_client"]
        status = "healthy"
        
        if any(components.get(comp) == "not_initialized" for comp in critical_components):
            status = "degraded"
        
        return HealthResponse(
            status=status,
            timestamp=datetime.now().isoformat(),
            components=components,
            memory_usage=memory_usage,
            active_sessions=len(ACTIVE_SESSIONS)
        )
        
    except Exception as e:
        logger.error(f"❌ Health check error: {e}")
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.now().isoformat(),
            components={"error": str(e)},
            memory_usage={},
            active_sessions=0
        )

@app.post("/process-documents", response_model=ProcessDocumentsResponse)
async def process_documents(request: ProcessDocumentsRequest):
    """Enhanced document processing endpoint"""
    start_time = time.time()
    temp_files = []
    
    try:
        if not request.file_urls:
            raise HTTPException(status_code=400, detail="No file URLs provided")
        
        if LOG_VERBOSE:
            logger.info(f"📄 Processing {len(request.file_urls)} documents")
        
        # Download files
        downloader = UniversalURLDownloader()
        downloaded_files = []
        
        for url in request.file_urls:
            try:
                content, filename = await downloader.download_from_url(str(url))
                
                # Save to temporary file
                temp_file = tempfile.NamedTemporaryFile(
                    delete=False, 
                    suffix=os.path.splitext(filename)[1]
                )
                temp_file.write(content)
                temp_file.close()
                
                temp_files.append(temp_file.name)
                downloaded_files.append(filename)
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to download {url}: {e}")
                continue
        
        if not temp_files:
            raise HTTPException(status_code=400, detail="No files could be downloaded")
        
        # Get or create session
        if request.session_id:
            try:
                rag_session = await EnhancedSessionManager.get_or_create_session(request.session_id)
            except Exception:
                # Create new session if requested session doesn't exist
                rag_session = EnhancedRAGSystem()
        else:
            rag_session = EnhancedRAGSystem()
        
        # Process documents
        result = await rag_session.process_documents(temp_files)
        
        # Apply domain override if specified
        if request.domain_override and request.domain_override in DOMAIN_CONFIGS:
            rag_session.domain = request.domain_override
            rag_session.domain_config = DOMAIN_CONFIGS[request.domain_override].copy()
            result['domain'] = request.domain_override
            result['domain_override'] = True
        
        # Store session
        session_obj = SessionObject(result['session_id'], rag_session)
        ACTIVE_SESSIONS[result['session_id']] = session_obj
        
        processing_time = time.time() - start_time
        
        response = ProcessDocumentsResponse(
            **result,
            processing_time=processing_time,
            status="success"
        )
        
        if LOG_VERBOSE:
            logger.info(f"✅ Document processing completed in {processing_time:.2f}s")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Document processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    
    finally:
        # Cleanup temporary files
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception as e:
                logger.warning(f"⚠️ Failed to cleanup {temp_file}: {e}")

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Enhanced query endpoint with confidence-based fallbacks"""
    start_time = time.time()
    
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        if request.session_id not in ACTIVE_SESSIONS:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get session
        session_obj = ACTIVE_SESSIONS[request.session_id]
        if session_obj.is_expired():
            ACTIVE_SESSIONS.pop(request.session_id)
            raise HTTPException(status_code=404, detail="Session expired")
        
        rag_session = session_obj.get_data()
        
        if not rag_session.documents:
            raise HTTPException(status_code=400, detail="No documents in session")
        
        if LOG_VERBOSE:
            logger.info(f"🔍 Processing query: {request.query}")
        
        # Enhanced retrieval and reranking
        retrieved_docs, similarity_scores = await rag_session.enhanced_retrieve_and_rerank(
            request.query,
            top_k=rag_session.domain_config["context_docs"]
        )
        
        if not retrieved_docs:
            raise HTTPException(status_code=404, detail="No relevant documents found")
        
        # Enhanced decision processing with fallback
        result = await DECISION_ENGINE.process_query_with_fallback(
            query=request.query,
            retrieved_docs=retrieved_docs,
            similarity_scores=similarity_scores,
            domain=rag_session.domain,
            domain_confidence=0.8,  # Default confidence
            query_type=request.query_type,
            rag_system=rag_session
        )
        
        processing_time = time.time() - start_time
        result['processing_time'] = processing_time
        result['status'] = "success"
        
        response = QueryResponse(**result)
        
        if LOG_VERBOSE:
            logger.info(f"✅ Query processed in {processing_time:.2f}s with confidence {result['confidence']:.2f}")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Query processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a specific session"""
    try:
        if session_id not in ACTIVE_SESSIONS:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session_obj = ACTIVE_SESSIONS.pop(session_id)
        await session_obj.get_data().cleanup()
        
        return {"message": f"Session {session_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Session deletion error: {e}")
        raise HTTPException(status_code=500, detail=f"Session deletion failed: {str(e)}")

@app.post("/clear-cache")
async def clear_cache():
    """Clear all caches"""
    try:
        await REDIS_CACHE.clear_cache()
        EMBEDDING_CACHE.clear()
        RESPONSE_CACHE.clear()
        
        return {"message": "All caches cleared successfully"}
        
    except Exception as e:
        logger.error(f"❌ Cache clearing error: {e}")
        raise HTTPException(status_code=500, detail=f"Cache clearing failed: {str(e)}")

@app.get("/sessions")
async def list_sessions():
    """List all active sessions"""
    try:
        current_time = time.time()
        sessions_info = []
        
        for session_id, session_obj in ACTIVE_SESSIONS.items():
            rag_session = session_obj.data
            sessions_info.append({
                "session_id": session_id,
                "domain": rag_session.domain,
                "document_count": len(rag_session.documents),
                "processed_files": rag_session.processed_files,
                "created_at": datetime.fromtimestamp(session_obj.created_at).isoformat(),
                "last_accessed": datetime.fromtimestamp(session_obj.last_accessed).isoformat(),
                "expires_in_seconds": max(0, int(session_obj.ttl - (current_time - session_obj.last_accessed)))
            })
        
        return {
            "total_sessions": len(sessions_info),
            "sessions": sessions_info
        }
        
    except Exception as e:
        logger.error(f"❌ Session listing error: {e}")
        raise HTTPException(status_code=500, detail=f"Session listing failed: {str(e)}")

# ================================
# EXCEPTION HANDLERS
# ================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Enhanced HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Enhanced general exception handler"""
    logger.error(f"❌ Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "details": str(exc) if LOG_VERBOSE else "An unexpected error occurred",
            "status_code": 500,
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )

# Add this endpoint to your existing FastAPI endpoints section

@app.post("/hackrx/run", response_model=HackRxRunResponse)
async def hackrx_run(request: HackRxRunRequest):
    """HackRx run endpoint - Process document and answer multiple questions"""
    start_time = time.time()
    temp_files = []
    
    try:
        if not request.documents.strip():
            raise HTTPException(status_code=400, detail="Document URL cannot be empty")
        
        if not request.questions:
            raise HTTPException(status_code=400, detail="No questions provided")
        
        if LOG_VERBOSE:
            logger.info(f"🚀 HackRx Run: Processing document and {len(request.questions)} questions")
        
        # Step 1: Download and process the document
        downloader = UniversalURLDownloader()
        
        try:
            content, filename = await downloader.download_from_url(request.documents)
            
            # Save to temporary file
            temp_file = tempfile.NamedTemporaryFile(
                delete=False, 
                suffix=os.path.splitext(filename)[1]
            )
            temp_file.write(content)
            temp_file.close()
            temp_files.append(temp_file.name)
            
        except Exception as e:
            logger.error(f"❌ Failed to download document: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to download document: {str(e)}")
        
        # Step 2: Create RAG session and process document
        rag_session = EnhancedRAGSystem()
        
        try:
            processing_result = await rag_session.process_documents([temp_file.name])
            
            if LOG_VERBOSE:
                logger.info(f"✅ Document processed: {processing_result['total_chunks']} chunks, domain: {processing_result['domain']}")
                
        except Exception as e:
            logger.error(f"❌ Document processing failed: {e}")
            raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")
        
        # Step 3: Process all questions in parallel for better performance
        async def process_single_question(question: str) -> HackRxQuestionAnswer:
            """Process a single question"""
            try:
                if LOG_VERBOSE:
                    logger.info(f"🔍 Processing question: {question}")
                
                # Enhanced retrieval and reranking
                retrieved_docs, similarity_scores = await rag_session.enhanced_retrieve_and_rerank(
                    question,
                    top_k=rag_session.domain_config["context_docs"]
                )
                
                if not retrieved_docs:
                    return HackRxQuestionAnswer(
                        question=question,
                        answer="No relevant information found in the document for this question.",
                        confidence=0.0,
                        reasoning_chain=["No relevant documents retrieved"]
                    )
                
                # Enhanced decision processing with fallback
                result = await DECISION_ENGINE.process_query_with_fallback(
                    query=question,
                    retrieved_docs=retrieved_docs,
                    similarity_scores=similarity_scores,
                    domain=rag_session.domain,
                    domain_confidence=processing_result.get('domain_confidence', 0.8),
                    query_type="general",
                    rag_system=rag_session
                )
                
                return HackRxQuestionAnswer(
                    question=question,
                    answer=result['answer'],
                    confidence=result['confidence'],
                    reasoning_chain=result['reasoning_chain']
                )
                
            except Exception as e:
                logger.error(f"❌ Error processing question '{question}': {e}")
                return HackRxQuestionAnswer(
                    question=question,
                    answer=f"Error processing question: {str(e)}",
                    confidence=0.0,
                    reasoning_chain=[f"Processing error: {str(e)}"]
                )
        
        # Process all questions concurrently with controlled concurrency
        semaphore = asyncio.Semaphore(5)  # Limit concurrent processing to 5 questions
        
        async def process_with_semaphore(question: str) -> HackRxQuestionAnswer:
            async with semaphore:
                return await process_single_question(question)
        
        # Process all questions
        if LOG_VERBOSE:
            logger.info(f"🔄 Processing {len(request.questions)} questions concurrently...")
        
        question_results = await asyncio.gather(
            *[process_with_semaphore(q) for q in request.questions],
            return_exceptions=True
        )
        
        # Handle any exceptions in results
        final_results = []
        for i, result in enumerate(question_results):
            if isinstance(result, Exception):
                logger.error(f"❌ Exception in question {i}: {result}")
                final_results.append(HackRxQuestionAnswer(
                    question=request.questions[i],
                    answer=f"Error processing question: {str(result)}",
                    confidence=0.0,
                    reasoning_chain=[f"Exception: {str(result)}"]
                ))
            else:
                final_results.append(result)
        
        # Step 4: Prepare response
        processing_time = time.time() - start_time
        
        response = HackRxRunResponse(
            document_url=request.documents,
            total_questions=len(request.questions),
            processing_time=processing_time,
            domain=processing_result['domain'],
            domain_confidence=processing_result.get('domain_confidence', 0.8),
            session_id=processing_result['session_id'],
            results=final_results,
            status="success",
            enhanced_features={
                "intelligent_chunking": True,
                "mmr_diversity": True,
                "enhanced_reranking": True,
                "confidence_fallback": True,
                "metadata_aware": True,
                "concurrent_processing": True,
                "domain_detection": processing_result['domain']
            }
        )
        
        if LOG_VERBOSE:
            successful_answers = sum(1 for r in final_results if r.confidence > 0.5)
            logger.info(f"✅ HackRx Run completed in {processing_time:.2f}s: {successful_answers}/{len(request.questions)} questions answered successfully")
        
        # Cleanup session after processing (optional - remove if you want to keep sessions)
        try:
            await rag_session.cleanup()
        except Exception as e:
            logger.warning(f"⚠️ Cleanup warning: {e}")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ HackRx Run error: {e}")
        raise HTTPException(status_code=500, detail=f"HackRx Run failed: {str(e)}")
    
    finally:
        # Cleanup temporary files
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception as e:
                logger.warning(f"⚠️ Failed to cleanup {temp_file}: {e}")

# ================================
# MAIN ENTRY POINT
# ================================

if __name__ == "__main__":
    import uvicorn
    
    # Production configuration
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8080)),
        workers=1,
        loop="asyncio",
        access_log=LOG_VERBOSE,
        log_level="info" if LOG_VERBOSE else "warning"
    )
