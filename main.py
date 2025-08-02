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

# Configuration
SESSION_TTL = int(os.getenv("SESSION_TTL", 3600))  # 1 hour
MAX_CACHE_SIZE = int(os.getenv("MAX_CACHE_SIZE", 1000))
EMBEDDING_CACHE_SIZE = int(os.getenv("EMBEDDING_CACHE_SIZE", 10000))
LOG_VERBOSE = os.getenv("LOG_VERBOSE", "true").lower() == "true"

# Environment variables for API keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "enhanced-rag-system")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Authentication token
HACKRX_TOKEN = "9a1163c13e8927960b857a674794a62c57baf588998981151b0753a4d6d17905"

# Enhanced caching system
ACTIVE_SESSIONS = {}
EMBEDDING_CACHE = LRUCache(maxsize=EMBEDDING_CACHE_SIZE)
RESPONSE_CACHE = TTLCache(maxsize=MAX_CACHE_SIZE, ttl=300)

# UNIVERSAL DOMAIN CONFIGURATION - Works for ALL document types
UNIVERSAL_CONFIG = {
    "chunk_size": 1200,
    "chunk_overlap": 200,
    "semantic_search_k": 25,
    "context_docs": 18,
    "confidence_threshold": 0.45,  # Lower for universal coverage
    "use_mmr": True,
    "mmr_lambda": 0.75,
    "use_metadata_filtering": True,
    "rerank_top_k": 35
}

# Domain-specific optimizations (inherits from UNIVERSAL_CONFIG)
DOMAIN_CONFIGS = {
    "insurance": {**UNIVERSAL_CONFIG, "confidence_threshold": 0.40, "semantic_search_k": 30},
    "legal": {**UNIVERSAL_CONFIG, "chunk_size": 1500, "chunk_overlap": 300, "semantic_search_k": 28},
    "medical": {**UNIVERSAL_CONFIG, "confidence_threshold": 0.50, "semantic_search_k": 22},
    "financial": {**UNIVERSAL_CONFIG, "confidence_threshold": 0.48, "semantic_search_k": 20},
    "technical": {**UNIVERSAL_CONFIG, "confidence_threshold": 0.52, "semantic_search_k": 24},
    "academic": {**UNIVERSAL_CONFIG, "chunk_size": 1300, "chunk_overlap": 250},
    "business": {**UNIVERSAL_CONFIG, "confidence_threshold": 0.47},
    "general": UNIVERSAL_CONFIG,
    "default": UNIVERSAL_CONFIG  # Fallback for any domain
}

# Universal keywords for comprehensive domain detection
DOMAIN_KEYWORDS = {
    "insurance": ['policy', 'premium', 'claim', 'coverage', 'benefit', 'exclusion', 'waiting period',
                  'pre-existing condition', 'maternity', 'critical illness', 'hospitalization', 'cashless',
                  'network provider', 'sum insured', 'policyholder', 'deductible', 'co-payment', 'room rent',
                  'sub-limit', 'renewal', 'grace period', 'nominee', 'cataract', 'PED', 'NCD', 'AYUSH'],
    "legal": ['clause', 'section', 'article', 'provision', 'terms', 'conditions', 'agreement', 'contract',
              'liability', 'jurisdiction', 'compliance', 'regulation', 'statute', 'law', 'legal'],
    "medical": ['medical', 'healthcare', 'patient', 'diagnosis', 'treatment', 'clinical', 'therapy',
                'medicine', 'hospital', 'physician', 'doctor', 'surgery', 'pharmaceutical', 'health'],
    "financial": ['financial', 'banking', 'investment', 'economics', 'business', 'finance', 'accounting',
                  'audit', 'tax', 'revenue', 'profit', 'loss', 'balance sheet', 'portfolio', 'market'],
    "technical": ['technical', 'engineering', 'software', 'development', 'programming', 'code',
                  'architecture', 'system', 'design', 'specifications', 'API', 'database', 'network'],
    "academic": ['academic', 'research', 'study', 'analysis', 'methodology', 'scholarly', 'scientific',
                 'paper', 'thesis', 'journal', 'publication', 'university', 'education', 'curriculum'],
    "business": ['business', 'corporate', 'strategy', 'management', 'operations', 'marketing', 'sales',
                 'human resources', 'organizational', 'project management', 'leadership', 'productivity']
}

# ================================
# AUTHENTICATION MIDDLEWARE (FIXED)
# ================================

security = HTTPBearer(auto_error=False)

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """FIXED: Verify Bearer token authentication"""
    if not credentials or credentials.credentials != HACKRX_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

# ================================
# COMPONENT READINESS CHECKER (NEW)
# ================================

async def ensure_components_ready(timeout: float = 60.0):
    """FIXED: Ensure all components are loaded before processing"""
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        # Check critical components
        critical_ready = (
            openai_client is not None and
            components_ready.get("openai_client", False)
        )
        
        # Optional components (can work without them)
        optional_ready = (
            base_sentence_model is not None or
            components_ready.get("base_sentence_model", False)
        )
        
        if critical_ready:
            logger.info(f"✅ Critical components ready. Optional components: {optional_ready}")
            return True
            
        await asyncio.sleep(0.5)
    
    # If we reach here, not all components are ready
    if openai_client is None:
        raise HTTPException(
            status_code=503, 
            detail="System not ready: OpenAI client not initialized"
        )
    
    logger.warning("⚠️ Some optional components not ready, continuing with available components")
    return True

# ================================
# INTELLIGENT TEXT SPLITTER (FIXED)
# ================================

class UniversalTextSplitter:
    """FIXED: Universal text splitter that works for ALL document types"""

    def __init__(self, chunk_size: int = 1200, chunk_overlap: int = 200, domain: str = "general"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.domain = domain
        
        # Universal separators for all document types
        self.separators = [
            "\n\n### ",    # Section headers
            "\n\n## ",     # Sub-section headers
            "\n\n# ",      # Main headers
            "\n\nChapter ", # Chapters
            "\n\nSection ", # Sections
            "\n\nArticle ", # Articles
            "\n\nClause ",  # Legal clauses
            "\n\nPart ",    # Parts
            "\n\n",         # Double newlines
            "\n",           # Single newlines
            ". ",           # Sentences
            "! ",           # Exclamations
            "? ",           # Questions
            " ",            # Words
            ""              # Characters
        ]
        
        # Universal section patterns
        self.section_patterns = [
            r'(?i)^(chapter|section|article|clause|part|appendix|annex)\s+(\d+(?:\.\d+)*)',
            r'(?i)^(\d+(?:\.\d+)*)\s*(chapter|section|article|clause|part)',
            r'(?i)^([A-Z][^.:!?]*[.:])\s*$',  # Headings ending with period or colon
            r'(?i)^([IVX]+\.)\s+',             # Roman numerals
            r'(?i)^(\([a-zA-Z0-9]\))\s+',      # (a) or (1) style numbering
            r'(?i)^(\d+\.\d+)\s+',             # Numbered sections like 1.1
            r'(?i)^([A-Z\s]{3,})\s*$'          # All caps titles
        ]

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """FIXED: Split documents with enhanced section awareness"""
        all_chunks = []
        for doc in documents:
            try:
                chunks = self._split_single_document(doc)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.warning(f"⚠️ Error splitting document: {e}")
                # Fallback to basic recursive splitting
                chunks = self._fallback_split(doc)
                all_chunks.extend(chunks)
        
        # Filter out very short chunks
        all_chunks = [chunk for chunk in all_chunks if len(chunk.page_content.strip()) >= 50]
        
        logger.info(f"📄 Split into {len(all_chunks)} chunks")
        return all_chunks

    def _split_single_document(self, document: Document) -> List[Document]:
        """FIXED: Split single document with section-aware chunking"""
        text = document.page_content
        
        # Try to identify sections first
        sections = self._identify_sections(text)
        
        if len(sections) > 1:
            chunks = []
            for section in sections:
                section_chunks = self._chunk_section(section, document.metadata)
                chunks.extend(section_chunks)
            return chunks
        else:
            # Use enhanced recursive splitting
            return self._enhanced_recursive_split(document)

    def _identify_sections(self, text: str) -> List[Dict[str, Any]]:
        """FIXED: Identify document sections using universal patterns"""
        sections = []
        lines = text.split('\n')
        current_section = {"header": "", "content": "", "start_line": 0}
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            is_header = self._is_section_header(line)
            
            if is_header and current_section["content"].strip():
                # Save previous section
                sections.append(current_section)
                current_section = {"header": line, "content": "", "start_line": i}
            elif is_header and not current_section["content"].strip():
                # Update header if no content yet
                current_section["header"] = line
                current_section["start_line"] = i
            else:
                current_section["content"] += line + "\n"
        
        # Add the last section
        if current_section["content"].strip():
            sections.append(current_section)
        
        # If no clear sections found, return entire text
        if not sections or len(sections) == 1:
            return [{"header": "Document", "content": text, "start_line": 0}]
        
        return sections

    def _is_section_header(self, line: str) -> bool:
        """FIXED: Universal section header detection"""
        if not line.strip() or len(line) > 200:  # Too long to be header
            return False
        
        # Check patterns
        for pattern in self.section_patterns:
            if re.match(pattern, line):
                return True
        
        # Additional universal heuristics
        line_upper = line.upper()
        if (len(line) < 100 and
            (line == line_upper or  # All caps
             line.endswith((':', '.')) or  # Ends with colon or period
             re.match(r'^\d+[\.\)]\s+', line) or  # Starts with number
             re.match(r'^[A-Z][a-z]*[\s\w]*$', line))):  # Title case
            return True
        
        return False

    def _chunk_section(self, section: Dict[str, Any], base_metadata: Dict) -> List[Document]:
        """FIXED: Chunk a single section preserving context"""
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
        
        # Split large section
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators
        )
        
        section_chunks = splitter.split_text(content)
        chunks = []
        
        for i, chunk_text in enumerate(section_chunks):
            enhanced_metadata = base_metadata.copy()
            enhanced_metadata.update({
                "section_header": header,
                "section_type": self._classify_section_type(header),
                "chunk_type": "section_part",
                "chunk_index": i,
                "total_chunks_in_section": len(section_chunks)
            })
            
            # Add header to first chunk
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
        """FIXED: Universal section type classification"""
        header_lower = header.lower()
        
        # Universal classification
        if any(word in header_lower for word in ['summary', 'abstract', 'overview']):
            return "summary"
        elif any(word in header_lower for word in ['introduction', 'background', 'preface']):
            return "introduction"
        elif any(word in header_lower for word in ['conclusion', 'summary', 'findings']):
            return "conclusion"
        elif any(word in header_lower for word in ['method', 'procedure', 'process', 'approach']):
            return "methodology"
        elif any(word in header_lower for word in ['result', 'outcome', 'finding']):
            return "results"
        elif any(word in header_lower for word in ['discussion', 'analysis', 'interpretation']):
            return "discussion"
        elif any(word in header_lower for word in ['reference', 'bibliography', 'citation']):
            return "references"
        elif any(word in header_lower for word in ['appendix', 'annex', 'supplement']):
            return "appendix"
        elif any(word in header_lower for word in ['definition', 'glossary', 'terminology']):
            return "definition"
        elif any(word in header_lower for word in ['table', 'figure', 'chart', 'diagram']):
            return "visual"
        else:
            return "content"

    def _enhanced_recursive_split(self, document: Document) -> List[Document]:
        """FIXED: Enhanced recursive splitting as fallback"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators
        )
        
        chunks = splitter.split_documents([document])
        
        # Enhance metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "chunk_index": i,
                "total_chunks": len(chunks),
                "chunk_type": "recursive_split",
                "section_type": "general"
            })
        
        return chunks

    def _fallback_split(self, document: Document) -> List[Document]:
        """FIXED: Simple fallback splitting"""
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            return splitter.split_documents([document])
        except Exception as e:
            logger.error(f"❌ Even fallback splitting failed: {e}")
            # Return document as single chunk
            return [document]

# ================================
# REDIS CACHE IMPLEMENTATION (FIXED)
# ================================

class RedisCache:
    """FIXED: Production-ready Redis cache with proper error handling"""
    
    def __init__(self):
        self.redis = None
        self.fallback_cache = TTLCache(maxsize=1000, ttl=300)
    
    async def initialize(self):
        """FIXED: Initialize Redis with proper error handling"""
        try:
            self.redis = redis.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", 6379)),
                db=0,
                socket_connect_timeout=5,
                socket_timeout=5,
                decode_responses=True
            )
            
            # Test connection
            await asyncio.wait_for(self.redis.ping(), timeout=5)
            components_ready["redis"] = True
            logger.info("✅ Redis cache initialized successfully")
            
        except Exception as e:
            logger.warning(f"⚠️ Redis not available, using fallback cache: {e}")
            self.redis = None
            components_ready["redis"] = False
    
    async def get_cached_response(self, query_hash: str) -> Optional[Dict]:
        """FIXED: Get cached response with fallback"""
        if self.redis:
            try:
                cached = await asyncio.wait_for(
                    self.redis.get(f"response:{query_hash}"), 
                    timeout=2
                )
                return json.loads(cached) if cached else None
            except Exception as e:
                logger.warning(f"⚠️ Redis get error: {e}")
        
        # Fallback to memory cache
        return self.fallback_cache.get(query_hash)
    
    async def cache_response(self, query_hash: str, response: dict, ttl: int = 300):
        """FIXED: Cache response with fallback"""
        # Always cache in memory
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
# OPTIMIZED EMBEDDING SERVICE (FIXED)
# ================================

class OptimizedEmbeddingService:
    """FIXED: Optimized embedding service with proper async handling"""
    
    def __init__(self):
        self.embedding_cache = LRUCache(maxsize=5000)
        self.batch_queue = []
        self.processing_lock = asyncio.Lock()
    
    async def get_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """FIXED: Process embeddings in optimized batches with fallback"""
        if not texts:
            return []
        
        async with self.processing_lock:
            # Check cache first
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
            
            # Process uncached texts
            if uncached_texts:
                try:
                    if base_sentence_model:
                        # Use sentence transformer
                        embeddings = await asyncio.to_thread(
                            base_sentence_model.encode,
                            uncached_texts,
                            batch_size=min(64, len(uncached_texts)),
                            show_progress_bar=False,
                            convert_to_numpy=True
                        )
                        
                        # Cache new embeddings
                        for text, embedding in zip(uncached_texts, embeddings):
                            text_hash = hashlib.md5(text.encode()).hexdigest()
                            self.embedding_cache[text_hash] = embedding
                        
                        # Add to results
                        for i, embedding in zip(uncached_indices, embeddings):
                            results.append((i, embedding))
                    
                    else:
                        # Fallback: return zero vectors
                        logger.warning("⚠️ No embedding model available, using zero vectors")
                        for i in uncached_indices:
                            results.append((i, np.zeros(384)))
                
                except Exception as e:
                    logger.error(f"❌ Embedding error: {e}")
                    # Return zero vectors as fallback
                    for i in uncached_indices:
                        results.append((i, np.zeros(384)))
            
            # Sort results by original index
            results.sort(key=lambda x: x[0])
            return [embedding for _, embedding in results]
    
    async def get_query_embedding(self, query: str) -> np.ndarray:
        """FIXED: Get single query embedding with proper error handling"""
        if not query.strip():
            return np.zeros(384)
        
        query_hash = hashlib.md5(query.encode()).hexdigest()
        
        if query_hash in self.embedding_cache:
            return self.embedding_cache[query_hash]
        
        try:
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
# OPTIMIZED OPENAI CLIENT (FIXED)
# ================================

class OptimizedOpenAIClient:
    """FIXED: OpenAI client with proper connection pooling and error handling"""
    
    def __init__(self):
        self.client = None
        self.prompt_cache = TTLCache(maxsize=1000, ttl=600)
        self.rate_limit_delay = 1.0
        self.request_semaphore = asyncio.Semaphore(10)  # Limit concurrent requests
    
    async def initialize(self, api_key: str):
        """FIXED: Initialize OpenAI client with proper timeout and pooling"""
        if not api_key:
            raise ValueError("OpenAI API key is required")
        
        try:
            # Create client with connection pooling
            self.client = AsyncOpenAI(
                api_key=api_key,
                timeout=httpx.Timeout(60.0),
                max_retries=3
            )
            
            # Test the connection
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
        """FIXED: Generate hash for prompt caching"""
        prompt_data = {
            "messages": json.dumps(messages, sort_keys=True),
            "model": kwargs.get("model", "gpt-4o"),
            "temperature": kwargs.get("temperature", 0.1),
            "max_tokens": kwargs.get("max_tokens", 1000)
        }
        return hashlib.md5(json.dumps(prompt_data, sort_keys=True).encode()).hexdigest()
    
    async def optimized_completion(self, messages: List[Dict], **kwargs) -> str:
        """FIXED: Optimized completion with proper error handling and caching"""
        if not self.client:
            raise ValueError("OpenAI client not initialized")
        
        # Check cache
        prompt_hash = self._get_prompt_hash(messages, **kwargs)
        cached = self.prompt_cache.get(prompt_hash)
        if cached:
            return cached
        
        # Also check Redis cache
        redis_cached = await REDIS_CACHE.get_cached_response(prompt_hash)
        if redis_cached and 'content' in redis_cached:
            result = redis_cached['content']
            self.prompt_cache[prompt_hash] = result
            return result
        
        # Make API call with semaphore to limit concurrency
        async with self.request_semaphore:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = await self.client.chat.completions.create(
                        messages=messages,
                        model=kwargs.get("model", "gpt-4o"),
                        temperature=kwargs.get("temperature", 0.1),
                        max_tokens=kwargs.get("max_tokens", 1000),
                        timeout=60
                    )
                    
                    result = response.choices[0].message.content
                    
                    # Cache successful response
                    self.prompt_cache[prompt_hash] = result
                    await REDIS_CACHE.cache_response(prompt_hash, {"content": result})
                    
                    return result
                    
                except openai.RateLimitError as e:
                    if attempt < max_retries - 1:
                        delay = self.rate_limit_delay * (2 ** attempt)
                        logger.warning(f"⏰ Rate limit hit, waiting {delay}s...")
                        await asyncio.sleep(delay)
                        continue
                    raise
                    
                except openai.APITimeoutError as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"⏰ Timeout, retrying... (attempt {attempt + 1})")
                        await asyncio.sleep(2 ** attempt)
                        continue
                    raise
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"⚠️ API error, retrying: {e}")
                        await asyncio.sleep(2 ** attempt)
                        continue
                    raise
        
        raise Exception("Max retries exceeded for OpenAI API")

# ================================
# UNIVERSAL DOMAIN DETECTOR (FIXED)
# ================================

class UniversalDomainDetector:
    """FIXED: Universal domain detector that works for ALL document types"""
    
    def __init__(self):
        self.domain_embeddings = {}
        self.fallback_cache = LRUCache(maxsize=500)
        
        # Enhanced universal domain descriptions
        self.domain_descriptions = {
            "insurance": "insurance policy premium claim coverage benefit exclusion waiting period pre-existing condition maternity critical illness hospitalization cashless network provider sum insured policyholder deductible co-payment room rent sub-limit renewal grace period nominee life insurance health insurance motor insurance travel insurance cataract PED clause medical expenses organ donor NCD AYUSH",
            "legal": "legal law regulation statute constitution court judicial legislation clause article provision contract agreement litigation compliance regulatory framework terms conditions liability jurisdiction attorney lawyer legal document court case legal proceedings legal advice legal opinion legal contract legal agreement",
            "medical": "medical healthcare patient diagnosis treatment clinical therapy medicine hospital physician doctor surgery pharmaceutical health medical condition disease symptoms medical history medical report clinical study medical research healthcare provider medical treatment medical procedure medical diagnosis",
            "financial": "financial banking investment policy economics business finance accounting audit tax revenue profit loss balance sheet financial planning investment portfolio market analysis stock bonds financial report financial statement financial data economic analysis financial strategy",
            "technical": "technical documentation engineering software development programming code architecture system design specifications API database network infrastructure configuration deployment technical manual technical guide technical specifications system requirements technical documentation software documentation",
            "academic": "academic research study analysis methodology scholarly scientific paper thesis journal publication university education learning curriculum dissertation peer review citation bibliography research paper academic study research methodology academic research scientific study",
            "business": "business corporate strategy management operations marketing sales human resources organizational development project management leadership team collaboration productivity efficiency business plan business strategy business operations business management corporate governance",
            "general": "general information document content text data knowledge base manual guide instructions reference material information overview summary general documentation general information general content"
        }
    
    def initialize_embeddings(self):
        """FIXED: Initialize domain embeddings with proper error handling"""
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
        """FIXED: Universal domain detection that works for ANY document type"""
        if not documents:
            return "general", 0.5
        
        # Create cache key
        combined_text = ' '.join([doc.page_content[:500] for doc in documents[:10]])
        cache_key = hashlib.md5(combined_text.encode()).hexdigest()[:16]
        
        if cache_key in self.fallback_cache:
            return self.fallback_cache[cache_key]
        
        try:
            # Enhanced keyword-based detection (primary method)
            domain_scores = self._keyword_based_detection(documents)
            
            # Semantic detection (if embeddings available)
            if self.domain_embeddings and base_sentence_model:
                semantic_scores = self._semantic_detection(documents)
                # Combine keyword and semantic scores
                for domain in domain_scores:
                    if domain in semantic_scores:
                        domain_scores[domain] = 0.7 * domain_scores[domain] + 0.3 * semantic_scores[domain]
            
            # Get best domain
            if domain_scores:
                best_domain = max(domain_scores, key=domain_scores.get)
                best_score = domain_scores[best_domain]
                
                # Apply threshold
                if best_score < confidence_threshold:
                    best_domain = "general"
                    best_score = confidence_threshold
                
                result = (best_domain, best_score)
                self.fallback_cache[cache_key] = result
                
                if LOG_VERBOSE:
                    logger.info(f"🔍 Domain detected: {best_domain} (confidence: {best_score:.2f})")
                
                return result
            
            # Fallback
            return "general", confidence_threshold
            
        except Exception as e:
            logger.warning(f"⚠️ Domain detection error: {e}")
            return "general", confidence_threshold
    
    def _keyword_based_detection(self, documents: List[Document]) -> Dict[str, float]:
        """FIXED: Enhanced keyword-based domain detection"""
        combined_text = ' '.join([doc.page_content for doc in documents[:15]])[:8000].lower()
        
        domain_scores = {}
        total_words = len(combined_text.split())
        
        for domain, keywords in DOMAIN_KEYWORDS.items():
            matches = 0
            for keyword in keywords:
                # Count keyword occurrences
                matches += combined_text.count(keyword.lower())
            
            # Calculate score based on keyword density
            score = min(1.0, matches / max(1, len(keywords) * 0.1))  # Normalize
            domain_scores[domain] = score
        
        return domain_scores
    
    def _semantic_detection(self, documents: List[Document]) -> Dict[str, float]:
        """FIXED: Semantic domain detection"""
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
# COMPONENT INITIALIZATION (FIXED)
# ================================

async def initialize_components():
    """FIXED: Robust component initialization"""
    global embedding_model, base_sentence_model, reranker, openai_client, pinecone_index
    
    logger.info("🚀 Initializing Enhanced RAG System...")
    
    try:
        # 1. Initialize OpenAI client (CRITICAL)
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            openai_client = OptimizedOpenAIClient()
            await openai_client.initialize(openai_api_key)
        else:
            logger.error("❌ OPENAI_API_KEY not found - system will not work properly")
        
        # 2. Initialize Redis (non-blocking)
        try:
            await REDIS_CACHE.initialize()
        except Exception as e:
            logger.warning(f"⚠️ Redis initialization failed: {e}")
        
        # 3. Initialize Pinecone (non-blocking)
        try:
            if PINECONE_API_KEY:
                pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
                
                # Create or get index
                if PINECONE_INDEX_NAME not in pinecone.list_indexes():
                    pinecone.create_index(
                        name=PINECONE_INDEX_NAME,
                        dimension=384,
                        metric="cosine"
                    )
                
                pinecone_index = pinecone.Index(PINECONE_INDEX_NAME)
                components_ready["pinecone"] = True
                logger.info("✅ Pinecone initialized")
        except Exception as e:
            logger.warning(f"⚠️ Pinecone initialization failed: {e}")
        
        # 4. Start background loading of heavy models
        asyncio.create_task(load_heavy_components())
        
        logger.info("✅ Component initialization complete")
        
    except Exception as e:
        logger.error(f"❌ Component initialization failed: {e}")
        raise

async def load_heavy_components():
    """FIXED: Load heavy components in background with proper error handling"""
    global base_sentence_model, embedding_model, reranker
    
    logger.info("🔄 Loading heavy components in background...")
    
    # Load sentence transformer
    try:
        logger.info("📊 Loading sentence transformer...")
        base_sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        components_ready["base_sentence_model"] = True
        logger.info("✅ Sentence transformer loaded")
    except Exception as e:
        logger.warning(f"⚠️ Failed to load sentence transformer: {e}")
        components_ready["base_sentence_model"] = False
    
    # Load embedding model
    try:
        logger.info("📊 Loading embedding model...")
        embedding_model = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        components_ready["embedding_model"] = True
        logger.info("✅ Embedding model loaded")
    except Exception as e:
        logger.warning(f"⚠️ Failed to load embedding model: {e}")
        components_ready["embedding_model"] = False
    
    # Load reranker
    try:
        logger.info("📊 Loading reranker...")
        reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-2-v1')
        components_ready["reranker"] = True
        logger.info("✅ Reranker loaded")
    except Exception as e:
        logger.warning(f"⚠️ Failed to load reranker: {e}")
        components_ready["reranker"] = False
    
    # Initialize domain detector
    try:
        if base_sentence_model:
            DOMAIN_DETECTOR.initialize_embeddings()
            logger.info("✅ Domain detector initialized")
    except Exception as e:
        logger.warning(f"⚠️ Domain detector initialization failed: {e}")
    
    logger.info("✅ Background component loading complete")

# ================================
# TOKEN OPTIMIZATION (FIXED)
# ================================

class TokenOptimizedProcessor:
    """FIXED: Advanced token optimization for efficiency"""
    
    def __init__(self):
        self.max_context_tokens = 4000
        self.token_buffer = 500  # Reserve for response
    
    @lru_cache(maxsize=2000)
    def estimate_tokens(self, text: str) -> int:
        """FIXED: Accurate token estimation"""
        if not text:
            return 0
        
        # More accurate token estimation
        # Account for different content types
        words = text.split()
        avg_chars_per_token = 3.5  # Better estimate for general text
        
        # Adjust for technical content
        if any(term in text.lower() for term in ['api', 'json', 'xml', 'code', 'function']):
            avg_chars_per_token = 3.0  # Technical content has more tokens
        
        estimated = len(text) / avg_chars_per_token
        return max(1, int(estimated * 1.1))  # Add 10% buffer
    
    def calculate_relevance_score(self, doc: Document, query: str) -> float:
        """FIXED: Calculate document relevance with multiple factors"""
        try:
            # Keyword overlap
            query_terms = set(query.lower().split())
            doc_terms = set(doc.page_content.lower().split())
            keyword_overlap = len(query_terms.intersection(doc_terms)) / max(len(query_terms), 1)
            
            # Position boost (earlier chunks often more relevant)
            position_boost = 1.0
            chunk_index = doc.metadata.get('chunk_index', 0)
            total_chunks = doc.metadata.get('total_chunks', 1)
            if total_chunks > 1:
                position_boost = 1.2 - (chunk_index / total_chunks) * 0.4  # Boost earlier chunks
            
            # Section type boost
            section_boost = 1.0
            section_type = doc.metadata.get('section_type', '')
            if section_type in ['summary', 'introduction', 'conclusion']:
                section_boost = 1.15
            elif section_type in ['content', 'methodology']:
                section_boost = 1.1
            
            # Complete section boost
            if doc.metadata.get('chunk_type') == 'complete_section':
                section_boost *= 1.05
            
            # Try semantic similarity if possible
            semantic_score = 0.5  # Default
            if base_sentence_model:
                try:
                    doc_embedding = self._get_cached_embedding(doc.page_content[:512])
                    query_embedding = self._get_cached_embedding(query)
                    semantic_score = float(util.cos_sim(doc_embedding, query_embedding)[0][0])
                except Exception:
                    pass  # Use default if semantic fails
            
            # Combine scores
            final_score = (
                0.4 * semantic_score +
                0.3 * keyword_overlap +
                0.2 * position_boost +
                0.1 * section_boost
            )
            
            return min(1.0, final_score)
            
        except Exception as e:
            logger.warning(f"⚠️ Error calculating relevance: {e}")
            return 0.5  # Fallback score
    
    @lru_cache(maxsize=3000)
    def _get_cached_embedding(self, text: str) -> np.ndarray:
        """FIXED: Get cached embedding with proper fallback"""
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
        
        # Return zero vector as fallback
        return np.zeros(384)
    
    def optimize_context_intelligently(self, documents: List[Document], query: str, max_tokens: int = None) -> str:
        """FIXED: Intelligent context optimization"""
        if not documents:
            return ""
        
        if max_tokens is None:
            max_tokens = self.max_context_tokens
        
        # Calculate scores for all documents
        doc_scores = []
        for doc in documents:
            relevance = self.calculate_relevance_score(doc, query)
            tokens = self.estimate_tokens(doc.page_content)
            
            # Efficiency score (relevance per token)
            efficiency = relevance / max(tokens, 1)
            
            doc_scores.append((doc, relevance, tokens, efficiency))
        
        # Sort by efficiency
        doc_scores.sort(key=lambda x: x[3], reverse=True)
        
        # Build context within budget
        context_parts = []
        token_budget = max_tokens - self.token_buffer
        
        for doc, relevance, tokens, efficiency in doc_scores:
            if tokens <= token_budget:
                context_parts.append(doc.page_content)
                token_budget -= tokens
            elif token_budget > 200:  # Partial inclusion for important docs
                if relevance > 0.7:  # Only for high-relevance docs
                    partial_content = self._truncate_content(doc.page_content, token_budget)
                    context_parts.append(partial_content)
                break
        
        context = "\n\n".join(context_parts)
        
        if LOG_VERBOSE:
            estimated_tokens = self.estimate_tokens(context)
            logger.info(f"📝 Context optimized: {len(context_parts)} documents, ~{estimated_tokens} tokens")
        
        return context
    
    def _truncate_content(self, content: str, max_tokens: int) -> str:
        """FIXED: Smart content truncation"""
        max_chars = max_tokens * 4
        
        if len(content) <= max_chars:
            return content
        
        # Keep beginning and end
        keep_chars = max_chars - 100  # Reserve for truncation message
        first_part = content[:keep_chars//2]
        last_part = content[-keep_chars//2:]
        
        return f"{first_part}\n\n[... content truncated for token efficiency ...]\n\n{last_part}"

# ================================
# ENHANCED RAG SYSTEM (FIXED)
# ================================

class EnhancedRAGSystem:
    """FIXED: Enhanced RAG system that works for ALL document types"""
    
    def __init__(self, session_id: str = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.documents = []
        self.vector_store = None
        self.bm25_retriever = None
        self.domain = "general"
        self.domain_config = UNIVERSAL_CONFIG.copy()
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
        """FIXED: Proper cleanup with error handling"""
        try:
            # Clear vector store references
            self.vector_store = None
            self.bm25_retriever = None
            
            # Clear document references
            self.documents.clear()
            self.processed_files.clear()
            
            if LOG_VERBOSE:
                logger.info(f"🧹 Session {self.session_id} cleaned up")
                
        except Exception as e:
            logger.warning(f"⚠️ Cleanup error: {e}")
    
    def calculate_document_hash(self, documents: List[Document]) -> str:
        """FIXED: Calculate unique hash for documents"""
        content_sample = "".join([doc.page_content[:100] for doc in documents[:5]])
        return hashlib.sha256(content_sample.encode()).hexdigest()[:16]
    
    async def process_documents(self, file_paths: List[str]) -> Dict[str, Any]:
        """FIXED: Process documents with universal support"""
        async with self._processing_lock:
            start_time = time.time()
            
            try:
                if not file_paths:
                    raise HTTPException(status_code=400, detail="No file paths provided")
                
                logger.info(f"📄 Processing {len(file_paths)} documents")
                
                # Load documents with better error handling
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
                self.domain_config = DOMAIN_CONFIGS.get(domain, UNIVERSAL_CONFIG).copy()
                
                logger.info(f"🔍 Detected domain: {domain} (confidence: {domain_confidence:.2f})")
                
                # Initialize universal text splitter
                self.text_splitter = UniversalTextSplitter(
                    chunk_size=self.domain_config["chunk_size"],
                    chunk_overlap=self.domain_config["chunk_overlap"],
                    domain=domain
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
                
                # Universal document chunking
                logger.info("🔄 Starting universal document chunking...")
                self.documents = self.text_splitter.split_documents(raw_documents)
                
                # Filter very short chunks
                self.documents = [doc for doc in self.documents if len(doc.page_content.strip()) >= 30]
                
                self.document_hash = self.calculate_document_hash(self.documents)
                self.processed_files = [os.path.basename(fp) for fp in file_paths]
                
                # Setup retrievers
                await self._setup_universal_retrievers()
                
                processing_time = time.time() - start_time
                
                result = {
                    'session_id': self.session_id,
                    'document_hash': self.document_hash,
                    'domain': domain,
                    'domain_confidence': domain_confidence,
                    'total_chunks': len(self.documents),
                    'processed_files': self.processed_files,
                    'chunk_size': self.domain_config["chunk_size"],
                    'chunk_overlap': self.domain_config["chunk_overlap"],
                    'processing_time': processing_time,
                    'enhanced_features': {
                        'universal_chunking': True,
                        'domain_adaptive': True,
                        'metadata_enrichment': True,
                        'mmr_enabled': self.domain_config.get("use_mmr", True),
                        'reranking_enabled': True
                    }
                }
                
                if LOG_VERBOSE:
                    logger.info(f"✅ Processing complete in {processing_time:.2f}s: {result}")
                
                return result
                
            except Exception as e:
                logger.error(f"❌ Document processing error: {e}")
                raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    
    async def _load_document(self, file_path: str) -> List[Document]:
        """FIXED: Universal document loader"""
        try:
            file_extension = os.path.splitext(file_path)[1].lower()
            
            # Support all document types
            if file_extension == '.pdf':
                loader = PyMuPDFLoader(file_path)
            elif file_extension in ['.docx', '.doc']:
                loader = Docx2txtLoader(file_path)
            elif file_extension in ['.txt', '.md', '.csv']:
                loader = TextLoader(file_path, encoding='utf-8')
            else:
                # Try to load as text
                logger.info(f"📄 Unknown extension {file_extension}, trying as text...")
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
            '.csv': 'csv'
        }
        return type_mapping.get(extension, 'unknown')
    
    async def _setup_universal_retrievers(self):
        """FIXED: Setup universal retrievers that work for all domains"""
        try:
            logger.info("🔧 Setting up universal retrievers...")
            
            # Setup vector store if components available
            if pinecone_index and embedding_model and embedding_model != "initializing":
                try:
                    namespace = f"{self.domain}_{self.document_hash}"
                    
                    self.vector_store = Pinecone(
                        index=pinecone_index,
                        embedding=embedding_model,
                        text_key="text",
                        namespace=namespace
                    )
                    
                    # Check if documents already exist
                    stats = pinecone_index.describe_index_stats()
                    current_count = stats.get('namespaces', {}).get(namespace, {}).get('vector_count', 0)
                    
                    if current_count < len(self.documents):
                        logger.info(f"📊 Adding {len(self.documents)} documents to vector store")
                        
                        # Batch process for efficiency
                        batch_size = 50
                        for i in range(0, len(self.documents), batch_size):
                            batch = self.documents[i:i + batch_size]
                            await asyncio.to_thread(self.vector_store.add_documents, batch)
                            
                            if LOG_VERBOSE:
                                logger.info(f"📊 Processed batch {i//batch_size + 1}/{(len(self.documents)-1)//batch_size + 1}")
                    
                    logger.info("✅ Vector store setup complete")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Vector store setup failed: {e}")
            
            # Setup BM25 retriever
            try:
                if self.documents:
                    self.bm25_retriever = await asyncio.to_thread(
                        BM25Retriever.from_documents, 
                        self.documents
                    )
                    self.bm25_retriever.k = min(self.domain_config["rerank_top_k"], len(self.documents))
                    logger.info(f"✅ BM25 retriever setup complete (k={self.bm25_retriever.k})")
            
            except Exception as e:
                logger.warning(f"⚠️ BM25 retriever setup failed: {e}")
            
        except Exception as e:
            logger.error(f"❌ Retriever setup error: {e}")
    
    async def enhanced_retrieve_and_rerank(self, query: str, top_k: int = None) -> Tuple[List[Document], List[float]]:
        """FIXED: Enhanced retrieval that works universally"""
        if top_k is None:
            top_k = self.domain_config["context_docs"]
        
        try:
            if not self.documents:
                return [], []
            
            # Get retrieval parameters
            semantic_k = min(self.domain_config["semantic_search_k"], len(self.documents))
            rerank_k = min(self.domain_config["rerank_top_k"], len(self.documents))
            
            if LOG_VERBOSE:
                logger.info(f"🔍 Retrieving: semantic_k={semantic_k}, rerank_k={rerank_k}, final_k={top_k}")
            
            # Parallel retrieval
            tasks = []
            
            # Vector search
            if self.vector_store:
                if self.domain_config.get("use_mmr", True):
                    tasks.append(self._mmr_search(query, semantic_k))
                else:
                    tasks.append(self._vector_search(query, semantic_k))
            
            # BM25 search
            if self.bm25_retriever:
                tasks.append(self._bm25_search(query))
            
            # Execute searches
            if tasks:
                search_results = await asyncio.gather(*tasks, return_exceptions=True)
            else:
                # Fallback to document similarity
                search_results = [await self._fallback_search(query, semantic_k)]
            
            # Merge and rerank
            return await self._merge_and_rerank(query, search_results, top_k, rerank_k)
            
        except Exception as e:
            logger.error(f"❌ Retrieval error: {e}")
            # Return top documents as fallback
            return self.documents[:top_k], [0.5] * min(len(self.documents), top_k)
    
    async def _mmr_search(self, query: str, k: int) -> List[Tuple[Document, float]]:
        """FIXED: MMR search with proper error handling"""
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
        """FIXED: Vector search with error handling"""
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
        """FIXED: BM25 search"""
        try:
            return await asyncio.to_thread(self.bm25_retriever.get_relevant_documents, query)
        except Exception as e:
            logger.warning(f"⚠️ BM25 search error: {e}")
            return []
    
    async def _fallback_search(self, query: str, k: int) -> List[Tuple[Document, float]]:
        """FIXED: Fallback search using keyword matching"""
        try:
            query_terms = set(query.lower().split())
            doc_scores = []
            
            for doc in self.documents:
                doc_terms = set(doc.page_content.lower().split())
                overlap = len(query_terms.intersection(doc_terms))
                score = overlap / max(len(query_terms), 1)
                doc_scores.append((doc, score))
            
            # Sort by score and return top k
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            return doc_scores[:k]
            
        except Exception as e:
            logger.error(f"❌ Fallback search error: {e}")
            return [(doc, 0.5) for doc in self.documents[:k]]
    
    async def _merge_and_rerank(self, query: str, search_results: List, top_k: int, rerank_k: int) -> Tuple[List[Document], List[float]]:
        """FIXED: Merge and rerank results"""
        all_docs = []
        all_scores = []
        seen_content = set()
        
        # Process vector search results
        if search_results and not isinstance(search_results[0], Exception):
            for doc, score in search_results[0]:
                content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
                if content_hash not in seen_content:
                    all_docs.append(doc)
                    # Normalize score to 0-1
                    normalized_score = max(0.0, min(1.0, (2.0 - score) / 2.0)) if score > 1.0 else score
                    all_scores.append(normalized_score)
                    seen_content.add(content_hash)
        
        # Process BM25 results
        if len(search_results) > 1 and not isinstance(search_results[1], Exception):
            for doc in search_results[1][:rerank_k]:
                content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
                if content_hash not in seen_content:
                    all_docs.append(doc)
                    all_scores.append(0.6)  # Default BM25 score
                    seen_content.add(content_hash)
        
        # If no results, use top documents
        if not all_docs:
            all_docs = self.documents[:rerank_k]
            all_scores = [0.4] * len(all_docs)
        
        # Rerank if reranker available
        if reranker and len(all_docs) > 1:
            try:
                return await self._semantic_rerank(query, all_docs, all_scores, top_k)
            except Exception as e:
                logger.warning(f"⚠️ Reranking failed: {e}")
        
        # Sort by score and return top k
        scored_docs = list(zip(all_docs, all_scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        final_docs = [doc for doc, _ in scored_docs[:top_k]]
        final_scores = [score for _, score in scored_docs[:top_k]]
        
        return final_docs, final_scores
    
    async def _semantic_rerank(self, query: str, documents: List[Document], scores: List[float], top_k: int) -> Tuple[List[Document], List[float]]:
        """FIXED: Semantic reranking"""
        try:
            if len(documents) <= 2:  # Skip reranking for small sets
                return documents[:top_k], scores[:top_k]
            
            # Prepare query-document pairs
            pairs = [[query, doc.page_content[:512]] for doc in documents[:25]]  # Limit for efficiency
            
            # Get reranking scores
            rerank_scores = await asyncio.to_thread(reranker.predict, pairs)
            
            # Normalize and combine scores
            normalized_rerank = [(score + 1) / 2 for score in rerank_scores]  # Convert [-1,1] to [0,1]
            
            combined_scores = []
            for i, (orig_score, rerank_score) in enumerate(zip(scores[:len(normalized_rerank)], normalized_rerank)):
                # Metadata boost
                doc = documents[i]
                boost = 1.0
                
                section_type = doc.metadata.get('section_type', '')
                if section_type in ['summary', 'introduction', 'conclusion']:
                    boost = 1.1
                elif doc.metadata.get('chunk_type') == 'complete_section':
                    boost = 1.05
                
                # Combine: 70% rerank, 30% original
                combined = (0.7 * rerank_score + 0.3 * orig_score) * boost
                combined_scores.append(min(1.0, combined))
            
            # Add remaining documents
            if len(documents) > len(combined_scores):
                combined_scores.extend(scores[len(combined_scores):])
            
            # Sort by combined score
            scored_docs = list(zip(documents, combined_scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            final_docs = [doc for doc, _ in scored_docs[:top_k]]
            final_scores = [score for _, score in scored_docs[:top_k]]
            
            return final_docs, final_scores
            
        except Exception as e:
            logger.error(f"❌ Semantic reranking error: {e}")
            return documents[:top_k], scores[:top_k]

# ================================
# UNIVERSAL DECISION ENGINE (FIXED)
# ================================

class UniversalDecisionEngine:
    """FIXED: Universal decision engine for all domains"""
    
    def __init__(self):
        self.token_processor = TokenOptimizedProcessor()
        self.confidence_cache = LRUCache(maxsize=2000)
        self.response_cache = LRUCache(maxsize=1000)
    
    def calculate_confidence_score(self, query: str, similarity_scores: List[float], 
                                 retrieved_docs: List[Document], domain_confidence: float = 1.0) -> float:
        """FIXED: Universal confidence calculation"""
        if not similarity_scores:
            return 0.0
        
        # Cache key
        scores_str = str(sorted(similarity_scores))
        cache_key = hashlib.md5(f"{query}_{scores_str}_{domain_confidence}".encode()).hexdigest()[:12]
        
        if cache_key in self.confidence_cache:
            return self.confidence_cache[cache_key]
        
        try:
            scores_array = np.array(similarity_scores)
            
            # Basic metrics
            max_score = np.max(scores_array)
            avg_score = np.mean(scores_array)
            score_std = np.std(scores_array)
            
            # Quality metrics
            high_quality_docs = np.sum(scores_array > 0.6) / len(scores_array)
            score_consistency = max(0.0, 1.0 - (score_std * 1.5))
            
            # Query-document match
            query_match = self._calculate_query_match(query, retrieved_docs)
            
            # Universal confidence formula
            confidence = (
                0.35 * max_score +           # Best match
                0.25 * avg_score +           # Overall quality
                0.20 * query_match +         # Query relevance
                0.15 * score_consistency +   # Consistency
                0.05 * domain_confidence     # Domain confidence
            )
            
            # Boost for multiple high-quality matches
            confidence += 0.1 * high_quality_docs
            
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
        for doc in docs[:5]:  # Check top 5 docs
            doc_terms = set(doc.page_content.lower().split())
            overlap = len(query_terms.intersection(doc_terms))
            match_score = overlap / len(query_terms)
            match_scores.append(match_score)
        
        return np.mean(match_scores) if match_scores else 0.5
    
    async def process_query_with_fallback(self, query: str, retrieved_docs: List[Document],
                                        similarity_scores: List[float], domain: str,
                                        domain_confidence: float = 1.0, query_type: str = "general",
                                        rag_system: 'EnhancedRAGSystem' = None) -> Dict[str, Any]:
        """FIXED: Universal query processing with fallback"""
        start_time = time.time()
        
        try:
            if not retrieved_docs:
                return self._empty_response(query, domain)
            
            # Calculate confidence
            confidence = self.calculate_confidence_score(query, similarity_scores, retrieved_docs, domain_confidence)
            
            # Get confidence threshold for domain
            confidence_threshold = DOMAIN_CONFIGS.get(domain, UNIVERSAL_CONFIG)["confidence_threshold"]
            
            # Fallback logic if confidence is low
            if confidence < confidence_threshold and rag_system:
                logger.info(f"🔄 Low confidence ({confidence:.2f}), attempting fallback")
                
                fallback_result = await self._attempt_fallback(
                    query, domain, rag_system, confidence_threshold
                )
                
                if fallback_result:
                    retrieved_docs, similarity_scores, confidence = fallback_result
                    logger.info(f"✅ Fallback improved confidence to {confidence:.2f}")
            
            # Optimize context
            context = self.token_processor.optimize_context_intelligently(
                retrieved_docs, query, max_tokens=4000
            )
            
            # Generate response
            response = await self._generate_universal_response(query, context, domain, confidence)
            
            processing_time = time.time() - start_time
            
            result = {
                "query": query,
                "answer": response,
                "confidence": confidence,
                "domain": domain,
                "domain_confidence": domain_confidence,
                "query_type": query_type,
                "reasoning_chain": [
                    f"Retrieved {len(retrieved_docs)} documents",
                    f"Confidence: {confidence:.1%} (threshold: {confidence_threshold:.1%})",
                    f"Domain: {domain} ({domain_confidence:.1%})",
                    f"Context optimized: {len(context)} chars"
                ],
                "source_documents": list(set([
                    doc.metadata.get('source', 'Unknown') for doc in retrieved_docs
                ])),
                "retrieved_chunks": len(retrieved_docs),
                "processing_time": processing_time,
                "enhanced_features": {
                    "universal_processing": True,
                    "confidence_fallback": confidence < confidence_threshold,
                    "context_optimization": True,
                    "multi_domain_support": True
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Query processing error: {e}")
            return self._error_response(query, domain, str(e))
    
    async def _attempt_fallback(self, query: str, domain: str, rag_system: 'EnhancedRAGSystem',
                              threshold: float) -> Optional[Tuple[List[Document], List[float], float]]:
        """FIXED: Attempt fallback retrieval strategies"""
        try:
            # Strategy 1: Retrieve more documents
            expanded_docs, expanded_scores = await rag_system.enhanced_retrieve_and_rerank(
                query, top_k=min(25, len(rag_system.documents))
            )
            
            if len(expanded_docs) > len(rag_system.documents[:rag_system.domain_config["context_docs"]]):
                new_confidence = self.calculate_confidence_score(query, expanded_scores, expanded_docs)
                if new_confidence > threshold * 0.9:
                    return expanded_docs, expanded_scores, new_confidence
            
            # Strategy 2: Query expansion
            expanded_query = self._expand_query(query, domain)
            if expanded_query != query:
                fallback_docs, fallback_scores = await rag_system.enhanced_retrieve_and_rerank(
                    expanded_query, top_k=rag_system.domain_config["context_docs"]
                )
                
                if fallback_docs:
                    new_confidence = self.calculate_confidence_score(expanded_query, fallback_scores, fallback_docs)
                    if new_confidence > threshold * 0.85:
                        return fallback_docs, fallback_scores, new_confidence
            
            return None
            
        except Exception as e:
            logger.warning(f"⚠️ Fallback error: {e}")
            return None
    
    def _expand_query(self, query: str, domain: str) -> str:
        """Universal query expansion"""
        query_lower = query.lower()
        
        # Universal expansions
        expansions = []
        
        if "what" in query_lower and "is" in query_lower:
            expansions.extend(["definition", "meaning", "explanation"])
        elif "how" in query_lower:
            expansions.extend(["process", "procedure", "method"])
        elif "when" in query_lower:
            expansions.extend(["time", "period", "duration"])
        elif "why" in query_lower:
            expansions.extend(["reason", "cause", "purpose"])
        elif "where" in query_lower:
            expansions.extend(["location", "place", "position"])
        
        # Domain-specific expansions
        domain_expansions = {
            "insurance": ["policy", "coverage", "benefit", "claim"],
            "legal": ["law", "regulation", "clause", "agreement"],
            "medical": ["treatment", "diagnosis", "medical", "healthcare"],
            "financial": ["financial", "investment", "economic", "money"],
            "technical": ["technical", "system", "process", "specification"],
            "academic": ["research", "study", "academic", "analysis"],
            "business": ["business", "corporate", "management", "strategy"]
        }
        
        if domain in domain_expansions:
            expansions.extend(domain_expansions[domain])
        
        if expansions:
            # Add top 2 relevant expansions
            return f"{query} {' '.join(expansions[:2])}"
        
        return query
    
    async def _generate_universal_response(self, query: str, context: str, domain: str, confidence: float) -> str:
        """FIXED: Generate universal response for any domain"""
        try:
            if not openai_client:
                return "System is still initializing. Please wait a moment and try again."
            
            # Universal prompt template
            system_prompt = f"""You are an expert document analyst specializing in {domain} documents. Your task is to provide accurate, extractive answers based strictly on the provided context.

CRITICAL INSTRUCTIONS:
1. Answer ONLY based on information explicitly stated in the provided context
2. If information is not in the context, clearly state "This information is not mentioned in the provided document"
3. Be specific and cite relevant details when available
4. Do not add external knowledge or make assumptions
5. If multiple relevant sections exist, synthesize them clearly
6. Provide confidence level based on available information

CONFIDENCE GUIDANCE:
- High confidence ({confidence:.0%}): Provide comprehensive answer
- Moderate confidence: Be clear about limitations
- Low confidence: Be cautious and explicit about uncertainty"""

            confidence_instruction = "High confidence: Provide detailed answer" if confidence >= 0.7 else \
                                   "Moderate confidence: Provide clear answer with noted limitations" if confidence >= 0.5 else \
                                   "Low confidence: Be cautious and explicit about uncertainty"

            user_prompt = f"""DOCUMENT CONTEXT:
{context}

QUESTION: {query}

INSTRUCTIONS: {confidence_instruction}

ANALYSIS AND ANSWER:"""

            response = await openai_client.optimized_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model="gpt-4o",
                max_tokens=1500,
                temperature=0.05  # Low temperature for consistency
            )
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Response generation error: {e}")
            return f"Error generating response: {str(e)}. Please try again."
    
    def _empty_response(self, query: str, domain: str) -> Dict[str, Any]:
        """Generate response when no documents found"""
        return {
            "query": query,
            "answer": "No relevant information found in the provided documents for this query.",
            "confidence": 0.0,
            "domain": domain,
            "query_type": "no_results",
            "reasoning_chain": ["No relevant documents retrieved"],
            "source_documents": [],
            "retrieved_chunks": 0,
            "processing_time": 0.0
        }
    
    def _error_response(self, query: str, domain: str, error_msg: str) -> Dict[str, Any]:
        """Generate error response"""
        return {
            "query": query,
            "answer": f"Error processing query: {error_msg}",
            "confidence": 0.0,
            "domain": domain,
            "query_type": "error",
            "reasoning_chain": [f"Error: {error_msg}"],
            "source_documents": [],
            "retrieved_chunks": 0,
            "processing_time": 0.0
        }

# ================================
# SESSION MANAGEMENT (FIXED)
# ================================

T = TypeVar('T')

class SessionObject(Generic[T]):
    """FIXED: Generic session object with proper TTL management"""
    
    def __init__(self, session_id: str, data: T, ttl: int = SESSION_TTL):
        self.session_id = session_id
        self.data = data
        self.created_at = time.time()
        self.last_accessed = time.time()
        self.ttl = ttl
    
    def is_expired(self) -> bool:
        return (time.time() - self.last_accessed) > self.ttl
    
    def touch(self):
        self.last_accessed = time.time()
    
    def get_data(self) -> T:
        self.touch()
        return self.data

class EnhancedSessionManager:
    """FIXED: Enhanced session manager"""
    
    @staticmethod
    async def get_or_create_session(document_hash: str) -> EnhancedRAGSystem:
        """Get existing session or create new one with proper cleanup"""
        current_time = time.time()
        
        # Clean expired sessions
        expired_sessions = [
            session_id for session_id, session_obj in ACTIVE_SESSIONS.items()
            if session_obj.is_expired()
        ]
        
        if expired_sessions:
            cleanup_tasks = []
            for session_id in expired_sessions:
                session_obj = ACTIVE_SESSIONS.pop(session_id, None)
                if session_obj:
                    cleanup_tasks.append(session_obj.get_data().cleanup())
            
            if cleanup_tasks:
                await asyncio.gather(*cleanup_tasks, return_exceptions=True)
                logger.info(f"🗑️ Cleaned {len(expired_sessions)} expired sessions")
        
        # Get or create session
        if document_hash in ACTIVE_SESSIONS:
            session_obj = ACTIVE_SESSIONS[document_hash]
            session = session_obj.get_data()
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
# UNIVERSAL URL DOWNLOADER (FIXED)
# ================================

class UniversalURLDownloader:
    """FIXED: Universal URL downloader with comprehensive blob support"""
    
    def __init__(self, timeout: float = 120.0):
        self.timeout = timeout
    
    async def download_from_url(self, url: str) -> Tuple[bytes, str]:
        """FIXED: Download from any URL type with robust error handling"""
        try:
            download_url, filename = self._prepare_url_and_filename(url)
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (compatible; DocumentProcessor/2.0)',
                'Accept': '*/*',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive'
            }
            
            # Use longer timeout for large files
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                follow_redirects=True,
                headers=headers,
                limits=httpx.Limits(max_connections=20)
            ) as client:
                
                if LOG_VERBOSE:
                    logger.info(f"📥 Downloading: {download_url}")
                
                response = await client.get(download_url)
                response.raise_for_status()
                
                if not response.content:
                    raise HTTPException(status_code=400, detail="Downloaded file is empty")
                
                # Validate file type
                content_type = response.headers.get('content-type', '').lower()
                if content_type and not any(ct in content_type for ct in [
                    'pdf', 'document', 'text', 'application', 'octet-stream'
                ]):
                    logger.warning(f"⚠️ Unexpected content type: {content_type}")
                
                logger.info(f"✅ Downloaded {len(response.content)} bytes as {filename}")
                return response.content, filename
                
        except httpx.RequestError as e:
            logger.error(f"❌ Network error: {e}")
            raise HTTPException(status_code=400, detail=f"Network error: {str(e)}")
        
        except httpx.HTTPStatusError as e:
            logger.error(f"❌ HTTP {e.response.status_code}: {e}")
            raise HTTPException(status_code=400, detail=f"HTTP {e.response.status_code}: Download failed")
        
        except Exception as e:
            logger.error(f"❌ Download error: {e}")
            raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")
    
    def _prepare_url_and_filename(self, url: str) -> Tuple[str, str]:
        """FIXED: Handle all types of URLs including Azure Blob Storage"""
        parsed_url = urlparse(url)
        
        # Azure Blob Storage (like your sample URL)
        if 'blob.core.windows.net' in parsed_url.netloc:
            # Extract filename from path
            path_parts = parsed_url.path.split('/')
            filename = path_parts[-1] if path_parts else "azure_blob_file"
            
            # Ensure extension
            if '.' not in filename:
                filename += '.pdf'
            
            return url, filename  # Use URL as-is for Azure blob
        
        # Google Drive URLs
        elif 'drive.google.com' in parsed_url.netloc:
            if '/file/d/' in url:
                file_id = url.split('/file/d/')[1].split('/')[0]
                download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
                filename = f"google_drive_{file_id}.pdf"
            else:
                raise HTTPException(status_code=400, detail="Invalid Google Drive URL")
            return download_url, filename
        
        # Dropbox URLs
        elif 'dropbox.com' in parsed_url.netloc:
            download_url = url.replace('?dl=0', '?dl=1')
            if '?dl=1' not in download_url:
                download_url += '?dl=1'
            filename = parsed_url.path.split('/')[-1] or "dropbox_file.pdf"
            return download_url, filename
        
        # OneDrive URLs
        elif any(domain in parsed_url.netloc for domain in ['onedrive.live.com', '1drv.ms', 'sharepoint.com']):
            if '1drv.ms' in parsed_url.netloc:
                download_url = url + "&download=1"
            else:
                download_url = url.replace('view.aspx', 'download.aspx')
            filename = "onedrive_file.pdf"
            return download_url, filename
        
        # AWS S3 URLs
        elif 's3.amazonaws.com' in parsed_url.netloc or 's3-' in parsed_url.netloc:
            filename = parsed_url.path.split('/')[-1] or "s3_file.pdf"
            return url, filename
        
        # Generic URLs
        else:
            filename = parsed_url.path.split('/')[-1] or "downloaded_file"
            
            # Ensure extension
            if '.' not in filename:
                filename += '.pdf'
            
            return url, filename

# ================================
# GLOBAL INSTANCES (FIXED)
# ================================

# Initialize global instances
REDIS_CACHE = RedisCache()
EMBEDDING_SERVICE = OptimizedEmbeddingService()
DECISION_ENGINE = UniversalDecisionEngine()
DOMAIN_DETECTOR = UniversalDomainDetector()

# ================================
# FASTAPI APPLICATION (FIXED)
# ================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FIXED: Application lifespan manager"""
    logger.info("🚀 Starting Enhanced Universal RAG System...")
    
    try:
        # Initialize components
        await initialize_components()
        yield
    finally:
        # Cleanup on shutdown
        logger.info("🛑 Shutting down...")
        try:
            cleanup_tasks = []
            for session_obj in ACTIVE_SESSIONS.values():
                cleanup_tasks.append(session_obj.get_data().cleanup())
            
            if cleanup_tasks:
                await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            
            # Clear caches
            EMBEDDING_CACHE.clear()
            RESPONSE_CACHE.clear()
            
            logger.info("✅ Cleanup completed")
        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}")

# Create FastAPI app
app = FastAPI(
    title="Enhanced Universal RAG System",
    description="Production-ready RAG system with universal document support, intelligent chunking, and multi-domain processing",
    version="3.0.0",
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
# PYDANTIC MODELS (UPDATED)
# ================================

class HackRxRunRequest(BaseModel):
    """FIXED: Request model for HackRx run endpoint"""
    documents: str = Field(..., description="Document URL to process")
    questions: List[str] = Field(..., description="List of questions to ask")

class ProcessDocumentsRequest(BaseModel):
    """Request model for document processing"""
    file_urls: List[HttpUrl] = Field(..., description="List of file URLs to process")
    domain_override: Optional[str] = Field(None, description="Override domain detection")
    session_id: Optional[str] = Field(None, description="Reuse existing session")

class QueryRequest(BaseModel):
    """Request model for queries"""
    query: str = Field(..., description="Question to ask")
    session_id: str = Field(..., description="Session ID from document processing")
    query_type: Optional[str] = Field("general", description="Type of query")

# Response models
class HealthResponse(BaseModel):
    status: str
    timestamp: str
    components: Dict[str, str]
    memory_usage: Dict[str, str]
    active_sessions: int

# ================================
# API ENDPOINTS (FIXED)
# ================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Enhanced Universal RAG System",
        "version": "3.0.0",
        "status": "operational",
        "features": "universal-documents,intelligent-chunking,multi-domain,confidence-fallbacks",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """FIXED: Comprehensive health check"""
    try:
        # Check component status
        components = {}
        for comp, ready in components_ready.items():
            components[comp] = "ready" if ready else "loading"
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_usage = {
            "total_gb": f"{memory.total / (1024**3):.1f}",
            "available_gb": f"{memory.available / (1024**3):.1f}",
            "used_percent": f"{memory.percent:.1f}%"
        }
        
        # Overall status
        critical_ready = components_ready.get("openai_client", False)
        status = "healthy" if critical_ready else "initializing"
        
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

# ================================
# MAIN HACKRX ENDPOINT (FIXED)
# ================================

@app.post("/hackrx/run", dependencies=[Depends(verify_token)])
async def hackrx_run(request: HackRxRunRequest):
    """FIXED: HackRx endpoint with proper response format"""
    start_time = time.time()
    temp_files = []
    
    try:
        # Ensure components are ready
        await ensure_components_ready()
        
        if not request.documents.strip():
            raise HTTPException(status_code=400, detail="Document URL cannot be empty")
        
        if not request.questions:
            raise HTTPException(status_code=400, detail="No questions provided")
        
        logger.info(f"🚀 HackRx: Processing document and {len(request.questions)} questions")
        
        # Download document
        downloader = UniversalURLDownloader()
        try:
            content, filename = await downloader.download_from_url(request.documents)
            
            # Save to temporary file
            temp_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=os.path.splitext(filename)[1] or '.pdf'
            )
            temp_file.write(content)
            temp_file.close()
            temp_files.append(temp_file.name)
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to download document: {str(e)}")
        
        # Create RAG session
        rag_session = EnhancedRAGSystem()
        
        try:
            processing_result = await rag_session.process_documents([temp_file.name])
            logger.info(f"✅ Document processed: {processing_result['total_chunks']} chunks, domain: {processing_result['domain']}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")
        
        # Process questions in controlled parallel batches
        async def process_single_question(question: str) -> str:
            """Process a single question and return answer"""
            try:
                # Retrieve relevant documents
                retrieved_docs, similarity_scores = await rag_session.enhanced_retrieve_and_rerank(
                    question, 
                    top_k=rag_session.domain_config["context_docs"]
                )
                
                # Process with decision engine
                result = await DECISION_ENGINE.process_query_with_fallback(
                    query=question,
                    retrieved_docs=retrieved_docs,
                    similarity_scores=similarity_scores,
                    domain=processing_result['domain'],
                    domain_confidence=processing_result['domain_confidence'],
                    query_type="hackrx",
                    rag_system=rag_session
                )
                
                return result["answer"]
                
            except Exception as e:
                logger.error(f"❌ Question processing error: {e}")
                return f"Error processing question: {str(e)}"
        
        # Process questions with controlled concurrency
        semaphore = asyncio.Semaphore(3)  # Limit to 3 concurrent questions
        
        async def process_with_semaphore(question: str) -> str:
            async with semaphore:
                return await process_single_question(question)
        
        # Execute all questions
        logger.info(f"🔄 Processing {len(request.questions)} questions...")
        
        try:
            # Process questions in parallel with timeout
            question_tasks = [process_with_semaphore(q) for q in request.questions]
            answers = await asyncio.wait_for(
                asyncio.gather(*question_tasks),
                timeout=25.0  # 25 second timeout for all questions
            )
            
        except asyncio.TimeoutError:
            logger.error("❌ Questions processing timed out")
            raise HTTPException(status_code=408, detail="Processing timeout - questions took too long")
        
        except Exception as e:
            logger.error(f"❌ Error processing questions: {e}")
            raise HTTPException(status_code=500, detail=f"Question processing failed: {str(e)}")
        
        # Cleanup session
        await rag_session.cleanup()
        
        processing_time = time.time() - start_time
        logger.info(f"✅ HackRx completed in {processing_time:.2f}s")
        
        # CRITICAL: Return in the exact format expected by the platform
        return {"answers": answers}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ HackRx endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"System error: {str(e)}")
    
    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception as e:
                logger.warning(f"⚠️ Failed to delete temp file {temp_file}: {e}")

# ================================
# ADDITIONAL ENDPOINTS (OPTIONAL)
# ================================

@app.post("/process-documents")
async def process_documents_endpoint(request: ProcessDocumentsRequest):
    """Process documents endpoint"""
    try:
        await ensure_components_ready()
        
        # Download files
        temp_files = []
        downloader = UniversalURLDownloader()
        
        for url in request.file_urls:
            try:
                content, filename = await downloader.download_from_url(str(url))
                temp_file = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=os.path.splitext(filename)[1] or '.pdf'
                )
                temp_file.write(content)
                temp_file.close()
                temp_files.append(temp_file.name)
            except Exception as e:
                logger.warning(f"⚠️ Failed to download {url}: {e}")
        
        if not temp_files:
            raise HTTPException(status_code=400, detail="No files could be downloaded")
        
        # Process documents
        if request.session_id:
            document_hash = request.session_id
        else:
            # Generate hash from URLs
            urls_str = "".join([str(url) for url in request.file_urls])
            document_hash = hashlib.md5(urls_str.encode()).hexdigest()[:16]
        
        rag_session = await EnhancedSessionManager.get_or_create_session(document_hash)
        
        try:
            result = await rag_session.process_documents(temp_files)
            return result
        finally:
            # Clean up temp files
            for temp_file in temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
                except Exception:
                    pass
        
    except Exception as e:
        logger.error(f"❌ Process documents error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    """Query documents endpoint"""
    try:
        await ensure_components_ready()
        
        if request.session_id not in ACTIVE_SESSIONS:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session_obj = ACTIVE_SESSIONS[request.session_id]
        rag_session = session_obj.get_data()
        
        # Retrieve documents
        retrieved_docs, similarity_scores = await rag_session.enhanced_retrieve_and_rerank(
            request.query,
            top_k=rag_session.domain_config["context_docs"]
        )
        
        # Process query
        result = await DECISION_ENGINE.process_query_with_fallback(
            query=request.query,
            retrieved_docs=retrieved_docs,
            similarity_scores=similarity_scores,
            domain=rag_session.domain,
            domain_confidence=0.8,
            query_type=request.query_type,
            rag_system=rag_session
        )
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Query error: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.get("/sessions")
async def list_sessions():
    """List active sessions"""
    sessions = []
    for session_id, session_obj in ACTIVE_SESSIONS.items():
        sessions.append({
            "session_id": session_id,
            "created_at": session_obj.created_at,
            "last_accessed": session_obj.last_accessed,
            "expires_in": session_obj.ttl - (time.time() - session_obj.last_accessed)
        })
    
    return {"active_sessions": len(sessions), "sessions": sessions}

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a specific session"""
    if session_id in ACTIVE_SESSIONS:
        session_obj = ACTIVE_SESSIONS.pop(session_id)
        await session_obj.get_data().cleanup()
        return {"message": f"Session {session_id} deleted"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")

@app.get("/status")
async def system_status():
    """Detailed system status"""
    try:
        memory = psutil.virtual_memory()
        
        return {
            "system": "Enhanced Universal RAG System",
            "version": "3.0.0",
            "status": "operational",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "openai_client": components_ready.get("openai_client", False),
                "embedding_model": components_ready.get("embedding_model", False),
                "base_sentence_model": components_ready.get("base_sentence_model", False),
                "reranker": components_ready.get("reranker", False),
                "pinecone": components_ready.get("pinecone", False),
                "redis": components_ready.get("redis", False)
            },
            "performance": {
                "active_sessions": len(ACTIVE_SESSIONS),
                "embedding_cache_size": len(EMBEDDING_CACHE),
                "response_cache_size": len(RESPONSE_CACHE),
                "memory_usage_percent": memory.percent,
                "available_memory_gb": memory.available / (1024**3)
            },
            "configuration": {
                "max_sessions": 100,
                "session_ttl_minutes": SESSION_TTL // 60,
                "supported_domains": list(DOMAIN_CONFIGS.keys()),
                "supported_file_types": ["pdf", "docx", "doc", "txt", "md", "csv"]
            }
        }
    except Exception as e:
        return {"error": str(e), "status": "error"}

# ================================
# ERROR HANDLERS
# ================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    logger.error(f"HTTP {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    error_id = str(uuid.uuid4())[:8]
    logger.error(f"Unhandled exception {error_id}: {str(exc)}\n{traceback.format_exc()}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "error_id": error_id,
            "message": "Please contact support with this error ID"
        }
    )

# ================================
# DEVELOPMENT SERVER
# ================================

if __name__ == "__main__":
    import uvicorn
    
    # Configuration for development
    config = {
        "host": "0.0.0.0",
        "port": int(os.getenv("PORT", 8000)),
        "workers": 1,  # Single worker for development
        "loop": "asyncio",
        "log_level": "info",
        "access_log": True,
        "reload": os.getenv("RELOAD", "false").lower() == "true"
    }
    
    logger.info(f"🚀 Starting Enhanced Universal RAG System on {config['host']}:{config['port']}")
    
    try:
        uvicorn.run("__main__:app", **config)
    except KeyboardInterrupt:
        logger.info("👋 Server shutdown requested")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
