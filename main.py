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
import random

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
# ENHANCED API KEY MANAGEMENT WITH ROTATION
# ================================

HACKRX_TOKEN = "9a1163c13e8927960b857a674794a62c57baf588998981151b0753a4d6d17905"

# Enhanced Gemini API Key Pool with Rotation
GEMINI_API_KEY_POOL = [
    'AIzaSyAv1KkRE-xS_HXylwhRAqz8ky1zRGsc3Jg',
    'AIzaSyDXNgHVcZuCJrs-qzucydlTdO7hX-BgV8Y',
    'AIzaSyDWFKzuKaGmKqSYlqjQTbEtrnGsx4SJ9lo',
    # Add more API keys here for better load distribution
]

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Dynamic Gemini Model Selection
GEMINI_MODELS = {
    "simple": "gemini-2.0-flash-exp",  # Fast, cost-effective
    "complex": "gemini-2.0-flash",     # High accuracy
}

class APIKeyRotator:
    """Smart API key rotation with failure tracking"""
    
    def __init__(self, api_keys: List[str]):
        self.api_keys = api_keys
        self.current_index = 0
        self.failure_counts = {key: 0 for key in api_keys}
        self.last_used = {key: 0 for key in api_keys}
        self.lock = threading.Lock()
        self.max_failures = 3
        self.cooldown_period = 300  # 5 minutes
    
    def get_next_key(self) -> str:
        """Get next available API key with smart rotation"""
        with self.lock:
            current_time = time.time()
            
            # Find available keys (not in cooldown)
            available_keys = []
            for key in self.api_keys:
                if (self.failure_counts[key] < self.max_failures or 
                    current_time - self.last_used[key] > self.cooldown_period):
                    available_keys.append(key)
            
            if not available_keys:
                # Reset all keys if none available
                self.failure_counts = {key: 0 for key in self.api_keys}
                available_keys = self.api_keys
            
            # Select key with least recent usage
            selected_key = min(available_keys, key=lambda k: self.last_used[k])
            self.last_used[selected_key] = current_time
            
            return selected_key
    
    def report_failure(self, api_key: str):
        """Report API key failure"""
        with self.lock:
            self.failure_counts[api_key] += 1
            logger.warning(f"⚠️ API key failure reported. Count: {self.failure_counts[api_key]}")
    
    def report_success(self, api_key: str):
        """Report API key success"""
        with self.lock:
            self.failure_counts[api_key] = max(0, self.failure_counts[api_key] - 1)

# Global API key rotator
api_key_rotator = APIKeyRotator(GEMINI_API_KEY_POOL)

def get_gemini_client_with_rotation(fallback_ok=True):
    """Get Gemini client with smart key rotation"""
    api_key = api_key_rotator.get_next_key()
    
    try:
        return AsyncOpenAI(
            api_key=api_key,
            base_url=GEMINI_BASE_URL,
            timeout=httpx.Timeout(connect=15.0, read=120.0, write=30.0, pool=10.0),
            max_retries=2
        ), api_key
    except Exception as e:
        api_key_rotator.report_failure(api_key)
        if fallback_ok and len(GEMINI_API_KEY_POOL) > 1:
            # Try next key
            backup_key = api_key_rotator.get_next_key()
            return AsyncOpenAI(
                api_key=backup_key,
                base_url=GEMINI_BASE_URL,
                timeout=httpx.Timeout(connect=15.0, read=120.0, write=30.0, pool=10.0),
                max_retries=2
            ), backup_key
        raise

def select_gemini_model(complexity: float) -> str:
    """Select appropriate Gemini model based on query complexity"""
    return GEMINI_MODELS["simple"] if complexity < 0.3 else GEMINI_MODELS["complex"]

# ================================
# ENHANCED RATE LIMITING SYSTEM
# ================================

class AdaptiveRateLimiter:
    """Adaptive rate limiting with backoff strategies"""
    
    def __init__(self):
        self.request_counts = defaultdict(int)
        self.request_times = defaultdict(list)
        self.backoff_delays = defaultdict(float)
        self.lock = threading.Lock()
        
        # Rate limiting configuration
        self.max_requests_per_minute = 60
        self.max_requests_per_second = 2
        self.base_backoff = 1.0
        self.max_backoff = 60.0
        self.backoff_multiplier = 2.0
    
    async def wait_if_needed(self, key: str = "default"):
        """Wait if rate limit is exceeded"""
        with self.lock:
            current_time = time.time()
            
            # Clean old requests (older than 1 minute)
            if key in self.request_times:
                self.request_times[key] = [
                    t for t in self.request_times[key] 
                    if current_time - t < 60
                ]
            
            # Check per-minute limit
            if len(self.request_times[key]) >= self.max_requests_per_minute:
                wait_time = 60 - (current_time - self.request_times[key][0])
                if wait_time > 0:
                    logger.info(f"⏳ Rate limit: waiting {wait_time:.2f}s (per-minute limit)")
                    await asyncio.sleep(wait_time)
            
            # Check per-second limit
            recent_requests = [
                t for t in self.request_times[key] 
                if current_time - t < 1
            ]
            if len(recent_requests) >= self.max_requests_per_second:
                wait_time = 1.0
                logger.info(f"⏳ Rate limit: waiting {wait_time:.2f}s (per-second limit)")
                await asyncio.sleep(wait_time)
            
            # Apply backoff if needed
            if key in self.backoff_delays and self.backoff_delays[key] > 0:
                wait_time = self.backoff_delays[key]
                logger.info(f"⏳ Backoff: waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
                # Reduce backoff after successful wait
                self.backoff_delays[key] = max(0, self.backoff_delays[key] / 2)
            
            # Record this request
            self.request_times[key].append(current_time)
    
    def trigger_backoff(self, key: str = "default"):
        """Trigger exponential backoff for a key"""
        with self.lock:
            current_backoff = self.backoff_delays.get(key, 0)
            if current_backoff == 0:
                self.backoff_delays[key] = self.base_backoff
            else:
                self.backoff_delays[key] = min(
                    self.max_backoff, 
                    current_backoff * self.backoff_multiplier
                )
            logger.warning(f"⚠️ Triggered backoff for {key}: {self.backoff_delays[key]:.2f}s")

# Global rate limiter
rate_limiter = AdaptiveRateLimiter()

# ================================
# PARALLEL PROCESSING WITH CONTROLLED CONCURRENCY
# ================================

class ConcurrencyController:
    """Control concurrent operations with semaphores"""
    
    def __init__(self):
        self.embedding_semaphore = asyncio.Semaphore(4)  # Max 4 concurrent embedding operations
        self.llm_semaphore = asyncio.Semaphore(2)        # Max 2 concurrent LLM calls
        self.processing_semaphore = asyncio.Semaphore(3)  # Max 3 concurrent document processing
        
    async def embedding_operation(self, coro):
        """Execute embedding operation with concurrency control"""
        async with self.embedding_semaphore:
            return await coro
    
    async def llm_operation(self, coro):
        """Execute LLM operation with concurrency control"""
        async with self.llm_semaphore:
            return await coro
    
    async def processing_operation(self, coro):
        """Execute processing operation with concurrency control"""
        async with self.processing_semaphore:
            return await coro

# Global concurrency controller
concurrency_controller = ConcurrencyController()

# ================================
# ENHANCED RETRY LOGIC WITH SMART BATCHING
# ================================

class SmartRetryManager:
    """Smart retry manager with exponential backoff and batch optimization"""
    
    def __init__(self):
        self.max_retries = 3
        self.base_delay = 1.0
        self.max_delay = 30.0
        self.exponential_base = 2.0
        self.jitter_range = 0.1
    
    async def retry_with_backoff(self, operation, *args, **kwargs):
        """Retry operation with exponential backoff and jitter"""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                # Apply rate limiting
                await rate_limiter.wait_if_needed()
                
                return await operation(*args, **kwargs)
                
            except Exception as e:
                last_exception = e
                
                if attempt == self.max_retries:
                    logger.error(f"❌ All {self.max_retries + 1} attempts failed: {e}")
                    break
                
                # Calculate delay with exponential backoff and jitter
                delay = min(
                    self.max_delay,
                    self.base_delay * (self.exponential_base ** attempt)
                )
                jitter = random.uniform(-self.jitter_range, self.jitter_range) * delay
                total_delay = delay + jitter
                
                logger.warning(f"⚠️ Attempt {attempt + 1} failed: {e}. Retrying in {total_delay:.2f}s")
                
                # Trigger rate limiter backoff on certain errors
                if "rate" in str(e).lower() or "quota" in str(e).lower():
                    rate_limiter.trigger_backoff()
                
                await asyncio.sleep(total_delay)
        
        raise last_exception

# Global retry manager
retry_manager = SmartRetryManager()

# ================================
# QUESTION BATCHING SYSTEM
# ================================

class QuestionBatcher:
    """Smart question batching with optimized processing"""
    
    def __init__(self):
        self.max_batch_size = 5
        self.batch_timeout = 2.0  # seconds
        
    async def process_questions_batch(self, questions: List[str], rag_system) -> List[Dict[str, Any]]:
        """Process multiple questions with intelligent batching"""
        if len(questions) <= 1:
            # Single question - no batching needed
            if questions:
                return [await rag_system.query(questions[0])]
            return []
        
        logger.info(f"📦 Processing {len(questions)} questions in optimized batches")
        
        # Group questions by complexity for better batching
        simple_questions = []
        complex_questions = []
        
        for question in questions:
            analysis = rag_system.complexity_analyzer.analyze_query_complexity(question)
            if analysis['complexity'] < 0.3:
                simple_questions.append(question)
            else:
                complex_questions.append(question)
        
        # Process simple questions first (faster)
        results = []
        
        # Process simple questions in larger batches
        if simple_questions:
            simple_results = await self._process_batch_parallel(
                simple_questions, rag_system, max_concurrent=3
            )
            results.extend(simple_results)
        
        # Process complex questions in smaller batches
        if complex_questions:
            complex_results = await self._process_batch_parallel(
                complex_questions, rag_system, max_concurrent=2
            )
            results.extend(complex_results)
        
        # Restore original order
        question_to_result = {r['query']: r for r in results}
        ordered_results = [question_to_result[q] for q in questions]
        
        return ordered_results
    
    async def _process_batch_parallel(self, questions: List[str], rag_system, max_concurrent: int) -> List[Dict[str, Any]]:
        """Process batch of questions with controlled parallelism"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single_question(question: str):
            async with semaphore:
                return await concurrency_controller.llm_operation(
                    rag_system.query(question)
                )
        
        # Create tasks for all questions
        tasks = [process_single_question(q) for q in questions]
        
        # Process with timeout protection
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=60.0  # 1 minute timeout for batch
            )
            
            # Handle any exceptions in results
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"❌ Question {i} failed: {result}")
                    processed_results.append({
                        'query': questions[i],
                        'answer': f"Processing failed: {str(result)}",
                        'confidence': 0.0,
                        'domain': 'unknown',
                        'processing_time': 0.0,
                        'error': True
                    })
                else:
                    processed_results.append(result)
            
            return processed_results
            
        except asyncio.TimeoutError:
            logger.error("❌ Batch processing timeout")
            return [
                {
                    'query': q,
                    'answer': "Processing timeout - please try again",
                    'confidence': 0.0,
                    'domain': 'unknown',
                    'processing_time': 0.0,
                    'timeout': True
                }
                for q in questions
            ]

# Global question batcher
question_batcher = QuestionBatcher()

# ================================
# ENHANCED CONFIGURATION WITH INCREASED TIMEOUTS
# ================================

# Enhanced timeouts for complex legal queries
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Improved chunking parameters
CHUNK_SIZE = 900
CHUNK_OVERLAP = 100
SEMANTIC_SEARCH_K = 5
CONTEXT_DOCS = 4
CONFIDENCE_THRESHOLD = 0.15

# Enhanced timeout configuration
BASE_QUESTION_TIMEOUT = 10.0  # Increased from 5.0
COMPLEX_QUESTION_TIMEOUT = 20.0  # For complex legal queries
BATCH_PROCESSING_TIMEOUT = 60.0  # For batch operations
EMBEDDING_TIMEOUT = 90.0  # Increased from 60.0
DOCUMENT_PROCESSING_TIMEOUT = 300.0  # 5 minutes for large documents

# Dynamic reranking parameters
BASE_RERANK_TOP_K = 8
MAX_RERANK_TOP_K = 16

# Token budget management
MAX_CONTEXT_TOKENS = 8000
TOKEN_SAFETY_MARGIN = 300
MAX_FILE_SIZE_MB = 50

# Parallel processing optimization
OPTIMAL_BATCH_SIZE = 32
MAX_PARALLEL_BATCHES = 4

# Supported formats
SUPPORTED_EXTENSIONS = ['.pdf', '.docx', '.doc', '.txt', '.md', '.csv']
SUPPORTED_URL_SCHEMES = ['http', 'https', 'blob', 'drive', 'dropbox']

# Domain detection keywords (unchanged)
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

# Global model state
_models_loaded = False
_model_lock = asyncio.Lock()
_startup_complete = False

# Cache for document processing
_document_cache = {}
_cache_ttl = 1800

# Global models
base_sentence_model = None
reranker = None

# ================================
# ENHANCED QUERY ANALYSIS WITH COMPLEXITY SCORING
# ================================

class QueryComplexityAnalyzer:
    """Enhanced query complexity analyzer with token-aware scoring"""
    
    def __init__(self):
        self.analytical_keywords = [
            'analyze', 'compare', 'contrast', 'evaluate', 'assess', 'why',
            'how does', 'what causes', 'relationship', 'impact', 'effect',
            'trends', 'patterns', 'implications', 'significance', 'explain',
            'describe', 'discuss', 'elaborate', 'detail'
        ]
        
        self.simple_patterns = [
            r'^what is\s+\w+',
            r'^define\s+\w+',
            r'^who is\s+\w+',
            r'^\w+\s+means?$'
        ]
        
        # Initialize tokenizer
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.tokenizer = None
    
    def analyze_query_complexity(self, query: str) -> Dict[str, Any]:
        """Enhanced query complexity analysis with timeout consideration"""
        query_lower = query.lower().strip()
        
        # Pattern-based complexity detection
        is_simple_pattern = any(re.match(pattern, query_lower) for pattern in self.simple_patterns)
        is_analytical = any(keyword in query_lower for keyword in self.analytical_keywords)
        
        # Token-based complexity scoring
        token_count = self._count_tokens(query)
        word_count = len(query.split())
        
        # Calculate complexity score
        complexity_factors = {
            'analytical_keywords': 0.3 if is_analytical else 0.0,
            'token_length': min(0.3, token_count / 100),
            'word_length': min(0.2, word_count / 20),
            'question_marks': min(0.1, query.count('?') * 0.05),
            'pattern_penalty': -0.2 if is_simple_pattern else 0.0
        }
        
        complexity_score = sum(complexity_factors.values())
        complexity_score = max(0.0, min(1.0, complexity_score))
        
        # Determine appropriate timeout based on complexity
        if complexity_score > 0.7:
            timeout = COMPLEX_QUESTION_TIMEOUT
        else:
            timeout = BASE_QUESTION_TIMEOUT
        
        # Determine query type
        if is_simple_pattern and complexity_score < 0.3:
            query_type = 'simple'
        elif is_analytical or complexity_score > 0.6:
            query_type = 'analytical'
        else:
            query_type = 'factual'
        
        return {
            'type': query_type,
            'complexity': complexity_score,
            'token_count': token_count,
            'word_count': word_count,
            'is_longform': token_count > 50 or word_count > 15,
            'requires_deep_context': complexity_score > 0.6,
            'recommended_timeout': timeout
        }
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens with fallback"""
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception:
                pass
        return max(1, int(len(text) / 3.8))

# ================================
# ENHANCED CACHING SYSTEM
# ================================

class SmartCacheManager:
    """Smart cache manager with TTL/LRU and thread safety"""
    
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
            logger.info("📦 Using dict fallback caching")
        
        self._lock = threading.RLock()
    
    def clear_all_caches(self):
        """Clear all caches for new documents"""
        with self._lock:
            self.embedding_cache.clear()
            self.document_chunk_cache.clear()
            self.domain_cache.clear()
            logger.info("🧹 All caches cleared")
    
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
    
    def get_cache_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "embedding_cache_size": len(self.embedding_cache),
                "document_chunk_cache_size": len(self.document_chunk_cache),
                "domain_cache_size": len(self.domain_cache),
                "primary_cache_available": self.primary_available,
                "cache_type": "TTLCache/LRUCache" if self.primary_available else "dict_fallback"
            }

# Additional utility classes (keeping the same structure but with enhanced timeout handling)

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

class AdaptiveChunkLimitCalculator:
    """Calculate dynamic chunk limits based on query complexity and domain"""
    
    @staticmethod
    def calculate_chunk_limit(domain: str, complexity: float, query_analysis: Dict[str, Any]) -> int:
        """Calculate adaptive chunk limit with timeout consideration"""
        domain_base_limits = {
            "legal": 150,
            "academic": 140,
            "medical": 130,
            "technical": 120,
            "insurance": 110,
            "financial": 110,
            "business": 100,
            "general": 100
        }
        
        base_limit = domain_base_limits.get(domain, 100)
        
        if complexity > 0.7:
            multiplier = 1.5
        elif complexity > 0.5:
            multiplier = 1.25
        elif complexity < 0.3:
            multiplier = 0.8
        else:
            multiplier = 1.0
        
        if query_analysis.get('is_longform', False):
            multiplier *= 1.2
        
        calculated_limit = int(base_limit * multiplier)
        final_limit = max(50, min(250, calculated_limit))
        
        logger.info(f"📊 Adaptive chunk limit: {final_limit} (domain: {domain}, complexity: {complexity:.2f})")
        return final_limit

class TokenAwareContextProcessor:
    """Token-aware context processor with budget management"""
    
    def __init__(self):
        self.max_context_tokens = MAX_CONTEXT_TOKENS
        self.safety_margin = TOKEN_SAFETY_MARGIN
        self.available_tokens = self.max_context_tokens - self.safety_margin
        
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.tokenizer = None
    
    def select_context_with_budget(self, documents: List[Document], query: str, complexity: float) -> str:
        """Select optimal context within token budget"""
        if not documents:
            return ""
        
        # Adjust parameters based on complexity
        if complexity > 0.7:
            max_docs = 6
            priority_boost = 1.3
        elif complexity > 0.5:
            max_docs = 5
            priority_boost = 1.1
        else:
            max_docs = 4
            priority_boost = 1.0
        
        # Score and rank documents
        scored_docs = []
        query_lower = query.lower()
        
        for i, doc in enumerate(documents[:max_docs * 2]):
            content = doc.page_content
            
            base_score = 1.0 / (i + 1)
            query_matches = sum(1 for word in query.split() if word.lower() in content.lower())
            match_score = min(0.5, query_matches * 0.1)
            length_score = min(0.2, len(content) / 5000)
            
            total_score = (base_score + match_score + length_score) * priority_boost
            scored_docs.append((doc, total_score))
        
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Select within token budget
        selected_parts = []
        current_tokens = 0
        
        for doc, score in scored_docs[:max_docs]:
            content = doc.page_content
            content_tokens = self._estimate_tokens(content)
            
            if current_tokens + content_tokens <= self.available_tokens:
                selected_parts.append(content)
                current_tokens += content_tokens
            else:
                remaining_tokens = self.available_tokens - current_tokens
                if remaining_tokens > 100:
                    partial_content = self._truncate_to_tokens(content, remaining_tokens)
                    if partial_content:
                        selected_parts.append(partial_content + "...")
                break
        
        context = "\n\n".join(selected_parts)
        final_tokens = self._estimate_tokens(context)
        
        logger.info(f"🎯 Context selected: {final_tokens}/{self.max_context_tokens} tokens ({len(selected_parts)} chunks)")
        return context
    
    def _estimate_tokens(self, text: str) -> int:
        if not text:
            return 0
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception:
                pass
        return max(1, int(len(text) / 3.5))
    
    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        if not text or max_tokens <= 0:
            return ""
        if self.tokenizer:
            try:
                tokens = self.tokenizer.encode(text)
                if len(tokens) <= max_tokens:
                    return text
                truncated_tokens = tokens[:max_tokens]
                return self.tokenizer.decode(truncated_tokens)
            except Exception:
                pass
        estimated_chars = int(max_tokens * 3.5)
        return text[:estimated_chars] if len(text) > estimated_chars else text

class AdaptiveReranker:
    """Context-aware reranking with dynamic parameters"""
    
    @staticmethod
    def calculate_rerank_params(complexity: float, query_analysis: Dict[str, Any]) -> Dict[str, int]:
        """Calculate adaptive reranking parameters"""
        if complexity > 0.7 or query_analysis.get('is_longform', False):
            rerank_top_k = MAX_RERANK_TOP_K
            context_docs = 6
        elif complexity > 0.5:
            rerank_top_k = 12
            context_docs = 5
        elif query_analysis.get('type') == 'analytical':
            rerank_top_k = 10
            context_docs = 5
        else:
            rerank_top_k = BASE_RERANK_TOP_K
            context_docs = 4
        
        return {
            'rerank_top_k': rerank_top_k,
            'context_docs': context_docs
        }

# ================================
# CONTINUE WITH REMAINING CLASSES
# ================================

class AdaptiveTextSplitter:
    """Enhanced adaptive text splitter with balanced chunking"""
    
    def __init__(self):
        self.separators = [
            "\n\n## ", "\n\n# ", "\n\n### ", "\n\n#### ",
            "\n\n", "\n", ". ", "! ", "? ", "; ", " ", ""
        ]
    
    def split_documents(self, documents: List[Document], detected_domain: str = "general") -> List[Document]:
        if not documents:
            return []
        
        content_hash = self._calculate_content_hash(documents)
        cache_key = f"chunks_{content_hash}_{detected_domain}_v3"
        
        cached_chunks = CACHE_MANAGER.get_document_chunks(cache_key)
        if cached_chunks is not None:
            logger.info(f"📄 Using cached chunks: {len(cached_chunks)} chunks")
            return cached_chunks
        
        chunk_size, chunk_overlap = self._get_balanced_chunk_params(detected_domain)
        all_chunks = []
        
        for doc in documents:
            try:
                chunks = self._split_document_balanced(doc, chunk_size, chunk_overlap)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.warning(f"⚠️ Error splitting document: {e}")
                chunks = self._simple_split(doc, chunk_size, chunk_overlap)
                all_chunks.extend(chunks)
        
        all_chunks = [chunk for chunk in all_chunks if len(chunk.page_content.strip()) >= 100]
        CACHE_MANAGER.set_document_chunks(cache_key, all_chunks)
        
        logger.info(f"📄 Created {len(all_chunks)} balanced chunks (size: {chunk_size}, overlap: {chunk_overlap})")
        return all_chunks
    
    def _get_balanced_chunk_params(self, detected_domain: str) -> Tuple[int, int]:
        domain_adjustments = {
            "legal": 1.1,
            "academic": 1.05,
            "medical": 1.0,
            "technical": 0.95,
            "insurance": 1.0,
            "financial": 1.0,
            "business": 0.9,
            "general": 1.0
        }
        
        adjustment = domain_adjustments.get(detected_domain, 1.0)
        adjusted_size = int(CHUNK_SIZE * adjustment)
        adjusted_overlap = CHUNK_OVERLAP
        adjusted_size = max(600, min(1200, adjusted_size))
        
        return adjusted_size, adjusted_overlap
    
    def _split_document_balanced(self, document: Document, chunk_size: int, chunk_overlap: int) -> List[Document]:
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
                "chunk_type": "balanced_split",
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap
            })
        
        return chunks
    
    def _simple_split(self, document: Document, chunk_size: int, chunk_overlap: int) -> List[Document]:
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            return splitter.split_documents([document])
        except Exception as e:
            logger.error(f"❌ Even simple splitting failed: {e}")
            return [document]
    
    def _calculate_content_hash(self, documents: List[Document]) -> str:
        content_sample = "".join([doc.page_content[:100] for doc in documents[:5]])
        return hashlib.sha256(content_sample.encode()).hexdigest()[:16]

class UnifiedLoader:
    """Unified document loader with enhanced timeout handling"""
    
    def __init__(self):
        self.mime_detector = magic if HAS_MAGIC else None
        self.google_patterns = [
            r'drive\.google\.com/file/d/([a-zA-Z0-9-_]+)',
            r'docs\.google\.com/document/d/([a-zA-Z0-9-_]+)',
            r'docs\.google\.com/spreadsheets/d/([a-zA-Z0-9-_]+)',
        ]
        self.dropbox_patterns = [
            r'dropbox\.com/s/([a-zA-Z0-9]+)',
            r'dropbox\.com/sh/([a-zA-Z0-9]+)',
            r'dropbox\.com/scl/fi/([a-zA-Z0-9-_]+)',
        ]
    
    async def load_document(self, source: str) -> List[Document]:
        """Universal document loader with enhanced timeout"""
        try:
            if self._is_url(source):
                docs = await asyncio.wait_for(
                    self._load_from_url(source),
                    timeout=DOCUMENT_PROCESSING_TIMEOUT
                )
            else:
                docs = await asyncio.wait_for(
                    self._load_from_file(source),
                    timeout=DOCUMENT_PROCESSING_TIMEOUT
                )
            
            for doc in docs:
                doc.metadata.update({
                    'source': source,
                    'load_time': time.time(),
                    'loader_version': '3.0'
                })
            
            logger.info(f"✅ Loaded {len(docs)} documents from {sanitize_pii(source)}")
            return docs
            
        except asyncio.TimeoutError:
            logger.error(f"❌ Document loading timeout for {sanitize_pii(source)}")
            raise HTTPException(status_code=408, detail="Document loading timeout")
        except Exception as e:
            logger.error(f"❌ Failed to load {sanitize_pii(source)}: {e}")
            raise
    
    def _is_url(self, source: str) -> bool:
        return source.startswith(('http://', 'https://', 'blob:', 'drive:', 'dropbox:'))
    
    async def _load_from_url(self, url: str) -> List[Document]:
        """Enhanced URL loading with retry and timeout"""
        parsed = urlparse(url)
        scheme = parsed.scheme.lower()
        
        if scheme in ["drive", "dropbox"]:
            if scheme == "drive":
                url = url.replace("drive:", "https://")
            elif scheme == "dropbox":
                url = url.replace("dropbox:", "https://")
        
        if not validate_url_scheme(url):
            raise ValueError(f"Unsupported URL scheme: {scheme}")
        
        download_url = self._transform_special_url(url)
        
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/117.0 Safari/537.36"
            )
        }
        
        # Use retry manager for robust downloading
        async def download_operation():
            timeout = httpx.Timeout(
                timeout=180.0,  # Increased timeout
                connect=30.0,
                read=180.0,
                write=30.0,
                pool=10.0
            )
            
            async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
                response = await client.get(download_url, follow_redirects=True)
                response.raise_for_status()
                return response.content
        
        content = await retry_manager.retry_with_backoff(download_operation)
        
        file_ext = (
            self._get_extension_from_url(url) or 
            self._detect_extension_from_content(content)
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
        for pattern in self.google_patterns:
            match = re.search(pattern, url)
            if match:
                file_id = match.group(1)
                return f"https://drive.google.com/uc?export=download&id={file_id}"
        
        # Dropbox transformation
        for pattern in self.dropbox_patterns:
            if re.search(pattern, url):
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
        """Load document from file with enhanced error handling"""
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
        
        # PDF loading
        if mime_type == 'application/pdf' or file_extension == '.pdf':
            try:
                loader = PyMuPDFLoader(file_path)
                docs = await asyncio.to_thread(loader.load)
                loader_used = "PyMuPDFLoader"
            except Exception as e:
                logger.warning(f"⚠️ PyMuPDF failed: {e}")
        
        # Word document loading
        elif ('word' in (mime_type or '') or 
              'officedocument' in (mime_type or '') or 
              file_extension in ['.docx', '.doc']):
            try:
                loader = Docx2txtLoader(file_path)
                docs = await asyncio.to_thread(loader.load)
                loader_used = "Docx2txtLoader"
            except Exception as e:
                logger.warning(f"⚠️ DOCX loader failed: {e}")
        
        # Text file loading
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
        
        # Fallback loading
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

class OptimizedFAISSVectorStore:
    """Optimized FAISS vector store with enhanced error handling"""
    
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
    
    async def add_documents_batch(self, documents: List[Document], embeddings: List[np.ndarray]):
        try:
            if not self.is_trained:
                self.initialize()
            
            if len(documents) != len(embeddings):
                raise ValueError("Number of documents must match number of embeddings")
            
            all_embeddings = np.array(embeddings, dtype=np.float32)
            faiss.normalize_L2(all_embeddings)
            
            self.index.add(all_embeddings)
            self.documents.extend(documents)
            
            logger.info(f"⚡ Added {len(documents)} documents in single batch")
        except Exception as e:
            logger.error(f"❌ Batch FAISS add error: {e}")
            raise
    
    async def add_documents(self, documents: List[Document], embeddings: List[np.ndarray]):
        await self.add_documents_batch(documents, embeddings)
    
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

class DomainDetector:
    """Universal domain detector with caching"""
    
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

# Document state and memory management classes remain the same structure
class DocumentStateManager:
    def __init__(self):
        self.current_doc_hash = None
        self.current_doc_timestamp = None
    
    def generate_doc_signature(self, sources: List[str]) -> str:
        signature_data = {
            'sources': sorted(sources),
            'timestamp': time.time(),
            'system_version': '3.0'
        }
        return hashlib.sha256(json.dumps(signature_data, sort_keys=True).encode()).hexdigest()
    
    def should_invalidate_cache(self, new_doc_hash: str) -> bool:
        if self.current_doc_hash is None:
            return True
        return self.current_doc_hash != new_doc_hash
    
    def invalidate_all_caches(self):
        CACHE_MANAGER.clear_all_caches()
        QUERY_CACHE.cache.clear()

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

class PerformanceMonitor:
    def __init__(self):
        self.metrics = defaultdict(list)
    
    def record_timing(self, operation: str, duration: float):
        self.metrics[operation].append(duration)
        if len(self.metrics[operation]) > 100:
            self.metrics[operation] = self.metrics[operation][-50:]
    
    def get_average_timing(self, operation: str) -> float:
        return np.mean(self.metrics.get(operation, [0]))

# ================================
# ENHANCED FAST RAG SYSTEM
# ================================

class FastRAGSystem:
    """Enhanced RAG system with all improvements"""
    
    def __init__(self):
        self.documents = []
        self.quick_chunks = []
        self.vector_store = None
        self.bm25_retriever = None
        self.domain = "general"
        self.loader = UnifiedLoader()
        self.text_splitter = AdaptiveTextSplitter()
        self.complexity_analyzer = QueryComplexityAnalyzer()
        self.context_processor = TokenAwareContextProcessor()
        self.chunk_calculator = AdaptiveChunkLimitCalculator()
        self.adaptive_reranker = AdaptiveReranker()
        self.doc_state_manager = DocumentStateManager()
    
    async def cleanup(self):
        self.documents.clear()
        self.quick_chunks.clear()
        if self.vector_store:
            self.vector_store.clear()
        logger.info("🧹 FastRAGSystem cleaned up")
    
    async def process_documents_fast(self, sources: List[str]) -> Dict[str, Any]:
        """Enhanced document processing with timeout protection"""
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
            # Load documents with controlled concurrency
            async def load_single_document(source):
                return await concurrency_controller.processing_operation(
                    self.loader.load_document(source)
                )
            
            # Load all documents with timeout
            raw_documents = []
            for source in sources:
                docs = await asyncio.wait_for(
                    load_single_document(source),
                    timeout=DOCUMENT_PROCESSING_TIMEOUT
                )
                raw_documents.extend(docs)
            
            if not raw_documents:
                raise ValueError("No documents could be loaded")
            
            # Domain detection
            domain, domain_confidence = DOMAIN_DETECTOR.detect_domain(raw_documents)
            self.domain = domain
            
            # Document processing with enhanced chunking
            all_chunks = self.text_splitter.split_documents(raw_documents, domain)
            self.documents = all_chunks
            
            # Calculate adaptive quick chunk limit
            default_query_analysis = {'is_longform': False, 'type': 'factual'}
            quick_chunk_limit = self.chunk_calculator.calculate_chunk_limit(
                domain, 0.5, default_query_analysis
            )
            
            self.quick_chunks = all_chunks[:quick_chunk_limit]
            
            # Setup retrievers with timeout protection
            await asyncio.wait_for(
                self._setup_quick_retrievers(),
                timeout=120.0  # 2 minute timeout for retriever setup
            )
            
            processing_time = time.time() - start_time
            
            logger.info(f"⚡ Enhanced processing complete in {processing_time:.2f}s")
            
            return {
                'domain': domain,
                'domain_confidence': float(domain_confidence),
                'total_chunks': len(all_chunks),
                'quick_chunks': len(self.quick_chunks),
                'quick_chunk_limit': quick_chunk_limit,
                'processing_time': processing_time,
                'enhanced': True
            }
            
        except asyncio.TimeoutError:
            logger.error("❌ Document processing timeout")
            raise HTTPException(status_code=408, detail="Document processing timeout")
        except Exception as e:
            logger.error(f"❌ Enhanced document processing error: {e}")
            raise
    
    async def _setup_quick_retrievers(self):
        """Setup retrievers with enhanced error handling"""
        try:
            logger.info("🔧 Setting up optimized retrievers...")
            
            # FAISS setup with concurrency control
            if HAS_FAISS and self.quick_chunks:
                try:
                    await ensure_models_ready()
                    self.vector_store = OptimizedFAISSVectorStore(dimension=384)
                    self.vector_store.initialize()
                    
                    quick_texts = [doc.page_content for doc in self.quick_chunks]
                    embeddings = await concurrency_controller.embedding_operation(
                        get_embeddings_batch_optimized(quick_texts)
                    )
                    
                    await self.vector_store.add_documents_batch(self.quick_chunks, embeddings)
                    logger.info("✅ Optimized FAISS vector store setup complete")
                except Exception as e:
                    logger.warning(f"⚠️ FAISS setup failed: {e}")
                    self.vector_store = None
            
            # BM25 setup with timeout
            try:
                if self.quick_chunks:
                    self.bm25_retriever = await asyncio.wait_for(
                        asyncio.to_thread(BM25Retriever.from_documents, self.quick_chunks),
                        timeout=30.0
                    )
                    self.bm25_retriever.k = min(5, len(self.quick_chunks))
                    logger.info(f"✅ BM25 retriever setup complete (k={self.bm25_retriever.k})")
            except asyncio.TimeoutError:
                logger.warning("⚠️ BM25 retriever setup timeout")
            except Exception as e:
                logger.warning(f"⚠️ BM25 retriever setup failed: {e}")
                
        except Exception as e:
            logger.error(f"❌ Quick retriever setup error: {e}")
    
    async def query_express_lane(self, query: str) -> Dict[str, Any]:
        """Ultra-fast processing for simple queries"""
        start_time = time.time()
        
        # Use rate limiting for express lane too
        await rate_limiter.wait_if_needed("express_lane")
        
        if self.vector_store and len(self.quick_chunks) > 0:
            query_embedding = await get_query_embedding(query)
            vector_results = await self.vector_store.similarity_search_with_score(
                query_embedding, k=3
            )
            retrieved_docs = [doc for doc, score in vector_results]
        else:
            retrieved_docs = self.quick_chunks[:3]
        
        context = self._optimize_context_fast(retrieved_docs, query)
        
        query_analysis = self.complexity_analyzer.analyze_query_complexity(query)
        answer = await self._generate_response_fast(query, context, self.domain, 0.85, query_analysis)
        
        processing_time = time.time() - start_time
        
        logger.info(f"⚡ Express lane complete in {processing_time:.2f}s")
        
        return {
            "query": query,
            "answer": answer,
            "confidence": 0.85,
            "domain": self.domain,
            "processing_time": processing_time,
            "express_lane": True
        }
    
    def _optimize_context_fast(self, documents: List[Document], query: str) -> str:
        if not documents:
            return ""
        
        context_parts = []
        total_chars = 0
        max_chars = 8000
        
        for doc in documents[:4]:
            if total_chars + len(doc.page_content) <= max_chars:
                context_parts.append(doc.page_content)
                total_chars += len(doc.page_content)
            else:
                remaining = max_chars - total_chars
                if remaining > 200:
                    context_parts.append(doc.page_content[:remaining] + "...")
                break
        
        return "\n\n".join(context_parts)
    
    async def _generate_response_fast(self, query: str, context: str, domain: str,
                                    confidence: float, query_analysis: Dict[str, Any]) -> str:
        """Faster response generation with API key rotation"""
        try:
            complexity = query_analysis['complexity']
            model_name = select_gemini_model(complexity)
            
            # Apply rate limiting
            await rate_limiter.wait_if_needed("fast_response")
            
            # Use API key rotation
            client, api_key = get_gemini_client_with_rotation()
            
            system_prompt = f"""Expert {domain} analyst. Answer concisely based on context provided.

Context: {context[:1200]}

Question: {query}

Provide a direct, accurate answer."""
            
            # Use retry manager for LLM calls
            async def llm_operation():
                return await client.chat.completions.create(
                    messages=[{"role": "user", "content": system_prompt}],
                    model=model_name,
                    temperature=0.1,
                    max_tokens=250
                )
            
            response = await asyncio.wait_for(
                retry_manager.retry_with_backoff(llm_operation),
                timeout=BASE_QUESTION_TIMEOUT
            )
            
            # Report success for API key
            api_key_rotator.report_success(api_key)
            
            return response.choices[0].message.content.strip()
            
        except asyncio.TimeoutError:
            logger.error("⚡ Fast response timeout")
            return "Based on the available information, I can provide a quick response, but please try again for a more detailed answer."
        except Exception as e:
            # Report failure for API key
            if 'api_key' in locals():
                api_key_rotator.report_failure(api_key)
            logger.error(f"❌ Fast response error: {e}")
            return f"I found relevant information but encountered a processing error: {str(e)}"
    
    async def query(self, query: str) -> Dict[str, Any]:
        """Enhanced query processing with all improvements"""
        start_time = time.time()
        
        try:
            # Enhanced query analysis with timeout consideration
            query_analysis = self.complexity_analyzer.analyze_query_complexity(query)
            complexity = query_analysis['complexity']
            recommended_timeout = query_analysis['recommended_timeout']
            
            logger.info(f"🔍 Query analysis: type={query_analysis['type']}, "
                       f"complexity={complexity:.2f}, timeout={recommended_timeout}s")
            
            # Check cache
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
            
            # Simple query routing (express lane)
            if query_analysis['type'] == 'simple' and complexity < 0.3:
                result = await self.query_express_lane(query)
                QUERY_CACHE.cache_answer(query, doc_hash, result['answer'])
                return result
            
            # Apply rate limiting
            await rate_limiter.wait_if_needed("standard_query")
            
            # Enhanced retrieval with adaptive parameters
            retrieved_docs, similarity_scores = await asyncio.wait_for(
                self.retrieve_and_rerank_enhanced(query, complexity, query_analysis),
                timeout=recommended_timeout * 0.6  # Reserve time for generation
            )
            
            if not retrieved_docs:
                return {
                    "query": query,
                    "answer": "No relevant documents found for your query.",
                    "confidence": 0.0,
                    "domain": self.domain,
                    "processing_time": time.time() - start_time
                }
            
            # Token-aware context selection
            context = self.context_processor.select_context_with_budget(
                retrieved_docs, query, complexity
            )
            
            # Enhanced response generation with timeout
            remaining_time = recommended_timeout - (time.time() - start_time)
            answer = await asyncio.wait_for(
                self._generate_response_enhanced(query, context, self.domain, 0.8, query_analysis),
                timeout=max(5.0, remaining_time * 0.8)
            )
            
            processing_time = time.time() - start_time
            
            result = {
                "query": query,
                "answer": answer,
                "confidence": 0.8,
                "domain": self.domain,
                "retrieved_chunks": len(retrieved_docs),
                "processing_time": processing_time,
                "complexity": complexity,
                "query_type": query_analysis['type'],
                "enhanced_accuracy": True
            }
            
            QUERY_CACHE.cache_answer(query, doc_hash, answer)
            return sanitize_for_json(result)
            
        except asyncio.TimeoutError:
            logger.error(f"❌ Query processing timeout after {recommended_timeout}s")
            return {
                "query": query,
                "answer": "The query processing took longer than expected. Please try again or simplify your question.",
                "confidence": 0.0,
                "domain": self.domain,
                "processing_time": time.time() - start_time,
                "timeout": True
            }
        except Exception as e:
            logger.error(f"❌ Enhanced query processing error: {e}")
            return {
                "query": query,
                "answer": f"An error occurred while processing your query: {sanitize_pii(str(e))}",
                "confidence": 0.0,
                "domain": self.domain,
                "processing_time": time.time() - start_time
            }
    
    async def retrieve_and_rerank_enhanced(self, query: str, complexity: float,
                                         query_analysis: Dict[str, Any]) -> Tuple[List[Document], List[float]]:
        """Enhanced retrieval with controlled concurrency"""
        if not self.documents:
            return [], []
        
        # Calculate adaptive parameters
        rerank_params = self.adaptive_reranker.calculate_rerank_params(complexity, query_analysis)
        rerank_top_k = rerank_params['rerank_top_k']
        context_docs = rerank_params['context_docs']
        
        search_k = min(8, max(5, int(complexity * 10)))
        
        logger.info(f"🎯 Adaptive retrieval: search_k={search_k}, rerank_k={rerank_top_k}, "
                   f"context_docs={context_docs}")
        
        # Controlled concurrent retrieval
        async def vector_search_task():
            if self.vector_store:
                try:
                    query_embedding = await get_query_embedding(query)
                    vector_search_results = await self.vector_store.similarity_search_with_score(
                        query_embedding, k=search_k
                    )
                    return [(doc, score) for doc, score in vector_search_results]
                except Exception as e:
                    logger.warning(f"⚠️ Vector search failed: {e}")
                    return []
            return []
        
        async def bm25_search_task():
            if self.bm25_retriever:
                try:
                    bm25_search_results = await asyncio.to_thread(self.bm25_retriever.invoke, query) or []
                    return [(doc, 0.7) for doc in bm25_search_results[:4]]
                except Exception as e:
                    logger.warning(f"⚠️ BM25 search failed: {e}")
                    return []
            return []
        
        # Execute searches concurrently with timeout
        vector_results, bm25_results = await asyncio.gather(
            asyncio.wait_for(vector_search_task(), timeout=10.0),
            asyncio.wait_for(bm25_search_task(), timeout=10.0),
            return_exceptions=True
        )
        
        # Handle exceptions
        if isinstance(vector_results, Exception):
            logger.warning(f"⚠️ Vector search exception: {vector_results}")
            vector_results = []
        if isinstance(bm25_results, Exception):
            logger.warning(f"⚠️ BM25 search exception: {bm25_results}")
            bm25_results = []
        
        # Apply Reciprocal Rank Fusion
        if vector_results and bm25_results:
            logger.info("🔄 Applying Reciprocal Rank Fusion")
            try:
                fused_results = reciprocal_rank_fusion([vector_results, bm25_results])
                unique_docs = [doc for doc, score in fused_results[:rerank_top_k]]
            except Exception as e:
                logger.warning(f"⚠️ RRF failed, using simple combination: {e}")
                all_docs = [doc for doc, _ in vector_results[:4]] + [doc for doc, _ in bm25_results[:2]]
                unique_docs = self._deduplicate_docs(all_docs)
        else:
            all_docs = [doc for doc, _ in vector_results[:4]] + [doc for doc, _ in bm25_results[:2]]
            unique_docs = self._deduplicate_docs(all_docs)
        
        # Enhanced reranking with timeout
        if reranker and len(unique_docs) > 3:
            try:
                context_length = 500 if complexity > 0.7 else 300 if complexity > 0.5 else 200
                pairs = [[query, doc.page_content[:context_length]] for doc in unique_docs[:rerank_top_k]]
                
                scores = await asyncio.wait_for(
                    asyncio.to_thread(reranker.predict, pairs),
                    timeout=15.0
                )
                
                scored_docs = list(zip(unique_docs[:len(scores)], scores))
                scored_docs.sort(key=lambda x: x[1], reverse=True)
                
                final_docs = [doc for doc, _ in scored_docs[:context_docs]]
                final_scores = [score for _, score in scored_docs[:context_docs]]
                
                logger.info(f"🎯 Enhanced reranking applied: {len(pairs)} candidates → {len(final_docs)} selected")
                return final_docs, final_scores
                
            except asyncio.TimeoutError:
                logger.warning("⚠️ Reranking timeout, using original ranking")
                return unique_docs[:context_docs], [0.8] * min(len(unique_docs), context_docs)
            except Exception as e:
                logger.warning(f"⚠️ Enhanced reranking failed: {e}")
                return unique_docs[:context_docs], [0.8] * min(len(unique_docs), context_docs)
        
        return unique_docs[:context_docs], [0.8] * min(len(unique_docs), context_docs)
    
    async def _generate_response_enhanced(self, query: str, context: str, domain: str,
                                        confidence: float, query_analysis: Dict[str, Any]) -> str:
        """Enhanced response generation with API key rotation and retry"""
        try:
            complexity = query_analysis['complexity']
            model_name = select_gemini_model(complexity)
            
            # Apply rate limiting
            await rate_limiter.wait_if_needed("enhanced_response")
            
            system_prompt = f"""You are an expert document analyst specializing in {domain} content.

INSTRUCTIONS:
1. Provide accurate, comprehensive answers based strictly on the provided context
2. If information is not available in the context, clearly state this limitation
3. Cite specific details and evidence from the context when relevant
4. Maintain professional accuracy and avoid speculation beyond the provided information
5. Structure your response clearly for complex queries

Context Quality: {confidence:.1%}"""
            
            user_message = f"""Context Information:
{context}

Question: {query}

Please provide a detailed, accurate answer based on the context above. Focus on directly answering the question with evidence from the provided information."""
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
            
            # Use API key rotation and retry manager
            async def llm_operation():
                client, api_key = get_gemini_client_with_rotation()
                response = await client.chat.completions.create(
                    messages=messages,
                    model=model_name,
                    temperature=0.1,
                    max_tokens=1000
                )
                api_key_rotator.report_success(api_key)
                return response
            
            response = await retry_manager.retry_with_backoff(llm_operation)
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"❌ Enhanced response generation error: {e}")
            return f"I apologize, but I encountered an error while processing your query: {str(e)}"
    
    def _deduplicate_docs(self, docs: List[Document]) -> List[Document]:
        """Quick deduplication of documents"""
        seen_content = set()
        unique_docs = []
        for doc in docs:
            content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_docs.append(doc)
        return unique_docs

# ================================
# GLOBAL INSTANCES AND UTILITY FUNCTIONS
# ================================

# Initialize global instances
CACHE_MANAGER = SmartCacheManager()
QUERY_CACHE = QueryResultCache()
DOC_STATE_MANAGER = DocumentStateManager()
MEMORY_MANAGER = MemoryManager()
PERFORMANCE_MONITOR = PerformanceMonitor()
DOMAIN_DETECTOR = DomainDetector()

# Utility functions
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
    """Process embeddings with caching and batch optimization"""
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
    
    # Process uncached embeddings in batch
    if uncached_texts:
        await ensure_models_ready()
        if base_sentence_model:
            embeddings = await asyncio.to_thread(
                base_sentence_model.encode,
                uncached_texts,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            # Cache new embeddings
            for text, embedding in zip(uncached_texts, embeddings):
                text_hash = hashlib.md5(text.encode()).hexdigest()
                CACHE_MANAGER.set_embedding(text_hash, embedding)
            
            # Add to results
            for i, embedding in zip(uncached_indices, embeddings):
                results.append((i, embedding))
    
    # Sort results by original order
    results.sort(key=lambda x: x[0])
    return [embedding for _, embedding in results]

async def get_query_embedding(query: str) -> np.ndarray:
    """Get single query embedding with caching"""
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

async def ensure_models_ready():
    """Load models with enhanced error handling"""
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

# Utility functions for validation and sanitization
def simple_auth_check(request: Request) -> bool:
    """Simple authentication check"""
    auth = request.headers.get("Authorization", "")
    expected = f"Bearer {HACKRX_TOKEN}"
    return auth == expected

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
# FASTAPI APPLICATION
# ================================


# ================================
# FASTAPI APPLICATION (Corrected)
# ================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager with enhanced startup"""
    logger.info("🚀 Starting Enhanced HackRx RAG System v3.0...")
    try:
        await ensure_models_ready()
        logger.info("✅ Application startup complete")
        yield
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    finally:
        logger.info("🔄 Application shutdown")

app = FastAPI(
    title="Enhanced HackRx RAG System",
    description="Advanced RAG system with improved performance and reliability",
    version="3.0.0",
    lifespan=lifespan
)

# CORS middleware
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

class DocumentRequest(BaseModel):
    sources: List[str]

class QueryRequest(BaseModel):
    query: str

class BatchQueryRequest(BaseModel):
    queries: List[str]

class HackRxRequest(BaseModel):
    documents: Optional[str] = None  # Single document URL (new format)
    questions: Optional[List[str]] = None  # Multiple questions (new format)
    question: Optional[str] = None  # Single question (backward compatibility)
    sources: Optional[List[str]] = None  # Multiple sources (backward compatibility)

class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: bool
    cache_stats: Dict[str, Any]
    performance_stats: Dict[str, Any]

# ================================
# GLOBAL RAG SYSTEM INSTANCE
# ================================

rag_system = FastRAGSystem()

# ================================
# API ENDPOINTS
# ================================

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with system information"""
    return {
        "message": "Enhanced HackRx RAG System v3.0",
        "status": "operational",
        "features": "rate_limiting,api_rotation,parallel_processing,smart_batching,enhanced_timeouts"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Enhanced health check with detailed metrics"""
    try:
        cache_stats = CACHE_MANAGER.get_cache_stats()
        performance_stats = {
            "embedding_avg_time": PERFORMANCE_MONITOR.get_average_timing("embedding"),
            "query_avg_time": PERFORMANCE_MONITOR.get_average_timing("query"),
            "retrieval_avg_time": PERFORMANCE_MONITOR.get_average_timing("retrieval")
        }
        
        # Add memory stats if available
        if HAS_PSUTIL:
            performance_stats.update({
                "memory_usage_percent": psutil.virtual_memory().percent,
                "cpu_usage_percent": psutil.cpu_percent()
            })
        
        return HealthResponse(
            status="healthy",
            version="3.0.0",
            models_loaded=_models_loaded,
            cache_stats=cache_stats,
            performance_stats=performance_stats
        )
    except Exception as e:
        logger.error(f"❌ Health check error: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.post("/hackrx/run")
async def hackrx_run_endpoint(request: Request):
    """Main HackRx endpoint - handles both single and batch processing"""
    start_time = time.time()
    
    try:
        # Authentication check
        if not simple_auth_check(request):
            raise HTTPException(status_code=401, detail="Unauthorized")
        
        # Parse request data
        data = await request.json()
        
        # Handle both new format and backward compatibility
        sources = []
        questions = []
        
        # New format: documents + questions (from judges)
        if "documents" in data and "questions" in data:
            document_url = data.get("documents", "").strip()
            questions = data.get("questions", [])
            
            if not document_url:
                raise HTTPException(status_code=400, detail="Document URL is required")
            
            if not questions or not isinstance(questions, list):
                raise HTTPException(status_code=400, detail="Questions array is required")
            
            sources = [document_url]
            
        # Backward compatibility: question + sources (old format)
        elif "question" in data and "sources" in data:
            question = data.get("question", "").strip()
            sources = data.get("sources", [])
            
            if not question:
                raise HTTPException(status_code=400, detail="Question is required")
            
            if not sources:
                raise HTTPException(status_code=400, detail="At least one source is required")
            
            questions = [question]
            
        else:
            raise HTTPException(
                status_code=400, 
                detail="Invalid request format. Expected: {documents, questions} or {question, sources}"
            )
        
        # Validate sources
        for source in sources:
            if not source or len(source.strip()) == 0:
                raise HTTPException(status_code=400, detail="Empty source provided")
            if len(source) > 2000:
                raise HTTPException(status_code=400, detail="Source URL/path too long")
            
            # Validate URL scheme if it's a URL
            if source.startswith(('http://', 'https://')):
                if not validate_url_scheme(source):
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Unsupported URL scheme: {sanitize_pii(source)}"
                    )
            else:
                # Validate file extension for local files
                if not validate_file_extension(source):
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Unsupported file type: {sanitize_pii(source)}"
                    )
        
        # Validate questions
        if len(questions) > 20:
            raise HTTPException(status_code=400, detail="Too many questions (max 20)")
        
        for question in questions:
            if not question.strip():
                raise HTTPException(status_code=400, detail="Empty question provided")
            if len(question) > 1000:
                raise HTTPException(status_code=400, detail="Question too long (max 1000 characters)")
        
        logger.info(f"🚀 HackRx processing: {len(sources)} sources, {len(questions)} questions")
        
        # Apply rate limiting
        await rate_limiter.wait_if_needed("hackrx_run")
        
        # Process documents with timeout protection
        doc_result = await asyncio.wait_for(
            rag_system.process_documents_fast(sources),
            timeout=DOCUMENT_PROCESSING_TIMEOUT
        )
        
        # Process questions with smart batching
        if len(questions) == 1:
            # Single question - direct processing
            query_result = await asyncio.wait_for(
                rag_system.query(questions[0]),
                timeout=COMPLEX_QUESTION_TIMEOUT
            )
            query_results = [query_result]
        else:
            # Multiple questions - use smart batching
            query_results = await asyncio.wait_for(
                question_batcher.process_questions_batch(questions, rag_system),
                timeout=BATCH_PROCESSING_TIMEOUT
            )
        
        processing_time = time.time() - start_time
        PERFORMANCE_MONITOR.record_timing("hackrx_run", processing_time)
        
        # Memory cleanup
        MEMORY_MANAGER.cleanup_if_needed()
        
        # Prepare response based on input format
        if len(questions) == 1:
            # Single question - backward compatible response
            query_result = query_results[0]
            response_data = {
                "answer": query_result['answer'],
                "confidence": query_result['confidence'],
                "domain": query_result['domain'],
                "processing_time": processing_time,
                "enhanced_features": {
                    "rate_limiting": True,
                    "api_key_rotation": True,
                    "parallel_processing": True,
                    "smart_batching": True,
                    "enhanced_timeouts": True,
                    "adaptive_chunking": True,
                    "memory_management": HAS_PSUTIL,
                    "advanced_caching": HAS_CACHETOOLS
                },
                "document_stats": {
                    "total_chunks": doc_result['total_chunks'],
                    "quick_chunks": doc_result['quick_chunks'],
                    "domain_confidence": doc_result['domain_confidence']
                },
                "query_stats": {
                    "cached": query_result.get('cached', False),
                    "express_lane": query_result.get('express_lane', False),
                    "complexity": query_result.get('complexity', 0.5),
                    "query_type": query_result.get('query_type', 'unknown')
                }
            }
        else:
            # Multiple questions - batch response
            response_data = {
                "results": [
                    {
                        "question": questions[i],
                        "answer": result['answer'],
                        "confidence": result['confidence'],
                        "domain": result['domain'],
                        "processing_time": result.get('processing_time', 0.0),
                        "cached": result.get('cached', False),
                        "express_lane": result.get('express_lane', False),
                        "complexity": result.get('complexity', 0.5),
                        "query_type": result.get('query_type', 'unknown')
                    }
                    for i, result in enumerate(query_results)
                ],
                "total_processing_time": processing_time,
                "document_stats": {
                    "total_chunks": doc_result['total_chunks'],
                    "quick_chunks": doc_result['quick_chunks'],
                    "domain_confidence": doc_result['domain_confidence'],
                    "domain": doc_result['domain']
                },
                "enhanced_features": {
                    "rate_limiting": True,
                    "api_key_rotation": True,
                    "parallel_processing": True,
                    "smart_batching": True,
                    "enhanced_timeouts": True,
                    "adaptive_chunking": True,
                    "memory_management": HAS_PSUTIL,
                    "advanced_caching": HAS_CACHETOOLS,
                    "batch_processing": True
                }
            }
        
        logger.info(f"✅ HackRx processing completed in {processing_time:.2f}s")
        
        return JSONResponse(
            status_code=200,
            content=sanitize_for_json(response_data)
        )
        
    except asyncio.TimeoutError:
        logger.error("❌ HackRx processing timeout")
        processing_time = time.time() - start_time
        return JSONResponse(
            status_code=408,
            content={
                "error": "Processing timeout",
                "message": "The request took too long to process. Please try with simpler questions or fewer documents.",
                "processing_time": processing_time,
                "timeout": True
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ HackRx endpoint error: {e}")
        processing_time = time.time() - start_time
        return JSONResponse(
            status_code=500,
            content={
                "error": "Processing failed",
                "message": f"An error occurred: {sanitize_pii(str(e))}",
                "processing_time": processing_time
            }
        )

@app.post("/process-documents")
async def process_documents_endpoint(request: DocumentRequest, http_request: Request):
    """Enhanced document processing endpoint"""
    if not simple_auth_check(http_request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    if not request.sources:
        raise HTTPException(status_code=400, detail="No sources provided")
    
    if len(request.sources) > 10:
        raise HTTPException(status_code=400, detail="Too many sources (max 10)")
    
    try:
        start_time = time.time()
        
        # Validate sources
        for source in request.sources:
            if len(source.strip()) == 0:
                raise HTTPException(status_code=400, detail="Empty source provided")
            if len(source) > 2000:
                raise HTTPException(status_code=400, detail="Source URL/path too long")
        
        # Apply rate limiting
        await rate_limiter.wait_if_needed("document_processing")
        
        # Process with timeout protection
        result = await asyncio.wait_for(
            rag_system.process_documents_fast(request.sources),
            timeout=DOCUMENT_PROCESSING_TIMEOUT
        )
        
        processing_time = time.time() - start_time
        PERFORMANCE_MONITOR.record_timing("document_processing", processing_time)
        
        # Memory cleanup if needed
        MEMORY_MANAGER.cleanup_if_needed()
        
        logger.info(f"✅ Document processing completed in {processing_time:.2f}s")
        
        return {
            "message": "Documents processed successfully",
            "result": sanitize_for_json(result),
            "processing_time": processing_time,
            "enhanced_features": True
        }
        
    except asyncio.TimeoutError:
        logger.error("❌ Document processing timeout")
        raise HTTPException(status_code=408, detail="Document processing timeout")
    except Exception as e:
        logger.error(f"❌ Document processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {sanitize_pii(str(e))}")

@app.post("/query")
async def query_endpoint(request: QueryRequest, http_request: Request):
    """Enhanced query endpoint with improved error handling"""
    if not simple_auth_check(http_request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Empty query provided")
    
    if len(request.query) > 1000:
        raise HTTPException(status_code=400, detail="Query too long (max 1000 characters)")
    
    try:
        start_time = time.time()
        
        # Apply rate limiting
        await rate_limiter.wait_if_needed("query")
        
        # Query analysis for timeout selection
        query_analysis = rag_system.complexity_analyzer.analyze_query_complexity(request.query)
        timeout = query_analysis['recommended_timeout']
        
        # Process query with timeout
        result = await asyncio.wait_for(
            rag_system.query(request.query),
            timeout=timeout
        )
        
        processing_time = time.time() - start_time
        PERFORMANCE_MONITOR.record_timing("query", processing_time)
        
        logger.info(f"✅ Query processed in {processing_time:.2f}s")
        
        return sanitize_for_json(result)
        
    except asyncio.TimeoutError:
        logger.error(f"❌ Query timeout after {timeout}s")
        return {
            "query": request.query,
            "answer": "Query processing timeout. Please try a simpler question or try again later.",
            "confidence": 0.0,
            "domain": getattr(rag_system, 'domain', 'unknown'),
            "processing_time": time.time() - start_time,
            "timeout": True
        }
    except Exception as e:
        logger.error(f"❌ Query processing error: {e}")
        return {
            "query": request.query,
            "answer": f"An error occurred while processing your query: {sanitize_pii(str(e))}",
            "confidence": 0.0,
            "domain": getattr(rag_system, 'domain', 'unknown'),
            "processing_time": time.time() - start_time,
            "error": True
        }

@app.post("/batch-query")
async def batch_query_endpoint(request: BatchQueryRequest, http_request: Request):
    """Enhanced batch query endpoint with smart processing"""
    if not simple_auth_check(http_request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    if not request.queries:
        raise HTTPException(status_code=400, detail="No queries provided")
    
    if len(request.queries) > 20:
        raise HTTPException(status_code=400, detail="Too many queries (max 20)")
    
    for query in request.queries:
        if not query.strip():
            raise HTTPException(status_code=400, detail="Empty query in batch")
        if len(query) > 500:
            raise HTTPException(status_code=400, detail="Query too long in batch (max 500 characters)")
    
    try:
        start_time = time.time()
        
        # Apply rate limiting for batch processing
        await rate_limiter.wait_if_needed("batch_query")
        
        # Use smart question batcher
        results = await asyncio.wait_for(
            question_batcher.process_questions_batch(request.queries, rag_system),
            timeout=BATCH_PROCESSING_TIMEOUT
        )
        
        processing_time = time.time() - start_time
        PERFORMANCE_MONITOR.record_timing("batch_query", processing_time)
        
        logger.info(f"✅ Batch query processed: {len(request.queries)} questions in {processing_time:.2f}s")
        
        return {
            "message": "Batch queries processed successfully",
            "results": sanitize_for_json(results),
            "total_queries": len(request.queries),
            "processing_time": processing_time,
            "batch_optimized": True
        }
        
    except asyncio.TimeoutError:
        logger.error("❌ Batch query timeout")
        raise HTTPException(status_code=408, detail="Batch processing timeout")
    except Exception as e:
        logger.error(f"❌ Batch query error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {sanitize_pii(str(e))}")

@app.post("/clear-cache")
async def clear_cache_endpoint(http_request: Request):
    """Clear all caches endpoint"""
    if not simple_auth_check(http_request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    try:
        # Clear all caches
        CACHE_MANAGER.clear_all_caches()
        QUERY_CACHE.cache.clear()
        DOC_STATE_MANAGER.invalidate_all_caches()
        
        # Clean up RAG system
        await rag_system.cleanup()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        logger.info("🧹 All caches cleared successfully")
        
        return {
            "message": "All caches cleared successfully",
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"❌ Cache clearing error: {e}")
        raise HTTPException(status_code=500, detail=f"Cache clearing failed: {str(e)}")

@app.get("/system-stats")
async def system_stats_endpoint(http_request: Request):
    """Get detailed system statistics"""
    if not simple_auth_check(http_request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    try:
        stats = {
            "version": "3.0.0",
            "uptime": time.time() - (getattr(app, '_start_time', time.time())),
            "models_loaded": _models_loaded,
            "cache_stats": CACHE_MANAGER.get_cache_stats(),
            "performance_stats": {
                "document_processing_avg": PERFORMANCE_MONITOR.get_average_timing("document_processing"),
                "query_avg": PERFORMANCE_MONITOR.get_average_timing("query"),
                "batch_query_avg": PERFORMANCE_MONITOR.get_average_timing("batch_query"),
                "embedding_avg": PERFORMANCE_MONITOR.get_average_timing("embedding"),
                "retrieval_avg": PERFORMANCE_MONITOR.get_average_timing("retrieval"),
                "hackrx_run_avg": PERFORMANCE_MONITOR.get_average_timing("hackrx_run")
            },
            "rate_limiter_stats": {
                "total_requests": len(rate_limiter.request_times),
                "backoff_delays": len([d for d in rate_limiter.backoff_delays.values() if d > 0])
            },
            "api_key_rotation": {
                "total_keys": len(GEMINI_API_KEY_POOL),
                "failure_counts": dict(api_key_rotator.failure_counts)
            },
            "features": {
                "rate_limiting": True,
                "api_key_rotation": True,
                "parallel_processing": True,
                "smart_batching": True,
                "enhanced_timeouts": True,
                "adaptive_chunking": True,
                "memory_management": HAS_PSUTIL,
                "advanced_caching": HAS_CACHETOOLS,
                "faiss_support": HAS_FAISS,
                "hackrx_endpoint": True
            }
        }
        
        # Add memory stats if available
        if HAS_PSUTIL:
            memory_info = psutil.virtual_memory()
            stats["system_resources"] = {
                "memory_total_gb": memory_info.total / (1024**3),
                "memory_available_gb": memory_info.available / (1024**3),
                "memory_usage_percent": memory_info.percent,
                "cpu_usage_percent": psutil.cpu_percent(),
                "cpu_count": psutil.cpu_count()
            }
        
        return sanitize_for_json(stats)
        
    except Exception as e:
        logger.error(f"❌ System stats error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get system stats: {str(e)}")

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Enhanced HTTP exception handler"""
    logger.warning(f"⚠️ HTTP {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "status_code": exc.status_code,
            "message": exc.detail,
            "timestamp": time.time()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Enhanced general exception handler"""
    logger.error(f"❌ Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "status_code": 500,
            "message": f"Internal server error: {sanitize_pii(str(exc))}",
            "timestamp": time.time()
        }
    )

# ================================
# APPLICATION STARTUP
# ================================

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment variable (Cloud Run/Docker requirement)
    port = int(os.getenv("PORT", 8000))
    
    # Record start time for uptime calculation
    app._start_time = time.time()
    
    logger.info("🚀 Starting Enhanced HackRx RAG System v3.0...")
    logger.info(f"🌐 Server will run on port: {port}")
    logger.info("🎯 Features enabled:")
    logger.info("   ✅ Smart rate limiting with exponential backoff")
    logger.info("   ✅ API key rotation with failure tracking")
    logger.info("   ✅ Parallel processing with controlled concurrency")
    logger.info("   ✅ Enhanced timeouts for complex legal queries")
    logger.info("   ✅ Question batching with smart retry logic")
    logger.info("   ✅ Adaptive chunking and reranking")
    logger.info("   ✅ Memory management and optimization")
    logger.info("   ✅ Advanced caching with TTL/LRU")
    logger.info("   ✅ HackRx endpoint: /hackrx/run")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
        access_log=True,
        timeout_keep_alive=30,
        timeout_graceful_shutdown=30
    )
