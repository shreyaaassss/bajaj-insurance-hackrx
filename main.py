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
# OPTIMIZED CONFIGURATION FOR ACCURACY & CONCISENESS
# ================================

# Fixed optimal configuration
HACKRX_TOKEN = "9a1163c13e8927960b857a674794a62c57baf588998981151b0753a4d6d17905"
GEMINI_API_KEY = 'AIzaSyC7S0IKMVnr0E6E57Ojt5p8aiM0GCfGs34'

# ENHANCED CONFIGURATION WITH ACCURACY IMPROVEMENTS
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ðŸ”¹ OPTIMIZED CONFIGURATION FOR ACCURACY & CONCISENESS
CHUNK_SIZE = 800  # Slightly reduced from 900
CHUNK_OVERLAP = 80  # Reduced from 100

# Retrieval parameters
SEMANTIC_SEARCH_K = 12  # Reduced from 15 for better precision
CONTEXT_DOCS = 8  # Increased from 6
CONFIDENCE_THRESHOLD = 0.25  # Reduced from 0.3

# Reranking
BASE_RERANK_TOP_K = 8   # Reduced from 10
MAX_RERANK_TOP_K = 24   # Reduced from 32

# Token management for conciseness
MAX_CONTEXT_TOKENS = 6000  # Reduced from 8000 for more focused context
TOKEN_SAFETY_MARGIN = 200   # Reduced

MAX_FILE_SIZE_MB = 100
QUESTION_TIMEOUT = 20.0

# PARALLEL PROCESSING - OPTIMIZED
OPTIMAL_BATCH_SIZE = 32
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

# Cache for document processing
_document_cache = {}
_cache_ttl = 1800  # 30 minutes

# Global models
base_sentence_model = None
reranker = None
gemini_client = None

class HackRxRunRequest(BaseModel):
    documents: str  # Single document URL as string
    questions: List[str]  # List of questions

class HackRxRunResponse(BaseModel):
    results: List[Dict[str, Any]]


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
        
        # Initialize tokenizer for complexity scoring
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.tokenizer = None

    def analyze_query_complexity(self, query: str) -> Dict[str, Any]:
        """Enhanced query complexity analysis"""
        query_lower = query.lower().strip()
        
        # ðŸ”¹ IMPROVED: Better pattern recognition
        simple_patterns = [
            r'^what is\s+\w+',
            r'^define\s+\w+',
            r'^who is\s+\w+',
            r'^\w+\s+means?$',
            r'^which article',
            r'^article \d+'
        ]
        
        analytical_patterns = [
            r'how does.*differ',
            r'compare.*with',
            r'analyze.*impact',
            r'explain.*relationship',
            r'discuss.*implications'
        ]
        
        is_simple_pattern = any(re.match(pattern, query_lower) for pattern in simple_patterns)
        is_analytical = any(re.search(pattern, query_lower) for pattern in analytical_patterns)
        
        # Additional complexity factors
        has_multiple_questions = query.count('?') > 1
        has_complex_keywords = any(keyword in query_lower for keyword in 
                                  ['comprehensive', 'detailed', 'elaborate', 'in depth'])
        
        token_count = self._count_tokens(query)
        word_count = len(query.split())

        # Calculate complexity score
        complexity_factors = {
            'analytical_keywords': 0.4 if is_analytical else 0.0,
            'token_length': min(0.3, token_count / 80),
            'word_length': min(0.2, word_count / 15),
            'multiple_questions': 0.2 if has_multiple_questions else 0.0,
            'complex_keywords': 0.1 if has_complex_keywords else 0.0,
            'pattern_penalty': -0.3 if is_simple_pattern else 0.0
        }

        complexity_score = sum(complexity_factors.values())
        complexity_score = max(0.0, min(1.0, complexity_score))

        # ðŸ”¹ IMPROVED: Better query type determination
        if is_simple_pattern and complexity_score < 0.25:
            query_type = 'simple'
        elif is_analytical or complexity_score > 0.65:
            query_type = 'analytical'
        else:
            query_type = 'factual'

        return {
            'type': query_type,
            'complexity': complexity_score,
            'token_count': token_count,
            'word_count': word_count,
            'is_longform': token_count > 40 or word_count > 12,
            'requires_deep_context': complexity_score > 0.6
        }

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text with fallback"""
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception:
                pass
        # Fallback estimation
        return max(1, int(len(text) / 3.8))

# ================================
# ENHANCED CHUNK LIMIT CALCULATOR
# ================================

class AdaptiveChunkLimitCalculator:
    """Calculate dynamic chunk limits based on query complexity and domain"""
    
    @staticmethod
    def calculate_chunk_limit(domain: str, complexity: float, query_analysis: Dict[str, Any]) -> int:
        """Calculate adaptive chunk limit based on domain and complexity"""
        # Base limits by domain
        domain_base_limits = {
            "legal": 150,     # Legal documents need more context
            "academic": 140,  # Academic papers need comprehensive context
            "medical": 130,   # Medical documents need detailed context
            "technical": 120, # Technical docs need thorough context
            "insurance": 110, # Insurance policies need detailed context
            "financial": 110, # Financial documents need context
            "business": 100,  # Business docs standard context
            "general": 100    # General documents standard
        }
        
        base_limit = domain_base_limits.get(domain, 100)
        
        # Complexity multipliers
        if complexity > 0.7:
            multiplier = 1.5   # High complexity needs more chunks
        elif complexity > 0.5:
            multiplier = 1.25  # Medium complexity needs some more chunks
        elif complexity < 0.3:
            multiplier = 0.8   # Simple queries need fewer chunks for speed
        else:
            multiplier = 1.0   # Standard complexity
        
        # Adjust for longform queries
        if query_analysis.get('is_longform', False):
            multiplier *= 1.2
        
        calculated_limit = int(base_limit * multiplier)
        
        # Enforce reasonable bounds (50 to 250)
        final_limit = max(50, min(250, calculated_limit))
        
        logger.info(f"ðŸ“Š Adaptive chunk limit: {final_limit} (domain: {domain}, complexity: {complexity:.2f})")
        return final_limit

# ================================
# ENHANCED TOKEN-AWARE CONTEXT PROCESSOR
# ================================

class TokenAwareContextProcessor:
    """Token-aware context processor with budget management"""
    
    def __init__(self):
        self.max_context_tokens = MAX_CONTEXT_TOKENS  # Now 6000
        self.safety_margin = TOKEN_SAFETY_MARGIN      # Now 200
        self.available_tokens = self.max_context_tokens - self.safety_margin  # 5800 tokens
        
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.tokenizer = None
            logger.warning("âš ï¸ Token encoder not available, using estimation")

    def select_context_with_budget(self, documents: List[Document], query: str,
                                  complexity: float) -> str:
        """Select optimal context within token budget with better relevance scoring"""
        if not documents:
            return ""

        # ðŸ”¹ OPTIMIZATION: Dynamic context parameters based on complexity
        if complexity > 0.7:
            max_docs = 10  # Increased for complex queries
            priority_boost = 1.4
        elif complexity > 0.5:
            max_docs = 8   # Increased
            priority_boost = 1.2
        else:
            max_docs = 6
            priority_boost = 1.0

        # ðŸ”¹ IMPROVED: Better relevance scoring
        scored_docs = []
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        for i, doc in enumerate(documents[:max_docs * 2]):
            content = doc.page_content
            content_lower = content.lower()
            
            # Base position score
            base_score = 1.0 / (i + 1)
            
            # ðŸ”¹ IMPROVED: Better query term matching
            content_words = set(content_lower.split())
            exact_matches = len(query_words.intersection(content_words))
            partial_matches = sum(1 for qword in query_words 
                                if any(qword in cword for cword in content_words))
            
            match_score = (exact_matches * 0.3 + partial_matches * 0.1)
            
            # Boost for content quality and length
            length_score = min(0.25, len(content) / 4000)
            
            # ðŸ”¹ NEW: Boost for constitutional articles/sections
            article_boost = 0.2 if any(word in content_lower 
                                     for word in ['article', 'section', 'clause', 'amendment']) else 0
            
            total_score = (base_score + match_score + length_score + article_boost) * priority_boost
            scored_docs.append((doc, total_score))

        # Sort by score
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # ðŸ”¹ OPTIMIZATION: Better context selection
        selected_parts = []
        current_tokens = 0
        
        for doc, score in scored_docs[:max_docs]:
            content = doc.page_content
            content_tokens = self._estimate_tokens(content)
            
            if current_tokens + content_tokens <= self.available_tokens:
                selected_parts.append(content)
                current_tokens += content_tokens
            else:
                # Try to fit partial content
                remaining_tokens = self.available_tokens - current_tokens
                if remaining_tokens > 200:
                    partial_content = self._truncate_to_tokens(content, remaining_tokens - 50)
                    if partial_content and len(partial_content) > 100:
                        selected_parts.append(partial_content + "...")
                break

        context = "\n\n".join(selected_parts)
        final_tokens = self._estimate_tokens(context)
        
        logger.info(f"ðŸŽ¯ Context selected: {final_tokens}/{self.max_context_tokens} tokens "
                    f"({len(selected_parts)} chunks, avg_score: {sum(s for _, s in scored_docs[:len(selected_parts)])/max(1, len(selected_parts)):.2f})")
        
        return context

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count with fallback"""
        if not text:
            return 0
        
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception:
                pass
        
        # Fallback estimation (conservative)
        return max(1, int(len(text) / 3.5))

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit"""
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
        
        # Fallback: character-based truncation
        estimated_chars = int(max_tokens * 3.5)
        return text[:estimated_chars] if len(text) > estimated_chars else text

# ================================
# ENHANCED ADAPTIVE RERANKER
# ================================

class AdaptiveReranker:
    """Context-aware reranking with dynamic parameters"""
    
    @staticmethod
    def calculate_rerank_params(complexity: float, query_analysis: Dict[str, Any]) -> Dict[str, int]:
        """Calculate adaptive reranking parameters"""
        # Base reranking parameters
        if complexity > 0.7 or query_analysis.get('is_longform', False):
            # High complexity or longform queries need extensive reranking
            rerank_top_k = MAX_RERANK_TOP_K  # 24
            context_docs = 8
        elif complexity > 0.5:
            # Medium complexity queries need moderate reranking
            rerank_top_k = 20
            context_docs = 6
        elif query_analysis.get('type') == 'analytical':
            # Analytical queries benefit from more reranking even if not complex
            rerank_top_k = 15
            context_docs = 6
        else:
            # Simple/factual queries use base parameters for speed
            rerank_top_k = BASE_RERANK_TOP_K  # 8
            context_docs = 5
        
        return {
            'rerank_top_k': rerank_top_k,
            'context_docs': context_docs
        }

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
                logger.info("âœ… Advanced caching with TTL/LRU enabled")
            else:
                raise ImportError("cachetools not available")
        except ImportError:
            self.embedding_cache = {}
            self.document_chunk_cache = {}
            self.domain_cache = {}
            self.primary_available = False
            logger.info("ðŸ“¦ Using dict fallback caching (cachetools not available)")
        
        self._lock = threading.RLock()

    def clear_all_caches(self):
        """Clear ALL caches when new document is uploaded - prevents stale answers"""
        with self._lock:
            self.embedding_cache.clear()
            self.document_chunk_cache.clear()
            self.domain_cache.clear()
        logger.info("ðŸ§¹ All caches cleared for new document upload")

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

# Query Router for simple query detection
class QueryRouter:
    def __init__(self):
        self.simple_patterns = [
            r'^what is\s+\w+',
            r'^define\s+\w+',
            r'^who is\s+\w+',
            r'^\w+\s+means?$'
        ]

    def is_simple_query(self, query: str) -> bool:
        """Detect queries that need minimal context"""
        query_lower = query.lower().strip()
        
        # Pattern matching for simple queries
        for pattern in self.simple_patterns:
            if re.match(pattern, query_lower):
                return True
        
        # Word count and complexity heuristics
        words = query.split()
        return (
            len(words) <= 6 and
            not any(word in query_lower for word in ['compare', 'analyze', 'explain', 'difference']) and
            query.count('?') <= 1
        )

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
        CACHE_MANAGER.clear_all_caches()
        QUERY_CACHE.cache.clear()

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
            logger.info("ðŸ§¹ Memory cleanup performed")

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

# ================================
# ENHANCED ADAPTIVE TEXT SPLITTER
# ================================

class AdaptiveTextSplitter:
    """Enhanced adaptive text splitter with balanced chunking"""
    
    def __init__(self):
        self.separators = [
            "\n\n## ", "\n\n# ", "\n\n### ", "\n\n#### ",
            "\n\n", "\n", ". ", "! ", "? ", "; ", " ", ""
        ]

    def split_documents(self, documents: List[Document], detected_domain: str = "general") -> List[Document]:
        """Split documents with enhanced balanced chunking"""
        if not documents:
            return []
        
        content_hash = self._calculate_content_hash(documents)
        cache_key = f"chunks_{content_hash}_{detected_domain}_v2"  # Updated cache version
        
        cached_chunks = CACHE_MANAGER.get_document_chunks(cache_key)
        if cached_chunks is not None:
            logger.info(f"ðŸ“„ Using cached chunks: {len(cached_chunks)} chunks")
            return cached_chunks
        
        # ENHANCED: Use improved chunking parameters
        chunk_size, chunk_overlap = self._get_balanced_chunk_params(detected_domain)
        
        all_chunks = []
        for doc in documents:
            try:
                chunks = self._split_document_balanced(doc, chunk_size, chunk_overlap)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.warning(f"âš ï¸ Error splitting document: {e}")
                chunks = self._simple_split(doc, chunk_size, chunk_overlap)
                all_chunks.extend(chunks)
        
        # Filter very short chunks
        all_chunks = [chunk for chunk in all_chunks if len(chunk.page_content.strip()) >= 100]
        
        # Cache the results
        CACHE_MANAGER.set_document_chunks(cache_key, all_chunks)
        
        logger.info(f"ðŸ“„ Created {len(all_chunks)} balanced chunks (size: {chunk_size}, overlap: {chunk_overlap})")
        return all_chunks

    def _get_balanced_chunk_params(self, detected_domain: str) -> Tuple[int, int]:
        """Get balanced chunking parameters"""
        # Domain-specific adjustments to the base parameters
        domain_adjustments = {
            "legal": 1.1,      # Legal documents benefit from slightly larger chunks
            "academic": 1.05,  # Academic papers need good context
            "medical": 1.0,    # Medical documents standard
            "technical": 0.95, # Technical docs can be slightly smaller for precision
            "insurance": 1.0,  # Insurance standard
            "financial": 1.0,  # Financial standard
            "business": 0.9,   # Business docs can be more concise
            "general": 1.0     # General documents standard
        }
        
        adjustment = domain_adjustments.get(detected_domain, 1.0)
        
        # Apply domain adjustment to base parameters
        adjusted_size = int(CHUNK_SIZE * adjustment)
        adjusted_overlap = CHUNK_OVERLAP  # Keep overlap consistent
        
        # Ensure reasonable bounds
        adjusted_size = max(600, min(1200, adjusted_size))
        
        return adjusted_size, adjusted_overlap

    def _split_document_balanced(self, document: Document, chunk_size: int, chunk_overlap: int) -> List[Document]:
        """Split single document with balanced parameters"""
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
        """Simple fallback splitting"""
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            return splitter.split_documents([document])
        except Exception as e:
            logger.error(f"âŒ Even simple splitting failed: {e}")
            return [document]

    def _calculate_content_hash(self, documents: List[Document]) -> str:
        """Calculate hash for content caching"""
        content_sample = "".join([doc.page_content[:100] for doc in documents[:5]])
        return hashlib.sha256(content_sample.encode()).hexdigest()[:16]

# ================================
# UNIFIED LOADER (UNCHANGED)
# ================================

class UnifiedLoader:
    """Unified document loader with enhanced URL support"""
    
    def __init__(self):
        self.mime_detector = magic if HAS_MAGIC else None
        self.google_patterns = [
            r'drive\.google\.com/file/d/([a-zA-Z0-9-_]+)',
            r'docs\.google\.com/document/d/([a-zA-Z0-9-_]+)',
            r'docs\.google\.com/spreadsheets/d/([a-zA-Z0-9-_]+)',
            r'drive\.google\.com/open\?id=([a-zA-Z0-9-_]+)',
            r'[?&]id=([a-zA-Z0-9-_]+)',  # Generic ID parameter
        ]
        
        self.dropbox_patterns = [
            r'dropbox\.com/s/([a-zA-Z0-9]+)',
            r'dropbox\.com/sh/([a-zA-Z0-9]+)',
            r'dropbox\.com/scl/fi/([a-zA-Z0-9-_]+)',
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
            
            logger.info(f"âœ… Loaded {len(docs)} documents from {sanitize_pii(source)}")
            return docs
        
        except Exception as e:
            logger.error(f"âŒ Failed to load {sanitize_pii(source)}: {e}")
            raise

    def _is_url(self, source: str) -> bool:
        """Check if source is a URL"""
        return source.startswith(('http://', 'https://', 'blob:', 'drive:', 'dropbox:'))

    async def _load_from_url(self, url: str) -> List[Document]:
        """Enhanced URL loading with retry logic."""
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
        
        # Enhanced headers for Google Drive compatibility
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                timeout = httpx.Timeout(
                    timeout=180.0,  # Increased for large files
                    connect=20.0,
                    read=180.0,
                    write=60.0,
                    pool=10.0
                )
                
                async with httpx.AsyncClient(timeout=timeout, headers=headers, follow_redirects=True) as client:
                    response = await client.get(download_url)
                    
                    # Handle Google Drive virus scan warning
                    if 'drive.google.com' in download_url and response.status_code == 200:
                        if 'Google Drive - Virus scan warning' in response.text:
                            # Extract confirm token and retry
                            confirm_match = re.search(r'name="confirm" value="([^"]+)"', response.text)
                            if confirm_match:
                                confirm_token = confirm_match.group(1)
                                confirm_url = f"{download_url}&confirm={confirm_token}"
                                response = await client.get(confirm_url)
                    
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
                
            except Exception:
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)

    def _transform_special_url(self, url: str) -> str:
        """Enhanced URL transformation for Google Drive and Dropbox"""
        # Enhanced Google Drive transformation
        for pattern in self.google_patterns:
            match = re.search(pattern, url)
            if match:
                file_id = match.group(1)
                # Use direct download URL that works better
                return f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t"
        
        # Additional Google Drive patterns for edge cases
        if 'drive.google.com' in url:
            # Handle sharing URLs like: https://drive.google.com/open?id=FILE_ID
            open_match = re.search(r'[?&]id=([a-zA-Z0-9-_]+)', url)
            if open_match:
                file_id = open_match.group(1)
                return f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t"
            
            # Handle view URLs like: https://drive.google.com/file/d/FILE_ID/view
            view_match = re.search(r'/file/d/([a-zA-Z0-9-_]+)/view', url)
            if view_match:
                file_id = view_match.group(1)
                return f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t"
        
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
        
        if content.startswith(b'%PDF'):
            return '.pdf'
        elif b'PK' in content[:10]:
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
        
        logger.info(f"ðŸ“„ Loading {file_extension} file ({file_size} bytes): {sanitize_pii(file_path)}")
        
        mime_type = None
        if self.mime_detector:
            try:
                mime_type = magic.from_file(file_path, mime=True)
            except Exception as e:
                logger.warning(f"âš ï¸ MIME detection failed: {e}")
        
        docs = None
        loader_used = None
        
        if mime_type == 'application/pdf' or file_extension == '.pdf':
            try:
                loader = PyMuPDFLoader(file_path)
                docs = await asyncio.to_thread(loader.load)
                loader_used = "PyMuPDFLoader"
            except Exception as e:
                logger.warning(f"âš ï¸ PyMuPDF failed: {e}")
        
        elif ('word' in (mime_type or '') or
              'officedocument' in (mime_type or '') or
              file_extension in ['.docx', '.doc']):
            try:
                loader = Docx2txtLoader(file_path)
                docs = await asyncio.to_thread(loader.load)
                loader_used = "Docx2txtLoader"
            except Exception as e:
                logger.warning(f"âš ï¸ DOCX loader failed: {e}")
        
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
                    logger.warning(f"âš ï¸ Text loader failed with {encoding}: {e}")
        
        if not docs:
            try:
                loader = TextLoader(file_path, encoding='utf-8')
                docs = await asyncio.to_thread(loader.load)
                loader_used = "TextLoader (fallback)"
            except Exception as e:
                logger.error(f"âŒ All loaders failed: {e}")
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
        
        logger.info(f"âœ… Loaded {len(docs)} documents using {loader_used}")
        return docs

# ================================
# OPTIMIZED FAISS VECTOR STORE (UNCHANGED)
# ================================

class OptimizedFAISSVectorStore:
    """OPTIMIZED FAISS-based vector store with batch processing"""
    
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
            logger.info("âœ… FAISS vector store initialized")
        except Exception as e:
            logger.error(f"âŒ Failed to initialize FAISS: {e}")
            raise

    async def add_documents_batch(self, documents: List[Document], embeddings: List[np.ndarray]):
        """Add all documents in single batch operation"""
        try:
            if not self.is_trained:
                self.initialize()
            
            if len(documents) != len(embeddings):
                raise ValueError("Number of documents must match number of embeddings")
            
            # Convert all embeddings to numpy array at once
            all_embeddings = np.array(embeddings, dtype=np.float32)
            faiss.normalize_L2(all_embeddings)
            
            # Single batch add - MUCH faster than individual adds
            self.index.add(all_embeddings)
            self.documents.extend(documents)
            
            logger.info(f"âš¡ Added {len(documents)} documents in single batch")
        
        except Exception as e:
            logger.error(f"âŒ Batch FAISS add error: {e}")
            raise

    async def add_documents(self, documents: List[Document], embeddings: List[np.ndarray]):
        """Wrapper for batch processing"""
        await self.add_documents_batch(documents, embeddings)

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
            logger.error(f"âŒ Error in FAISS similarity search: {e}")
            return []

    def clear(self):
        """Clear the vector store"""
        self.documents.clear()
        if self.index:
            self.index.reset()

# ================================
# DOMAIN DETECTOR (UNCHANGED)
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
            logger.info(f"ðŸ” Using cached domain: {cached_result[0]} (confidence: {cached_result[1]:.2f})")
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
                
                logger.info(f"ðŸ” Domain detected: {best_domain} (confidence: {best_score:.2f})")
                return result
            
            return "general", confidence_threshold
        
        except Exception as e:
            logger.warning(f"âš ï¸ Domain detection error: {e}")
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
# ENHANCED FAST RAG SYSTEM WITH ACCURACY IMPROVEMENTS
# ================================

class FastRAGSystem:
    """Enhanced RAG system with accuracy improvements and performance preservation"""
    
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
        """RAGSystem cleanup method"""
        self.documents.clear()
        self.quick_chunks.clear()
        if self.vector_store:
            self.vector_store.clear()
        logger.info("ðŸ§¹ FastRAGSystem cleaned up")

    async def process_documents_fast(self, sources: List[str]) -> Dict[str, Any]:
        """Enhanced document processing with adaptive chunk limits"""
        start_time = time.time()
        
        doc_signature = self.doc_state_manager.generate_doc_signature(sources)
        
        if hasattr(self, '_last_doc_signature') and self._last_doc_signature == doc_signature:
            logger.info("ðŸ“„ Documents already processed, skipping...")
            return {"cached": True, "processing_time": 0.001}
        
        if self.doc_state_manager.should_invalidate_cache(doc_signature):
            logger.info("ðŸ§¹ New documents detected - invalidating all caches")
            self.doc_state_manager.invalidate_all_caches()
            self.doc_state_manager.current_doc_hash = doc_signature
            self.doc_state_manager.current_doc_timestamp = time.time()
        
        self._last_doc_signature = doc_signature
        
        try:
            # Load documents
            raw_documents = []
            for source in sources:
                docs = await self.loader.load_document(source)
                raw_documents.extend(docs)
            
            if not raw_documents:
                raise ValueError("No documents could be loaded")
            
            # Domain detection
            domain, domain_confidence = DOMAIN_DETECTOR.detect_domain(raw_documents)
            self.domain = domain
            
            # Enhanced document processing with balanced chunking
            all_chunks = self.text_splitter.split_documents(raw_documents, domain)
            self.documents = all_chunks
            
            # ENHANCED: Calculate adaptive quick chunk limit based on domain
            # Use medium complexity (0.5) as default for initial processing
            default_query_analysis = {'is_longform': False, 'type': 'factual'}
            quick_chunk_limit = self.chunk_calculator.calculate_chunk_limit(
                domain, 0.5, default_query_analysis
            )
            
            # Keep adaptive number of chunks for quick processing
            self.quick_chunks = all_chunks[:quick_chunk_limit]
            
            # Setup quick retrievers
            await self._setup_quick_retrievers()
            
            processing_time = time.time() - start_time
            
            logger.info(f"âš¡ Enhanced processing complete in {processing_time:.2f}s")
            
            return {
                'domain': domain,
                'domain_confidence': float(domain_confidence),
                'total_chunks': len(all_chunks),
                'quick_chunks': len(self.quick_chunks),
                'quick_chunk_limit': quick_chunk_limit,
                'processing_time': processing_time,
                'enhanced': True
            }
        
        except Exception as e:
            logger.error(f"âŒ Enhanced document processing error: {e}")
            raise

    async def _setup_quick_retrievers(self):
        """Setup retrievers with minimal chunks for instant response"""
        try:
            logger.info("ðŸ”§ Setting up optimized retrievers...")
            
            if HAS_FAISS and self.quick_chunks:
                try:
                    await ensure_models_ready()
                    self.vector_store = OptimizedFAISSVectorStore(dimension=384)
                    self.vector_store.initialize()
                    
                    # Process embeddings for quick chunks only
                    quick_texts = [doc.page_content for doc in self.quick_chunks]
                    embeddings = await get_embeddings_batch_optimized(quick_texts)
                    
                    # Use batch processing
                    await self.vector_store.add_documents_batch(self.quick_chunks, embeddings)
                    
                    logger.info("âœ… Optimized FAISS vector store setup complete")
                
                except Exception as e:
                    logger.warning(f"âš ï¸ FAISS setup failed: {e}")
                    self.vector_store = None
            
            # Setup BM25 with quick chunks
            try:
                if self.quick_chunks:
                    self.bm25_retriever = await asyncio.to_thread(
                        BM25Retriever.from_documents, self.quick_chunks
                    )
                    self.bm25_retriever.k = min(5, len(self.quick_chunks))
                    logger.info(f"âœ… Optimized BM25 retriever setup complete (k={self.bm25_retriever.k})")
            
            except Exception as e:
                logger.warning(f"âš ï¸ BM25 retriever setup failed: {e}")
        
        except Exception as e:
            logger.error(f"âŒ Quick retriever setup error: {e}")

    async def query_express_lane(self, query: str) -> Dict[str, Any]:
        """Ultra-fast processing for simple queries"""
        start_time = time.time()
        
        # Use only first 3-5 most relevant chunks
        if self.vector_store and len(self.quick_chunks) > 0:
            query_embedding = await get_query_embedding(query)
            vector_results = await self.vector_store.similarity_search_with_score(
                query_embedding, k=3
            )
            retrieved_docs = [doc for doc, score in vector_results]
        else:
            retrieved_docs = self.quick_chunks[:3]
        
        # Skip reranking for express lane
        context = self._optimize_context_fast(retrieved_docs, query)
        
        # Generate response with reduced context
        answer = await self._generate_response_fast(query, context, self.domain, 0.85)
        
        processing_time = time.time() - start_time
        
        logger.info(f"âš¡ Express lane complete in {processing_time:.2f}s")
        
        return {
            "query": query,
            "answer": answer,
            "confidence": 0.85,
            "domain": self.domain,
            "processing_time": processing_time,
            "express_lane": True
        }

    def _optimize_context_fast(self, documents: List[Document], query: str) -> str:
        """Fast context optimization for simple queries"""
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

    async def _generate_response_fast(self, query: str, context: str, domain: str, confidence: float) -> str:
        """Faster response generation for simple queries with better prompting"""
        try:
            await ensure_gemini_ready()

            # ðŸ”¹ IMPROVED: Better prompt for simple questions
            system_prompt = f"""Expert {domain} analyst. Answer directly and concisely.

Context: {context[:2000]}

Question: {query}

Provide a direct, accurate answer in 1-3 sentences maximum."""

            response = await asyncio.wait_for(
                gemini_client.chat.completions.create(
                    messages=[{"role": "user", "content": system_prompt}],
                    model="gemini-2.0-flash",
                    temperature=0.1,
                    max_tokens=200  # ðŸ”¹ REDUCED for conciseness
                ),
                timeout=6.0  # Slightly increased
            )

            return response.choices[0].message.content.strip()

        except asyncio.TimeoutError:
            logger.error("âš¡ Fast response timeout")
            return "I can provide information on this topic, but need more processing time. Please try again."
        except Exception as e:
            logger.error(f"âŒ Fast response error: {e}")
            return f"I found relevant information but encountered a processing error: {str(e)}"

    async def query(self, query: str) -> Dict[str, Any]:
        """Enhanced query processing with token-aware context and adaptive parameters"""
        start_time = time.time()
        
        try:
            # Enhanced query analysis
            query_analysis = self.complexity_analyzer.analyze_query_complexity(query)
            complexity = query_analysis['complexity']
            
            logger.info(f"ðŸ” Query analysis: type={query_analysis['type']}, "
                       f"complexity={complexity:.2f}, tokens={query_analysis['token_count']}")
            
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
            
            # Enhanced retrieval with adaptive parameters
            retrieved_docs, similarity_scores = await self.retrieve_and_rerank_enhanced(
                query, complexity, query_analysis
            )
            
            if not retrieved_docs:
                # ðŸ”¹ ENHANCED FALLBACK: Try with different search strategies
                logger.warning("âš ï¸ No documents retrieved, attempting enhanced fallback...")
                
                try:
                    # Try with relaxed parameters and broader search
                    fallback_analysis = {
                        'type': 'analytical', 
                        'is_longform': True, 
                        'complexity': 0.8
                    }
                    
                    # Use more documents for fallback
                    if hasattr(self, 'documents') and len(self.documents) > 0:
                        # Use text similarity as last resort
                        query_words = set(query.lower().split())
                        scored_docs = []
                        
                        for doc in self.documents[:50]:  # Check more documents
                            content_words = set(doc.page_content.lower().split())
                            overlap = len(query_words.intersection(content_words))
                            if overlap > 0:
                                score = overlap / len(query_words)
                                scored_docs.append((doc, score))
                        
                        if scored_docs:
                            scored_docs.sort(key=lambda x: x[1], reverse=True)
                            retrieved_docs = [doc for doc, _ in scored_docs[:8]]
                            similarity_scores = [score for _, score in scored_docs[:8]]
                            logger.info(f"âœ… Text similarity fallback retrieved {len(retrieved_docs)} documents")
                    
                    if not retrieved_docs:
                        # Final fallback - provide helpful message
                        return {
                            "query": query,
                            "answer": "I couldn't find specific information about this question in the provided document. The document may not contain relevant details on this topic, or the question may need to be rephrased to better match the available content.",
                            "confidence": 0.0,
                            "domain": self.domain,
                            "processing_time": time.time() - start_time,
                            "fallback_used": "text_similarity_failed"
                        }
                        
                except Exception as e:
                    logger.error(f"âŒ Enhanced fallback failed: {e}")
                    return {
                        "query": query,
                        "answer": "I apologize, but I couldn't find relevant information for this question in the provided document.",
                        "confidence": 0.0,
                        "domain": self.domain,
                        "processing_time": time.time() - start_time,
                        "fallback_error": str(e)
                    }
            
            # ENHANCED: Token-aware context selection
            context = self.context_processor.select_context_with_budget(
                retrieved_docs, query, complexity
            )
            
            # ðŸ”¹ OPTIMIZATION 5: CHECK CONTEXT CONFIDENCE
            if len(context.strip()) < 200:  # Context too short
                logger.warning("âš ï¸ Context too short, attempting fallback retrieval...")
                try:
                    fallback_docs, _ = await self.retrieve_and_rerank_enhanced(
                        query,
                        complexity=min(1.0, complexity + 0.3),  # Increase complexity
                        query_analysis={'type': 'analytical', 'is_longform': True}
                    )
                    
                    if fallback_docs:
                        context = self.context_processor.select_context_with_budget(
                            fallback_docs, query, complexity + 0.3
                        )
                        logger.info("âœ… Fallback context selection successful")
                except Exception as e:
                    logger.warning(f"âš ï¸ Fallback context selection failed: {e}")
            
            answer = await self._generate_response_enhanced(query, context, self.domain, 0.8)
            
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
        
        except Exception as e:
            logger.error(f"âŒ Enhanced query processing error: {e}")
            return {
                "query": query,
                "answer": f"An error occurred while processing your query: {sanitize_pii(str(e))}",
                "confidence": 0.0,
                "domain": self.domain,
                "processing_time": time.time() - start_time
            }

    async def retrieve_and_rerank_enhanced(self, query: str, complexity: float,
                                         query_analysis: Dict[str, Any]) -> Tuple[List[Document], List[float]]:
        """Enhanced retrieval with adaptive parameters and token awareness"""
        if not self.documents:
            return [], []
        
        # ENHANCED: Calculate adaptive parameters
        rerank_params = self.adaptive_reranker.calculate_rerank_params(complexity, query_analysis)
        rerank_top_k = rerank_params['rerank_top_k']
        context_docs = rerank_params['context_docs']
        
        # Adaptive search parameters based on OPTIMIZATION 1
        search_k = min(SEMANTIC_SEARCH_K, max(10, int(complexity * 20)))  # Increased range
        
        logger.info(f"ðŸŽ¯ Adaptive retrieval: search_k={search_k}, rerank_k={rerank_top_k}, "
                   f"context_docs={context_docs}")
        
        query_embedding = await get_query_embedding(query)
        
        # Vector search
        vector_docs = []
        vector_results = []
        if self.vector_store:
            try:
                vector_search_results = await self.vector_store.similarity_search_with_score(
                    query_embedding, k=search_k
                )
                vector_results = [(doc, score) for doc, score in vector_search_results]
                vector_docs = [doc for doc, score in vector_results]
            except Exception as e:
                logger.warning(f"âš ï¸ Vector search failed: {e}")
        
        # BM25 search
        bm25_docs = []
        bm25_results = []
        if self.bm25_retriever and len(vector_docs) < search_k:
            try:
                bm25_search_results = await asyncio.to_thread(self.bm25_retriever.invoke, query) or []
                bm25_limit = min(6, search_k - len(vector_docs))  # Increased limit
                bm25_docs = bm25_search_results[:bm25_limit]
                bm25_results = [(doc, 0.7) for doc in bm25_docs]
            except Exception as e:
                logger.warning(f"âš ï¸ BM25 search failed: {e}")
        
        # Apply Reciprocal Rank Fusion if we have both result sets
        if vector_results and bm25_results:
            logger.info("ðŸ”„ Applying Reciprocal Rank Fusion")
            try:
                fused_results = reciprocal_rank_fusion([vector_results, bm25_results])
                unique_docs = [doc for doc, score in fused_results[:rerank_top_k]]
            except Exception as e:
                logger.warning(f"âš ï¸ RRF failed, using simple combination: {e}")
                all_docs = vector_docs[:6] + bm25_docs[:4]  # Increased limits
                unique_docs = self._deduplicate_docs(all_docs)
        else:
            all_docs = vector_docs[:6] + bm25_docs[:4]  # Increased limits
            unique_docs = self._deduplicate_docs(all_docs)
        
        # ENHANCED: Context-aware reranking with dynamic parameters
        if reranker and len(unique_docs) > 3:
            try:
                # Use longer context for complex queries
                context_length = 500 if complexity > 0.7 else 300 if complexity > 0.5 else 200
                pairs = [[query, doc.page_content[:context_length]] for doc in unique_docs[:rerank_top_k]]
                
                scores = reranker.predict(pairs)
                scored_docs = list(zip(unique_docs[:len(scores)], scores))
                scored_docs.sort(key=lambda x: x[1], reverse=True)
                
                final_docs = [doc for doc, _ in scored_docs[:context_docs]]
                final_scores = [score for _, score in scored_docs[:context_docs]]
                
                logger.info(f"ðŸŽ¯ Enhanced reranking applied: {len(pairs)} candidates â†’ {len(final_docs)} selected")
                return final_docs, final_scores
            
            except Exception as e:
                logger.warning(f"âš ï¸ Enhanced reranking failed: {e}")
        
        return unique_docs[:context_docs], [0.8] * min(len(unique_docs), context_docs)

    def _calculate_confidence_enhanced(self, query: str, scores: List[float], docs: List[Document]) -> float:
        """Enhanced confidence calculation"""
        if not scores:
            return 0.0
        
        try:
            max_score = max(scores) if scores else 0.0
            avg_score = sum(scores) / len(scores) if scores else 0.0
            
            # Enhanced query match checking
            query_lower = query.lower()
            query_words = set(query_lower.split())
            
            match_scores = []
            for doc in docs[:3]:  # Check top 3 docs
                doc_words = set(doc.page_content.lower().split())
                overlap = len(query_words.intersection(doc_words))
                match_score = overlap / len(query_words) if query_words else 0
                match_scores.append(match_score)
            
            avg_match = sum(match_scores) / len(match_scores) if match_scores else 0
            
            # Weighted confidence calculation
            confidence = (
                0.4 * max_score +
                0.3 * avg_score +
                0.2 * avg_match +
                0.1  # Base confidence
            )
            
            return min(1.0, max(0.0, confidence))
        
        except Exception as e:
            logger.warning(f"âš ï¸ Enhanced confidence calculation error: {e}")
            return 0.3

    async def _generate_response_enhanced(self, query: str, context: str, domain: str, confidence: float) -> str:
        """Enhanced response generation with better prompting for concise, complete answers"""
        try:
            await ensure_gemini_ready()
            if not gemini_client:
                return "System is still initializing. Please wait a moment and try again."

            # ðŸ”¹ OPTIMIZATION: Improved system prompt for concise but complete answers
            system_prompt = f"""You are an expert {domain} analyst. Provide COMPLETE but CONCISE answers.

CRITICAL INSTRUCTIONS:
1. Answer the question DIRECTLY and COMPLETELY based on the provided context
2. Be CONCISE - avoid unnecessary elaboration while ensuring completeness
3. If context is insufficient, clearly state what information is missing
4. Structure complex answers with bullet points for clarity
5. Always provide a definitive answer when context allows
6. For constitutional questions: cite specific articles/sections mentioned in context

Context Quality: {confidence:.1%}"""

            # ðŸ”¹ OPTIMIZATION: Better user message structure
            user_message = f"""Context:
{context}

Question: {query}

Provide a complete but concise answer. Focus on directly answering what is asked without unnecessary details."""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]

            response = await asyncio.wait_for(
                gemini_client.chat.completions.create(
                    messages=messages,
                    model="gemini-2.0-flash",
                    temperature=0.05,  # ðŸ”¹ REDUCED for more focused answers
                    max_tokens=600,   # ðŸ”¹ REDUCED from 1000 for conciseness
                    top_p=0.9        # ðŸ”¹ ADDED for better focus
                ),
                timeout=QUESTION_TIMEOUT
            )

            return response.choices[0].message.content.strip()

        except asyncio.TimeoutError:
            logger.error(f"âŒ Enhanced response generation timeout after {QUESTION_TIMEOUT}s")
            return "I apologize, but the response generation took too long. Please try again with a simpler question."
        except Exception as e:
            logger.error(f"âŒ Enhanced response generation error: {e}")
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
QUERY_ROUTER = QueryRouter()

# ================================
# ENHANCED UTILITY FUNCTIONS WITH RRF
# ================================

def reciprocal_rank_fusion(results_list: List[List[Tuple[Document, float]]], k_value: int = 60) -> List[Tuple[Document, float]]:
    """ENHANCED: Implement Reciprocal Rank Fusion for combining multiple result sets"""
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
    """Process embeddings with simple batching"""
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
            logger.warning("âš ï¸ No embedding model available for query")
            return np.zeros(384)
    
    except Exception as e:
        logger.error(f"âŒ Query embedding error: {e}")
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
            logger.info("âœ… Gemini client initialized successfully")
        except Exception as e:
            logger.error(f"âŒ Failed to initialize Gemini client: {e}")
            raise HTTPException(status_code=503, detail="Gemini client not available")
    elif gemini_client is None:
        raise HTTPException(status_code=503, detail="Gemini API key not configured")

async def ensure_models_ready():
    """Load models only once per container lifecycle"""
    global base_sentence_model, reranker, _models_loaded, _startup_complete
    
    if _models_loaded and _startup_complete:
        return
    
    async with _model_lock:
        if _models_loaded and _startup_complete:
            return
        
        logger.info("ðŸ”„ Loading pre-downloaded models...")
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
                logger.info("âœ… Sentence transformer loaded and warmed up")
            
            if reranker is None:
                reranker = CrossEncoder(
                    RERANKER_MODEL_NAME,
                    max_length=128,
                    device='cpu'
                )
                _ = reranker.predict([["warmup", "test"]])
                logger.info("âœ… Reranker loaded and warmed up")
            
            _models_loaded = True
            _startup_complete = True
            
            load_time = time.time() - start_time
            logger.info(f"âœ… All models ready in {load_time:.2f}s")
        
        except Exception as e:
            logger.error(f"âŒ Failed to load models: {e}")
            raise

# ================================
# UTILITY FUNCTIONS
# ================================

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
# FASTAPI APPLICATION
# ================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("ðŸš€ Starting Enhanced HackRx RAG System...")
    start_time = time.time()
    
    try:
        await ensure_models_ready()
        if GEMINI_API_KEY:
            await ensure_gemini_ready()
        
        startup_time = time.time() - start_time
        logger.info(f"âœ… Enhanced system fully initialized in {startup_time:.2f}s")
    
    except Exception as e:
        logger.error(f"âŒ Enhanced startup failed: {e}")
        raise
    
    yield
    
    logger.info("ðŸ”„ Shutting down enhanced system...")
    _document_cache.clear()
    CACHE_MANAGER.clear_all_caches()
    
    if gemini_client and hasattr(gemini_client, 'close'):
        try:
            await gemini_client.close()
        except Exception:
            pass
    
    logger.info("âœ… Enhanced system shutdown complete")

app = FastAPI(
    title="Enhanced HackRx RAG System",
    description="Enhanced RAG System with Accuracy Improvements and Google Gemini",
    version="2.1.0",
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
        "message": "Enhanced HackRx RAG System v2.1.0",
        "models_loaded": _models_loaded,
        "startup_complete": _startup_complete,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    try:
        memory_info = {}
        if HAS_PSUTIL:
            memory = psutil.virtual_memory()
            memory_info = {
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "memory_used_gb": memory.used / (1024**3)
            }
        
        cache_stats = CACHE_MANAGER.get_cache_stats()
        
        return {
            "status": "healthy",
            "models_loaded": _models_loaded,
            "startup_complete": _startup_complete,
            "has_faiss": HAS_FAISS,
            "has_psutil": HAS_PSUTIL,
            "has_cachetools": HAS_CACHETOOLS,
            "cache_stats": cache_stats,
            "memory_info": memory_info,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/status")
async def detailed_status():
    """Enhanced status endpoint with performance metrics"""
    try:
        cache_stats = CACHE_MANAGER.get_cache_stats()
        
        performance_stats = {}
        for operation in ['embedding', 'retrieval', 'reranking', 'generation']:
            avg_time = PERFORMANCE_MONITOR.get_average_timing(operation)
            performance_stats[f"{operation}_avg_ms"] = round(avg_time * 1000, 2)
        
        memory_info = {}
        if HAS_PSUTIL:
            memory = psutil.virtual_memory()
            memory_info = {
                "memory_percent": memory.percent,
                "memory_available_gb": round(memory.available / (1024**3), 2),
                "memory_used_gb": round(memory.used / (1024**3), 2)
            }
        
        return {
            "system_status": "operational",
            "version": "2.1.0",
            "models_loaded": _models_loaded,
            "startup_complete": _startup_complete,
            "features": {
                "faiss_available": HAS_FAISS,
                "psutil_available": HAS_PSUTIL,
                "cachetools_available": HAS_CACHETOOLS,
                "magic_available": HAS_MAGIC
            },
            "cache_stats": cache_stats,
            "performance_stats": performance_stats,
            "memory_info": memory_info,
            "configuration": {
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
                "max_context_tokens": MAX_CONTEXT_TOKENS,
                "semantic_search_k": SEMANTIC_SEARCH_K,
                "context_docs": CONTEXT_DOCS,
                "confidence_threshold": CONFIDENCE_THRESHOLD
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"âŒ Status endpoint error: {e}")
        return {
            "system_status": "error",
            "error": sanitize_pii(str(e)),
            "timestamp": datetime.now().isoformat()
        }

# ================================
# REQUEST/RESPONSE MODELS
# ================================

class DocumentProcessRequest(BaseModel):
    sources: List[str]

class QueryRequest(BaseModel):
    query: str

class DocumentProcessResponse(BaseModel):
    status: str
    domain: str
    domain_confidence: float
    total_chunks: int
    quick_chunks: int
    processing_time: float
    message: str = ""
    error: str = ""

class QueryResponse(BaseModel):
    query: str
    answer: str
    confidence: float
    domain: str
    processing_time: float
    retrieved_chunks: int = 0
    query_type: str = ""
    complexity: float = 0.0
    cached: bool = False
    express_lane: bool = False
    enhanced_accuracy: bool = False

# ================================
# MAIN API ENDPOINTS
# ================================

@app.post("/process-documents", response_model=DocumentProcessResponse)
async def process_documents_endpoint(request: DocumentProcessRequest, http_request: Request):
    """Enhanced document processing endpoint"""
    if not simple_auth_check(http_request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    try:
        logger.info(f"ðŸ“¤ Processing {len(request.sources)} documents")
        
        # Validate sources
        for source in request.sources:
            if source.startswith(('http://', 'https://')):
                if not validate_url_scheme(source):
                    raise HTTPException(status_code=400, detail=f"Unsupported URL scheme: {source}")
            else:
                if not validate_file_extension(source):
                    raise HTTPException(status_code=400, detail=f"Unsupported file type: {source}")
        
        # Process documents
        result = await rag_system.process_documents_fast(request.sources)
        
        # Memory cleanup
        MEMORY_MANAGER.cleanup_if_needed()
        
        return DocumentProcessResponse(
            status="success",
            domain=result.get('domain', 'general'),
            domain_confidence=result.get('domain_confidence', 0.5),
            total_chunks=result.get('total_chunks', 0),
            quick_chunks=result.get('quick_chunks', 0),
            processing_time=result.get('processing_time', 0.0),
            message=f"Successfully processed {len(request.sources)} documents"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"âŒ Document processing error: {e}")
        return DocumentProcessResponse(
            status="error",
            domain="unknown",
            domain_confidence=0.0,
            total_chunks=0,
            quick_chunks=0,
            processing_time=0.0,
            error=sanitize_pii(str(e))
        )

@app.post("/hackrx/run")
async def hackrx_run_endpoint(request: HackRxRunRequest, http_request: Request):
    """HackRx competition endpoint - processes document and answers multiple questions"""
    if not simple_auth_check(http_request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    # Validate input
    if not request.documents.strip():
        raise HTTPException(status_code=400, detail="Document URL cannot be empty")
    
    if not request.questions or len(request.questions) == 0:
        raise HTTPException(status_code=400, detail="Questions list cannot be empty")
    
    # Increase limit from 20 to 50
    if len(request.questions) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 questions allowed per request")
    
    start_time = time.time()
    
    try:
        logger.info(f"🏆 HackRx processing: 1 document, {len(request.questions)} questions")
        
        # Step 1: Process the document
        try:
            # Convert single document string to list format expected by process_documents_fast
            sources = [request.documents]
            
            # Validate the document URL
            if not validate_url_scheme(request.documents):
                raise HTTPException(status_code=400, detail=f"Unsupported URL scheme: {request.documents}")
            
            # Process the document
            doc_result = await rag_system.process_documents_fast(sources)
            
            if doc_result.get('status') == 'error':
                raise HTTPException(status_code=400, detail=f"Document processing failed: {doc_result.get('error', 'Unknown error')}")
            
            logger.info(f"✅ Document processed: {doc_result.get('domain', 'general')} domain, {doc_result.get('total_chunks', 0)} chunks")
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"❌ Document processing error: {e}")
            raise HTTPException(status_code=500, detail=f"Document processing failed: {sanitize_pii(str(e))}")
        
        # Step 2: Process all questions
        results = []
        for i, question in enumerate(request.questions):
            if not question.strip():
                logger.warning(f"⚠️ Skipping empty question at index {i}")
                results.append({
                    "question": question,
                    "answer": "Question is empty",
                    "confidence": 0.0,
                    "processing_time": 0.0,
                    "error": "Empty question"
                })
                continue
            
            try:
                logger.info(f"📝 Processing question {i+1}/{len(request.questions)}: {sanitize_pii(question[:50])}...")
                
                # Process the question
                query_result = await rag_system.query(question)
                
                # Format result for HackRx response
                formatted_result = {
                    "question": question,
                    "answer": query_result.get('answer', ''),
                    "confidence": query_result.get('confidence', 0.0),
                    "domain": query_result.get('domain', 'unknown'),
                    "processing_time": query_result.get('processing_time', 0.0),
                    "retrieved_chunks": query_result.get('retrieved_chunks', 0),
                    "query_type": query_result.get('query_type', ''),
                    "complexity": query_result.get('complexity', 0.0),
                    "cached": query_result.get('cached', False),
                    "express_lane": query_result.get('express_lane', False)
                }
                
                results.append(formatted_result)
                
            except Exception as e:
                logger.error(f"❌ Question processing error for question {i+1}: {e}")
                results.append({
                    "question": question,
                    "answer": f"Error processing question: {sanitize_pii(str(e))}",
                    "confidence": 0.0,
                    "processing_time": 0.0,
                    "error": sanitize_pii(str(e))
                })
        
        total_processing_time = time.time() - start_time
        
        # Memory cleanup
        MEMORY_MANAGER.cleanup_if_needed()
        
        logger.info(f"🏆 HackRx batch completed: {len(results)} questions processed in {total_processing_time:.2f}s")
        
        # Return results in the expected format
        response_data = {
            "results": sanitize_for_json(results),
            "summary": {
                "total_questions": len(request.questions),
                "successful_answers": len([r for r in results if not r.get('error')]),
                "document_domain": doc_result.get('domain', 'general'),
                "total_processing_time": total_processing_time,
                "average_time_per_question": total_processing_time / max(1, len(request.questions))
            }
        }
        
        return sanitize_for_json(response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ HackRx endpoint error: {e}")
        total_time = time.time() - start_time
        
        return {
            "results": [],
            "error": sanitize_pii(str(e)),
            "processing_time": total_time,
            "status": "error"
        }

@app.post("/batch-query")
async def batch_query_endpoint(queries: List[str], http_request: Request):
    """Process multiple queries in batch"""
    if not simple_auth_check(http_request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    if len(queries) > 50:
        raise HTTPException(status_code=400, detail="Maximum  50 qeries per batch")
    
    results = []
    for query in queries:
        if query.strip():
            try:
                result = await rag_system.query(query)
                results.append(result)
            except Exception as e:
                logger.error(f"âŒ Batch query error for '{sanitize_pii(query)}': {e}")
                results.append({
                    "query": query,
                    "answer": f"Error processing query: {sanitize_pii(str(e))}",
                    "confidence": 0.0,
                    "domain": "unknown",
                    "processing_time": 0.0
                })
    
    return {"results": sanitize_for_json(results)}

@app.get("/cache/stats")
async def cache_stats_endpoint(http_request: Request):
    """Get cache statistics"""
    if not simple_auth_check(http_request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    return CACHE_MANAGER.get_cache_stats()

@app.post("/cache/clear")
async def clear_cache_endpoint(http_request: Request):
    """Clear all caches"""
    if not simple_auth_check(http_request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    try:
        CACHE_MANAGER.clear_all_caches()
        QUERY_CACHE.cache.clear()
        logger.info("ðŸ§¹ All caches cleared via API")
        return {"status": "success", "message": "All caches cleared"}
    except Exception as e:
        logger.error(f"âŒ Cache clear error: {e}")
        return {"status": "error", "error": str(e)}


    
    return stats

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"âŒ Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": sanitize_pii(str(exc)),
            "timestamp": datetime.now().isoformat()
        }
    )

# ================================
# GLOBAL RAG SYSTEM INSTANCE
# ================================

rag_system = FastRAGSystem()

# ================================
# MAIN ENTRY POINT
# ================================

if __name__ == "__main__":
    import uvicorn
    
    # Production configuration
    config = {
        "host": "0.0.0.0",
        "port": int(os.getenv("PORT", 8000)),
        "workers": 1,  # Single worker for model sharing
        "loop": "uvloop" if sys.platform != "win32" else "asyncio",
        "http": "httptools" if sys.platform != "win32" else "h11",
        "access_log": False,
        "log_level": "info",
        "timeout_keep_alive": 65,
        "timeout_graceful_shutdown": 30
    }
    
    logger.info(f"ðŸš€ Starting Enhanced HackRx RAG System on {config['host']}:{config['port']}")
    
    try:
        uvicorn.run("main:app", **config)
    except KeyboardInterrupt:
        logger.info("ðŸ‘‹ Graceful shutdown initiated")
    except Exception as e:
        logger.error(f"âŒ Server startup error: {e}")
        sys.exit(1)
