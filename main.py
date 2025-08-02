import os

# Add before ChromaDB usage - Performance optimization
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

import sys
import asyncio
import logging
import time
from typing import List, Dict, Any, Tuple, Optional, Union
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

# Core libraries
import pandas as pd
import numpy as np
import psutil

# FastAPI and web
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, status, APIRouter, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, HttpUrl
import httpx

# Document processing
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# AI and embeddings
from sentence_transformers import CrossEncoder
import openai
from openai import AsyncOpenAI

# Configure logging for GCP Cloud Run
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Global variables for shared resources and performance optimization
embedding_model = None
reranker = None
openai_client = None

# ENHANCED: Global session storage with TTL
ACTIVE_SESSIONS = {}
SESSION_TTL = 3600  # 1 hour
QUERY_CACHE = {}
CACHE_TTL = 1800  # 30 minutes

# FIXED: Document cache with TTL and size management
DOCUMENT_CACHE = {}
CACHE_TTL_HOURS = timedelta(hours=2) # 2 hour TTL
_cache_lock = asyncio.Lock()

# ENHANCED: Persistent directory for vector store
PERSISTENT_CHROMA_DIR = "/tmp/persistent_chroma"

# FIXED: Proper cache management with TTL
MAX_CACHE_SIZE = 10
CACHE_CLEANUP_INTERVAL = 300 # 5 minutes

# ENHANCED: Domain-adaptive document processing settings
DOMAIN_CONFIG = {
    "academic": {
        "chunk_size": 1500,
        "chunk_overlap": 300,
        "semantic_search_k": 8,
        "context_docs": 10
    },
    "legal": {
        "chunk_size": 1200,
        "chunk_overlap": 250,
        "semantic_search_k": 7,
        "context_docs": 8
    },
    "insurance": {
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "semantic_search_k": 7,
        "context_docs": 7
    },
    "general": {
        "chunk_size": 1000,
        "chunk_overlap": 150,
        "semantic_search_k": 6,
        "context_docs": 6
    },
    "physics": {
        "chunk_size": 1200,
        "chunk_overlap": 200,
        "semantic_search_k": 8,
        "context_docs": 9
    },
    "medical": {
        "chunk_size": 1100,
        "chunk_overlap": 220,
        "semantic_search_k": 7,
        "context_docs": 8
    }
}

# ENHANCED: Component keywords mapping for robust clause matching (Insurance specific)
COMPONENT_KEYWORDS = {
    "Diagnostics": ["diagnostic", "test", "scan", "lab", "investigation", "pathology", "radiology", "x-ray", "mri", "ct", "ultrasound", "blood test", "urine test", "biopsy", "ecg", "ekg"],
    "Surgery": ["surgery", "surgical", "operation", "procedure", "operative", "incision", "laparoscopic", "arthroscopic", "cardiac", "orthopedic", "neurosurgery"],
    "Room Rent": ["room", "bed", "rent", "accommodation", "ward", "icu", "intensive care", "private room", "semi-private", "general ward", "cabin", "suite"],
    "Hospitalization": ["hospitalization", "admission", "inpatient", "hospital stay", "confinement", "indoor treatment", "hospital expenses"],
    "Medication": ["medicine", "drug", "prescription", "injection", "tablet", "capsule", "pharmaceutical", "medical store", "pharmacy", "medication expenses"],
    "Physiotherapy": ["physiotherapy", "rehabilitation", "therapy", "physical therapy", "rehab", "exercise therapy", "occupational therapy"],
    "Ambulance": ["ambulance", "transport", "transfer", "emergency vehicle", "medical transport", "patient transport", "air ambulance"],
    "Post-discharge Labs": ["post-hospitalization", "follow-up", "aftercare", "lab", "post-discharge", "follow up test", "monitoring"],
    "Home Nurse": ["nurse", "nursing", "home care", "attendant", "caregiver", "home nursing", "private nursing"],
    "Dietary Supplements": ["nutrition", "supplement", "diet", "nutritional", "vitamin", "protein", "dietary aid"],
    "Consultation": ["consultation", "doctor", "physician", "specialist", "medical consultation", "opd", "outpatient", "visit"],
    "Emergency": ["emergency", "casualty", "trauma", "accident", "urgent care", "emergency room", "er"],
    "Maternity": ["maternity", "pregnancy", "delivery", "childbirth", "prenatal", "postnatal", "obstetric", "gynecology"],
    "Dental": ["dental", "tooth", "oral", "dentist", "orthodontic", "periodontal", "oral surgery"],
    "Mental Health": ["mental health", "psychiatry", "psychology", "counseling", "therapy", "psychiatric", "behavioral health"],
    "Alternative Medicine": ["ayurveda", "homeopathy", "unani", "alternative medicine", "traditional medicine", "naturopathy"],
    "Organ Transplant": ["transplant", "organ", "kidney", "liver", "heart", "bone marrow", "cornea"],
    "Cancer Treatment": ["cancer", "oncology", "chemotherapy", "radiation", "tumor", "malignant", "carcinoma"],
    "Chronic Disease": ["diabetes", "hypertension", "chronic", "lifestyle disease", "metabolic", "endocrine"],
    "Cosmetic Surgery": ["cosmetic", "plastic surgery", "aesthetic", "beauty", "enhancement", "reconstructive"],
    "Pre-existing": ["pre-existing", "pre existing", "previous", "prior", "history", "chronic condition"],
    "Exclusions": ["exclusion", "not covered", "excluded", "limitation", "restriction", "exception"],
    "Waiting Period": ["waiting period", "moratorium", "cooling period", "initial waiting", "specific waiting"],
    "Sub-limit": ["sub-limit", "sublimit", "ceiling", "maximum", "cap", "limit per condition"],
    "Deductible": ["deductible", "excess", "co-payment", "copay", "patient contribution", "out of pocket"],
    "Network Hospital": ["network", "cashless", "preferred provider", "panel hospital", "empaneled", "tie-up"]
}

# ENHANCED: Semantic clause types for better categorization (Insurance specific)
CLAUSE_TYPES = {
    "coverage": ["coverage", "benefit", "included", "covered", "payable", "eligible"],
    "exclusion": ["exclusion", "not covered", "excluded", "limitation", "restriction", "exception"],
    "condition": ["condition", "requirement", "criteria", "eligibility", "qualification"],
    "limit": ["limit", "maximum", "ceiling", "cap", "sub-limit", "sublimit"],
    "procedure": ["procedure", "process", "claim", "settlement", "payment", "documentation"],
    "waiting_period": ["waiting", "moratorium", "cooling", "initial", "specific waiting period"],
    "deductible": ["deductible", "excess", "co-payment", "copay", "patient share"]
}

# NEW: Query classification keywords (Insurance specific)
CLAIM_KEYWORDS = [
    "claim", "reimburse", "settle", "approve", "reject", "amount", "payment", "bill",
    "expenses", "cost", "invoice", "receipt", "settlement", "processing", "discharge",
    "hospitalization", "treatment", "medical bill", "claim amount", "payable"
]

POLICY_INFO_KEYWORDS = [
    "what is", "define", "definition", "meaning", "explain", "grace period", "waiting period",
    "coverage", "benefit", "terms", "conditions", "policy", "eligibility", "how does",
    "when does", "what does", "which", "why", "where", "who", "information about",
    "details about", "features", "provisions", "clauses", "sections", "covered under"
]

# ENHANCED: Domain detection keywords
DOMAIN_DETECTION_KEYWORDS = {
    "insurance": ["claim", "policy", "premium", "coverage", "deductible", "bajaj insurance", "sum insured", "cashless", "network hospital", "co-payment", "exclusion"],
    "physics": ["newton", "force", "motion", "velocity", "acceleration", "principia", "mechanics", "mass", "gravity", "inertia"],
    "legal": ["law", "legal", "rights", "article", "clause", "constitution", "court", "jurisdiction", "statute", "regulation", "provision"],
    "academic": ["research", "study", "analysis", "methodology", "conclusion", "hypothesis", "theory", "academic", "scholarly"],
    "medical": ["patient", "treatment", "diagnosis", "symptoms", "therapy", "clinical", "medical", "healthcare", "disease", "medicine"]
}

# ADDED: Bearer Token Authentication
security = HTTPBearer()

# ADDED: Expected Bearer Token (you'll set this as environment variable)
EXPECTED_BEARER_TOKEN = os.getenv("BEARER_TOKEN", "9a1163c13e8927960b857a674794a62c57baf588998981151b0753a4d6d17905")

def verify_bearer_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify the bearer token provided in the Authorization header."""
    if credentials.credentials != EXPECTED_BEARER_TOKEN:
        logger.warning(f"❌ Invalid bearer token attempted: {credentials.credentials[:10]}...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

# ENHANCED: Pydantic Models with missing methods
class PolicyInfoResponse(BaseModel):
    question: str
    answer: str
    confidence: float
    sources: List[str]
    reasoning_chain: List[str] = []
    matched_sections: List[str] = []

class ComponentAnalysisResponse(BaseModel):
    query: str
    decision: str
    confidence: float
    matched_components: Dict[str, Any]
    total_score: float
    reasoning_chain: List[str]
    clause_traceability: List[Dict[str, Any]]
    sources: List[str]

class ClaimDecisionResponse(BaseModel):
    claim_id: str
    decision: str  # "APPROVED", "REJECTED", "NEEDS_REVIEW"
    coverage_amount: Optional[float]
    deductible: Optional[float]
    reasoning: List[str]
    policy_clauses: List[str]
    confidence: float
    risk_factors: List[str]

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    conversation_history: Optional[List[Dict[str, str]]] = []

class QueryResponse(BaseModel):
    decision: str
    confidence_score: float
    reasoning_chain: List[str]
    evidence_sources: List[str]
    timestamp: str
    query_type: Optional[str] = None
    domain: Optional[str] = None

class GeneralDocumentResponse(BaseModel):
    query_type: str
    domain: str
    answer: str
    confidence: str
    source_documents: List[str]
    reasoning: Optional[str] = None

class HackRxRequest(BaseModel):
    documents: HttpUrl = Field(..., description="URL to the policy document (PDF, DOCX, or TXT).")
    questions: List[str] = Field(..., description="A list of questions to ask about the document.")

class HackRxResponse(BaseModel):
    answers: List[str]

class ClaimComponent(BaseModel):
    component: str
    amount: float
    reason: str

class ClaimBreakdown(BaseModel):
    approved: List[ClaimComponent] = []
    rejected: List[ClaimComponent] = []

class PolicyReference(BaseModel):
    title: str
    note: str

class StructuredClaimResponse(BaseModel):
    status: str
    confidence: float
    approvedAmount: float
    rejectedAmount: float
    breakdown: ClaimBreakdown
    keyPolicyReferences: List[PolicyReference]
    summary: str

class UnifiedResponse(BaseModel):
    query_type: str
    domain: str
    answer: str
    decision: Optional[str] = None
    approved_amount: Optional[float] = None
    policy_references: Optional[List[str]] = None
    confidence: str
    source_documents: List[str]

# UPDATED: Enhanced HackRx response model with required fields
class EnhancedHackRxResponse(BaseModel):
    success: bool = True
    processing_time_seconds: Optional[float] = None
    timestamp: Optional[str] = None
    message: Optional[str] = None
    answers: List[Union[StructuredClaimResponse, PolicyInfoResponse, GeneralDocumentResponse]]

# NEW: Cache Entry with TTL
class CacheEntry:
    def __init__(self, value: Any):
        self.value = value
        self.created_at = datetime.now()
        self.accessed_at = datetime.now()

# ENHANCED: Session Manager
class SessionManager:
    @staticmethod
    async def get_or_create_session(document_hash: str) -> 'RAGSystem':
        """Get existing session or create new one with proper cleanup."""
        current_time = time.time()
        
        # Clean expired sessions
        expired_sessions = [
            session_id for session_id, (session, timestamp) in ACTIVE_SESSIONS.items()
            if current_time - timestamp > SESSION_TTL
        ]
        
        for session_id in expired_sessions:
            session, _ = ACTIVE_SESSIONS.pop(session_id)
            await session.async_cleanup()
            logger.info(f"🗑️ Cleaned expired session: {session_id}")
        
        if document_hash in ACTIVE_SESSIONS:
            session, timestamp = ACTIVE_SESSIONS[document_hash]
            # Update timestamp
            ACTIVE_SESSIONS[document_hash] = (session, current_time)
            logger.info(f"♻️ Reusing existing session: {document_hash}")
            return session
        
        # Create new session
        session = RAGSystem(session_id=document_hash)
        ACTIVE_SESSIONS[document_hash] = (session, current_time)
        logger.info(f"🆕 Created new session: {document_hash}")
        return session

    @staticmethod
    def get_cached_response(query_hash: str, doc_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached response if available and not expired."""
        cache_key = f"{query_hash}_{doc_hash}"
        if cache_key in QUERY_CACHE:
            response, timestamp = QUERY_CACHE[cache_key]
            if time.time() - timestamp < CACHE_TTL:
                return response
            else:
                del QUERY_CACHE[cache_key]
        return None

    @staticmethod
    def cache_response(query_hash: str, doc_hash: str, response: Dict[str, Any]):
        """Cache response with timestamp."""
        cache_key = f"{query_hash}_{doc_hash}"
        QUERY_CACHE[cache_key] = (response, time.time())

# ENHANCED: Utility functions
async def cleanup_expired_cache():
    """Clean up expired cache entries."""
    async with _cache_lock:
        global DOCUMENT_CACHE
        current_time = datetime.now()
        expired_keys = []
        
        for key, entry in DOCUMENT_CACHE.items():
            if current_time - entry.created_at > CACHE_TTL_HOURS:
                expired_keys.append(key)
        
        for key in expired_keys:
            del DOCUMENT_CACHE[key]
            logger.info(f"🗑️ Expired cache entry removed: {key[:8]}")
        
        # Also enforce size limit
        if len(DOCUMENT_CACHE) > MAX_CACHE_SIZE:
            # Remove oldest entries (LRU)
            sorted_entries = sorted(
                DOCUMENT_CACHE.items(),
                key=lambda x: x[1].accessed_at
            )
            
            while len(DOCUMENT_CACHE) > MAX_CACHE_SIZE:
                oldest_key = sorted_entries.pop(0)[0]
                del DOCUMENT_CACHE[oldest_key]
                logger.info(f"🗑️ LRU cache entry removed: {oldest_key[:8]}")

async def safe_cache_get(key: str) -> Optional[Any]:
    """Thread-safe cache get operation with TTL check."""
    async with _cache_lock:
        entry = DOCUMENT_CACHE.get(key)
        if entry:
            # Check TTL
            if datetime.now() - entry.created_at > CACHE_TTL_HOURS:
                del DOCUMENT_CACHE[key]
                logger.info(f"🗑️ TTL expired for cache entry: {key[:8]}")
                return None
            
            # Update access time
            entry.accessed_at = datetime.now()
            return entry.value
        return None

async def safe_cache_set(key: str, value: Any) -> None:
    """Thread-safe cache set operation with automatic cleanup."""
    async with _cache_lock:
        DOCUMENT_CACHE[key] = CacheEntry(value)
        await cleanup_expired_cache()

def get_embedding_model():
    """Get the global embedding model."""
    global embedding_model
    return embedding_model

def get_document_persist_dir(document_hash: str) -> str:
    """Get document-specific persistence directory."""
    return f"{PERSISTENT_CHROMA_DIR}_{document_hash}"

# ENHANCED: Domain detection functions
def detect_document_domain_batch(documents: List[Document]) -> str:
    """Detect the domain of the document set based on combined content analysis."""
    # Combine content from first 10 documents for domain detection
    combined_content = ' '.join([doc.page_content for doc in documents[:10]])
    content_lower = combined_content.lower()
    
    domain_scores = {}
    for domain, keywords in DOMAIN_DETECTION_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword in content_lower)
        if score > 0:
            domain_scores[domain] = score
    
    if not domain_scores:
        return "general"
    
    # Return domain with highest score
    detected_domain = max(domain_scores, key=domain_scores.get)
    logger.info(f"🔍 Document domain detected: {detected_domain} (score: {domain_scores[detected_domain]})")
    return detected_domain

async def detect_document_domain_async(documents: List[Document]) -> str:
    """Async version of domain detection."""
    return await asyncio.to_thread(detect_document_domain_batch, documents)

# ENHANCED: Query classification with better logic
async def classify_query_enhanced(query: str, domain: str) -> Tuple[str, str]:
    """Enhanced query classification with better logic and fallback handling."""
    query_lower = query.lower()
    
    # Enhanced keyword sets with more comprehensive coverage
    amount_patterns = [
        r'\$[\d,]+\.?\d*', r'[\d,]+\s*dollars?', r'[\d,]+\s*USD',
        r'cost of', r'price of', r'amount of', r'total.*[\d,]+'
    ]
    
    claim_keywords = [
        'claim', 'damage', 'loss', 'accident', 'incident', 'injury',
        'broken', 'stolen', 'fire', 'flood', 'collision', 'liability',
        'coverage', 'deductible', 'reimbursement', 'compensation'
    ]
    
    policy_keywords = [
        'policy', 'coverage', 'benefit', 'premium', 'terms', 'conditions',
        'exclusion', 'limit', 'what is covered', 'does my policy',
        'eligible', 'qualification', 'requirement'
    ]
    
    component_keywords = [
        'component', 'part', 'section', 'clause', 'provision',
        'breakdown', 'analysis', 'structure', 'elements'
    ]
    
    # Check for amounts using enhanced currency detection
    extracted_amounts = extract_amounts_from_query(query)
    has_amounts = len(extracted_amounts) > 0

    # Log detected amounts with currency
    if extracted_amounts:
        amount_info = [f"{amt['formatted']}" for amt in extracted_amounts]
        logger.info(f"💰 Detected amounts: {', '.join(amount_info)}")

    
    # Calculate keyword scores
    claim_score = sum(1 for keyword in claim_keywords if keyword in query_lower)
    policy_score = sum(1 for keyword in policy_keywords if keyword in query_lower)
    component_score = sum(1 for keyword in component_keywords if keyword in query_lower)
    
    # Enhanced classification logic
    if domain == "insurance":
        # Priority 1: Explicit component analysis requests
        if component_score > 0 or 'analyze' in query_lower or 'break down' in query_lower:
            return "INSURANCE_COMPONENT_ANALYSIS", "insurance"
        
        # Priority 2: Claims processing (amounts + claim indicators)
        elif has_amounts and claim_score > 0:
            return "INSURANCE_CLAIM_PROCESSING", "insurance"
        
        # Priority 3: Claims without amounts but strong claim indicators
        elif claim_score >= 2:
            return "INSURANCE_CLAIM_PROCESSING", "insurance"
        
        # Priority 4: Policy information requests
        elif policy_score > 0 or any(phrase in query_lower for phrase in ['what does', 'am i covered', 'is covered']):
            return "INSURANCE_POLICY_INFO", "insurance"
        
        # Priority 5: General insurance queries with amounts
        elif has_amounts:
            return "INSURANCE_CLAIM_PROCESSING", "insurance"
        
        # Default: Policy information for insurance domain
        else:
            return "INSURANCE_POLICY_INFO", "insurance"
    
    # Non-insurance domains
    elif component_score > 0:
        return "COMPONENT_ANALYSIS", domain
    else:
        return "GENERAL_QA", domain

def extract_amounts_from_query(query: str) -> List[Dict[str, Any]]:
    """Extract monetary amounts from query text with currency detection."""
    import re
    
    # Enhanced patterns for multiple currencies
    currency_patterns = [
        # Indian Rupees
        (r'₹\s*([0-9,]+\.?[0-9]*)', 'INR', '₹'),
        (r'rs\.?\s*([0-9,]+\.?[0-9]*)', 'INR', 'Rs.'),
        (r'rupees?\s*([0-9,]+\.?[0-9]*)', 'INR', 'Rupees'),
        (r'([0-9,]+\.?[0-9]*)\s*rupees?', 'INR', 'Rupees'),
        (r'inr\s*([0-9,]+\.?[0-9]*)', 'INR', 'INR'),
        
        # US Dollars
        (r'\$\s*([0-9,]+\.?[0-9]*)', 'USD', '$'),
        (r'([0-9,]+\.?[0-9]*)\s*dollars?', 'USD', 'Dollars'),
        (r'usd\s*([0-9,]+\.?[0-9]*)', 'USD', 'USD'),
        
        # Euros
        (r'€\s*([0-9,]+\.?[0-9]*)', 'EUR', '€'),
        (r'([0-9,]+\.?[0-9]*)\s*euros?', 'EUR', 'Euros'),
        (r'eur\s*([0-9,]+\.?[0-9]*)', 'EUR', 'EUR'),
        
        # British Pounds
        (r'£\s*([0-9,]+\.?[0-9]*)', 'GBP', '£'),
        (r'([0-9,]+\.?[0-9]*)\s*pounds?', 'GBP', 'Pounds'),
        (r'gbp\s*([0-9,]+\.?[0-9]*)', 'GBP', 'GBP'),
        
        # Generic amount patterns (no specific currency)
        (r'amount\s*of\s*([0-9,]+\.?[0-9]*)', 'UNKNOWN', 'Amount'),
        (r'cost\s*of\s*([0-9,]+\.?[0-9]*)', 'UNKNOWN', 'Cost'),
        (r'price\s*of\s*([0-9,]+\.?[0-9]*)', 'UNKNOWN', 'Price'),
    ]
    
    amounts = []
    query_lower = query.lower()
    
    for pattern, currency_code, currency_name in currency_patterns:
        matches = re.finditer(pattern, query_lower, re.IGNORECASE)
        for match in matches:
            try:
                # Remove commas and convert to float
                amount_str = match.group(1).replace(',', '')
                amount = float(amount_str)
                amounts.append({
                    'amount': amount,
                    'currency_code': currency_code,
                    'currency_name': currency_name,
                    'original_text': match.group(0),
                    'formatted': f"{currency_name} {amount:,.2f}" if currency_code != 'UNKNOWN' else f"{amount:,.2f}"
                })
            except (ValueError, IndexError):
                continue
    
    return amounts

def detect_document_currency(documents: List[Document]) -> str:
    """Detect the primary currency used in the document."""
    currency_indicators = {
        'INR': ['₹', 'rupees', 'rupee', 'rs.', 'rs ', 'inr', 'indian rupee'],
        'USD': ['$', 'dollars', 'dollar', 'usd', 'us dollar'],
        'EUR': ['€', 'euros', 'euro', 'eur'],
        'GBP': ['£', 'pounds', 'pound', 'gbp', 'british pound']
    }
    
    # Combine content from documents
    combined_content = ' '.join([doc.page_content.lower() for doc in documents[:5]])
    
    currency_scores = {}
    for currency, indicators in currency_indicators.items():
        score = sum(1 for indicator in indicators if indicator in combined_content)
        if score > 0:
            currency_scores[currency] = score
    
    if currency_scores:
        detected_currency = max(currency_scores, key=currency_scores.get)
        logger.info(f"💱 Document currency detected: {detected_currency}")
        return detected_currency
    
    return 'INR'  # Default to INR for Indian insurance documents

def assess_query_complexity(query: str) -> str:
    """Assess query complexity for appropriate handling."""
    word_count = len(query.split())
    
    if word_count <= 5:
        return "SIMPLE"
    elif word_count <= 15:
        return "MEDIUM"
    else:
        return "COMPLEX"

# ENHANCED: RAG System with async support
class RAGSystem:
    def __init__(self, session_id: str = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.documents = []
        self.vector_store = None
        self.retrievers = {}
        self.current_domain = None
        self.document_hash = None
        self._processing_lock = asyncio.Lock()
        self.processed_files = []
        self.structured_info_cache = {}
        self.document_domain = "general"
        self.domain_config = DOMAIN_CONFIG["general"]
        self.bm25_retriever = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Don't cleanup in session mode - handled by SessionManager
        if not hasattr(self, 'session_id'):
            await self.async_cleanup()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup_documents()

    async def async_cleanup(self):
        """Proper async cleanup."""
        try:
            # Wait for any pending operations
            await asyncio.sleep(0.1)
            
            # Cleanup resources
            if hasattr(self, 'vector_store') and self.vector_store:
                try:
                    # Persist vector store before cleanup
                    if hasattr(self.vector_store, 'persist'):
                        self.vector_store.persist()
                except Exception as e:
                    logger.warning(f"⚠️ Error persisting vector store: {e}")
            
            self.cleanup_documents()
            logger.info(f"🧹 Async cleanup completed for session: {self.session_id}")
        except Exception as e:
            logger.error(f"❌ Error in async cleanup: {e}")

    def cleanup_documents(self):
        """Clear document memory to prevent leaks."""
        logger.info(f"🧹 Cleaning up RAG System session: {self.session_id}")
        
        # FIXED: Proper vector store cleanup
        if self.vector_store:
            try:
                if hasattr(self.vector_store, '_client') and self.vector_store._client:
                    self.vector_store._client.reset()
                logger.info("✅ Vector store client reset")
            except Exception as e:
                logger.warning(f"⚠️ Error resetting vector store client: {e}")
            finally:
                self.vector_store = None

        self.documents.clear()
        self.processed_files.clear()
        self.structured_info_cache.clear()
        logger.info(f"✅ RAG System session {self.session_id} cleaned up")

    def _calculate_content_hash(self, documents: List[Document]) -> str:
        """Calculate hash based on document content for unique identification."""
        # Use more content and longer hash to reduce collision risk
        sample_content = "".join([doc.page_content for doc in documents[:10]]) # More docs
        full_hash = hashlib.sha256(sample_content.encode()).hexdigest() # SHA256 instead of MD5
        return full_hash[:16] # 16 chars instead of 8 for lower collision probability

    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of file content for unique identification."""
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()[:16] # 16 chars for consistency
        except Exception as e:
            logger.error(f"❌ Error calculating file hash: {str(e)}")
            return str(hash(file_path))[:16] # Fallback to path hash

    async def _process_single_file(self, file_path: str) -> List[Document]:
        """Process a single file and return documents with structured information extraction."""
        try:
            file_extension = os.path.splitext(file_path)[1].lower()
            
            # FIXED: More specific error handling for different file types
            if file_extension == '.pdf':
                loader = PyMuPDFLoader(file_path)
            elif file_extension == '.docx':
                loader = Docx2txtLoader(file_path)
            elif file_extension == '.txt':
                loader = TextLoader(file_path, encoding='utf-8')
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")

            # FIXED: Wrap blocking operations in to_thread
            documents = await asyncio.to_thread(loader.load)
            
            # Extract structured information during processing
            for doc in documents:
                structured_info = self._extract_structured_info(doc.page_content)
                doc.metadata.update({
                    'source_file': os.path.basename(file_path),
                    'file_type': file_extension,
                    'processed_at': datetime.now().isoformat(),
                    'session_id': self.session_id,
                    **structured_info # Add extracted structured info
                })

            logger.info(f"✅ Processed {len(documents)} documents from {os.path.basename(file_path)}")
            return documents

        except (FileNotFoundError, PermissionError) as file_error:
            logger.error(f"❌ File access error for {file_path}: {str(file_error)}")
            raise HTTPException(status_code=400, detail=f"Cannot access file {os.path.basename(file_path)}: {str(file_error)}")
        except ValueError as value_error:
            logger.error(f"❌ File format error for {file_path}: {str(value_error)}")
            raise HTTPException(status_code=400, detail=f"Invalid file format {os.path.basename(file_path)}: {str(value_error)}")
        except Exception as e:
            logger.error(f"❌ Unexpected error processing {file_path}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to process document {os.path.basename(file_path)}: {str(e)}")

    def _extract_structured_info(self, text: str) -> Dict[str, Any]:
        """Extract common policy information patterns."""
        info = {}
        
        # Grace period extraction
        grace_patterns = [
            r"grace\s+period[^\d]*(\d+)\s*days?",
            r"(\d+)\s*days?\s+grace\s+period",
            r"grace\s+period[:\s]+(\d+)"
        ]
        
        for pattern in grace_patterns:
            if match := re.search(pattern, text, re.IGNORECASE):
                info["grace_period_days"] = match.group(1)
                break
        
        # Waiting period extraction
        waiting_patterns = [
            r"waiting\s+period[^\d]*(\d+)\s*(?:months?|years?)",
            r"(\d+)\s*(?:months?|years?)\s+waiting\s+period",
            r"initial\s+waiting\s+period[^\d]*(\d+)"
        ]
        
        for pattern in waiting_patterns:
            if match := re.search(pattern, text, re.IGNORECASE):
                info["waiting_period"] = match.group(0)
                break
        
        # Sum insured extraction
        sum_insured_patterns = [
            r"sum\s+insured[^\d]*(\d{1,3}(?:,\d{3})*)",
            r"coverage\s+amount[^\d]*(\d{1,3}(?:,\d{3})*)",
            r"insured\s+amount[^\d]*(\d{1,3}(?:,\d{3})*)"
        ]
        
        for pattern in sum_insured_patterns:
            if match := re.search(pattern, text, re.IGNORECASE):
                info["sum_insured"] = match.group(1)
                break
        
        return info

    async def setup_vector_store_optimized(self, documents: List[Document], domain: str) -> bool:
        """Optimized vector store setup with persistence check."""
        try:
            async with self._processing_lock:
                embedding_model = get_embedding_model()
                document_persist_dir = get_document_persist_dir(self.document_hash)
                
                # Check if valid persisted store exists
                if await self._load_existing_vector_store(document_persist_dir, embedding_model):
                    return True
                
                # Create new vector store with optimized chunking
                optimized_documents = await self._optimize_documents_for_processing(documents)
                
                # Use async thread for vector store creation
                self.vector_store = await asyncio.to_thread(
                    self._create_vector_store_sync,
                    optimized_documents,
                    embedding_model,
                    document_persist_dir
                )
                
                logger.info(f"✅ Created new vector store with {len(optimized_documents)} chunks")
                return True
                
        except Exception as e:
            logger.error(f"❌ Error setting up vector store: {e}")
            return False

    async def _load_existing_vector_store(self, persist_dir: str, embedding_model) -> bool:
        """Load existing vector store if valid."""
        if os.path.exists(f"{persist_dir}/chroma.sqlite3"):
            try:
                self.vector_store = await asyncio.to_thread(
                    Chroma,
                    persist_directory=persist_dir,
                    embedding_function=embedding_model
                )
                
                # Verify store is working
                test_search = await asyncio.to_thread(
                    self.vector_store.similarity_search,
                    "test query",
                    k=1
                )
                
                if test_search:
                    logger.info(f"✅ Loaded existing vector store: {persist_dir}")
                    return True
                else:
                    raise Exception("Empty search results")
                    
            except Exception as e:
                logger.warning(f"⚠️ Existing store corrupted, recreating: {e}")
                shutil.rmtree(persist_dir, ignore_errors=True)
        
        return False

    def _create_vector_store_sync(self, documents, embedding_model, persist_dir):
        """Synchronous vector store creation for thread execution."""
        return Chroma.from_documents(
            documents=documents,
            embedding=embedding_model,
            persist_directory=persist_dir
        )

    async def _optimize_documents_for_processing(self, documents: List[Document]) -> List[Document]:
        """Optimize document chunks for better retrieval."""
        optimized_docs = []
        
        for doc in documents:
            # Split large documents into smaller, overlapping chunks
            if len(doc.page_content) > 2000:
                chunks = self._smart_chunk_document(doc)
                optimized_docs.extend(chunks)
            else:
                optimized_docs.append(doc)
        
        return optimized_docs

    def _smart_chunk_document(self, doc: Document, chunk_size: int = 1500, overlap: int = 200) -> List[Document]:
        """Smart chunking that preserves context."""
        content = doc.page_content
        chunks = []
        
        for i in range(0, len(content), chunk_size - overlap):
            chunk_content = content[i:i + chunk_size]
            
            # Create new document with enhanced metadata
            chunk_metadata = doc.metadata.copy()
            chunk_metadata.update({
                'chunk_id': len(chunks),
                'chunk_start': i,
                'chunk_end': min(i + chunk_size, len(content)),
                'parent_doc_id': doc.metadata.get('source', 'unknown')
            })
            
            chunks.append(Document(
                page_content=chunk_content,
                metadata=chunk_metadata
            ))
        
        return chunks

    async def load_and_process_documents(self, file_paths: List[str]) -> Dict[str, Any]:
        """Process multiple files with enhanced error handling and domain-adaptive chunking."""
        try:
            # Use robust processing
            result = await self.robust_document_processing(file_paths)
            all_docs = result["documents"]

            # Extract processed and skipped files info
            processed_files = []
            skipped_files = []
            for res in result["results"]:
                if res["status"] == "success":
                    processed_files.append(os.path.basename(res["file"]))
                else:
                    skipped_files.append(os.path.basename(res["file"]))

            if not all_docs:
                logger.warning("⚠️ No documents were successfully processed")
                raise HTTPException(status_code=500, detail="No documents could be processed successfully")

            # FIXED: Domain detection runs once per document set, not per chunk
            self.document_domain = detect_document_domain_batch(all_docs)
            self.domain_config = DOMAIN_CONFIG.get(self.document_domain, DOMAIN_CONFIG["general"])

            # Update metadata with detected domain
            for doc in all_docs:
                doc.metadata['domain'] = self.document_domain

            # Detect document currency
            document_currency = detect_document_currency(all_docs)
            for doc in all_docs:
                doc.metadata['detected_currency'] = document_currency

            logger.info(f"📄 Document domain: {self.document_domain}")

            # ENHANCED: Domain-adaptive text splitter settings
            chunk_size = self.domain_config["chunk_size"]
            chunk_overlap = self.domain_config["chunk_overlap"]
            logger.info(f"🔧 Using domain-adaptive chunking for '{self.document_domain}': chunk_size={chunk_size}, overlap={chunk_overlap}")

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )

            # FIXED: Wrap blocking operations
            chunked_docs = await asyncio.to_thread(text_splitter.split_documents, all_docs)
            chunked_docs = [doc for doc in chunked_docs if len(doc.page_content.strip()) > 50]

            self.documents = chunked_docs
            self.processed_files = processed_files

            # Calculate document hash for unique identification
            self.document_hash = self._calculate_content_hash(chunked_docs)

            logger.info(f"📄 Created {len(chunked_docs)} chunks from {len(processed_files)} files (domain: {self.document_domain}, hash: {self.document_hash})")

            return {
                'documents': chunked_docs,
                'processed_files': processed_files,
                'skipped_files': skipped_files,
                'total_chunks': len(chunked_docs),
                'domain': self.document_domain,
                'document_hash': self.document_hash
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"❌ Error in document processing: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error during document processing: {str(e)}")

    async def robust_document_processing(self, file_paths: List[str]) -> Dict[str, Any]:
        """Improved error handling with detailed logging."""
        processing_results = []
        for file_path in file_paths:
            try:
                result = await self._process_single_file(file_path)
                processing_results.append({"file": file_path, "status": "success", "documents": result})
            except HTTPException:
                raise # Re-raise HTTP exceptions as-is
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                processing_results.append({"file": file_path, "status": "failed", "error": str(e)})

        # Don't fail entirely if some files processed successfully
        successful_docs = []
        for r in processing_results:
            if r["status"] == "success":
                successful_docs.extend(r["documents"])

        if not successful_docs:
            raise HTTPException(status_code=500, detail="No documents processed successfully")

        return {
            "results": processing_results,
            "documents": successful_docs,
            "total_successful": len([r for r in processing_results if r["status"] == "success"]),
            "total_failed": len([r for r in processing_results if r["status"] == "failed"])
        }

    async def setup_retrievers(self, persist_directory: str = PERSISTENT_CHROMA_DIR):
        """Initialize vector store and BM25 retriever with proper cleanup."""
        global embedding_model

        if not self.documents:
            logger.warning("⚠️ No documents available for retriever setup")
            return False

        try:
            # FIXED: Clean up existing vector store properly
            if self.vector_store:
                try:
                    if hasattr(self.vector_store, '_client') and self.vector_store._client:
                        self.vector_store._client.reset()
                    logger.info("✅ Previous vector store client reset")
                except Exception as e:
                    logger.warning(f"⚠️ Error resetting previous vector store: {e}")
                finally:
                    self.vector_store = None

            # FIXED: Document-specific storage to prevent cross-document contamination
            document_persist_dir = f"{persist_directory}_{self.document_domain}_{self.document_hash}"

            if os.path.exists(f"{document_persist_dir}/chroma.sqlite3"):
                logger.info(f"✅ Loading cached vector store for document {self.document_hash}")
                self.vector_store = Chroma(
                    persist_directory=document_persist_dir,
                    embedding_function=embedding_model
                )
            else:
                logger.info(f"🔍 Creating new vector store for document {self.document_hash}...")
                # FIXED: Wrap blocking operations in to_thread
                self.vector_store = await asyncio.to_thread(
                    Chroma.from_documents,
                    documents=self.documents,
                    embedding=embedding_model,
                    persist_directory=document_persist_dir
                )
                logger.info(f"✅ New vector store created and persisted for document {self.document_hash}")

            logger.info("🔍 Setting up BM25 retriever...")
            # FIXED: Wrap blocking operations
            self.bm25_retriever = await asyncio.to_thread(BM25Retriever.from_documents, self.documents)

            # Domain-adaptive retrieval settings
            self.bm25_retriever.k = self.domain_config["semantic_search_k"] + 3 # Slightly higher for BM25

            logger.info("✅ Retrievers setup complete")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to setup retrievers: {str(e)}")
            return False

    async def setup_retrievers_async(self):
        """Async version of setup_retrievers."""
        return await self.setup_retrievers()

    async def retrieve_and_rerank_optimized(self, query: str, k: int = 10) -> Tuple[List[Document], List[float]]:
        """Optimized retrieval with better scoring."""
        try:
            # Get initial candidates
            initial_docs = await asyncio.to_thread(
                self.vector_store.similarity_search_with_score,
                query,
                k=k*2  # Get more candidates for reranking
            )
            
            if not initial_docs:
                return [], []
            
            # Extract documents and scores
            documents = [doc for doc, score in initial_docs]
            similarity_scores = [1.0 - score for doc, score in initial_docs]  # Convert distance to similarity
            
            # Rerank based on query relevance and document quality
            reranked_results = await self._rerank_documents(query, documents, similarity_scores)
            
            # Return top k results
            final_docs = reranked_results[:k]
            final_scores = similarity_scores[:k]
            
            return final_docs, final_scores
            
        except Exception as e:
            logger.error(f"❌ Error in retrieval: {e}")
            return [], []

    async def _rerank_documents(self, query: str, documents: List[Document], scores: List[float]) -> List[Document]:
        """Advanced reranking based on multiple factors."""
        # Score documents based on multiple factors
        enhanced_scores = []
        
        for i, (doc, base_score) in enumerate(zip(documents, scores)):
            # Factor 1: Base similarity score
            similarity_factor = base_score
            
            # Factor 2: Document length appropriateness
            length_factor = self._calculate_length_factor(doc.page_content)
            
            # Factor 3: Metadata relevance
            metadata_factor = self._calculate_metadata_relevance(doc.metadata, query)
            
            # Factor 4: Content quality
            quality_factor = self._calculate_content_quality(doc.page_content)
            
            # Combined score with weights
            combined_score = (
                0.4 * similarity_factor +
                0.2 * length_factor +
                0.2 * metadata_factor +
                0.2 * quality_factor
            )
            
            enhanced_scores.append((combined_score, i, doc))
        
        # Sort by enhanced score
        enhanced_scores.sort(key=lambda x: x[0], reverse=True)
        
        return [doc for score, idx, doc in enhanced_scores]

    def _calculate_length_factor(self, content: str) -> float:
        """Calculate score based on content length appropriateness."""
        length = len(content)
        if 500 <= length <= 2000:  # Optimal length range
            return 1.0
        elif length < 500:
            return length / 500  # Penalize very short content
        else:
            return max(0.5, 2000 / length)  # Penalize very long content

    def _calculate_metadata_relevance(self, metadata: Dict, query: str) -> float:
        """Calculate relevance based on metadata."""
        relevance_score = 0.5  # Base score
        
        # Check if source filename is relevant
        source = metadata.get('source', '').lower()
        query_lower = query.lower()
        
        if any(term in source for term in query_lower.split()):
            relevance_score += 0.3
        
        # Check for section headers or titles
        if 'section' in metadata and any(term in metadata['section'].lower() for term in query_lower.split()):
            relevance_score += 0.2
        
        return min(1.0, relevance_score)

    def _calculate_content_quality(self, content: str) -> float:
        """Calculate content quality score."""
        # Basic quality indicators
        quality_score = 0.5
        
        # Has structured content (bullet points, numbers)
        if any(marker in content for marker in ['•', '-', '1.', '2.', 'a)', 'b)']):
            quality_score += 0.2
        
        # Has proper sentences
        sentences = content.split('.')
        if len(sentences) >= 2:
            quality_score += 0.2
        
        # Not too repetitive
        words = content.split()
        if len(set(words)) / max(len(words), 1) > 0.3:  # Unique word ratio
            quality_score += 0.1
        
        return min(1.0, quality_score)

    def retrieve_and_rerank(self, query: str, top_k: int = None) -> Tuple[List[Document], List[float]]:
        """Retrieve and rerank documents based on query with enhanced search."""
        global reranker

        if top_k is None:
            top_k = self.domain_config["context_docs"]

        if not self.vector_store or not self.bm25_retriever:
            logger.warning("⚠️ Retrievers not initialized")
            return [], []

        try:
            # Use enhanced search for better retrieval
            retrieved_docs = self.enhanced_document_search(query, top_k=15)

            if not retrieved_docs:
                logger.warning("⚠️ No documents retrieved")
                return [], []

            # Rerank using cross-encoder
            query_doc_pairs = [[query, doc.page_content] for doc in retrieved_docs]
            similarity_scores = reranker.predict(query_doc_pairs)

            # Sort by relevance score
            doc_score_pairs = sorted(
                list(zip(retrieved_docs, similarity_scores)),
                key=lambda x: x[1],
                reverse=True
            )

            top_docs = [pair[0] for pair in doc_score_pairs[:top_k]]
            top_scores = [float(pair[1]) for pair in doc_score_pairs[:top_k]]

            logger.info(f"🔍 Retrieved and reranked {len(top_docs)} documents for domain '{self.document_domain}'")
            return top_docs, top_scores

        except Exception as e:
            logger.error(f"❌ Error in retrieve_and_rerank: {str(e)}")
            return [], []

    def enhanced_document_search(self, query: str, top_k: int = None) -> List[Document]:
        """Multi-strategy document retrieval with domain adaptation."""
        if top_k is None:
            top_k = self.domain_config["semantic_search_k"]

        all_docs = []

        # Strategy 1: Semantic search
        if self.vector_store:
            semantic_results = self.vector_store.similarity_search(query, k=top_k)
            all_docs.extend(semantic_results)

        # Strategy 2: BM25 search
        if self.bm25_retriever:
            bm25_results = self.bm25_retriever.get_relevant_documents(query)
            all_docs.extend(bm25_results[:top_k])

        # Strategy 3: Domain-specific pattern matching
        query_lower = query.lower()
        if self.document_domain == "insurance":
            # Insurance-specific patterns
            if "grace period" in query_lower:
                pattern_docs = self._pattern_search(r"grace\s+period")
                all_docs.extend(pattern_docs[:5])
            if "waiting period" in query_lower:
                pattern_docs = self._pattern_search(r"waiting\s+period")
                all_docs.extend(pattern_docs[:5])

        elif self.document_domain == "physics":
            # Physics-specific patterns
            if "newton" in query_lower or "principia" in query_lower:
                pattern_docs = self._pattern_search(r"newton|principia|axiom|proposition")
                all_docs.extend(pattern_docs[:5])

        elif self.document_domain == "legal":
            # Legal-specific patterns
            if "article" in query_lower or "clause" in query_lower:
                pattern_docs = self._pattern_search(r"article|clause|section|subsection")
                all_docs.extend(pattern_docs[:5])

        # Remove duplicates while preserving order
        seen_content = set()
        unique_docs = []
        for doc in all_docs:
            if doc.page_content not in seen_content:
                unique_docs.append(doc)
                seen_content.add(doc.page_content)

        return unique_docs[:top_k]

    def _pattern_search(self, pattern: str, max_results: int = 5) -> List[Document]:
        """Search for documents containing specific patterns."""
        pattern_docs = []
        compiled_pattern = re.compile(pattern, re.IGNORECASE)

        for doc in self.documents:
            if compiled_pattern.search(doc.page_content):
                pattern_docs.append(doc)
                if len(pattern_docs) >= max_results:
                    break

        return pattern_docs

# ENHANCED: Decision Engine with all missing methods
class DecisionEngine:
    def __init__(self):
        self.insurance_components = {
            "coverage": ["coverage", "covered", "benefits", "protection", "insured"],
            "exclusions": ["exclude", "not covered", "exception", "limitation"],
            "deductible": ["deductible", "out-of-pocket", "copay", "self-insured"],
            "premium": ["premium", "cost", "price", "fee", "payment"],
            "claims": ["claim", "loss", "damage", "incident", "accident"],
            "policy_terms": ["term", "period", "duration", "effective", "expiry"],
            "eligibility": ["eligible", "qualify", "requirement", "condition"]
        }

    def calculate_confidence_score(self, similarity_scores: List[float], query_match_quality: float) -> float:
        """Calculate confidence based on retrieval quality and query matching."""
        if not similarity_scores:
            return 0.0
        
        # Base confidence from similarity scores
        avg_similarity = np.mean(similarity_scores)
        max_similarity = max(similarity_scores)
        
        # Query match quality factor
        query_factor = min(1.0, query_match_quality)
        
        # Number of relevant documents factor
        doc_count_factor = min(1.0, len(similarity_scores) / 5.0)  # Optimal around 5 docs
        
        # Consistency factor (how similar are the top scores)
        if len(similarity_scores) > 1:
            consistency_factor = 1.0 - np.std(similarity_scores[:3]) / max(np.mean(similarity_scores[:3]), 0.1)
        else:
            consistency_factor = 1.0
        
        # Combined confidence with weights
        confidence = (
            0.4 * avg_similarity +
            0.3 * max_similarity +
            0.15 * query_factor +
            0.1 * doc_count_factor +
            0.05 * consistency_factor
        )
        
        return min(1.0, max(0.0, confidence))

    def generate_reasoning_chain(self, query: str, matched_docs: List[Document], decision: str, confidence: float) -> List[str]:
        """Generate detailed reasoning chain for explainability."""
        reasoning = []
        
        # Query analysis
        query_terms = self._extract_key_terms(query)
        reasoning.append(f"Query analysis: Identified key terms - {', '.join(query_terms)}")
        
        # Document matching
        reasoning.append(f"Document retrieval: Found {len(matched_docs)} relevant sections")
        
        if matched_docs:
            sources = list(set([doc.metadata.get('source', 'Unknown') for doc in matched_docs]))
            reasoning.append(f"Primary sources: {', '.join(sources[:3])}")
        
        # Decision basis
        reasoning.append(f"Decision basis: {decision}")
        
        # Confidence factors
        if confidence > 0.8:
            reasoning.append("High confidence: Strong document matches with consistent information")
        elif confidence > 0.6:
            reasoning.append("Medium confidence: Good document matches with some uncertainty")
        else:
            reasoning.append("Low confidence: Limited or unclear document matches")
        
        return reasoning

    def _extract_key_terms(self, query: str) -> List[str]:
        """Extract key terms from query for analysis."""
        # Simple keyword extraction (can be enhanced with NLP)
        stop_words = {'the', 'is', 'at', 'which', 'on', 'and', 'or', 'but', 'in', 'with', 'a', 'an'}
        words = query.lower().split()
        key_terms = [word for word in words if word not in stop_words and len(word) > 2]
        return key_terms[:5]  # Return top 5 key terms

    async def get_policy_info(self, query: str, retrieved_docs: List[Document]) -> PolicyInfoResponse:
        """Get comprehensive policy information with detailed analysis."""
        if not retrieved_docs:
            return PolicyInfoResponse(
                question=query,
                answer="No relevant policy information found in the provided documents.",
                confidence=0.0,
                sources=[],
                reasoning_chain=["No documents retrieved for analysis"],
                matched_sections=[]
            )

        try:
            # Extract context with intelligent truncation
            context = self._optimize_context_for_tokens(retrieved_docs, max_tokens=3000)
            sources = [doc.metadata.get('source', 'Unknown') for doc in retrieved_docs]
            matched_sections = [doc.page_content[:200] + "..." for doc in retrieved_docs[:3]]
            
            # Calculate similarity scores for confidence
            similarity_scores = [0.8, 0.7, 0.6]  # Placeholder - should come from retrieval
            query_match_quality = self._assess_query_match_quality(query, retrieved_docs)
            confidence = self.calculate_confidence_score(similarity_scores, query_match_quality)
            
            # Generate reasoning chain
            reasoning_chain = self.generate_reasoning_chain(query, retrieved_docs, "Policy information extracted", confidence)
            
            # Optimized prompt for accuracy and efficiency
            prompt = f"""Analyze the following policy documents to answer the user's question accurately and comprehensively.

POLICY CONTEXT:
{context}

QUESTION: {query}

INSTRUCTIONS:
- Provide a clear, accurate answer based solely on the policy information
- Include specific policy terms, coverage amounts, and conditions when available
- If information is unclear or missing, state this explicitly
- Focus on factual information from the documents
- Maintain professional insurance terminology

ANSWER:"""

            global openai_client
            response = await openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert insurance policy analyst. Provide accurate, detailed responses based strictly on the provided policy documents."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1500
            )

            answer = response.choices[0].message.content

            return PolicyInfoResponse(
                question=query,
                answer=answer,
                confidence=confidence,
                sources=list(set(sources)),
                reasoning_chain=reasoning_chain,
                matched_sections=matched_sections
            )

        except Exception as e:
            logger.error(f"❌ Error in get_policy_info: {e}")
            return PolicyInfoResponse(
                question=query,
                answer=f"Error processing policy information: {str(e)}",
                confidence=0.0,
                sources=sources if 'sources' in locals() else [],
                reasoning_chain=[f"Error occurred during processing: {str(e)}"],
                matched_sections=[]
            )

    async def get_structured_decision(self, query: str, retrieved_docs: List[Document], similarity_scores: List[float]) -> Dict[str, Any]:
        """Get structured insurance claim decision with component breakdown."""
        if not retrieved_docs:
            return {
                "decision": "INSUFFICIENT_INFO",
                "confidence": 0.0,
                "components": {},
                "reasoning": ["No relevant policy documents found"],
                "risk_factors": ["Missing policy documentation"]
            }

        try:
            # Analyze claim components
            component_analysis = await self._analyze_insurance_components(query, retrieved_docs)
            
            # Extract key information
            context = self._optimize_context_for_tokens(retrieved_docs, max_tokens=2500)
            
            # Assess claim details
            claim_details = self._extract_claim_details(query)
            
            # Calculate comprehensive confidence
            query_match_quality = self._assess_query_match_quality(query, retrieved_docs)
            confidence = self.calculate_confidence_score(similarity_scores, query_match_quality)
            
            # Generate decision reasoning
            reasoning_chain = self.generate_reasoning_chain(query, retrieved_docs, "Claim analysis completed", confidence)
            
            # Enhanced prompt for claim processing
            prompt = f"""Analyze this insurance claim against the policy terms and provide a structured decision.

POLICY CONTEXT:
{context}

CLAIM DETAILS:
Query: {query}
Extracted amounts: {[f"{amt['formatted']}" for amt in claim_details.get('currency_info', [])]}
Detected currency: {claim_details.get('currency_info', [{}])[0].get('currency_code', 'INR') if claim_details.get('currency_info') else 'INR'}
Incident type: {claim_details.get('incident_type', 'General')}

COMPONENT ANALYSIS:
{json.dumps(component_analysis, indent=2)}

INSTRUCTIONS:
- Use the detected currency from the document/query
- Convert amounts to document's base currency if needed
- Determine if the claim should be APPROVED, REJECTED, or NEEDS_REVIEW
- Identify applicable coverage amounts and deductibles in the correct currency
- List specific policy clauses that apply
- Identify any risk factors or concerns
- Provide clear reasoning for the decision

Respond in JSON format with:
- decision: APPROVED/REJECTED/NEEDS_REVIEW
- coverage_amount: numeric value or null
- deductible: numeric value or null
- applicable_clauses: list of relevant policy sections
- risk_factors: list of identified concerns
- reasoning: list of decision factors

JSON RESPONSE:"""

            global openai_client
            response = await openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert insurance claims adjuster. Analyze claims thoroughly and provide structured decisions based on policy terms."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )

            # Parse JSON response
            try:
                decision_data = json.loads(response.choices[0].message.content)
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                decision_data = {
                    "decision": "NEEDS_REVIEW",
                    "coverage_amount": None,
                    "deductible": None,
                    "applicable_clauses": [],
                    "risk_factors": ["Unable to parse decision response"],
                    "reasoning": ["JSON parsing error in decision response"]
                }

            # Enhance with additional analysis
            decision_data.update({
                "confidence": confidence,
                "components": component_analysis,
                "query_analysis": claim_details,
                "reasoning_chain": reasoning_chain,
                "sources": list(set([doc.metadata.get('source', 'Unknown') for doc in retrieved_docs]))
            })

            return decision_data

        except Exception as e:
            logger.error(f"❌ Error in get_structured_decision: {e}")
            return {
                "decision": "ERROR",
                "confidence": 0.0,
                "components": {},
                "reasoning": [f"Error processing claim decision: {str(e)}"],
                "risk_factors": ["System error during processing"],
                "error": str(e)
            }

    def _optimize_context_for_tokens(self, retrieved_docs: List[Document], max_tokens: int = 3000) -> str:
        """Intelligently truncate context to stay within token limits while preserving key information."""
        # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
        max_chars = max_tokens * 4
        
        # Prioritize documents by relevance and extract key sections
        context_parts = []
        current_chars = 0
        
        for i, doc in enumerate(retrieved_docs):
            content = doc.page_content
            source = doc.metadata.get('source', f'Document {i+1}')
            
            # Add source header
            header = f"\n--- {source} ---\n"
            
            if current_chars + len(header) + len(content) <= max_chars:
                # Full document fits
                context_parts.append(header + content)
                current_chars += len(header) + len(content)
            else:
                # Partial document - take the most relevant part
                remaining_chars = max_chars - current_chars - len(header)
                if remaining_chars > 200:  # Only add if meaningful content can fit
                    truncated_content = content[:remaining_chars] + "..."
                    context_parts.append(header + truncated_content)
                break
        
        return "\n".join(context_parts)

    async def _analyze_insurance_components(self, query: str, retrieved_docs: List[Document]) -> Dict[str, Any]:
        """Analyze insurance-specific components in the documents."""
        component_scores = {}
        
        # Combine all document content for analysis
        full_content = " ".join([doc.page_content.lower() for doc in retrieved_docs])
        query_lower = query.lower()
        
        for component, keywords in self.insurance_components.items():
            # Calculate relevance score for each component
            keyword_matches = sum(1 for keyword in keywords if keyword in full_content)
            query_matches = sum(1 for keyword in keywords if keyword in query_lower)
            
            # Score based on matches and query relevance
            content_score = keyword_matches / len(keywords)
            query_relevance = query_matches / len(keywords)
            
            final_score = (0.7 * content_score) + (0.3 * query_relevance)
            
            if final_score > 0.2:  # Increased threshold for better precision
                component_scores[component] = {
                    "score": final_score,
                    "matched_keywords": [kw for kw in keywords if kw in full_content],
                    "query_relevance": query_relevance > 0
                }
        
        return component_scores

    def _extract_claim_details(self, query: str) -> Dict[str, Any]:
        """Extract structured claim details from query with enhanced currency support."""
        details = {
            "amounts": [],
            "currency_info": [],
            "incident_type": "General",
            "dates": [],
            "locations": []
        }
        
        # Extract monetary amounts with currency
        extracted_amounts = extract_amounts_from_query(query)
        details["amounts"] = [amt['amount'] for amt in extracted_amounts]
        details["currency_info"] = extracted_amounts
        
        # Identify incident type
        incident_keywords = {
            "accident": ["accident", "collision", "crash", "hit"],
            "theft": ["theft", "stolen", "burglar", "robbery"],
            "damage": ["damage", "broken", "destroyed", "loss"],
            "medical": ["medical", "injury", "hospital", "treatment"],
            "fire": ["fire", "burn", "smoke"],
            "water": ["water", "flood", "leak", "moisture"]
        }
        
        query_lower = query.lower()
        for incident_type, keywords in incident_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                details["incident_type"] = incident_type
                break
        
        return details

    def _assess_query_match_quality(self, query: str, retrieved_docs: List[Document]) -> float:
        """Assess how well the retrieved documents match the query intent."""
        query_terms = set(self._extract_key_terms(query))
        
        if not query_terms:
            return 0.5  # Default if no clear terms
        
        match_scores = []
        
        for doc in retrieved_docs:
            doc_terms = set(doc.page_content.lower().split())
            
            # Calculate term overlap
            overlap = len(query_terms.intersection(doc_terms))
            match_score = overlap / len(query_terms)
            match_scores.append(match_score)
        
        # Return average match quality
        return np.mean(match_scores) if match_scores else 0.0

    async def get_component_analysis(self, query: str, retrieved_docs: List[Document], similarity_scores: List[float]) -> ComponentAnalysisResponse:
        """Enhanced component analysis with detailed traceability."""
        component_analysis = await self._analyze_insurance_components(query, retrieved_docs)
        
        # Calculate total score
        total_score = sum(comp["score"] for comp in component_analysis.values())
        
        # Generate decision based on component analysis
        if total_score > 1.5 and len(component_analysis) >= 3:
            decision = "COMPREHENSIVE_MATCH"
        elif total_score > 0.8:
            decision = "PARTIAL_MATCH"
        else:
            decision = "LIMITED_MATCH"
        
        # Calculate confidence
        query_match_quality = self._assess_query_match_quality(query, retrieved_docs)
        confidence = self.calculate_confidence_score(similarity_scores, query_match_quality)
        
        # Generate reasoning chain
        reasoning_chain = self.generate_reasoning_chain(query, retrieved_docs, decision, confidence)
        
        # Create clause traceability
        clause_traceability = []
        for doc in retrieved_docs[:5]:  # Top 5 documents
            clause_traceability.append({
                "source": doc.metadata.get('source', 'Unknown'),
                "section": doc.metadata.get('section', 'Main content'),
                "content_preview": doc.page_content[:150] + "...",
                "relevance_score": similarity_scores[retrieved_docs.index(doc)] if doc in retrieved_docs else 0.0
            })
        
        return ComponentAnalysisResponse(
            query=query,
            decision=decision,
            confidence=confidence,
            matched_components=component_analysis,
            total_score=total_score,
            reasoning_chain=reasoning_chain,
            clause_traceability=clause_traceability,
            sources=list(set([doc.metadata.get('source', 'Unknown') for doc in retrieved_docs]))
        )

    async def get_general_document_answer(self, query: str, context_docs: List[Document], domain: str = "general") -> GeneralDocumentResponse:
        """Handle any type of document with robust error handling and fallback."""
        global openai_client

        if not context_docs:
            return GeneralDocumentResponse(
                query_type="GENERAL_DOCUMENT_QA",
                domain=domain,
                answer="No relevant documents found to answer your question.",
                confidence="low",
                source_documents=[],
                reasoning="No document content available for analysis."
            )

        # FIXED: Robust error handling with exponential backoff
        max_retries = 3
        base_delay = 1
        
        for attempt in range(max_retries):
            try:
                context = ""
                source_files = []
                for i, doc in enumerate(context_docs):
                    context += f"\n--- Document Section {i+1} | Source: {doc.metadata.get('source_file', 'Unknown')} ---\n"
                    context += doc.page_content + "\n"
                    if doc.metadata.get('source_file'):
                        source_files.append(doc.metadata.get('source_file'))

                prompt = f"""You are a helpful document analysis assistant specialized in {domain} documents. Answer the question accurately based on the provided document content.

Document Content:
{context}

Question: {query}

Instructions:
- Provide a clear, accurate answer based solely on the document content
- If the information isn't in the document, say so clearly
- Be comprehensive but concise
- Cite specific sections, propositions, articles, or clauses when relevant
- If information requires interpretation, explain your reasoning

Provide a direct, informative answer that addresses the user's question comprehensively."""

                logger.info(f"🤖 Calling OpenAI API for general document QA (domain: {domain}, attempt: {attempt + 1})...")

                response = await openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": f"You are a helpful document analysis assistant specializing in {domain} content. Provide clear, accurate information based on the document content without creating fictional scenarios."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.1,
                    max_tokens=2000
                )

                answer = response.choices[0].message.content

                # Determine confidence based on answer quality and context relevance
                confidence = "high"
                reasoning = f"Answer generated from {len(context_docs)} relevant document sections with {domain} domain analysis"

                if "not found" in answer.lower() or "no information" in answer.lower():
                    confidence = "low"
                    reasoning = "Limited information available in the document content"
                elif "unclear" in answer.lower() or "ambiguous" in answer.lower():
                    confidence = "medium"
                    reasoning = "Some ambiguity in the source material"

                return GeneralDocumentResponse(
                    query_type="GENERAL_DOCUMENT_QA",
                    domain=domain,
                    answer=answer,
                    confidence=confidence,
                    source_documents=list(set(source_files)), # Remove duplicates
                    reasoning=reasoning
                )

            except openai.RateLimitError as e:
                logger.warning(f"⏰ Rate limit hit on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) # Exponential backoff
                    logger.info(f"🔄 Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                    continue
                else:
                    return GeneralDocumentResponse(
                        query_type="GENERAL_DOCUMENT_QA",
                        domain=domain,
                        answer="I'm currently experiencing high demand. Please try again in a moment. Based on the document content, I can see relevant information is available but cannot process it right now due to API rate limits.",
                        confidence="low",
                        source_documents=list(set(source_files)) if 'source_files' in locals() else [],
                        reasoning="OpenAI API rate limit exceeded after retries"
                    )

            except openai.APIError as e:
                logger.error(f"❌ OpenAI API error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.info(f"🔄 Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                    continue
                else:
                    return GeneralDocumentResponse(
                        query_type="GENERAL_DOCUMENT_QA",
                        domain=domain,
                        answer="I encountered a technical issue while processing your question. The document content appears relevant, but I cannot provide a complete analysis at this time. Please try again shortly.",
                        confidence="low",
                        source_documents=list(set(source_files)) if 'source_files' in locals() else [],
                        reasoning="OpenAI API error after retries"
                    )

            except Exception as e:
                logger.error(f"❌ Unexpected error in general document QA on attempt {attempt + 1}: {str(e)}")
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.info(f"🔄 Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                    continue
                else:
                    return GeneralDocumentResponse(
                        query_type="GENERAL_DOCUMENT_QA",
                        domain=domain,
                        answer=f"An unexpected error occurred while processing this question. The system encountered: {str(e)}. Please try again or contact support if the issue persists.",
                        confidence="low",
                        source_documents=[],
                        reasoning="System error during processing after retries"
                    )

        # This should never be reached due to the loop structure, but just in case
        return GeneralDocumentResponse(
            query_type="GENERAL_DOCUMENT_QA",
            domain=domain,
            answer="Maximum retry attempts exceeded. Please try again later.",
            confidence="low",
            source_documents=[],
            reasoning="Maximum retry attempts exceeded"
        )

# ENHANCED: Document processing functions
async def process_document_async(file_content: bytes, filename: str) -> List[Document]:
    """Process document content asynchronously."""
    # Save to temp file first
    temp_file_path = f"/tmp/{uuid.uuid4().hex}_{filename}"
    
    try:
        with open(temp_file_path, "wb") as f:
            f.write(file_content)
        
        # Determine file type and process
        file_extension = os.path.splitext(filename)[1].lower()
        
        if file_extension == '.pdf':
            loader = PyMuPDFLoader(temp_file_path)
        elif file_extension == '.docx':
            loader = Docx2txtLoader(temp_file_path)
        elif file_extension == '.txt':
            loader = TextLoader(temp_file_path, encoding='utf-8')
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        # Load documents asynchronously
        documents = await asyncio.to_thread(loader.load)
        
        # Add metadata
        for doc in documents:
            doc.metadata.update({
                'source_file': filename,
                'file_type': file_extension,
                'processed_at': datetime.now().isoformat()
            })
        
        return documents
        
    finally:
        # Cleanup temp file
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception:
                pass

# ENHANCED: Main processing functions with optimizations
async def process_document_qa_optimized(
    file_content: bytes,
    filename: str,
    questions: List[str],
    session_id: str = None
) -> List[Dict[str, Any]]:
    """Optimized document Q&A processing with session management and caching."""
    
    try:
        # Generate document hash for session management
        document_hash = hashlib.sha256(file_content + filename.encode()).hexdigest()[:16]
        
        # Get or create session
        rag_system = await SessionManager.get_or_create_session(document_hash)
        
        async with rag_system:
            # Check if system is already set up for this document
            if not hasattr(rag_system, 'vector_store') or rag_system.vector_store is None:
                # Process document and set up RAG system
                documents = await process_document_async(file_content, filename)
                if not documents:
                    return [{"error": "Failed to process document", "question": q} for q in questions]
                
                # Detect domain
                domain = await detect_document_domain_async(documents)
                
                # Set up vector store
                rag_system.documents = documents
                rag_system.document_hash = document_hash
                rag_system.current_domain = domain
                
                success = await rag_system.setup_vector_store_optimized(documents, domain)
                if not success:
                    return [{"error": "Failed to setup vector store", "question": q} for q in questions]
                
                # Setup retrievers
                await rag_system.setup_retrievers_async()
                
                logger.info(f"✅ RAG system initialized for {filename}")
            else:
                logger.info(f"♻️ Reusing existing RAG system for {filename}")
            
            # Process questions in parallel batches
            batch_size = 3  # Optimal batch size for memory management
            results = []
            
            for i in range(0, len(questions), batch_size):
                batch_questions = questions[i:i + batch_size]
                
                # Process batch in parallel
                batch_results = await process_question_batch_optimized(
                    batch_questions, rag_system, document_hash
                )
                results.extend(batch_results)
                
                # Small delay between batches to prevent overload
                if i + batch_size < len(questions):
                    await asyncio.sleep(0.1)
            
            return results
            
    except Exception as e:
        logger.error(f"❌ Error in process_document_qa_optimized: {e}")
        return [{"error": str(e), "question": q} for q in questions]

async def process_question_batch_optimized(
    questions: List[str], 
    rag_system: RAGSystem, 
    document_hash: str
) -> List[Dict[str, Any]]:
    """Process a batch of questions with caching and optimization."""
    
    results = []
    decision_engine = DecisionEngine()
    
    for question in questions:
        try:
            # Check cache first
            question_hash = hashlib.sha256(question.encode()).hexdigest()[:16]
            cached_response = SessionManager.get_cached_response(question_hash, document_hash)
            
            if cached_response:
                logger.info(f"📋 Using cached response for: {question[:50]}...")
                results.append(cached_response)
                continue
            
            # Classify query
            query_type, domain = await classify_query_enhanced(question, rag_system.current_domain)
            
            # Retrieve relevant documents
            retrieved_docs, similarity_scores = await rag_system.retrieve_and_rerank_optimized(question)
            
            if not retrieved_docs:
                response = {
                    "question": question,
                    "answer": "No relevant information found in the document.",
                    "confidence": 0.0,
                    "sources": [],
                    "query_type": query_type
                }
            else:
                # Process based on query type
                if query_type == "INSURANCE_POLICY_INFO":
                    policy_response = await decision_engine.get_policy_info(question, retrieved_docs)
                    response = {
                        "question": policy_response.question,
                        "answer": policy_response.answer,
                        "confidence": policy_response.confidence,
                        "sources": policy_response.sources,
                        "reasoning_chain": policy_response.reasoning_chain,
                        "matched_sections": policy_response.matched_sections,
                        "query_type": query_type
                    }
                
                elif query_type == "INSURANCE_CLAIM_PROCESSING":
                    claim_decision = await decision_engine.get_structured_decision(
                        question, retrieved_docs, similarity_scores
                    )
                    response = {
                        "question": question,
                        "decision": claim_decision.get("decision", "NEEDS_REVIEW"),
                        "confidence": claim_decision.get("confidence", 0.0),
                        "coverage_amount": claim_decision.get("coverage_amount"),
                        "deductible": claim_decision.get("deductible"),
                        "reasoning": claim_decision.get("reasoning", []),
                        "risk_factors": claim_decision.get("risk_factors", []),
                        "sources": claim_decision.get("sources", []),
                        "query_type": query_type
                    }
                
                elif query_type == "INSURANCE_COMPONENT_ANALYSIS":
                    component_response = await decision_engine.get_component_analysis(
                        question, retrieved_docs, similarity_scores
                    )
                    response = {
                        "question": component_response.query,
                        "analysis": component_response.decision,
                        "components": component_response.matched_components,
                        "confidence": component_response.confidence,
                        "total_score": component_response.total_score,
                        "reasoning_chain": component_response.reasoning_chain,
                        "clause_traceability": component_response.clause_traceability,
                        "sources": component_response.sources,
                        "query_type": query_type
                    }
                
                else:
                    # General Q&A fallback
                    response = await process_general_qa(question, retrieved_docs, decision_engine)
                    response["query_type"] = query_type
            
            # Cache the response
            SessionManager.cache_response(question_hash, document_hash, response)
            results.append(response)
            
        except Exception as e:
            logger.error(f"❌ Error processing question '{question}': {e}")
            response = {
                "question": question,
                "answer": f"Error processing question: {str(e)}",
                "confidence": 0.0,
                "sources": [],
                "query_type": "ERROR",
                "error": str(e)
            }
            results.append(response)
    
    return results

async def process_general_qa(question: str, retrieved_docs: List[Document], decision_engine: DecisionEngine) -> Dict[str, Any]:
    """Process general Q&A with fallback handling."""
    try:
        general_response = await decision_engine.get_general_document_answer(
            question, retrieved_docs, "general"
        )
        return {
            "question": question,
            "answer": general_response.answer,
            "confidence": general_response.confidence,
            "sources": general_response.source_documents,
            "reasoning": general_response.reasoning,
            "domain": general_response.domain
        }
    except Exception as e:
        logger.error(f"❌ Error in general Q&A: {e}")
        return {
            "question": question,
            "answer": f"Error processing question: {str(e)}",
            "confidence": "low",
            "sources": [],
            "error": str(e)
        }

# ENHANCED: Global initialization
@lru_cache(maxsize=1)
def initialize_models():
    """Initialize models with caching to prevent reloading."""
    global embedding_model, reranker, openai_client
    
    try:
        logger.info("🚀 Initializing AI models...")
        
        # Initialize OpenAI client
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        openai_client = AsyncOpenAI(api_key=openai_api_key)
        
        # Initialize embedding model
        logger.info("📊 Loading embedding model...")
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize reranker
        logger.info("🔄 Loading reranker model...")
        reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-2-v2')
        
        logger.info("✅ All models initialized successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize models: {e}")
        raise

# ENHANCED: Lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Enhanced lifespan management with proper initialization and cleanup."""
    # Startup
    logger.info("🚀 Application startup initiated...")
    
    try:
        # Check system resources
        memory = psutil.virtual_memory()
        logger.info(f"💾 Available memory: {memory.available / (1024**3):.2f} GB")
        
        # Initialize models
        initialize_models()
        
        # Ensure persistent directory exists
        os.makedirs(PERSISTENT_CHROMA_DIR, exist_ok=True)
        logger.info(f"📁 Persistent storage ready: {PERSISTENT_CHROMA_DIR}")
        
        # Start background cleanup task
        cleanup_task = asyncio.create_task(periodic_cleanup())
        
        logger.info("✅ Application startup complete!")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    finally:
        # Shutdown
        logger.info("🛑 Application shutdown initiated...")
        
        try:
            # Cancel background tasks
            if 'cleanup_task' in locals():
                cleanup_task.cancel()
                try:
                    await cleanup_task
                except asyncio.CancelledError:
                    pass
            
            # Cleanup active sessions
            cleanup_count = 0
            for session_id, (session, _) in list(ACTIVE_SESSIONS.items()):
                try:
                    await session.async_cleanup()
                    cleanup_count += 1
                except Exception as e:
                    logger.warning(f"⚠️ Error cleaning session {session_id}: {e}")
            
            ACTIVE_SESSIONS.clear()
            logger.info(f"🧹 Cleaned up {cleanup_count} active sessions")
            
            # Clear caches
            QUERY_CACHE.clear()
            DOCUMENT_CACHE.clear()
            
            logger.info("✅ Application shutdown complete!")
            
        except Exception as e:
            logger.error(f"❌ Shutdown error: {e}")

async def periodic_cleanup():
    """Background task for periodic cleanup."""
    while True:
        try:
            await asyncio.sleep(CACHE_CLEANUP_INTERVAL)  # 5 minutes
            await cleanup_expired_cache()
            logger.info("🧹 Periodic cleanup completed")
        except asyncio.CancelledError:
            logger.info("🛑 Cleanup task cancelled")
            break
        except Exception as e:
            logger.error(f"❌ Error in periodic cleanup: {e}")

# ENHANCED: FastAPI app with proper configuration
app = FastAPI(
    title="Enhanced RAG Document Q&A API",
    description="Advanced document processing and Q&A system with multi-domain support",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# ENHANCED: CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ENHANCED: API Routes with comprehensive error handling

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with system status."""
    return {
        "message": "Enhanced RAG Document Q&A API v2.0",
        "status": "operational",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=Dict[str, Any])
async def health_check():
    """Comprehensive health check endpoint."""
    try:
        # Check system resources
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Check models
        models_status = {
            "embedding_model": embedding_model is not None,
            "reranker": reranker is not None,
            "openai_client": openai_client is not None
        }
        
        # Check storage
        storage_available = os.path.exists(PERSISTENT_CHROMA_DIR)
        
        status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "system": {
                "memory_usage_percent": memory.percent,
                "cpu_usage_percent": cpu_percent,
                "storage_available": storage_available
            },
            "models": models_status,
            "active_sessions": len(ACTIVE_SESSIONS),
            "cached_queries": len(QUERY_CACHE),
            "cached_documents": len(DOCUMENT_CACHE),
            "version": "2.0.0"
        }
        
        # Determine overall health
        if all(models_status.values()) and memory.percent < 90 and cpu_percent < 90:
            status["status"] = "healthy"
        elif memory.percent > 95 or cpu_percent > 95:
            status["status"] = "degraded"
        else:
            status["status"] = "warning"
        
        return status
        
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.post("/upload_and_query", response_model=List[Dict[str, Any]])
async def upload_and_query(
    file: UploadFile = File(..., description="Document file (PDF, DOCX, or TXT)"),
    questions: str = Form(..., description="JSON array of questions"),
    bearer_token: str = Depends(verify_bearer_token)
):
    """Upload document and ask multiple questions with enhanced processing."""
    start_time = time.time()
    
    try:
        # Validate file type
        allowed_extensions = {'.pdf', '.docx', '.txt'}
        file_extension = os.path.splitext(file.filename)[1].lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_extension}. Supported types: {', '.join(allowed_extensions)}"
            )
        
        # Parse questions
        try:
            questions_list = json.loads(questions)
            if not isinstance(questions_list, list) or not questions_list:
                raise ValueError("Questions must be a non-empty list")
            
            # Validate each question
            for i, q in enumerate(questions_list):
                if not isinstance(q, str) or not q.strip():
                    raise ValueError(f"Question {i+1} must be a non-empty string")
                if len(q) > 1000:
                    raise ValueError(f"Question {i+1} exceeds 1000 character limit")
            
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON format for questions")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        # Validate file size
        file_content = await file.read()
        if len(file_content) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        max_file_size = 50 * 1024 * 1024  # 50MB
        if len(file_content) > max_file_size:
            raise HTTPException(
                status_code=400, 
                detail=f"File too large. Maximum size: {max_file_size / (1024*1024)}MB"
            )
        
        logger.info(f"📄 Processing {file.filename} ({len(file_content)/1024:.1f}KB) with {len(questions_list)} questions")
        
        # Process document and questions
        results = await process_document_qa_optimized(
            file_content, 
            file.filename, 
            questions_list
        )
        
        processing_time = time.time() - start_time
        logger.info(f"✅ Completed processing in {processing_time:.2f}s")
        
        # Add metadata to results
        for result in results:
            result.update({
                "file_name": file.filename,
                "processing_time_seconds": processing_time,
                "timestamp": datetime.now().isoformat()
            })
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"❌ Error in upload_and_query after {processing_time:.2f}s: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(
    request: QueryRequest,
    bearer_token: str = Depends(verify_bearer_token)
):
    """Query endpoint for active sessions."""
    try:
        # This endpoint would work with active sessions
        # For now, return a message directing to upload_and_query
        return QueryResponse(
            decision="USE_UPLOAD_ENDPOINT",
            confidence_score=1.0,
            reasoning_chain=["Please use /upload_and_query endpoint to upload document and ask questions"],
            evidence_sources=[],
            timestamp=datetime.now().isoformat(),
            query_type="REDIRECT",
            domain="system"
        )
        
    except Exception as e:
        logger.error(f"❌ Error in query endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# UPDATED: Fixed endpoint path and enhanced response model for HackRx compliance
@app.post("/hackrx/run", response_model=EnhancedHackRxResponse)
async def hackrx_run_endpoint(
    request: HackRxRequest,
    bearer_token: str = Depends(verify_bearer_token)
):
    """HackRx run endpoint with enhanced responses - compliant with submission requirements."""
    start_time = time.time()
    
    try:
        # Download document from URL
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(str(request.documents))
            if response.status_code != 200:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to download document: HTTP {response.status_code}"
                )
            
            file_content = response.content
            
            # Determine filename from URL
            filename = str(request.documents).split('/')[-1]
            if not filename or '.' not in filename:
                filename = "document.pdf"  # Default assumption
        
        logger.info(f"📄 HackRx: Processing document from URL with {len(request.questions)} questions")
        
        # Process using the optimized pipeline
        results = await process_document_qa_optimized(
            file_content,
            filename,
            request.questions
        )
        
        # Convert results to HackRx format
        enhanced_answers = []
        for result in results:
            if result.get("query_type") == "INSURANCE_CLAIM_PROCESSING":
                enhanced_answer = StructuredClaimResponse(
                    status=result.get("decision", "NEEDS_REVIEW"),
                    confidence=result.get("confidence", 0.0),
                    approvedAmount=result.get("coverage_amount", 0.0) or 0.0,
                    rejectedAmount=0.0,
                    breakdown=ClaimBreakdown(approved=[], rejected=[]),
                    keyPolicyReferences=[
                        PolicyReference(title=source, note="Referenced policy document")
                        for source in result.get("sources", [])[:3]
                    ],
                    summary=result.get("answer", "Claim analysis completed")
                )
            elif result.get("query_type") == "INSURANCE_POLICY_INFO":
                enhanced_answer = PolicyInfoResponse(
                    question=result.get("question", ""),
                    answer=result.get("answer", ""),
                    confidence=result.get("confidence", 0.0),
                    sources=result.get("sources", []),
                    reasoning_chain=result.get("reasoning_chain", []),
                    matched_sections=result.get("matched_sections", [])
                )
            else:
                enhanced_answer = GeneralDocumentResponse(
                    query_type=result.get("query_type", "GENERAL_QA"),
                    domain=result.get("domain", "general"),
                    answer=result.get("answer", ""),
                    confidence=result.get("confidence", "medium"),
                    source_documents=result.get("sources", []),
                    reasoning=result.get("reasoning", None)
                )
            
            enhanced_answers.append(enhanced_answer)
        
        processing_time = time.time() - start_time
        
        return EnhancedHackRxResponse(
            success=True,
            processing_time_seconds=processing_time,
            timestamp=datetime.now().isoformat(),
            message=f"Successfully processed {len(request.questions)} questions from document",
            answers=enhanced_answers
        )
        
    except httpx.RequestError as e:
        processing_time = time.time() - start_time
        logger.error(f"❌ Network error downloading document: {e}")
        return EnhancedHackRxResponse(
            success=False,
            processing_time_seconds=processing_time,
            timestamp=datetime.now().isoformat(),
            message=f"Failed to download document: {str(e)}",
            answers=[]
        )
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"❌ Error in HackRx endpoint: {e}")
        return EnhancedHackRxResponse(
            success=False,
            processing_time_seconds=processing_time,
            timestamp=datetime.now().isoformat(),
            message=f"Processing error: {str(e)}",
            answers=[]
        )

@app.get("/sessions", response_model=Dict[str, Any])
async def list_sessions(bearer_token: str = Depends(verify_bearer_token)):
    """List active sessions with details."""
    try:
        sessions_info = {}
        current_time = time.time()
        
        for session_id, (session, timestamp) in ACTIVE_SESSIONS.items():
            time_active = current_time - timestamp
            sessions_info[session_id] = {
                "time_active_seconds": time_active,
                "document_count": len(session.documents) if hasattr(session, 'documents') else 0,
                "domain": getattr(session, 'current_domain', 'unknown'),
                "processed_files": getattr(session, 'processed_files', [])
            }
        
        return {
            "active_sessions": len(ACTIVE_SESSIONS),
            "sessions": sessions_info,
            "cache_stats": {
                "query_cache_size": len(QUERY_CACHE),
                "document_cache_size": len(DOCUMENT_CACHE)
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Error listing sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    bearer_token: str = Depends(verify_bearer_token)
):
    """Delete a specific session."""
    try:
        if session_id in ACTIVE_SESSIONS:
            session, _ = ACTIVE_SESSIONS.pop(session_id)
            await session.async_cleanup()
            logger.info(f"🗑️ Deleted session: {session_id}")
            return {"message": f"Session {session_id} deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Session not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error deleting session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear_cache")
async def clear_cache(bearer_token: str = Depends(verify_bearer_token)):
    """Clear all caches."""
    try:
        # Clear query cache
        query_count = len(QUERY_CACHE)
        QUERY_CACHE.clear()
        
        # Clear document cache
        doc_count = len(DOCUMENT_CACHE)
        async with _cache_lock:
            DOCUMENT_CACHE.clear()
        
        logger.info(f"🧹 Cleared {query_count} query cache entries and {doc_count} document cache entries")
        
        return {
            "message": "Cache cleared successfully",
            "cleared_query_entries": query_count,
            "cleared_document_entries": doc_count
        }
        
    except Exception as e:
        logger.error(f"❌ Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Enhanced HTTP exception handler."""
    logger.warning(f"⚠️ HTTP {exc.status_code}: {exc.detail} - {request.url}")
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
    """Enhanced general exception handler."""
    logger.error(f"❌ Unhandled exception: {str(exc)} - {request.url}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )

# ENHANCED: Main entry point
if __name__ == "__main__":
    import uvicorn
    
    # Configure logging for development
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Get configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    workers = int(os.getenv("WORKERS", 1))
    
    logger.info(f"🚀 Starting Enhanced RAG API on {host}:{port}")
    
    if workers > 1:
        # Use gunicorn for production with multiple workers
        uvicorn.run(
            "main:app",
            host=host,
            port=port,
            workers=workers,
            log_level="info",
            access_log=True
        )
    else:
        # Single worker for development
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="info",
            access_log=True,
            reload=False  # Disable reload in production
        )
