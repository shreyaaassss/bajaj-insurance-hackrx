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

# Core libraries
import pandas as pd
import numpy as np
from datetime import datetime

# FastAPI and web - ADDED SECURITY IMPORTS
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, status, APIRouter, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, HttpUrl
import httpx

# Document processing - Fixed imports
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

# ENHANCED: Document-specific cache with proper isolation
DOCUMENT_CACHE = {}
# ENHANCED: Base directory for all vector stores
VECTORSTORE_BASE_DIR = "/tmp/vectorstores"
# ENHANCED: Memory Management for Cache
MAX_CACHE_SIZE = 5  # Reduced to prevent memory issues

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

# ENHANCED: Domain detection keywords
DOMAIN_DETECTION_KEYWORDS = {
    "insurance": ["claim", "policy", "premium", "coverage", "deductible", "bajaj insurance", "sum insured", "cashless", "network hospital", "co-payment", "exclusion"],
    "physics": ["newton", "force", "motion", "velocity", "acceleration", "principia", "mechanics", "mass", "gravity", "inertia"],
    "legal": ["law", "legal", "rights", "article", "clause", "constitution", "court", "jurisdiction", "statute", "regulation", "provision"],
    "academic": ["research", "study", "analysis", "methodology", "conclusion", "hypothesis", "theory", "academic", "scholarly"],
    "medical": ["patient", "treatment", "diagnosis", "symptoms", "therapy", "clinical", "medical", "healthcare", "disease", "medicine"]
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

def manage_cache():
    """Remove oldest entries if cache exceeds limit."""
    global DOCUMENT_CACHE
    if len(DOCUMENT_CACHE) > MAX_CACHE_SIZE:
        # Remove oldest entry (simple FIFO)
        oldest_key = next(iter(DOCUMENT_CACHE))
        del DOCUMENT_CACHE[oldest_key]
        logger.info(f"🗑️ Removed oldest cache entry: {oldest_key[:8]}")

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

# 🔧 FIXED: Document ID generation function
def generate_document_id(content: bytes, filename: str = None) -> str:
    """Generate a unique document ID based on content and filename."""
    content_hash = hashlib.md5(content).hexdigest()[:8]
    if filename:
        filename_hash = hashlib.md5(filename.encode()).hexdigest()[:4]
        return f"{content_hash}_{filename_hash}"
    return content_hash

# 🔧 FIXED: Vector store path generation
def get_vectorstore_path(document_id: str, domain: str) -> str:
    """Generate isolated vector store path for document."""
    return os.path.join(VECTORSTORE_BASE_DIR, f"{domain}_{document_id}")

# 🔧 NEW: Clear stale vector stores
def clear_stale_vectorstores():
    """Clear old vector stores to prevent memory issues."""
    try:
        if os.path.exists(VECTORSTORE_BASE_DIR):
            # Remove directories older than 1 hour
            current_time = time.time()
            for item in os.listdir(VECTORSTORE_BASE_DIR):
                item_path = os.path.join(VECTORSTORE_BASE_DIR, item)
                if os.path.isdir(item_path):
                    item_age = current_time - os.path.getctime(item_path)
                    if item_age > 3600:  # 1 hour
                        shutil.rmtree(item_path)
                        logger.info(f"🗑️ Removed stale vector store: {item}")
    except Exception as e:
        logger.warning(f"⚠️ Failed to clear stale vector stores: {e}")

# NEW: Enhanced domain detection
def detect_document_domain(content: str) -> str:
    """Detect the domain of the document based on content analysis."""
    content_lower = content.lower()
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

# 🔧 FIXED: Enhanced query classification with proper domain respect
def classify_query_type(query: str, document_content: str = None) -> Tuple[str, str]:
    """
    Classify query type and detect domain - FIXED VERSION
    Returns (query_type, domain)
    """
    query_lower = query.lower()
    
    # First detect domain from document content if available
    domain = "general"
    if document_content:
        domain = detect_document_domain(document_content)
        logger.info(f"🔍 Document domain detected: {domain}")
    
    # 🔧 FIX: Only route to insurance if domain is actually insurance
    if domain == "insurance":
        logger.info("📋 Processing insurance domain query")
        # Only then check for insurance-specific query types
        claim_score = sum(1 for keyword in CLAIM_KEYWORDS if keyword in query_lower)
        policy_score = sum(1 for keyword in POLICY_INFO_KEYWORDS if keyword in query_lower)
        has_amounts = bool(re.search(r'[\₹Rs\.]*\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?', query))
        
        logger.info(f"🔍 Insurance query analysis - Claim score: {claim_score}, Policy score: {policy_score}, Has amounts: {has_amounts}")
        
        if has_amounts and claim_score > 0:
            logger.info("💰 Classified as INSURANCE_CLAIM_PROCESSING")
            return "INSURANCE_CLAIM_PROCESSING", "insurance"
        elif policy_score > claim_score:
            logger.info("📖 Classified as INSURANCE_POLICY_INFO")
            return "INSURANCE_POLICY_INFO", "insurance"
        else:
            logger.info("🏥 Classified as INSURANCE_GENERAL")
            return "INSURANCE_GENERAL", "insurance"
    
    # 🔧 FIX: For all non-insurance domains, use general document QA
    logger.info(f"📄 Classified as GENERAL_DOCUMENT_QA for domain: {domain}")
    return "GENERAL_DOCUMENT_QA", domain

# NEW: Content type detection utility
def detect_content_type_from_url(url: str) -> str:
    """Detect content type from URL patterns."""
    url_lower = url.lower()
    
    # Google Docs export patterns
    if "export?format=pdf" in url_lower or "exportFormat=pdf" in url_lower:
        return "application/pdf"
    elif "export?format=docx" in url_lower or "exportFormat=docx" in url_lower:
        return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    elif "export?format=txt" in url_lower or "exportFormat=txt" in url_lower:
        return "text/plain"
    
    # File extension patterns
    if url_lower.endswith('.pdf'):
        return "application/pdf"
    elif url_lower.endswith('.docx'):
        return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    elif url_lower.endswith('.txt'):
        return "text/plain"
    
    # Default to PDF if unclear
    return "application/pdf"

def get_file_extension_from_content_type(content_type: str) -> str:
    """Get appropriate file extension based on content type."""
    content_type_mapping = {
        "application/pdf": ".pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
        "text/plain": ".txt"
    }
    return content_type_mapping.get(content_type, ".pdf")

# FIXED: Amount parsing utility to handle currency and commas
def parse_amount(value) -> float:
    """Parse amount string with currency symbols and commas to float."""
    if value is None:
        return 0.0
    
    if isinstance(value, (int, float)):
        return float(value)
    
    if isinstance(value, str):
        # Remove currency symbols, commas, and whitespace
        cleaned_value = value.replace("₹", "").replace("Rs.", "").replace("Rs", "").replace(",", "").strip()
        if not cleaned_value or cleaned_value == "":
            return 0.0
        try:
            return float(cleaned_value)
        except ValueError as e:
            logger.warning(f"⚠️ Failed to parse amount '{value}': {e}")
            return 0.0
    
    return 0.0

# FIXED: Amount validation utility
def validate_amounts(breakdown: Dict, total_claim_amount: float = None) -> Tuple[bool, str]:
    """Validate that breakdown amounts are consistent and valid."""
    try:
        # Validate individual components
        for item in breakdown.get("approved", []):
            if not isinstance(item.get("amount"), (float, int)):
                return False, f"Invalid approved amount type: {item.get('amount')}"
            if item.get("amount", 0) < 0:
                return False, f"Negative approved amount: {item.get('amount')}"
        
        for item in breakdown.get("rejected", []):
            if not isinstance(item.get("amount"), (float, int)):
                return False, f"Invalid rejected amount type: {item.get('amount')}"
            if item.get("amount", 0) < 0:
                return False, f"Negative rejected amount: {item.get('amount')}"
        
        # Calculate totals
        approved_total = sum(item.get("amount", 0) for item in breakdown.get("approved", []))
        rejected_total = sum(item.get("amount", 0) for item in breakdown.get("rejected", []))
        calculated_total = approved_total + rejected_total
        
        # Validate against total claim amount if provided
        if total_claim_amount is not None and total_claim_amount > 0:
            if abs(calculated_total - total_claim_amount) > 1e-2:
                return False, f"Breakdown total ({calculated_total}) doesn't match claim amount ({total_claim_amount})"
        
        return True, "Valid amounts"
    
    except Exception as e:
        return False, f"Amount validation error: {str(e)}"

# NEW: Extract structured information from text
def extract_structured_info(text: str) -> Dict[str, Any]:
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

# Insurance-specific clause mapping functions
def calculate_keyword_similarity_score(component_type: str, clause_text: str) -> float:
    """Calculate similarity score between component and clause based on keywords."""
    keywords = COMPONENT_KEYWORDS.get(component_type, [])
    if not keywords:
        return 0.0
    
    clause_lower = clause_text.lower()
    matches = sum(1 for keyword in keywords if keyword in clause_lower)
    
    # Normalize score by number of keywords
    return matches / len(keywords) if keywords else 0.0

def calculate_semantic_clause_score(clause_text: str, clause_type: str = None) -> float:
    """Calculate semantic relevance score for clause type."""
    if not clause_type or clause_type not in CLAUSE_TYPES:
        return 0.0
    
    clause_lower = clause_text.lower()
    type_keywords = CLAUSE_TYPES[clause_type]
    matches = sum(1 for keyword in type_keywords if keyword in clause_lower)
    
    return matches / len(type_keywords) if type_keywords else 0.0

def extract_policy_clauses_from_context(context_text: str) -> List[Dict[str, str]]:
    """Extract structured policy clauses from context text."""
    clauses = []
    
    # Split context into sections
    sections = context_text.split('---')
    
    for section in sections:
        if not section.strip():
            continue
        
        lines = [line.strip() for line in section.split('\n') if line.strip()]
        if not lines:
            continue
        
        # Try to identify clause structure
        for i, line in enumerate(lines):
            if any(keyword in line.lower() for keyword in ['clause', 'section', 'article', 'benefit', 'coverage', 'exclusion']):
                title = line
                note_lines = lines[i+1:i+5]  # Take next few lines as note
                note = ' '.join(note_lines) if note_lines else title
                
                clauses.append({
                    "title": title,
                    "note": note[:300],  # Limit note length
                    "full_text": title + " " + note
                })
    
    # If no structured clauses found, create general clauses from paragraphs
    if not clauses:
        paragraphs = [p.strip() for p in context_text.split('\n\n') if p.strip() and len(p.strip()) > 50]
        for i, para in enumerate(paragraphs[:10]):  # Limit to first 10 paragraphs
            clauses.append({
                "title": f"Policy Provision {i+1}",
                "note": para[:300],
                "full_text": para
            })
    
    logger.info(f"📋 Extracted {len(clauses)} policy clauses from context")
    return clauses

def map_clauses_to_components(components: List[Dict], context_text: str) -> List[Dict[str, str]]:
    """Map policy clauses to claim components using advanced matching."""
    # Extract clauses from context
    available_clauses = extract_policy_clauses_from_context(context_text)
    
    if not available_clauses:
        logger.warning("⚠️ No policy clauses extracted from context")
        return [{
            "title": "General Policy Terms",
            "note": "Refer to complete policy document for detailed terms and conditions."
        }]
    
    mapped_clauses = []
    used_clause_indices = set()
    
    logger.info(f"🔗 Mapping {len(components)} components to {len(available_clauses)} available clauses")
    
    for comp in components:
        comp_type = comp.get("component", "Unknown")
        best_clause = None
        best_score = 0.0
        best_clause_index = -1
        
        # Find best matching clause for this component
        for idx, clause in enumerate(available_clauses):
            if idx in used_clause_indices:
                continue  # Avoid reusing clauses
            
            clause_text = clause.get("full_text", "")
            
            # Calculate similarity scores
            keyword_score = calculate_keyword_similarity_score(comp_type, clause_text)
            
            # Bonus for exact component name match
            exact_match_bonus = 0.3 if comp_type.lower() in clause_text.lower() else 0.0
            
            # Penalty for exclusion-type clauses when component is approved
            exclusion_penalty = 0.0
            if comp.get("status") == "approved" and any(excl in clause_text.lower() for excl in ["exclusion", "not covered", "excluded"]):
                exclusion_penalty = 0.2
            
            # Calculate final score
            final_score = keyword_score + exact_match_bonus - exclusion_penalty
            
            if final_score > best_score:
                best_score = final_score
                best_clause = clause
                best_clause_index = idx
        
        # Add best matching clause if found
        if best_clause and best_score > 0.1:  # Minimum threshold
            used_clause_indices.add(best_clause_index)
            mapped_clauses.append({
                "title": best_clause["title"],
                "note": best_clause["note"],
                "relevance_score": round(best_score, 3),
                "component": comp_type
            })
            logger.debug(f"✅ Mapped '{comp_type}' to '{best_clause['title']}' (score: {best_score:.3f})")
        else:
            # Create generic clause for unmapped component
            mapped_clauses.append({
                "title": f"Policy Terms for {comp_type}",
                "note": f"Refer to policy terms and conditions regarding {comp_type.lower()} coverage and applicable limits.",
                "relevance_score": 0.0,
                "component": comp_type
            })
            logger.debug(f"⚠️ No good match for '{comp_type}' - created generic clause")
    
    # Add fallback if no clauses mapped
    if not mapped_clauses:
        mapped_clauses.append({
            "title": "General Policy Terms and Conditions",
            "note": "Please refer to the complete policy document for detailed coverage terms, exclusions, and claim procedures.",
            "relevance_score": 0.0,
            "component": "General"
        })
        logger.warning("⚠️ No clauses mapped - added fallback clause")
    
    # Remove duplicates while preserving order
    unique_clauses = []
    seen_titles = set()
    for clause in mapped_clauses:
        if clause["title"] not in seen_titles:
            unique_clauses.append({
                "title": clause["title"],
                "note": clause["note"]
            })
            seen_titles.add(clause["title"])
    
    logger.info(f"✅ Successfully mapped {len(unique_clauses)} unique policy clauses")
    return unique_clauses

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize shared resources at startup."""
    global embedding_model, reranker, openai_client
    
    logger.info("🚀 Initializing Multi-Domain Document QA System...")
    
    try:
        # PERFORMANCE OPTIMIZATION: Initialize embedding model with optimized settings
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-TinyBERT-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        logger.info("✅ TinyBERT embedding model loaded with optimized settings")
        
        # PERFORMANCE OPTIMIZATION: Initialize lighter reranker model
        reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')
        logger.info("✅ Lightweight reranker model loaded")
        
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("❌ OPENAI_API_KEY environment variable not set")
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        openai_client = AsyncOpenAI(api_key=api_key)
        logger.info("✅ OpenAI client initialized")
        
        # 🔧 FIXED: Create base vectorstore directory
        os.makedirs(VECTORSTORE_BASE_DIR, exist_ok=True)
        logger.info(f"✅ Base vector store directory ready: {VECTORSTORE_BASE_DIR}")
        
        # Clear stale vector stores on startup
        clear_stale_vectorstores()
        logger.info("✅ Stale vector stores cleared")
        
        logger.info("🎉 Multi-Domain System initialization complete!")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize system: {str(e)}")
        raise
    
    yield
    
    # Cleanup
    logger.info("🔄 Shutting down Multi-Domain system...")

# --- Pydantic Models ---

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

# NEW: General Document QA Response Model
class GeneralDocumentResponse(BaseModel):
    query_type: str
    domain: str
    answer: str
    confidence: str
    source_documents: List[str]
    reasoning: Optional[str] = None

# Insurance-specific models (preserved)
class PolicyInfoResponse(BaseModel):
    question: str
    answer: str
    policy_section: Optional[str] = None
    additional_notes: Optional[str] = None
    confidence: float
    sources: List[str] = []

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

# NEW: Unified Response Model
class UnifiedResponse(BaseModel):
    query_type: str
    domain: str
    answer: str
    # Insurance-specific fields (only populated for insurance queries)
    decision: Optional[str] = None
    approved_amount: Optional[float] = None
    policy_references: Optional[List[str]] = None
    # General fields
    confidence: str
    source_documents: List[str]

class EnhancedHackRxResponse(BaseModel):
    answers: List[Union[StructuredClaimResponse, PolicyInfoResponse, GeneralDocumentResponse]]

# Initialize FastAPI app
app = FastAPI(
    title="Multi-Domain Document QA API",
    description="AI-powered system for analyzing any document type with specialized insurance capabilities and intelligent query classification.",
    version="4.0.0",
    lifespan=lifespan
)

api_v1_router = APIRouter(prefix="/api/v1")

# --- Middleware ---

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log requests without sensitive content."""
    start_time = time.time()
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Request: {request.method} {request.url.path}")
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"[{request_id}] Response: {response.status_code} - Time: {process_time:.2f}s")
    return response

# --- File and System Utilities ---

def validate_uploaded_file(file: UploadFile) -> Tuple[bool, str]:
    """Validate uploaded files for security and format compliance."""
    max_size = 50 * 1024 * 1024  # 50MB
    allowed_types = [
        'application/pdf',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'text/plain'
    ]
    allowed_extensions = {'.pdf', '.docx', '.txt'}
    
    if hasattr(file, 'size') and file.size and file.size > max_size:
        return False, f"File {file.filename} exceeds 50MB limit"
    
    if file.content_type not in allowed_types:
        return False, f"File {file.filename} has unsupported content type: {file.content_type}"
    
    if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
        return False, f"File {file.filename} has unsupported extension"
    
    return True, "Valid file"

async def save_uploaded_file(file: UploadFile, upload_dir: str = "uploads") -> str:
    """Save uploaded file to disk with proper naming."""
    os.makedirs(upload_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_extension = os.path.splitext(file.filename)[1]
    safe_filename = f"{timestamp}_{uuid.uuid4().hex[:8]}{file_extension}"
    file_path = os.path.join(upload_dir, safe_filename)
    
    try:
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        logger.info(f"📁 File saved: {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"❌ Failed to save file {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

# --- Core RAG and Decision Logic ---

class RAGSystem:
    """🔧 FIXED: Enhanced Retrieval-Augmented Generation system with proper document isolation."""

    def __init__(self):
        self.vector_store = None
        self.bm25_retriever = None
        self.documents = []
        self.processed_files = []
        self.structured_info_cache = {}
        self.document_domain = "general"
        self.domain_config = DOMAIN_CONFIG["general"]
        # 🔧 NEW: Document isolation fields
        self.document_id = None
        self.vectorstore_path = None
        self.document_content_hash = None

    async def _process_single_file(self, file_path: str, document_content: bytes = None) -> List[Document]:
        """🔧 FIXED: Process a single file with document ID tracking."""
        try:
            # 🔧 NEW: Generate document ID for proper isolation
            if document_content is None:
                with open(file_path, 'rb') as f:
                    document_content = f.read()
            
            filename = os.path.basename(file_path)
            self.document_id = generate_document_id(document_content, filename)
            self.document_content_hash = hashlib.md5(document_content).hexdigest()
            
            logger.info(f"🔧 Generated document ID: {self.document_id} for file: {filename}")
            
            file_extension = os.path.splitext(file_path)[1].lower()
            
            if file_extension == '.pdf':
                loader = PyMuPDFLoader(file_path)
            elif file_extension == '.docx':
                loader = Docx2txtLoader(file_path)
            elif file_extension == '.txt':
                loader = TextLoader(file_path, encoding='utf-8')
            else:
                logger.warning(f"⚠️ Unsupported file type: {file_extension}")
                return []

            documents = await asyncio.to_thread(loader.load)

            # Detect document domain from content
            if documents:
                sample_content = ' '.join([doc.page_content for doc in documents[:3]])  # First 3 docs
                self.document_domain = detect_document_domain(sample_content)
                self.domain_config = DOMAIN_CONFIG.get(self.document_domain, DOMAIN_CONFIG["general"])
                
                # 🔧 NEW: Set vectorstore path with document isolation
                self.vectorstore_path = get_vectorstore_path(self.document_id, self.document_domain)
                
                logger.info(f"📄 Document domain: {self.document_domain}, Vector store: {self.vectorstore_path}")

            # Extract structured information during processing
            for doc in documents:
                structured_info = extract_structured_info(doc.page_content)
                doc.metadata.update({
                    'source_file': filename,
                    'file_type': file_extension,
                    'processed_at': datetime.now().isoformat(),
                    'domain': self.document_domain,
                    'document_id': self.document_id,  # 🔧 NEW: Add document ID to metadata
                    'content_hash': self.document_content_hash,  # 🔧 NEW: Add content hash
                    **structured_info  # Add extracted structured info
                })

            logger.info(f"✅ Processed {len(documents)} documents from {filename} (domain: {self.document_domain}, ID: {self.document_id})")
            return documents

        except Exception as e:
            logger.error(f"❌ Error processing {file_path}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to process document {os.path.basename(file_path)}: {str(e)}")

    async def robust_document_processing(self, file_paths: List[str], file_contents: List[bytes] = None) -> Dict[str, Any]:
        """🔧 FIXED: Improved error handling with document content tracking."""
        processing_results = []
        
        for i, file_path in enumerate(file_paths):
            try:
                content = file_contents[i] if file_contents and i < len(file_contents) else None
                result = await self._process_single_file(file_path, content)
                processing_results.append({"file": file_path, "status": "success", "documents": result})
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

    async def load_and_process_documents(self, file_paths: List[str], file_contents: List[bytes] = None) -> Dict[str, Any]:
        """🔧 FIXED: Process multiple files with document isolation and enhanced error handling."""
        try:
            # Use robust processing with content tracking
            result = await self.robust_document_processing(file_paths, file_contents)
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

            chunked_docs = await asyncio.to_thread(text_splitter.split_documents, all_docs)
            chunked_docs = [doc for doc in chunked_docs if len(doc.page_content.strip()) > 50]

            self.documents = chunked_docs
            self.processed_files = processed_files

            logger.info(f"📄 Created {len(chunked_docs)} chunks from {len(processed_files)} files (domain: {self.document_domain}, ID: {self.document_id})")

            return {
                'documents': chunked_docs,
                'processed_files': processed_files,
                'skipped_files': skipped_files,
                'total_chunks': len(chunked_docs),
                'domain': self.document_domain,
                'document_id': self.document_id  # 🔧 NEW: Return document ID
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"❌ Error in document processing: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error during document processing: {str(e)}")

    async def setup_retrievers(self):
        """🔧 FIXED: Initialize isolated vector store and BM25 retriever."""
        global embedding_model

        if not self.documents:
            logger.warning("⚠️ No documents available for retriever setup")
            return False

        if not self.document_id:
            logger.error("❌ Document ID not set - cannot create isolated vector store")
            return False

        try:
            # 🔧 FIXED: Use isolated vector store path
            if os.path.exists(f"{self.vectorstore_path}/chroma.sqlite3"):
                logger.info(f"✅ Loading existing isolated vector store: {self.vectorstore_path}")
                self.vector_store = Chroma(
                    persist_directory=self.vectorstore_path,
                    embedding_function=embedding_model
                )
                
                # 🔧 NEW: Verify this vector store contains our document
                try:
                    # Test query to verify content
                    test_results = self.vector_store.similarity_search("test", k=1)
                    if test_results and test_results[0].metadata.get('document_id') != self.document_id:
                        logger.warning(f"⚠️ Vector store contains different document. Recreating for document ID: {self.document_id}")
                        shutil.rmtree(self.vectorstore_path)
                        raise FileNotFoundError("Stale vector store detected")
                except Exception:
                    logger.warning(f"⚠️ Unable to verify vector store content. Recreating for document ID: {self.document_id}")
                    if os.path.exists(self.vectorstore_path):
                        shutil.rmtree(self.vectorstore_path)
                    raise FileNotFoundError("Vector store verification failed")
            else:
                logger.info(f"🔍 Creating new isolated vector store: {self.vectorstore_path}")
                
                # Ensure directory exists
                os.makedirs(os.path.dirname(self.vectorstore_path), exist_ok=True)
                
                self.vector_store = await asyncio.to_thread(
                    Chroma.from_documents,
                    documents=self.documents,
                    embedding=embedding_model,
                    persist_directory=self.vectorstore_path
                )
                
                logger.info(f"✅ New isolated vector store created: {self.vectorstore_path}")

            logger.info("🔍 Setting up BM25 retriever with document isolation...")
            self.bm25_retriever = BM25Retriever.from_documents(self.documents)

            # Domain-adaptive retrieval settings
            self.bm25_retriever.k = self.domain_config["semantic_search_k"] + 3  # Slightly higher for BM25

            logger.info("✅ Isolated retrievers setup complete")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to setup isolated retrievers: {str(e)}")
            return False

    def enhanced_document_search(self, query: str, top_k: int = None) -> List[Document]:
        """🔧 FIXED: Multi-strategy document retrieval with document isolation."""
        if top_k is None:
            top_k = self.domain_config["semantic_search_k"]

        all_docs = []

        # Strategy 1: Semantic search with document filtering
        if self.vector_store:
            try:
                # 🔧 NEW: Filter by document ID to prevent cross-document contamination
                semantic_results = self.vector_store.similarity_search(
                    query, 
                    k=top_k,
                    filter={"document_id": self.document_id}  # 🔧 KEY FIX: Filter by document ID
                )
                all_docs.extend(semantic_results)
                logger.info(f"🔍 Semantic search returned {len(semantic_results)} results for document {self.document_id}")
            except Exception as e:
                logger.warning(f"⚠️ Semantic search failed: {e}, falling back to unfiltered search")
                # Fallback to unfiltered search if filtering fails
                semantic_results = self.vector_store.similarity_search(query, k=top_k)
                # Manually filter by document_id
                filtered_results = [doc for doc in semantic_results if doc.metadata.get('document_id') == self.document_id]
                all_docs.extend(filtered_results)
                logger.info(f"🔍 Fallback search returned {len(filtered_results)} filtered results")

        # Strategy 2: BM25 search (already isolated to current documents)
        if self.bm25_retriever:
            bm25_results = self.bm25_retriever.get_relevant_documents(query)
            # Additional filtering for safety
            filtered_bm25 = [doc for doc in bm25_results[:top_k] if doc.metadata.get('document_id') == self.document_id]
            all_docs.extend(filtered_bm25)
            logger.info(f"🔍 BM25 search returned {len(filtered_bm25)} results for document {self.document_id}")

        # Strategy 3: Domain-specific pattern matching (with document filtering)
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

        # Remove duplicates while preserving order and ensuring document isolation
        seen_content = set()
        unique_docs = []
        for doc in all_docs:
            # 🔧 CRITICAL FIX: Only include docs from current document
            if (doc.page_content not in seen_content and 
                doc.metadata.get('document_id') == self.document_id):
                unique_docs.append(doc)
                seen_content.add(doc.page_content)

        logger.info(f"🔍 Document search for '{self.document_domain}' domain returned {len(unique_docs)} unique results from document {self.document_id}")
        return unique_docs[:top_k]

    def _pattern_search(self, pattern: str, max_results: int = 5) -> List[Document]:
        """🔧 FIXED: Search for documents containing specific patterns with document isolation."""
        pattern_docs = []
        compiled_pattern = re.compile(pattern, re.IGNORECASE)
        
        for doc in self.documents:
            # 🔧 CRITICAL: Only search within current document
            if (doc.metadata.get('document_id') == self.document_id and
                compiled_pattern.search(doc.page_content)):
                pattern_docs.append(doc)
                if len(pattern_docs) >= max_results:
                    break
        
        logger.info(f"🔍 Pattern search for '{pattern}' found {len(pattern_docs)} results in document {self.document_id}")
        return pattern_docs

    def retrieve_and_rerank(self, query: str, top_k: int = None) -> Tuple[List[Document], List[float]]:
        """🔧 FIXED: Retrieve and rerank documents with proper isolation."""
        global reranker

        if top_k is None:
            top_k = self.domain_config["context_docs"]

        if not self.vector_store or not self.bm25_retriever:
            logger.warning("⚠️ Retrievers not initialized")
            return [], []

        try:
            # Use enhanced search with document isolation
            retrieved_docs = self.enhanced_document_search(query, top_k=15)

            if not retrieved_docs:
                logger.warning(f"⚠️ No documents retrieved for document {self.document_id}")
                return [], []

            # 🔧 VERIFICATION: Ensure all retrieved docs are from current document
            verified_docs = []
            for doc in retrieved_docs:
                if doc.metadata.get('document_id') == self.document_id:
                    verified_docs.append(doc)
                else:
                    logger.warning(f"⚠️ Filtered out document from different source: {doc.metadata.get('source_file')} (ID: {doc.metadata.get('document_id')})")
            
            if not verified_docs:
                logger.error(f"❌ No documents from current document {self.document_id} after verification")
                return [], []

            # Rerank using cross-encoder
            query_doc_pairs = [[query, doc.page_content] for doc in verified_docs]
            similarity_scores = reranker.predict(query_doc_pairs)

            # Sort by relevance score
            doc_score_pairs = sorted(
                list(zip(verified_docs, similarity_scores)),
                key=lambda x: x[1],
                reverse=True
            )

            top_docs = [pair[0] for pair in doc_score_pairs[:top_k]]
            top_scores = [float(pair[1]) for pair in doc_score_pairs[:top_k]]

            logger.info(f"🔍 Retrieved and reranked {len(top_docs)} documents for document {self.document_id} (domain: {self.document_domain})")
            
            # 🔧 FINAL VERIFICATION: Log source files to confirm isolation
            source_files = [doc.metadata.get('source_file', 'Unknown') for doc in top_docs]
            unique_sources = list(set(source_files))
            logger.info(f"🔍 Retrieved content from sources: {unique_sources}")
            
            return top_docs, top_scores

        except Exception as e:
            logger.error(f"❌ Error in retrieve_and_rerank: {str(e)}")
            return [], []

class DecisionEngine:
    """Enhanced decision engine with multi-domain support and query type classification."""

    def __init__(self):
        pass

    def _get_domain_specific_prompt_instructions(self, domain: str, query: str) -> str:
        """Get domain-specific instructions for better accuracy."""
        domain_instructions = {
            "physics": """
When explaining physics concepts:
- Include mathematical relationships and cite specific propositions from Principia
- Break down complex concepts into understandable parts
- Reference Newton's laws, definitions, and axioms explicitly
- Use precise scientific terminology
- Explain the historical context when relevant
""",
            "legal": """
When discussing legal matters:
- Reference specific articles, clauses, or legal principles mentioned in the document
- Explain legal terminology clearly
- Cite exact section numbers or constitutional provisions
- Discuss rights, procedures, and legal implications
- Maintain precision in legal interpretations
""",
            "academic": """
When analyzing academic content:
- Reference specific studies, methodologies, or theories mentioned
- Explain technical concepts clearly
- Cite authors, publications, or research findings
- Discuss implications and conclusions
- Maintain scholarly accuracy
""",
            "medical": """
When discussing medical content:
- Use appropriate medical terminology
- Reference specific conditions, treatments, or procedures
- Explain medical concepts in accessible language
- Cite clinical guidelines or medical standards
- Maintain medical accuracy and caution
""",
            "insurance": """
When analyzing insurance content:
- Apply insurance-specific terminology and concepts
- Reference policy clauses, exclusions, and benefits
- Calculate amounts precisely with applicable deductions
- Consider regulatory compliance and industry standards
- Provide structured decision-making rationale
"""
        }

        base_instruction = domain_instructions.get(domain, "")

        # Add query-specific enhancements
        if "newton" in query.lower() or "physics" in query.lower():
            base_instruction += "\nFor physics questions, include mathematical relationships and cite specific propositions from Principia."
        elif "law" in query.lower() or "rights" in query.lower():
            base_instruction += "\nFor legal questions, reference specific articles, clauses, or legal principles mentioned in the document."

        return base_instruction

    async def get_general_document_answer(self, query: str, context_docs: List[Document], domain: str = "general") -> GeneralDocumentResponse:
        """Handle any type of document with generic QA and domain adaptation."""
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

        try:
            context = ""
            source_files = []
            
            for i, doc in enumerate(context_docs):
                # 🔧 NEW: Include document ID in context for verification
                doc_id = doc.metadata.get('document_id', 'Unknown')
                source_file = doc.metadata.get('source_file', 'Unknown')
                context += f"\n--- Document Section {i+1} | Source: {source_file} | ID: {doc_id} ---\n"
                context += doc.page_content + "\n"
                if source_file not in source_files:
                    source_files.append(source_file)

            # Get domain-specific instructions
            domain_instructions = self._get_domain_specific_prompt_instructions(domain, query)

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

{domain_instructions}

Provide a direct, informative answer that addresses the user's question comprehensively."""

            logger.info(f"🤖 Calling OpenAI API for general document QA (domain: {domain})...")

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
            reasoning = f"Answer generated from {len(context_docs)} relevant document sections"

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
                source_documents=source_files,
                reasoning=reasoning
            )

        except Exception as e:
            logger.error(f"❌ Error in general document QA: {str(e)}")
            return GeneralDocumentResponse(
                query_type="GENERAL_DOCUMENT_QA",
                domain=domain,
                answer=f"An error occurred while processing this question: {str(e)}",
                confidence="low",
                source_documents=[],
                reasoning="System error during processing"
            )

    # Insurance-specific methods (preserved from original)
    def _get_policy_info_prompt(self) -> str:
        """Returns prompt template for policy information queries."""
        return """
**BAJAJ INSURANCE POLICY INFORMATION ASSISTANT**

**SYSTEM ROLE**
You are a policy information assistant specializing in Bajaj Insurance products. The user is asking for information about policy terms, definitions, or general coverage details.

**IMPORTANT: This is NOT a claim processing request. Do NOT create fictional claim amounts or processing scenarios.**

**CONTEXT (Policy Document Excerpts):**
{context}

**USER QUESTION:** "{query}"

**INSTRUCTIONS:**
- Provide clear, accurate information about the policy terms
- Reference specific policy sections when available
- Do NOT mention claim amounts, processing, or settlements
- Focus on explaining policy features, definitions, and coverage details
- If information is not available in the context, clearly state this
- Provide helpful guidance on where to find complete information

**RESPONSE FORMAT:**
Provide a clear, informative answer that directly addresses the user's question about policy information.
"""

    def _get_structured_prompt_template(self) -> str:
        """Returns a structured analysis prompt for cleaner output with enhanced clause mapping."""
        return """
**BAJAJ INSURANCE STRUCTURED CLAIM ANALYSIS WITH ADVANCED CLAUSE MAPPING**

**SYSTEM ROLE**
You are an expert BAJAJ Insurance claim analyst. Analyze the claim and provide a structured decision with clear component-wise breakdown and accurate policy clause references.

**CONTEXT (Policy Document Excerpts):**
{context}

**CLAIM QUERY:** "{query}"

**ANALYSIS REQUIREMENTS:**
1. **Determine overall claim status**: APPROVED, PARTIAL_PAYMENT, REJECTED, NEEDS_MANUAL_REVIEW, or INSUFFICIENT_INFORMATION
2. **Break down claim components** and determine approval/rejection for each with amounts
3. **Calculate total approved and rejected amounts**
4. **Identify relevant policy clauses** for each component using semantic matching
5. **Provide clear reasoning** for each component decision with specific policy references

**ENHANCED COMPONENT ANALYSIS:**
For each claim component, consider:
- Exact coverage terms in the policy
- Sub-limits and capping clauses
- Waiting period applicability
- Network vs non-network benefits
- Exclusion clauses
- Co-payment requirements
- Documentation requirements

**REQUIRED OUTPUT FORMAT (JSON):**
{{
    "status": "[APPROVED | PARTIAL_PAYMENT | REJECTED | NEEDS_MANUAL_REVIEW | INSUFFICIENT_INFORMATION]",
    "confidence": [Float between 0.0 and 1.0],
    "approvedAmount": [Total approved amount in INR],
    "rejectedAmount": [Total rejected amount in INR],
    "breakdown": {{
        "approved": [
            {{
                "component": "Component name (e.g., Hospitalization, Surgery, Medicines)",
                "amount": [Amount in INR],
                "reason": "Brief reason for approval with specific policy reference"
            }}
        ],
        "rejected": [
            {{
                "component": "Component name",
                "amount": [Amount in INR],
                "reason": "Brief reason for rejection (policy clause, exclusion, etc.) with specific reference"
            }}
        ]
    }},
    "keyPolicyReferences": [
        {{
            "title": "Specific policy section or clause name",
            "note": "Detailed explanation of how this clause applies to the claim components"
        }}
    ],
    "summary": "Comprehensive paragraph summary of the overall decision, key policy factors considered, specific clauses applied, and final outcome with total amounts and percentages."
}}

**CRITICAL INSTRUCTIONS:**
- Extract specific amounts from the query and categorize each component precisely
- Reference exact policy clauses with specific section names or numbers
- Ensure approvedAmount + rejectedAmount = total claimed amount
- Be precise with financial calculations including any co-payments or deductibles
- Provide clear, actionable decisions with policy justification
- All amounts must be parseable numbers without currency symbols or commas in the JSON
- Components must have positive amounts
- Policy references must be specific and relevant to the components
- Include sub-limits, waiting periods, and network benefits in reasoning
- If no components can be determined, provide meaningful error in status
- Prioritize accuracy over speed in clause mapping
"""

    async def get_policy_info(self, query: str, context_docs: List[Document]) -> PolicyInfoResponse:
        """Generate policy information response for non-claim queries."""
        global openai_client

        if not context_docs:
            return PolicyInfoResponse(
                question=query,
                answer="No relevant policy documents found to answer your question.",
                confidence=0.0,
                sources=[]
            )

        try:
            context_text = ""
            sources = []
            
            for i, doc in enumerate(context_docs):
                source_file = doc.metadata.get('source_file', 'Unknown')
                context_text += f"\n--- Document Excerpt {i+1} | Source: {source_file} ---\n"
                context_text += doc.page_content + "\n"
                if source_file not in sources:
                    sources.append(source_file)

            prompt = self._get_policy_info_prompt().format(
                query=query,
                context=context_text
            )

            logger.info("🤖 Calling OpenAI API for policy information...")

            response = await openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful policy information assistant. Provide clear, accurate information about insurance policy terms without creating fictional scenarios or amounts."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.0,
                max_tokens=2000
            )

            answer = response.choices[0].message.content

            # Extract policy section if mentioned
            policy_section = None
            if any(word in answer.lower() for word in ['section', 'clause', 'article']):
                policy_section = "Policy document referenced"

            return PolicyInfoResponse(
                question=query,
                answer=answer,
                policy_section=policy_section,
                confidence=0.8,  # High confidence for policy info
                sources=sources
            )

        except Exception as e:
            logger.error(f"❌ Error in policy info generation: {str(e)}")
            return PolicyInfoResponse(
                question=query,
                answer=f"An error occurred while retrieving policy information: {str(e)}",
                confidence=0.0,
                sources=[]
            )

    async def get_structured_decision(self, query: str, context_docs: List[Document],
                                    similarity_scores: List[float] = None) -> Dict[str, Any]:
        """Generate structured decision based on query and retrieved documents with enhanced clause mapping."""
        global openai_client

        if not context_docs:
            return {
                "status": "INSUFFICIENT_INFORMATION",
                "confidence": 0.0,
                "approvedAmount": 0.0,
                "rejectedAmount": 0.0,
                "breakdown": {"approved": [], "rejected": []},
                "keyPolicyReferences": [{"title": "No Documents", "note": "No relevant policy documents found to analyze the claim."}],
                "summary": "Insufficient information in policy documents to analyze the claim."
            }

        try:
            context_text = ""
            for i, doc in enumerate(context_docs):
                score_info = f" (Relevance: {similarity_scores[i]:.2f})" if similarity_scores else ""
                source_file = doc.metadata.get('source_file', 'Unknown')
                context_text += f"\n--- Document Excerpt {i+1}{score_info} | Source: {source_file} ---\n"
                context_text += doc.page_content + "\n"

            prompt = self._get_structured_prompt_template().format(
                query=query,
                context=context_text
            )

            logger.info("🤖 Calling OpenAI API for structured BAJAJ Insurance analysis with enhanced clause mapping...")

            response = await openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a BAJAJ Insurance expert with advanced policy clause mapping capabilities. Always respond with valid JSON in the exact format specified. Provide structured claim analysis with precise component-wise breakdown and accurate policy clause references."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.0,
                max_tokens=3500,
                response_format={"type": "json_object"}
            )

            response_content = response.choices[0].message.content
            logger.info("✅ Received structured response from OpenAI with clause mapping")

            try:
                decision_data = json.loads(response_content)
                
                # Validate, sanitize, and enhance clause mapping
                decision_data = await self._sanitize_and_enhance_structured_response(decision_data, query, context_text)

                logger.info(f"✅ Enhanced structured decision: {decision_data['status']} - Approved: ₹{decision_data['approvedAmount']:,.2f} | Clauses: {len(decision_data['keyPolicyReferences'])}")
                return decision_data

            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logger.error(f"❌ Failed to parse structured response: {str(e)}")
                logger.error(f"❌ Raw response content: {response_content[:500]}...")
                
                return {
                    "status": "NEEDS_MANUAL_REVIEW",
                    "confidence": 0.0,
                    "approvedAmount": 0.0,
                    "rejectedAmount": 0.0,
                    "breakdown": {"approved": [], "rejected": []},
                    "keyPolicyReferences": [{"title": "System Error", "note": f"Error parsing response: {str(e)}"}],
                    "summary": "System error occurred during analysis. Manual review required."
                }

        except Exception as e:
            logger.error(f"❌ Error in structured decision generation: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error during structured analysis: {str(e)}")

    async def _sanitize_and_enhance_structured_response(self, decision_data: Dict[str, Any], original_query: str, context_text: str) -> Dict[str, Any]:
        """Sanitize, validate, and enhance the structured response with improved clause mapping."""
        # Set defaults for missing fields
        defaults = {
            "status": "NEEDS_MANUAL_REVIEW",
            "confidence": 0.0,
            "approvedAmount": 0.0,
            "rejectedAmount": 0.0,
            "breakdown": {"approved": [], "rejected": []},
            "keyPolicyReferences": [],
            "summary": "Analysis completed with enhanced clause mapping."
        }

        # Ensure all required fields exist
        for field, default_value in defaults.items():
            if field not in decision_data:
                decision_data[field] = default_value
                logger.warning(f"⚠️ Missing field '{field}' - using default value")

        # Sanitize breakdown components
        if "breakdown" not in decision_data or not isinstance(decision_data["breakdown"], dict):
            decision_data["breakdown"] = {"approved": [], "rejected": []}

        if "approved" not in decision_data["breakdown"]:
            decision_data["breakdown"]["approved"] = []
        if "rejected" not in decision_data["breakdown"]:
            decision_data["breakdown"]["rejected"] = []

        # Parse and validate amounts in breakdown components
        all_components = []
        for component_type in ["approved", "rejected"]:
            sanitized_components = []
            for component in decision_data["breakdown"][component_type]:
                if isinstance(component, dict):
                    try:
                        sanitized_component = {
                            "component": str(component.get("component", "Unknown Component")),
                            "amount": parse_amount(component.get("amount", 0)),
                            "reason": str(component.get("reason", "No reason provided")),
                            "status": component_type  # Add status for clause mapping
                        }

                        # Validate amount is non-negative
                        if sanitized_component["amount"] < 0:
                            logger.warning(f"⚠️ Negative amount found: {sanitized_component['amount']}, setting to 0")
                            sanitized_component["amount"] = 0.0

                        sanitized_components.append(sanitized_component)
                        all_components.append(sanitized_component)
                    except Exception as e:
                        logger.warning(f"⚠️ Error sanitizing component {component}: {e}")
                        continue

            decision_data["breakdown"][component_type] = [
                {k: v for k, v in comp.items() if k != "status"}
                for comp in sanitized_components
            ]

        # Recalculate totals from breakdown
        approved_total = round(sum(item["amount"] for item in decision_data["breakdown"]["approved"]), 2)
        rejected_total = round(sum(item["amount"] for item in decision_data["breakdown"]["rejected"]), 2)
        decision_data["approvedAmount"] = approved_total
        decision_data["rejectedAmount"] = rejected_total

        # Apply advanced clause mapping to components
        try:
            if all_components:
                enhanced_clauses = map_clauses_to_components(all_components, context_text)
                
                # Merge with existing policy references if any
                existing_clauses = decision_data.get("keyPolicyReferences", [])
                
                # Combine and deduplicate clauses
                all_clause_titles = set()
                combined_clauses = []
                
                # Add existing clauses first
                for clause in existing_clauses:
                    if isinstance(clause, dict) and clause.get("title") not in all_clause_titles:
                        combined_clauses.append({
                            "title": clause.get("title", "Unknown Policy Section"),
                            "note": clause.get("note", "Policy reference")
                        })
                        all_clause_titles.add(clause.get("title", ""))
                
                # Add enhanced mapped clauses
                for clause in enhanced_clauses:
                    if clause.get("title") not in all_clause_titles:
                        combined_clauses.append({
                            "title": clause.get("title", "Policy Provision"),
                            "note": clause.get("note", "Relevant policy terms")
                        })
                        all_clause_titles.add(clause.get("title", ""))
                
                decision_data["keyPolicyReferences"] = combined_clauses
                logger.info(f"✅ Enhanced clause mapping: {len(combined_clauses)} policy references mapped")

        except Exception as clause_error:
            logger.error(f"❌ Error in enhanced clause mapping: {clause_error}")
            # Ensure at least some policy references exist
            if not decision_data.get("keyPolicyReferences"):
                decision_data["keyPolicyReferences"] = [{
                    "title": "Policy Terms and Conditions",
                    "note": f"Please refer to the complete policy document for detailed terms. Clause mapping error: {str(clause_error)}"
                }]

        # Validate amounts consistency
        total_calculated = approved_total + rejected_total
        if total_calculated > 0:
            # Extract claim amount from query if possible
            try:
                amount_patterns = [
                    r'[\₹Rs\.]*\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
                    r'amount[:\s]+[\₹Rs\.]*\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
                    r'claim[:\s]+[\₹Rs\.]*\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
                    r'total[:\s]+[\₹Rs\.]*\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
                ]
                
                for pattern in amount_patterns:
                    amount_matches = re.findall(pattern, original_query, re.IGNORECASE)
                    if amount_matches:
                        query_amount = parse_amount(amount_matches[0])
                        if query_amount > 0:
                            if abs(total_calculated - query_amount) > 1e-2:
                                logger.warning(f"⚠️ Breakdown total ({total_calculated}) doesn't match query amount ({query_amount})")
                                decision_data["keyPolicyReferences"].append({
                                    "title": "Amount Calculation Notice",
                                    "note": f"Breakdown components total ₹{total_calculated:,.2f}. Original claim amount: ₹{query_amount:,.2f}. Difference may be due to applicable deductibles, co-payments, or sub-limits."
                                })
                            break
            except Exception:
                pass

        # Ensure confidence is within bounds
        decision_data["confidence"] = max(0.0, min(1.0, float(decision_data.get("confidence", 0.0))))

        # Validate amounts using utility function
        is_valid, validation_message = validate_amounts(decision_data["breakdown"])
        if not is_valid:
            logger.error(f"❌ Amount validation failed: {validation_message}")
            decision_data["keyPolicyReferences"].append({
                "title": "Amount Validation Notice",
                "note": validation_message
            })

        # Enhance summary with clause mapping information
        if decision_data.get("summary") and len(decision_data.get("keyPolicyReferences", [])) > 0:
            clause_count = len(decision_data["keyPolicyReferences"])
            enhanced_summary = decision_data["summary"]
            if "policy" not in enhanced_summary.lower():
                enhanced_summary += f" This analysis references {clause_count} relevant policy clause(s) for comprehensive coverage determination."
            decision_data["summary"] = enhanced_summary

        return decision_data

    # 🔧 FIXED: Main routing method with corrected query classification
    async def process_query(self, query: str, rag_system: RAGSystem) -> Union[PolicyInfoResponse, Dict[str, Any], GeneralDocumentResponse]:
        """Route query to appropriate processing method based on query type and domain."""
        # Get document content sample for classification
        document_content = ""
        if rag_system.documents:
            document_content = ' '.join([doc.page_content for doc in rag_system.documents[:3]])  # First 3 docs

        # 🔧 USE FIXED CLASSIFICATION
        query_type, domain = classify_query_type(query, document_content)
        logger.info(f"🔍 Query classified as: {query_type}, Domain: {domain}")

        # Retrieve relevant documents with domain-adaptive settings
        top_k = rag_system.domain_config.get("context_docs", 7)
        retrieved_docs, similarity_scores = rag_system.retrieve_and_rerank(query, top_k=top_k)

        if query_type == "INSURANCE_POLICY_INFO":
            return await self.get_policy_info(query, retrieved_docs)
        elif query_type in ["INSURANCE_CLAIM_PROCESSING", "INSURANCE_GENERAL"]:
            # For insurance claim processing, use structured decision
            return await self.get_structured_decision(query, retrieved_docs, similarity_scores)
        else:  # GENERAL_DOCUMENT_QA
            # Use the new general document QA method
            return await self.get_general_document_answer(query, retrieved_docs, domain)

# ENHANCED: Parallel processing with multi-domain support
async def process_questions_parallel_structured(questions: List[str], rag_system: RAGSystem, decision_engine: DecisionEngine) -> List[Union[StructuredClaimResponse, PolicyInfoResponse, GeneralDocumentResponse]]:
    """Process multiple questions in parallel with structured output and multi-domain support."""
    
    async def process_single_question_structured(question: str) -> Union[StructuredClaimResponse, PolicyInfoResponse, GeneralDocumentResponse]:
        """Process a single question and return structured response."""
        logger.info(f"🔄 Processing structured question: '{question}'")
        
        try:
            # Use the enhanced routing method
            result = await decision_engine.process_query(question, rag_system)

            # Convert to appropriate response format
            if isinstance(result, (PolicyInfoResponse, GeneralDocumentResponse)):
                return result
            else:
                # Handle structured claim response
                if not isinstance(result, dict):
                    raise ValueError("Invalid decision result format")

                # Safe component creation with better error handling
                try:
                    approved_components = []
                    for comp in result.get("breakdown", {}).get("approved", []):
                        if isinstance(comp, dict) and all(key in comp for key in ["component", "amount", "reason"]):
                            approved_components.append(ClaimComponent(**comp))
                        else:
                            logger.warning(f"⚠️ Skipping invalid approved component: {comp}")

                    rejected_components = []
                    for comp in result.get("breakdown", {}).get("rejected", []):
                        if isinstance(comp, dict) and all(key in comp for key in ["component", "amount", "reason"]):
                            rejected_components.append(ClaimComponent(**comp))
                        else:
                            logger.warning(f"⚠️ Skipping invalid rejected component: {comp}")

                    policy_references = []
                    for ref in result.get("keyPolicyReferences", []):
                        if isinstance(ref, dict) and all(key in ref for key in ["title", "note"]):
                            policy_references.append(PolicyReference(**ref))
                        else:
                            logger.warning(f"⚠️ Skipping invalid policy reference: {ref}")

                    # Ensure we have at least one policy reference
                    if not policy_references:
                        policy_references.append(PolicyReference(
                            title="General Policy Terms",
                            note="Please refer to the complete policy document for detailed coverage terms and conditions."
                        ))

                except Exception as comp_error:
                    logger.error(f"❌ Error creating components: {comp_error}")
                    approved_components = []
                    rejected_components = []
                    policy_references = [PolicyReference(title="Component Processing Error", note=f"Error processing components: {str(comp_error)}")]

                # Create structured response with safe field access
                structured_response = StructuredClaimResponse(
                    status=result.get("status", "NEEDS_MANUAL_REVIEW"),
                    confidence=result.get("confidence", 0.0),
                    approvedAmount=result.get("approvedAmount", 0.0),
                    rejectedAmount=result.get("rejectedAmount", 0.0),
                    breakdown=ClaimBreakdown(
                        approved=approved_components,
                        rejected=rejected_components
                    ),
                    keyPolicyReferences=policy_references,
                    summary=result.get("summary", "Analysis completed with multi-domain support.")
                )

                logger.info(f"✅ Structured response: {len(approved_components)} approved, {len(rejected_components)} rejected, {len(policy_references)} policy references")
                return structured_response

        except Exception as e:
            logger.error(f"❌ Error in structured processing for question: {str(e)}")
            
            # Return error response based on detected domain
            document_content = ""
            if rag_system.documents:
                document_content = ' '.join([doc.page_content for doc in rag_system.documents[:3]])
            
            query_type, domain = classify_query_type(question, document_content)
            
            if query_type == "GENERAL_DOCUMENT_QA":
                return GeneralDocumentResponse(
                    query_type="GENERAL_DOCUMENT_QA",
                    domain=domain,
                    answer=f"An error occurred while processing this question: {str(e)}",
                    confidence="low",
                                        source_documents=[],
                    reasoning=f"System error during processing: {str(e)}"
                )
            else:
                # Return insurance error response
                return StructuredClaimResponse(
                    status="NEEDS_MANUAL_REVIEW",
                    confidence=0.0,
                    approvedAmount=0.0,
                    rejectedAmount=0.0,
                    breakdown=ClaimBreakdown(approved=[], rejected=[]),
                    keyPolicyReferences=[PolicyReference(
                        title="Processing Error",
                        note=f"An error occurred during analysis: {str(e)}"
                    )],
                    summary=f"Error processing question: {str(e)}"
                )

    # Process questions concurrently with controlled parallelism
    semaphore = asyncio.Semaphore(3)  # Limit concurrent processing

    async def process_with_semaphore(question: str):
        async with semaphore:
            return await process_single_question_structured(question)

    tasks = [process_with_semaphore(q) for q in questions]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Handle any exceptions in results
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"❌ Error processing question {i+1}: {str(result)}")
            # Create error response
            processed_results.append(GeneralDocumentResponse(
                query_type="GENERAL_DOCUMENT_QA",
                domain="general",
                answer=f"An error occurred while processing this question: {str(result)}",
                confidence="low",
                source_documents=[],
                reasoning="Error during parallel processing"
            ))
        else:
            processed_results.append(result)

    logger.info(f"✅ Completed parallel processing of {len(questions)} questions")
    return processed_results

# --- API Endpoints ---

@api_v1_router.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@api_v1_router.post("/upload-and-query", response_model=EnhancedHackRxResponse, dependencies=[Depends(verify_bearer_token)])
async def upload_and_query_documents(
    request: Request,
    documents: UploadFile = File(...),
    questions: str = Form(...)
):
    """🔧 FIXED: Upload documents and process questions with proper document isolation."""
    start_time = time.time()
    logger.info(f"📁 Upload and Query request received - File: {documents.filename}")

    try:
        # Parse questions
        try:
            question_list = json.loads(questions)
            if not isinstance(question_list, list):
                raise ValueError("Questions must be a list")
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON format for questions")

        if not question_list:
            raise HTTPException(status_code=400, detail="At least one question is required")

        # Validate uploaded file
        is_valid, validation_message = validate_uploaded_file(documents)
        if not is_valid:
            raise HTTPException(status_code=400, detail=validation_message)

        # Read file content for document ID generation
        file_content = await documents.read()
        await documents.seek(0)  # Reset file pointer

        # Save file with proper isolation
        file_path = await save_uploaded_file(documents)
        
        try:
            # Initialize RAG system with document isolation
            rag_system = RAGSystem()
            decision_engine = DecisionEngine()

            # 🔧 FIXED: Process documents with content tracking for proper isolation
            result = await rag_system.load_and_process_documents([file_path], [file_content])
            
            if not result['documents']:
                raise HTTPException(status_code=500, detail="No documents could be processed")

            # Setup isolated retrievers
            setup_success = await rag_system.setup_retrievers()
            if not setup_success:
                raise HTTPException(status_code=500, detail="Failed to initialize document retrieval system")

            # 🔧 VERIFICATION: Log document isolation details
            logger.info(f"🔧 Document processing complete:")
            logger.info(f"   Document ID: {rag_system.document_id}")
            logger.info(f"   Domain: {rag_system.document_domain}")
            logger.info(f"   Vector store: {rag_system.vectorstore_path}")
            logger.info(f"   Total chunks: {len(rag_system.documents)}")

            # Cache the RAG system with document ID
            if rag_system.document_id:
                manage_cache()  # Clean old entries if needed
                DOCUMENT_CACHE[rag_system.document_id] = {
                    'rag_system': rag_system,
                    'timestamp': time.time(),
                    'domain': rag_system.document_domain,
                    'file_info': result
                }
                logger.info(f"✅ Cached RAG system for document: {rag_system.document_id}")

            # Process questions with enhanced structured output
            answers = await process_questions_parallel_structured(question_list, rag_system, decision_engine)

            processing_time = time.time() - start_time
            logger.info(f"✅ Upload and Query completed in {processing_time:.2f}s - Document: {rag_system.document_id}")

            return EnhancedHackRxResponse(answers=answers)

        finally:
            # Cleanup temporary file
            if os.path.exists(file_path):
                os.unlink(file_path)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error in upload_and_query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@api_v1_router.post("/process-url", response_model=EnhancedHackRxResponse, dependencies=[Depends(verify_bearer_token)])
async def process_document_from_url(request: HackRxRequest):
    """🔧 FIXED: Process document from URL with proper isolation."""
    start_time = time.time()
    logger.info(f"🌐 URL processing request: {request.documents}")

    try:
        # Download document from URL
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(str(request.documents))
            response.raise_for_status()
            
            file_content = response.content
            content_type = response.headers.get('content-type', 'application/pdf')
            
            # Detect content type from URL if header is generic
            if content_type in ['application/octet-stream', 'binary/octet-stream']:
                content_type = detect_content_type_from_url(str(request.documents))

        # Create temporary file with appropriate extension
        file_extension = get_file_extension_from_content_type(content_type)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name

        try:
            # Initialize RAG system with document isolation
            rag_system = RAGSystem()
            decision_engine = DecisionEngine()

            # 🔧 FIXED: Process with content for proper document ID generation
            result = await rag_system.load_and_process_documents([temp_file_path], [file_content])
            
            if not result['documents']:
                raise HTTPException(status_code=500, detail="No documents could be processed from URL")

            # Setup isolated retrievers
            setup_success = await rag_system.setup_retrievers()
            if not setup_success:
                raise HTTPException(status_code=500, detail="Failed to initialize document retrieval system")

            # 🔧 VERIFICATION: Log document isolation details
            logger.info(f"🔧 URL document processing complete:")
            logger.info(f"   Document ID: {rag_system.document_id}")
            logger.info(f"   Domain: {rag_system.document_domain}")
            logger.info(f"   Vector store: {rag_system.vectorstore_path}")
            logger.info(f"   Source URL: {str(request.documents)}")

            # Cache the RAG system
            if rag_system.document_id:
                manage_cache()
                DOCUMENT_CACHE[rag_system.document_id] = {
                    'rag_system': rag_system,
                    'timestamp': time.time(),
                    'domain': rag_system.document_domain,
                    'file_info': result,
                    'source_url': str(request.documents)
                }
                logger.info(f"✅ Cached RAG system for URL document: {rag_system.document_id}")

            # Process questions
            answers = await process_questions_parallel_structured(request.questions, rag_system, decision_engine)

            processing_time = time.time() - start_time
            logger.info(f"✅ URL processing completed in {processing_time:.2f}s - Document: {rag_system.document_id}")

            return EnhancedHackRxResponse(answers=answers)

        finally:
            # Cleanup temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    except httpx.HTTPError as e:
        logger.error(f"❌ HTTP error downloading document: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to download document: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error in process_document_from_url: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@api_v1_router.post("/query", response_model=UnifiedResponse, dependencies=[Depends(verify_bearer_token)])
async def query_cached_documents(request: QueryRequest):
    """🔧 FIXED: Query against most recently cached document with proper isolation."""
    start_time = time.time()
    logger.info(f"❓ Query request: '{request.query[:100]}...'")

    try:
        if not DOCUMENT_CACHE:
            raise HTTPException(
                status_code=400, 
                detail="No documents cached. Please upload a document first using /upload-and-query or /process-url"
            )

        # Get most recent document from cache
        latest_doc_id = max(DOCUMENT_CACHE.keys(), key=lambda k: DOCUMENT_CACHE[k]['timestamp'])
        cached_data = DOCUMENT_CACHE[latest_doc_id]
        rag_system = cached_data['rag_system']
        domain = cached_data['domain']

        logger.info(f"🔍 Using cached document: {latest_doc_id} (domain: {domain})")

        # 🔧 VERIFICATION: Ensure we're querying the right document
        logger.info(f"🔧 Query verification:")
        logger.info(f"   Document ID: {rag_system.document_id}")
        logger.info(f"   Vector store: {rag_system.vectorstore_path}")
        logger.info(f"   Document count: {len(rag_system.documents)}")

        # Initialize decision engine
        decision_engine = DecisionEngine()

        # Process query with isolated retrieval
        result = await decision_engine.process_query(request.query, rag_system)

        # Convert to unified response format
        if isinstance(result, GeneralDocumentResponse):
            unified_response = UnifiedResponse(
                query_type=result.query_type,
                domain=result.domain,
                answer=result.answer,
                confidence=result.confidence,
                source_documents=result.source_documents
            )
        elif isinstance(result, PolicyInfoResponse):
            unified_response = UnifiedResponse(
                query_type="INSURANCE_POLICY_INFO",
                domain="insurance",
                answer=result.answer,
                confidence="high" if result.confidence > 0.7 else "medium",
                source_documents=result.sources
            )
        else:
            # Structured claim response
            unified_response = UnifiedResponse(
                query_type="INSURANCE_CLAIM_PROCESSING",
                domain="insurance",
                answer=result.get("summary", "Claim processed"),
                decision=result.get("status", "UNKNOWN"),
                approved_amount=result.get("approvedAmount", 0.0),
                policy_references=[ref.get("title", "") for ref in result.get("keyPolicyReferences", [])],
                confidence="high" if result.get("confidence", 0) > 0.7 else "medium",
                source_documents=[f"Document ID: {latest_doc_id}"]
            )

        processing_time = time.time() - start_time
        logger.info(f"✅ Query completed in {processing_time:.2f}s - Document: {latest_doc_id}")

        return unified_response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in query processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query processing error: {str(e)}")

@api_v1_router.get("/cache-status")
async def get_cache_status(token: str = Depends(verify_bearer_token)):
    """Get information about cached documents."""
    cache_info = {}
    
    for doc_id, data in DOCUMENT_CACHE.items():
        cache_info[doc_id] = {
            "domain": data.get("domain", "unknown"),
            "timestamp": datetime.fromtimestamp(data["timestamp"]).isoformat(),
            "file_info": {
                "processed_files": data.get("file_info", {}).get("processed_files", []),
                "total_chunks": data.get("file_info", {}).get("total_chunks", 0)
            },
            "source_url": data.get("source_url")  # Only present for URL-based documents
        }
    
    return {
        "cached_documents": len(DOCUMENT_CACHE),
        "cache_details": cache_info,
        "max_cache_size": MAX_CACHE_SIZE
    }

@api_v1_router.delete("/clear-cache")
async def clear_document_cache(token: str = Depends(verify_bearer_token)):
    """Clear all cached documents and vector stores."""
    global DOCUMENT_CACHE
    
    cleared_count = len(DOCUMENT_CACHE)
    DOCUMENT_CACHE.clear()
    
    # Also clear vector stores directory
    try:
        if os.path.exists(VECTORSTORE_BASE_DIR):
            shutil.rmtree(VECTORSTORE_BASE_DIR)
            os.makedirs(VECTORSTORE_BASE_DIR, exist_ok=True)
        logger.info("✅ Vector stores directory cleared")
    except Exception as e:
        logger.warning(f"⚠️ Failed to clear vector stores: {e}")
    
    return {
        "message": f"Cleared {cleared_count} cached documents and vector stores",
        "timestamp": datetime.now().isoformat()
    }

# Add the router to the main app
app.include_router(api_v1_router)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with system information."""
    return {
        "service": "Multi-Domain Document QA API",
        "version": "4.0.0",
        "description": "AI-powered system for analyzing any document type with specialized insurance capabilities and intelligent query classification.",
        "features": [
            "🔧 FIXED: Document Isolation - Each document gets its own vector store",
            "Multi-domain support (Physics, Legal, Insurance, Academic, Medical, General)",
            "Intelligent query type classification",
            "Domain-adaptive processing",
            "Bearer token authentication",
            "Parallel question processing",
            "Enhanced clause mapping for insurance",
            "Structured responses with confidence scoring"
        ],
        "endpoints": {
            "upload": "/api/v1/upload-and-query",
            "url_process": "/api/v1/process-url", 
            "query": "/api/v1/query",
            "cache_status": "/api/v1/cache-status",
            "clear_cache": "/api/v1/clear-cache",
            "health": "/api/v1/health"
        },
        "authentication": "Bearer token required (set BEARER_TOKEN environment variable)",
        "timestamp": datetime.now().isoformat()
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler."""
    logger.error(f"HTTP {exc.status_code}: {exc.detail}")
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
    """General exception handler."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment or use default
    port = int(os.getenv("PORT", 8000))
    
    logger.info(f"🚀 Starting Multi-Domain Document QA API on port {port}")
    logger.info("🔧 KEY FEATURES:")
    logger.info("   ✅ FIXED: Proper document isolation with unique vector stores")
    logger.info("   ✅ Enhanced domain detection and adaptive processing")
    logger.info("   ✅ Advanced query classification preventing cross-contamination")
    logger.info("   ✅ Bearer token authentication for security")
    logger.info("   ✅ Comprehensive error handling and logging")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,  # Set to False for production
        access_log=True,
        log_level="info"
    )

