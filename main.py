import os

# Add before ChromaDB usage - Performance optimization
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

import sys
import asyncio
import logging
import time
from typing import List, Dict, Any, Tuple, Optional
from contextlib import asynccontextmanager
import uuid
import json
import shutil
import tempfile
import hashlib

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

# PERFORMANCE OPTIMIZATION: Document cache for embeddings
DOCUMENT_CACHE = {}

# PERFORMANCE OPTIMIZATION: Persistent directory for vector store
PERSISTENT_CHROMA_DIR = "/tmp/persistent_chroma"

# ADDED: Memory Management for Cache
MAX_CACHE_SIZE = 10 # Maximum number of documents to cache

# FIXED: Enhanced component keywords mapping for robust clause matching
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

# ADDED: Semantic clause types for better categorization
CLAUSE_TYPES = {
    "coverage": ["coverage", "benefit", "included", "covered", "payable", "eligible"],
    "exclusion": ["exclusion", "not covered", "excluded", "limitation", "restriction", "exception"],
    "condition": ["condition", "requirement", "criteria", "eligibility", "qualification"],
    "limit": ["limit", "maximum", "ceiling", "cap", "sub-limit", "sublimit"],
    "procedure": ["procedure", "process", "claim", "settlement", "payment", "documentation"],
    "waiting_period": ["waiting", "moratorium", "cooling", "initial", "specific waiting period"],
    "deductible": ["deductible", "excess", "co-payment", "copay", "patient share"]
}

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

# NEW: Advanced clause mapping functions
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

# PERFORMANCE OPTIMIZATION: Embedding cache function (FIXED)
async def get_or_create_embeddings(file_content: bytes) -> Tuple[str, bool]:
    """
    Cache embeddings based on file hash to avoid recomputation.
    Returns the file hash and a boolean indicating if it was cached.
    """
    # Create hash of file content
    file_hash = hashlib.md5(file_content).hexdigest()
    is_cached = file_hash in DOCUMENT_CACHE
    
    if is_cached:
        logger.info(f"✅ Using cached embeddings for document {file_hash[:8]}")
    else:
        # Process only if not cached
        logger.info(f"🔄 Processing new document {file_hash[:8]}")
    
    return file_hash, is_cached

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize shared resources at startup."""
    global embedding_model, reranker, openai_client
    logger.info("🚀 Initializing BAJAJ Insurance Claim Analysis System...")
    
    try:
        # PERFORMANCE OPTIMIZATION: Initialize embedding model with optimized settings
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={
                'device': 'cpu'
            },
            encode_kwargs={
                'batch_size': 32, # Move batch_size here
                'show_progress_bar': False
            }
        )
        logger.info("✅ Embedding model loaded with optimized settings")
        
        # PERFORMANCE OPTIMIZATION: Initialize lighter reranker model
        reranker = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L2-v2') # Smaller, faster model
        logger.info("✅ Lightweight reranker model loaded")
        
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("❌ OPENAI_API_KEY environment variable not set")
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        openai_client = AsyncOpenAI(api_key=api_key)
        logger.info("✅ OpenAI client initialized")
        
        # Log bearer token status
        if EXPECTED_BEARER_TOKEN and EXPECTED_BEARER_TOKEN != "your-default-token-here":
            logger.info("✅ Bearer token authentication configured")
        else:
            logger.warning("⚠️ Bearer token not properly configured")
        
        # PERFORMANCE OPTIMIZATION: Create persistent directory
        os.makedirs(PERSISTENT_CHROMA_DIR, exist_ok=True)
        logger.info(f"✅ Persistent vector store directory ready: {PERSISTENT_CHROMA_DIR}")
        
        # Log clause mapping capabilities
        logger.info(f"✅ Enhanced clause mapping initialized with {len(COMPONENT_KEYWORDS)} component types")
        
        logger.info("🎉 BAJAJ Insurance System initialization complete!")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize BAJAJ Insurance system: {str(e)}")
        raise
    
    yield
    
    # Cleanup
    logger.info("🔄 Shutting down BAJAJ Insurance system...")

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

class HackRxRequest(BaseModel):
    documents: HttpUrl = Field(..., description="URL to the policy PDF document.")
    questions: List[str] = Field(..., description="A list of questions to ask about the document.")

class HackRxResponse(BaseModel):
    answers: List[str]

# NEW MODELS FOR STRUCTURED RESPONSES
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

class EnhancedHackRxResponse(BaseModel):
    answers: List[StructuredClaimResponse]

# Initialize FastAPI app
app = FastAPI(
    title="BAJAJ Insurance Claim Analysis API",
    description="AI-powered system for analyzing BAJAJ insurance claims and policy documents across all domains.",
    version="3.1.0",
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
    max_size = 50 * 1024 * 1024 # 50MB
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
    """Enhanced Retrieval-Augmented Generation system for BAJAJ insurance documents."""

    def __init__(self):
        self.vector_store = None
        self.bm25_retriever = None
        self.documents = []
        self.processed_files = []

    async def _process_single_file(self, file_path: str) -> List[Document]:
        """Process a single file and return documents."""
        try:
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
            
            for doc in documents:
                doc.metadata.update({
                    'source_file': os.path.basename(file_path),
                    'file_type': file_extension,
                    'processed_at': datetime.now().isoformat()
                })

            logger.info(f"✅ Processed {len(documents)} documents from {os.path.basename(file_path)}")
            return documents

        except Exception as e:
            logger.error(f"❌ Error processing {file_path}: {str(e)}")
            # FIXED: Raise specific exception for better error handling
            raise HTTPException(status_code=500, detail=f"Failed to process document {os.path.basename(file_path)}: {str(e)}")

    async def load_and_process_documents(self, file_paths: List[str]) -> Dict[str, Any]:
        """Process multiple files concurrently."""
        all_docs = []
        skipped_files = []
        self.processed_files = []

        tasks = [self._process_single_file(path) for path in file_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result, file_path in zip(results, file_paths):
            if isinstance(result, Exception):
                logger.error(f"❌ Exception processing {file_path}: {result}")
                skipped_files.append(os.path.basename(file_path))
                # FIXED: Don't silently continue, re-raise critical errors
                if isinstance(result, HTTPException):
                    raise result
            elif result:
                all_docs.extend(result)
                self.processed_files.append(os.path.basename(file_path))
            else:
                skipped_files.append(os.path.basename(file_path))

        if not all_docs:
            logger.warning("⚠️ No documents were successfully processed")
            # FIXED: Raise exception instead of returning empty result
            raise HTTPException(status_code=500, detail="No documents could be processed successfully")

        # PERFORMANCE OPTIMIZATION: Optimized text splitter settings
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, # Increased from 1000 for better performance
            chunk_overlap=150, # Reduced from 200 for better performance
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        chunked_docs = await asyncio.to_thread(text_splitter.split_documents, all_docs)
        chunked_docs = [doc for doc in chunked_docs if len(doc.page_content.strip()) > 50]

        self.documents = chunked_docs

        logger.info(f"📄 Created {len(chunked_docs)} chunks from {len(self.processed_files)} files")

        return {
            'documents': chunked_docs,
            'processed_files': self.processed_files,
            'skipped_files': skipped_files,
            'total_chunks': len(chunked_docs)
        }

    async def setup_retrievers(self, persist_directory: str = PERSISTENT_CHROMA_DIR):
        """Initialize vector store and BM25 retriever with persistence optimization."""
        global embedding_model

        if not self.documents:
            logger.warning("⚠️ No documents available for retriever setup")
            return False

        try:
            # PERFORMANCE OPTIMIZATION: Check if embeddings already exist
            if os.path.exists(f"{persist_directory}/chroma.sqlite3"):
                logger.info("✅ Loading existing vector store from persistent directory")
                self.vector_store = Chroma(
                    persist_directory=persist_directory,
                    embedding_function=embedding_model
                )
            else:
                logger.info(f"🔍 Creating new vector store in {persist_directory}...")
                self.vector_store = await asyncio.to_thread(
                    Chroma.from_documents,
                    documents=self.documents,
                    embedding=embedding_model,
                    persist_directory=persist_directory
                )
                logger.info("✅ New vector store created and persisted")

            logger.info("🔍 Setting up BM25 retriever...")
            self.bm25_retriever = BM25Retriever.from_documents(self.documents)
            self.bm25_retriever.k = 10

            logger.info("✅ Retrievers setup complete")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to setup retrievers: {str(e)}")
            return False

    def retrieve_and_rerank(self, query: str, top_k: int = 5) -> Tuple[List[Document], List[float]]:
        """Retrieve and rerank documents based on query."""
        global reranker

        if not self.vector_store or not self.bm25_retriever:
            logger.warning("⚠️ Retrievers not initialized")
            return [], []

        try:
            semantic_weight, bm25_weight = (0.7, 0.3) if len(query.split()) > 10 else (0.4, 0.6)

            chroma_retriever = self.vector_store.as_retriever(search_kwargs={'k': 15})
            ensemble_retriever = EnsembleRetriever(
                retrievers=[self.bm25_retriever, chroma_retriever],
                weights=[bm25_weight, semantic_weight]
            )

            retrieved_docs = ensemble_retriever.get_relevant_documents(query)

            if not retrieved_docs:
                logger.warning("⚠️ No documents retrieved")
                return [], []

            query_doc_pairs = [[query, doc.page_content] for doc in retrieved_docs]
            similarity_scores = reranker.predict(query_doc_pairs)

            doc_score_pairs = sorted(
                list(zip(retrieved_docs, similarity_scores)),
                key=lambda x: x[1],
                reverse=True
            )

            top_docs = [pair[0] for pair in doc_score_pairs[:top_k]]
            top_scores = [float(pair[1]) for pair in doc_score_pairs[:top_k]]

            logger.info(f"🔍 Retrieved and reranked {len(top_docs)} documents")
            return top_docs, top_scores

        except Exception as e:
            logger.error(f"❌ Error in retrieve_and_rerank: {str(e)}")
            return [], []

class DecisionEngine:
    """Decision engine using the integrated BAJAJ INSURANCE analysis prompt."""

    def _get_enhanced_prompt_template(self) -> str:
        """Returns the comprehensive, multi-domain BAJAJ INSURANCE analysis prompt."""
        return """
**BAJAJ INSURANCE CLAIM DECISION ENGINE - COMPREHENSIVE MULTI-DOMAIN ANALYSIS PROMPT**

**SYSTEM ROLE**
You are an expert insurance claim analyst specializing in Bajaj Insurance products across ALL domains including Health/Medical, Motor (Car/Bike), Travel, Home, Personal Accident, Gold, Cyber, Commercial, and specialized insurance policies. You have deep knowledge of Indian insurance regulations, IRDAI guidelines, and multi-domain claim processing. Analyze each claim with precision, fairness, and regulatory compliance regardless of policy type or data source format.

**CONTEXT (Policy Document Excerpts):**
{context}

**ANALYSIS FRAMEWORK**

**STEP 1: QUERY DECOMPOSITION**
- **Raw Query:** "{query}"
- **Format:** Extract data format from context (PDF, JSON, structured text, etc.)
- **Type:** Identify insurance type from context and query
- **Extract Key Details:** Age, Gender, Location, Claim Amount, Policy Duration, Policy Number, etc.
- **Instructions:** From the query and context, identify key details. If not present, state "Not specified".

**STEP 2: COVERAGE VERIFICATION**
- **Check:** Sum Insured, Sub-limits, Waiting Periods, Network Benefits, Specific Exclusions
- **Domain Rules Application:**
  - **Health:** Room rent caps, cashless facility eligibility, network hospital benefits
  - **Motor:** IDV (Insured Declared Value), NCB (No Claim Bonus), depreciation rates
  - **Travel:** Geographical coverage limits, trip duration limits
  - **Home:** Contents coverage, natural disaster coverage
  - **Personal Accident:** Occupation-based coverage, activity exclusions
  - **Gold:** Market value fluctuations, purity requirements
  - **Cyber:** Data breach limits, business interruption coverage
  - **Commercial:** Business type coverage, liability limits
- **Instructions:** Based on the context, verify if the core claim is covered under the specific policy type.

**STEP 3: ELIGIBILITY & CALCULATION**
- **Apply Deductions:** Deductibles, Co-payments, Depreciation, Sub-limits
- **Calculate Eligible Amount:** Claimed Amount - Non-Payable Items - Deductibles - Co-payment percentage
- **Domain-Specific Calculations:**
  - **Health:** Room rent capping, medical inflation adjustments
  - **Motor:** Depreciation on parts, betterment charges
  - **Travel:** Daily allowance limits, emergency evacuation costs
  - **Home:** Replacement cost vs actual cash value
- **Instructions:** Determine the final payable amount after applying all applicable deductions and limits.

**STEP 4: RISK ASSESSMENT**
- **Documentation Consistency:** Check for consistency in provided information across documents
- **Fraud Indicators:** Look for red flags, unusual patterns, contradictory information
- **Pattern Analysis:** Compare with typical claims for similar scenarios
- **Risk Level Classification:**
  - **Low (0-30%):** Clear-cut claim with complete documentation
  - **Medium (31-70%):** Minor ambiguities or missing non-critical information
  - **High (71-100%):** Contradictory information, suspicious patterns, or major gaps
- **Instructions:** Assess the overall risk level of the claim based on document consistency and fraud indicators.

**STEP 5: DECISION SYNTHESIS**
- **Primary Decision Options:** Accept / Reject / Partial Payment / Investigation Required / More Information Required
- **Financial Impact:** Calculate final settlement amount and percentage of claimed amount
- **Regulatory Compliance:** Ensure decision aligns with IRDAI guidelines and company policies
- **Instructions:** Formulate a final decision with clear justification based on all previous analysis steps.

**REQUIRED OUTPUT FORMAT (JSON):**
Provide your analysis *only* in the following JSON format. Do not add any text outside of this JSON structure.

{{
  "decision": "[One of: Approved, Rejected, Needs Manual Review, Insufficient Information, Partial Payment]",
  "confidence_score": [Float between 0.0 and 1.0, reflecting clarity of the policy language and decision certainty],
  "reasoning_chain": [
    "Query Decomposition: [Summary of extracted details including age, gender, location, claim amount, policy type, etc.]",
    "Coverage Verification: [Analysis of whether the claim is covered, applicable limits, waiting periods, exclusions]",
    "Eligibility Calculation: [Detailed calculation showing claimed amount, deductions, co-payments, final eligible amount]",
    "Risk Assessment: [Brief risk assessment with consistency check, fraud indicators, risk level classification]",
    "Decision Synthesis: [Final decision with justification, settlement amount, regulatory compliance notes]"
  ],
  "evidence_sources": [
    "Direct quote from policy document supporting coverage decision",
    "Reference to specific clause numbers or sections", 
    "Relevant exclusion or limitation clauses",
    "Premium calculation or benefit structure details"
  ],
  "final_answer": "[A comprehensive answer addressing the user's query with specific amounts, percentages, and clear decision rationale]"
}}

**CRITICAL INSTRUCTIONS:**
- Always extract and analyze ALL available information from the context
- Apply domain-specific rules based on the identified insurance type
- Provide specific monetary calculations where applicable
- Reference exact policy clauses and sections in evidence sources
- Maintain regulatory compliance with Indian insurance laws
- Consider both coverage and exclusions in equal detail
- Provide clear, actionable decisions that can be implemented immediately
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
                context_text += f"\n--- Document Excerpt {i+1}{score_info} | Source: {doc.metadata.get('source_file', 'Unknown')} ---\n"
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
                max_tokens=3500,  # Increased for enhanced clause mapping
                response_format={"type": "json_object"}
            )

            response_content = response.choices[0].message.content
            logger.info("✅ Received structured response from OpenAI with clause mapping")

            try:
                decision_data = json.loads(response_content)
                
                # ENHANCED: Validate, sanitize, and enhance clause mapping
                decision_data = await self._sanitize_and_enhance_structured_response(decision_data, query, context_text)
                
                logger.info(f"✅ Enhanced structured decision: {decision_data['status']} - Approved: ₹{decision_data['approvedAmount']:,.2f} | Clauses: {len(decision_data['keyPolicyReferences'])}")
                return decision_data

            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logger.error(f"❌ Failed to parse structured response: {str(e)}")
                logger.error(f"❌ Raw response content: {response_content[:500]}...") # Log first 500 chars for debugging
                
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
        """ENHANCED: Sanitize, validate, and enhance the structured response with improved clause mapping."""
        
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

        # FIXED: Sanitize breakdown components
        if "breakdown" not in decision_data or not isinstance(decision_data["breakdown"], dict):
            decision_data["breakdown"] = {"approved": [], "rejected": []}

        if "approved" not in decision_data["breakdown"]:
            decision_data["breakdown"]["approved"] = []
        if "rejected" not in decision_data["breakdown"]:
            decision_data["breakdown"]["rejected"] = []

        # FIXED: Parse and validate amounts in breakdown components
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
                        # Skip invalid component
                        continue
            decision_data["breakdown"][component_type] = [
                {k: v for k, v in comp.items() if k != "status"} 
                for comp in sanitized_components
            ]

        # FIXED: Recalculate totals from breakdown
        approved_total = round(sum(item["amount"] for item in decision_data["breakdown"]["approved"]), 2)
        rejected_total = round(sum(item["amount"] for item in decision_data["breakdown"]["rejected"]), 2)
        
        decision_data["approvedAmount"] = approved_total
        decision_data["rejectedAmount"] = rejected_total

        # ENHANCED: Apply advanced clause mapping to components
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

        # FIXED: Validate total amounts consistency
        total_calculated = approved_total + rejected_total
        if total_calculated > 0:
            # Extract claim amount from query if possible
            try:
                # Enhanced extraction logic with multiple patterns
                import re
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
                                # Add this information to policy references
                                decision_data["keyPolicyReferences"].append({
                                    "title": "Amount Calculation Notice",
                                    "note": f"Breakdown components total ₹{total_calculated:,.2f}. Original claim amount: ₹{query_amount:,.2f}. Difference may be due to applicable deductibles, co-payments, or sub-limits."
                                })
                            break
            except Exception:
                pass  # Continue without validation if extraction fails

        # Ensure confidence is within bounds
        decision_data["confidence"] = max(0.0, min(1.0, float(decision_data.get("confidence", 0.0))))

        # FIXED: Validate amounts using utility function
        is_valid, validation_message = validate_amounts(decision_data["breakdown"])
        if not is_valid:
            logger.error(f"❌ Amount validation failed: {validation_message}")
            # Add to policy references instead of failing
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

    async def get_decision(self, query: str, context_docs: List[Document], 
                          similarity_scores: List[float] = None) -> Dict[str, Any]:
        """Generate decision based on query and retrieved documents."""
        global openai_client

        if not context_docs:
            return {
                "decision": "Insufficient Information",
                "confidence_score": 0.0,
                "reasoning_chain": ["No relevant document sections found to analyze the query."],
                "evidence_sources": [],
                "timestamp": datetime.now().isoformat(),
                "final_answer": "I could not find any relevant information in the provided documents to answer your question."
            }

        try:
            context_text = ""
            for i, doc in enumerate(context_docs):
                score_info = f" (Relevance: {similarity_scores[i]:.2f})" if similarity_scores else ""
                context_text += f"\n--- Document Excerpt {i+1}{score_info} | Source: {doc.metadata.get('source_file', 'Unknown')} ---\n"
                context_text += doc.page_content + "\n"

            prompt = self._get_enhanced_prompt_template().format(
                query=query,
                context=context_text
            )

            logger.info("🤖 Calling OpenAI API for BAJAJ Insurance decision analysis...")

            response = await openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a BAJAJ Insurance expert claim analyst. Always respond with valid JSON as specified in the prompt. Apply comprehensive multi-domain insurance analysis."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.0,
                max_tokens=3000,
                response_format={"type": "json_object"}
            )

            response_content = response.choices[0].message.content
            logger.info("✅ Received response from OpenAI for BAJAJ Insurance analysis")

            try:
                decision_data = json.loads(response_content)

                required_fields = ["decision", "confidence_score", "reasoning_chain", "evidence_sources", "final_answer"]
                if not all(field in decision_data for field in required_fields):
                    raise ValueError("Missing required fields in LLM response.")

                confidence = float(decision_data["confidence_score"])
                decision_data["confidence_score"] = max(0.0, min(1.0, confidence))
                decision_data["timestamp"] = datetime.now().isoformat()

                logger.info(f"✅ BAJAJ Insurance Decision generated: {decision_data['decision']} (confidence: {decision_data['confidence_score']:.2f})")
                return decision_data

            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logger.error(f"❌ Failed to parse OpenAI response: {str(e)}")
                return {
                    "decision": "Needs Manual Review",
                    "confidence_score": 0.0,
                    "reasoning_chain": ["System encountered an error parsing the AI's response for BAJAJ Insurance analysis."],
                    "evidence_sources": [f"Error: {str(e)}"],
                    "timestamp": datetime.now().isoformat(),
                    "final_answer": "A system error occurred during BAJAJ Insurance analysis. Please try again or review manually."
                }

        except Exception as e:
            logger.error(f"❌ Error in BAJAJ Insurance decision generation: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error during BAJAJ Insurance decision generation: {str(e)}")

# PERFORMANCE OPTIMIZATION: Parallel processing function
async def process_questions_parallel(questions: List[str], rag_system: RAGSystem, decision_engine: DecisionEngine) -> List[str]:
    """Process multiple questions in parallel for better performance."""
    
    async def process_single_question(question: str) -> str:
        """Process a single question and return formatted answer."""
        logger.info(f"🔄 Processing question: '{question}'")
        
        # Retrieve relevant documents
        retrieved_docs, similarity_scores = rag_system.retrieve_and_rerank(question, top_k=7)
        
        # Generate decision using BAJAJ Insurance comprehensive analysis
        decision_result = await decision_engine.get_decision(
            query=question,
            context_docs=retrieved_docs,
            similarity_scores=similarity_scores
        )
        
        # Create comprehensive answer with decision status
        comprehensive_answer = {
            "status": decision_result.get("decision", "Unknown"),
            "confidence": decision_result.get("confidence_score", 0.0),
            "answer": decision_result.get("final_answer", "No specific answer could be formulated using BAJAJ Insurance analysis."),
            "reasoning": decision_result.get("reasoning_chain", []),
            "evidence": decision_result.get("evidence_sources", [])
        }
        
        # Format the response to show clear decision status
        status_text = f"**STATUS: {comprehensive_answer['status'].upper()}** (Confidence: {comprehensive_answer['confidence']:.2f})\n\n"
        formatted_answer = status_text + comprehensive_answer['answer']
        
        if comprehensive_answer['reasoning']:
            formatted_answer += f"\n\n**Decision Analysis:**\n" + "\n".join([f"• {reason}" for reason in comprehensive_answer['reasoning']])
        
        return formatted_answer

    # Process questions in parallel
    tasks = [process_single_question(question) for question in questions]
    results = await asyncio.gather(*tasks)

    return results

# ENHANCED: Structured parallel processing function with improved error handling and clause mapping
async def process_questions_parallel_structured(questions: List[str], rag_system: RAGSystem, decision_engine: DecisionEngine) -> List[StructuredClaimResponse]:
    """Process multiple questions in parallel with structured output and enhanced clause mapping."""
    
    async def process_single_question_structured(question: str) -> StructuredClaimResponse:
        """Process a single question and return structured response with enhanced clause mapping."""
        logger.info(f"🔄 Processing structured question with enhanced clause mapping: '{question}'")
        
        try:
            # Retrieve relevant documents
            retrieved_docs, similarity_scores = rag_system.retrieve_and_rerank(question, top_k=7)
            
            # FIXED: Check if no documents retrieved
            if not retrieved_docs:
                logger.warning(f"⚠️ No documents retrieved for question: {question}")
                return StructuredClaimResponse(
                    status="INSUFFICIENT_INFORMATION",
                    confidence=0.0,
                    approvedAmount=0.0,
                    rejectedAmount=0.0,
                    breakdown=ClaimBreakdown(approved=[], rejected=[]),
                    keyPolicyReferences=[PolicyReference(title="No Documents", note="No relevant documents found")],
                    summary="No relevant policy documents could be found to analyze this claim."
                )

            # Generate structured decision with enhanced clause mapping
            decision_result = await decision_engine.get_structured_decision(
                query=question,
                context_docs=retrieved_docs,
                similarity_scores=similarity_scores
            )

            # Add validation for decision_result structure
            if not isinstance(decision_result, dict):
                raise ValueError("Invalid decision result format")

            # ENHANCED: Safe component creation with better error handling
            try:
                approved_components = []
                for comp in decision_result.get("breakdown", {}).get("approved", []):
                    if isinstance(comp, dict) and all(key in comp for key in ["component", "amount", "reason"]):
                        approved_components.append(ClaimComponent(**comp))
                    else:
                        logger.warning(f"⚠️ Skipping invalid approved component: {comp}")

                rejected_components = []
                for comp in decision_result.get("breakdown", {}).get("rejected", []):
                    if isinstance(comp, dict) and all(key in comp for key in ["component", "amount", "reason"]):
                        rejected_components.append(ClaimComponent(**comp))
                    else:
                        logger.warning(f"⚠️ Skipping invalid rejected component: {comp}")

                policy_references = []
                for ref in decision_result.get("keyPolicyReferences", []):
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
                policy_references = [PolicyReference(title="Component Processing Error", note=f"Error processing components with enhanced clause mapping: {str(comp_error)}")]

            # Create structured response with safe field access
            structured_response = StructuredClaimResponse(
                status=decision_result.get("status", "NEEDS_MANUAL_REVIEW"),
                confidence=decision_result.get("confidence", 0.0),
                approvedAmount=decision_result.get("approvedAmount", 0.0),
                rejectedAmount=decision_result.get("rejectedAmount", 0.0),
                breakdown=ClaimBreakdown(
                    approved=approved_components,
                    rejected=rejected_components
                ),
                keyPolicyReferences=policy_references,
                summary=decision_result.get("summary", "Analysis completed with enhanced clause mapping and structured breakdown.")
            )

            logger.info(f"✅ Enhanced structured response: {len(approved_components)} approved, {len(rejected_components)} rejected, {len(policy_references)} policy references")
            return structured_response

        except Exception as e:
            logger.error(f"❌ Error in enhanced structured processing for question: {str(e)}")
            # Return error response in structured format
            return StructuredClaimResponse(
                status="NEEDS_MANUAL_REVIEW",
                confidence=0.0,
                approvedAmount=0.0,
                rejectedAmount=0.0,
                breakdown=ClaimBreakdown(approved=[], rejected=[]),
                keyPolicyReferences=[PolicyReference(title="Processing Error", note=f"Error in enhanced processing: {str(e)}")],
                summary=f"An error occurred while processing this question with enhanced clause mapping: {str(e)}"
            )

    # Process questions in parallel
    tasks = [process_single_question_structured(question) for question in questions]
    results = await asyncio.gather(*tasks)

    return results

# Global instances
decision_engine = DecisionEngine()

# --- API Endpoints ---
@app.get("/", tags=["General"])
async def root():
    """Health check endpoint - No authentication required."""
    return {
        "message": "BAJAJ Insurance Claim Analysis API is operational",
        "version": "3.1.0",
        "system": "BAJAJ Insurance Multi-Domain Analysis Engine with Enhanced Clause Mapping",
        "supported_domains": [
            "Health/Medical", "Motor (Car/Bike)", "Travel", "Home",
            "Personal Accident", "Gold", "Cyber", "Commercial"
        ],
        "authentication": "Bearer token required for /api/v1 endpoints",
        "performance_optimizations": [
            "Document embedding cache", "Persistent vector store",
            "Optimized chunk sizes", "Parallel processing", "Lightweight models"
        ],
        "fixes_implemented": [
            "Amount parsing with currency handling",
            "Total calculation validation",
            "Floating point precision handling",
            "Exception handling improvements",
            "Component validation logic"
        ],
        "enhanced_features": [
            "Advanced policy clause mapping",
            "Semantic keyword matching",
            "Component-to-clause association",
            "Fallback clause generation",
            "Synonym-based matching",
            f"{len(COMPONENT_KEYWORDS)} component types supported",
            "Multi-pattern amount extraction"
        ]
    }

# UPDATED: Main HackRx endpoint with enhanced clause mapping
@api_v1_router.post("/hackrx/run", response_model=EnhancedHackRxResponse, tags=["HackRx"])
async def hackrx_run(
    request: HackRxRequest,
    token: str = Depends(verify_bearer_token)
):
    """
    Main endpoint for HackRx competition - BAJAJ Insurance Analysis with Enhanced Clause Mapping.
    Returns structured claim analysis with component-wise breakdown and accurate policy clause references.
    """
    session_id = uuid.uuid4().hex
    temp_dir = tempfile.mkdtemp(prefix=f"bajaj_hackrx_{session_id}_")
    logger.info(f"[{session_id}] BAJAJ Insurance: Authenticated request received with enhanced clause mapping")

    try:
        # 1. Download the document
        file_url = str(request.documents)
        logger.info(f"[{session_id}] BAJAJ Insurance: Downloading document from: {file_url}")
        
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            response = await client.get(file_url)
            response.raise_for_status()
            file_content = response.content

        # PERFORMANCE OPTIMIZATION: Check if embeddings exist in cache (FIXED CALL)
        file_hash, is_cached = await get_or_create_embeddings(file_content)

        temp_pdf_path = os.path.join(temp_dir, "bajaj_policy.pdf")
        with open(temp_pdf_path, "wb") as f:
            f.write(file_content)
        logger.info(f"[{session_id}] BAJAJ Insurance document saved to {temp_pdf_path}")

        # 2. Setup session-specific RAG system
        rag_system = RAGSystem()

        # 3. Process documents and setup retrievers
        processing_result = await rag_system.load_and_process_documents([temp_pdf_path])
        
        if not processing_result or not processing_result['documents']:
            raise HTTPException(status_code=500, detail="Failed to process the BAJAJ Insurance document from the URL.")

        # PERFORMANCE OPTIMIZATION: Use persistent directory with file hash
        chroma_db_path = os.path.join(PERSISTENT_CHROMA_DIR, f"bajaj_{file_hash}")
        retriever_success = await rag_system.setup_retrievers(persist_directory=chroma_db_path)
        
        if not retriever_success:
            raise HTTPException(status_code=500, detail="Failed to set up BAJAJ Insurance document retrievers.")

        # 4. ENHANCED: Process questions in parallel with enhanced clause mapping
        logger.info(f"[{session_id}] BAJAJ Insurance: Processing {len(request.questions)} questions with enhanced clause mapping")
        structured_answers = await process_questions_parallel_structured(request.questions, rag_system, decision_engine)

        # Log enhanced clause mapping results
        total_policy_refs = sum(len(answer.keyPolicyReferences) for answer in structured_answers)
        logger.info(f"[{session_id}] BAJAJ Insurance: Enhanced analysis complete - {total_policy_refs} total policy references mapped")

        logger.info(f"[{session_id}] BAJAJ Insurance: Successfully processed all {len(request.questions)} questions with enhanced clause mapping.")

        # Store in cache only if it's a new document (FIXED CACHING LOGIC)
        if not is_cached:
            DOCUMENT_CACHE[file_hash] = {
                'processed_files': processing_result['processed_files'],
                'total_chunks': processing_result['total_chunks'],
                'timestamp': datetime.now().isoformat(),
                'enhanced_features': True
            }
            # ADDED: Memory Management
            manage_cache()

        return EnhancedHackRxResponse(answers=structured_answers)

    except httpx.RequestError as e:
        logger.error(f"[{session_id}] BAJAJ Insurance: Failed to download document: {e}")
        raise HTTPException(status_code=400, detail=f"Could not retrieve BAJAJ Insurance document from URL: {e}")
    
    except Exception as e:
        logger.error(f"[{session_id}] BAJAJ Insurance: An error occurred during enhanced /hackrx/run: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred in enhanced BAJAJ Insurance analysis: {str(e)}")

    finally:
        # 5. Cleanup temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info(f"[{session_id}] BAJAJ Insurance: Cleaned up temporary directory: {temp_dir}")

# UPDATED: Query endpoint with Bearer Token Authentication  
@api_v1_router.post("/query", response_model=QueryResponse, tags=["Query Analysis"])
async def process_query(
    request: QueryRequest,
    token: str = Depends(verify_bearer_token)
):
    """
    Process a single query against pre-loaded documents using BAJAJ Insurance analysis.
    Requires Bearer Token Authentication.
    """
    try:
        return QueryResponse(
            decision="Insufficient Information",
            confidence_score=0.0,
            reasoning_chain=["No documents currently loaded in the system. Please use /hackrx/run endpoint or upload documents first."],
            evidence_sources=[],
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"❌ Error in query processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

# Public endpoints (no authentication required)
@app.post("/reset", tags=["General"])
async def reset_system():
    """Reset the BAJAJ Insurance analysis system by clearing all persistent data."""
    try:
        # Clear persistent chroma directory
        if os.path.exists(PERSISTENT_CHROMA_DIR):
            shutil.rmtree(PERSISTENT_CHROMA_DIR)
        
        # Clear traditional chroma_db if it exists
        if os.path.exists("./chroma_db"):
            shutil.rmtree("./chroma_db")
        
        # Clear uploads
        if os.path.exists("uploads"):
            shutil.rmtree("uploads")
        
        # Clear document cache
        global DOCUMENT_CACHE
        DOCUMENT_CACHE = {}
        
        # Recreate persistent directory
        os.makedirs(PERSISTENT_CHROMA_DIR, exist_ok=True)
        
        logger.info("🔄 BAJAJ Insurance system artifacts cleared with enhanced clause mapping reset")
        
        return {
            "message": "BAJAJ Insurance system artifacts reset successfully",
            "timestamp": datetime.now().isoformat(),
            "system": "BAJAJ Insurance Multi-Domain Analysis Engine with Enhanced Clause Mapping",
            "cleared_items": ["Vector stores", "Document cache", "Upload files", "Persistent storage", "Clause mapping cache"],
            "enhanced_features_reset": True
        }

    except Exception as e:
        logger.error(f"❌ BAJAJ Insurance system reset failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to reset BAJAJ Insurance system: {str(e)}")

@app.get("/health", tags=["General"])
async def health_check():
    """Detailed health check for BAJAJ Insurance analysis system with enhanced clause mapping."""
    try:
        global embedding_model, reranker, openai_client
        
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "system": "BAJAJ Insurance Multi-Domain Analysis Engine with Enhanced Clause Mapping",
            "components": {
                "embedding_model": embedding_model is not None,
                "reranker": reranker is not None,
                "openai_client": openai_client is not None,
                "bearer_auth": EXPECTED_BEARER_TOKEN != "your-default-token-here",
                "persistent_storage": os.path.exists(PERSISTENT_CHROMA_DIR),
                "clause_mapping": len(COMPONENT_KEYWORDS) > 0
            },
            "performance_optimizations": {
                "document_cache_size": len(DOCUMENT_CACHE),
                "persistent_directory": PERSISTENT_CHROMA_DIR,
                "optimized_chunk_size": 1500,
                "optimized_overlap": 150,
                "batch_size": 32
            },
            "response_features": {
                "structured_claims": True,
                "component_breakdown": True,
                "policy_references": True,
                "financial_calculations": True,
                "confidence_scoring": True,
                "enhanced_clause_mapping": True
            },
            "fixes_implemented": {
                "amount_parsing": "Currency symbols and commas handled",
                "total_validation": "Approved + rejected totals validated",
                "precision_handling": "Floating point precision managed",
                "exception_handling": "Improved error catching and reporting",
                "component_validation": "Individual component validation added"
            },
            "enhanced_clause_mapping": {
                "component_types_supported": len(COMPONENT_KEYWORDS),
                "semantic_matching": True,
                "synonym_support": True,
                "fallback_clauses": True,
                "keyword_categories": list(COMPONENT_KEYWORDS.keys())[:10],  # First 10 for display
                "clause_types": len(CLAUSE_TYPES),
                "mapping_algorithm": "Keyword similarity + semantic scoring"
            },
            "supported_domains": [
                "Health/Medical", "Motor (Car/Bike)", "Travel", "Home",
                "Personal Accident", "Gold", "Cyber", "Commercial"
            ],
            "api_version": "3.1.0"
        }

        if not all(health_status["components"].values()):
            health_status["status"] = "degraded"

        return health_status

    except Exception as e:
        logger.error(f"❌ Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "system": "BAJAJ Insurance Multi-Domain Analysis Engine with Enhanced Clause Mapping"
        }

# NEW: Clause mapping diagnostic endpoint
@app.get("/clause-mapping-info", tags=["General"])
async def clause_mapping_info():
    """Get detailed information about the enhanced clause mapping system."""
    return {
        "system": "Enhanced Policy Clause Mapping System",
        "version": "3.1.0",
        "component_keywords": {
            "total_types": len(COMPONENT_KEYWORDS),
            "sample_mappings": {k: v[:3] for k, v in list(COMPONENT_KEYWORDS.items())[:5]},  # Show first 3 keywords for first 5 types
            "all_component_types": list(COMPONENT_KEYWORDS.keys())
        },
        "clause_types": {
            "total_types": len(CLAUSE_TYPES),
            "semantic_categories": list(CLAUSE_TYPES.keys()),
            "sample_keywords": {k: v[:3] for k, v in CLAUSE_TYPES.items()}
        },
        "mapping_features": [
            "Keyword similarity scoring",
            "Semantic clause categorization", 
            "Component-specific synonym matching",
            "Fallback clause generation",
            "Duplicate clause prevention",
            "Context-aware clause extraction",
            "Multi-pattern amount extraction"
        ],
        "supported_languages": ["English", "Mixed English-Hindi terms"],
        "accuracy_improvements": [
            "Reduced incorrect clause associations",
            "Better handling of synonyms and paraphrases",
            "Comprehensive fallback mechanisms",
            "Enhanced trust in reasoning chains"
        ]
    }

# Include the router in the main app
app.include_router(api_v1_router)

# GCP Cloud Run compatible entry point
if __name__ == "__main__":
    import uvicorn
    # Get port from environment variable for Cloud Run compatibility
    port = int(os.environ.get("PORT", 8000))
    
    uvicorn.run(
        "main:app", # Fixed module reference
        host="0.0.0.0",
        port=port,
        timeout_keep_alive=300,
        log_level="info"
    )
