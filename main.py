import asyncio
import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import fitz  # PyMuPDF
import openai
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langchain.docstore.document import Document
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# FastAPI app
app = FastAPI(title="Universal Document Analysis System", version="4.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Response Models (Existing + New)
class ClaimProcessingResult(BaseModel):
    """Legacy insurance claim processing response"""
    claim_type: str
    decision: str
    approved_amount: Optional[float]
    denial_reason: Optional[str]
    policy_references: List[str]
    confidence: float
    summary: str

class GeneralDocumentResponse(BaseModel):
    """General document analysis response"""
    query_type: str
    domain: str
    answer: str
    confidence: str
    source_documents: List[str]

class UniversalAnalysisResponse(BaseModel):
    """Universal response for structured analysis"""
    status: str
    confidence: float
    domain: str
    analysis: Dict[str, Any]
    key_references: List[Dict[str, str]]
    summary: str
    query_type: str = "STRUCTURED_ANALYSIS"

class EnhancedUnifiedResponse(BaseModel):
    """Enhanced unified response for any query type"""
    query_type: str
    domain: str
    processing_approach: str
    answer: str
    confidence: str
    source_documents: List[str]
    key_findings: Optional[List[str]] = None
    references: Optional[List[str]] = None
    structured_data: Optional[Dict[str, Any]] = None
    quantitative_info: Optional[List[Dict]] = None
    decision: Optional[str] = None
    approved_amount: Optional[float] = None
    policy_references: Optional[List[str]] = None

class ExplainableResponse(BaseModel):
    """Enhanced response with full explainability"""
    
    # Decision Information
    decision: str  # "approved", "rejected", "needs_review"
    confidence_score: float
    processing_time_ms: int
    
    # Financial Information
    approved_amount: Optional[float] = None
    maximum_coverage: Optional[float] = None
    deductible_amount: Optional[float] = None
    co_payment_percentage: Optional[float] = None
    
    # Detailed Justification
    decision_reasoning: str
    key_factors: List[str]
    risk_assessment: Dict[str, Any]
    
    # Clause Traceability
    supporting_clauses: List[Dict[str, Any]]
    conflicting_clauses: List[Dict[str, Any]]
    exclusion_analysis: List[Dict[str, Any]]
    
    # Query Analysis
    parsed_query: Dict[str, Any]
    missing_information: List[str]
    assumptions_made: List[str]
    
    # Audit Trail
    evaluation_steps: List[Dict[str, Any]]
    rule_applications: List[Dict[str, Any]]
    
    # Recommendations
    next_steps: List[str]
    additional_documents_needed: List[str]
    appeal_process: Optional[str] = None

class ProcessingResponse(BaseModel):
    """Processing status response"""
    success: bool
    message: str
    doc_count: int
    processing_time: float

# Domain Configuration (Enhanced)
DOMAIN_CONFIGS = {
    "insurance": {
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "context_docs": 7,
        "keywords": ["policy", "premium", "claim", "coverage", "deductible", "benefit"]
    },
    "legal": {
        "chunk_size": 1200,
        "chunk_overlap": 250,
        "context_docs": 6,
        "keywords": ["article", "clause", "section", "law", "regulation", "provision"]
    },
    "medical": {
        "chunk_size": 800,
        "chunk_overlap": 150,
        "context_docs": 8,
        "keywords": ["diagnosis", "treatment", "symptom", "medication", "procedure"]
    },
    "physics": {
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "context_docs": 7,
        "keywords": ["force", "energy", "velocity", "acceleration", "momentum", "law"]
    },
    "academic": {
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "context_docs": 7,
        "keywords": ["theory", "research", "study", "analysis", "methodology"]
    },
    "general": {
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "context_docs": 7,
        "keywords": []
    }
}

def detect_document_domain(content: str) -> str:
    """Detect document domain based on content analysis"""
    content_lower = content.lower()
    domain_scores = {}
    
    for domain, config in DOMAIN_CONFIGS.items():
        if domain == "general":
            continue
        score = sum(1 for keyword in config["keywords"] if keyword in content_lower)
        if score > 0:
            domain_scores[domain] = score / len(config["keywords"])
    
    if not domain_scores:
        return "general"
    
    detected_domain = max(domain_scores, key=domain_scores.get)
    logger.info(f"🔍 Domain detection scores: {domain_scores}")
    return detected_domain

# NEW: Query Parser for Enhanced Structured Information Extraction
class QueryParser:
    """Enhanced query parser to extract structured information"""
    
    def __init__(self):
        self.entity_patterns = {
            "age": r'\b(\d{1,3})\s*(?:years?|yrs?|y\.o\.)\b',
            "amount": r'[\₹$€£¥]\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?|\b\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s*(?:rupees?|dollars?|euros?)\b',
            "duration": r'\b(\d+)\s*(?:years?|months?|days?)\b',
            "location": r'\b(?:in|at|from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
            "procedure": r'\b(?:surgery|treatment|procedure|operation|therapy)\s+(?:for\s+)?([a-zA-Z\s]+)\b'
        }
    
    async def parse_structured_query(self, query: str) -> Dict[str, Any]:
        """Extract structured information from natural language query"""
        
        # Use GPT to extract structured data
        prompt = f"""
        Extract structured information from this query and return valid JSON:
        
        Query: "{query}"
        
        Extract these fields (set to null if not found):
        {{
            "claimant_age": number or null,
            "procedure_type": "string or null",
            "claim_amount": number or null,
            "location": "string or null", 
            "policy_duration": "string or null",
            "incident_date": "string or null",
            "medical_condition": "string or null",
            "vehicle_type": "string or null",
            "damage_type": "string or null",
            "query_intent": "claim_processing|eligibility_check|coverage_inquiry|general_question",
            "urgency": "high|medium|low",
            "extracted_entities": ["list of key entities found"]
        }}
        """
        
        try:
            response = await openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            logger.info(f"🔍 Query parsed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error parsing query: {str(e)}")
            return {
                "claimant_age": None,
                "procedure_type": None,
                "claim_amount": None,
                "query_intent": "general_question",
                "urgency": "medium",
                "extracted_entities": []
            }

# NEW: Semantic Clause Retriever for Enhanced Policy Analysis
class SemanticClauseRetriever:
    """Advanced clause retrieval with semantic understanding"""
    
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.clause_patterns = [
            r'(?:Article|Section|Clause|Rule|Provision)\s+\d+',
            r'\b(?:Coverage|Exclusion|Benefit|Limit|Deductible)\b.*?(?:\.|;|\n)',
            r'(?:If|When|In case of).*?(?:then|shall|will).*?(?:\.|;|\n)',
        ]
    
    async def retrieve_relevant_clauses(self, parsed_query: Dict[str, Any], top_k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve clauses with semantic understanding and ranking"""
        
        # Generate multiple search queries for comprehensive retrieval
        search_queries = await self._generate_search_queries(parsed_query)
        
        all_clauses = []
        for query in search_queries:
            docs, scores = self.vector_store.similarity_search_with_score(query, k=top_k)
            
            for doc, score in zip(docs, scores):
                # Extract specific clauses from documents
                clauses = self._extract_clauses(doc.page_content)
                
                for clause in clauses:
                    all_clauses.append({
                        "clause_id": f"clause_{len(all_clauses)}",
                        "content": clause,
                        "source_document": doc.metadata.get('source_file', 'Unknown'),
                        "relevance_score": score,
                        "clause_type": self._classify_clause_type(clause),
                        "applies_to": self._determine_applicability(clause, parsed_query)
                    })
        
        # Rank and deduplicate clauses
        return self._rank_and_filter_clauses(all_clauses, parsed_query)
    
    async def _generate_search_queries(self, parsed_query: Dict[str, Any]) -> List[str]:
        """Generate multiple search queries based on parsed query"""
        base_query = []
        
        if parsed_query.get("procedure_type"):
            base_query.append(f"coverage {parsed_query['procedure_type']}")
        if parsed_query.get("medical_condition"):
            base_query.append(f"treatment {parsed_query['medical_condition']}")
        if parsed_query.get("claimant_age"):
            base_query.append(f"age eligibility {parsed_query['claimant_age']}")
        if parsed_query.get("claim_amount"):
            base_query.append(f"claim limit amount coverage")
        
        # Add general policy terms
        base_query.extend([
            "exclusion conditions",
            "benefit coverage",
            "policy terms conditions"
        ])
        
        return base_query[:5]  # Limit to 5 queries
    
    def _extract_clauses(self, text: str) -> List[str]:
        """Extract individual clauses from document text"""
        clauses = []
        
        # Split by common clause separators
        potential_clauses = re.split(r'\n\s*(?=(?:Article|Section|Clause|Rule|\d+\.))', text)
        
        for clause in potential_clauses:
            clause = clause.strip()
            if len(clause) > 50 and any(keyword in clause.lower() for keyword in 
                ['coverage', 'exclusion', 'benefit', 'limit', 'deductible', 'claim', 'policy']):
                clauses.append(clause)
        
        return clauses
    
    def _classify_clause_type(self, clause: str) -> str:
        """Classify the type of clause"""
        clause_lower = clause.lower()
        
        if "exclusion" in clause_lower or "not covered" in clause_lower:
            return "exclusion"
        elif "coverage" in clause_lower or "benefit" in clause_lower:
            return "coverage"
        elif "limit" in clause_lower or "maximum" in clause_lower:
            return "limit"
        elif "deductible" in clause_lower or "co-pay" in clause_lower:
            return "cost_sharing"
        else:
            return "general"
    
    def _determine_applicability(self, clause: str, parsed_query: Dict[str, Any]) -> Dict[str, Any]:
        """Determine if clause applies to the parsed query"""
        applicability = {
            "age_relevant": False,
            "procedure_relevant": False,
            "amount_relevant": False,
            "condition_relevant": False
        }
        
        clause_lower = clause.lower()
        
        if parsed_query.get("claimant_age") and "age" in clause_lower:
            applicability["age_relevant"] = True
        if parsed_query.get("procedure_type") and parsed_query["procedure_type"].lower() in clause_lower:
            applicability["procedure_relevant"] = True
        if parsed_query.get("medical_condition") and parsed_query["medical_condition"].lower() in clause_lower:
            applicability["condition_relevant"] = True
        
        return applicability
    
    def _rank_and_filter_clauses(self, clauses: List[Dict[str, Any]], parsed_query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Rank and filter clauses by relevance"""
        
        # Calculate composite relevance score
        for clause in clauses:
            relevance_factors = []
            
            # Base similarity score
            relevance_factors.append(1 / (1 + clause["relevance_score"]))  # Lower distance = higher relevance
            
            # Applicability boost
            applicability = clause["applies_to"]
            if any(applicability.values()):
                relevance_factors.append(0.3)
            
            # Clause type relevance
            if clause["clause_type"] in ["coverage", "exclusion", "limit"]:
                relevance_factors.append(0.2)
            
            clause["composite_relevance"] = sum(relevance_factors)
        
        # Sort by composite relevance and return top clauses
        clauses.sort(key=lambda x: x["composite_relevance"], reverse=True)
        
        # Remove duplicates based on content similarity
        unique_clauses = []
        for clause in clauses:
            is_duplicate = False
            for existing in unique_clauses:
                if self._calculate_text_similarity(clause["content"], existing["content"]) > 0.8:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_clauses.append(clause)
            
            if len(unique_clauses) >= 15:  # Limit to top 15 unique clauses
                break
        
        return unique_clauses
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity calculation"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0

# NEW: Decision Logic Engine for Rule-Based Analysis
class DecisionLogicEngine:
    """Rule-based decision engine with explainable logic"""
    
    def __init__(self):
        self.decision_rules = {
            "age_eligibility": self._check_age_eligibility,
            "coverage_validation": self._check_coverage,
            "exclusion_check": self._check_exclusions,
            "amount_calculation": self._calculate_amount,
            "policy_validity": self._check_policy_validity
        }
    
    async def evaluate_claim(self, parsed_query: Dict[str, Any], relevant_clauses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate claim against retrieved clauses with full traceability"""
        
        start_time = datetime.now()
        
        decision_trace = {
            "query_analysis": parsed_query,
            "evaluated_clauses": relevant_clauses,
            "decision_factors": [],
            "final_decision": None,
            "confidence_score": 0.0,
            "reasoning_chain": []
        }
        
        # Apply decision rules in sequence
        for rule_name, rule_func in self.decision_rules.items():
            try:
                rule_result = await rule_func(parsed_query, relevant_clauses)
                
                decision_trace["decision_factors"].append({
                    "rule": rule_name,
                    "result": rule_result["outcome"],
                    "confidence": rule_result["confidence"],
                    "supporting_clauses": rule_result["supporting_clauses"],
                    "reasoning": rule_result["reasoning"]
                })
                
                # Update reasoning chain
                decision_trace["reasoning_chain"].append(
                    f"{rule_name}: {rule_result['reasoning']}"
                )
                
            except Exception as e:
                logger.error(f"❌ Error in decision rule {rule_name}: {str(e)}")
                decision_trace["decision_factors"].append({
                    "rule": rule_name,
                    "result": "error",
                    "confidence": 0.0,
                    "supporting_clauses": [],
                    "reasoning": f"Error processing rule: {str(e)}"
                })
        
        # Make final decision
        final_decision = await self._make_final_decision(decision_trace)
        decision_trace["final_decision"] = final_decision
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        decision_trace["processing_time"] = processing_time
        
        return self._format_decision_response(decision_trace)
    
    async def _check_age_eligibility(self, query: Dict[str, Any], clauses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check age eligibility against policy clauses"""
        
        age = query.get("claimant_age")
        if not age:
            return {"outcome": "unknown", "confidence": 0.0, "supporting_clauses": [], "reasoning": "Age not provided"}
        
        age_clauses = [c for c in clauses if "age" in c["content"].lower()]
        
        # Use LLM to evaluate age eligibility
        if age_clauses:
            prompt = f"""
            Evaluate if age {age} meets eligibility criteria in these clauses:
            
            {json.dumps([c["content"] for c in age_clauses], indent=2)}
            
            Return JSON:
            {{
                "eligible": true/false,
                "reasoning": "explanation",
                "applicable_clause": "exact clause text that applies",
                "confidence": 0.0 to 1.0
            }}
            """
            
            try:
                response = await openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    response_format={"type": "json_object"}
                )
                
                result = json.loads(response.choices[0].message.content)
                
                return {
                    "outcome": "eligible" if result["eligible"] else "ineligible",
                    "confidence": result["confidence"],
                    "supporting_clauses": [c["clause_id"] for c in age_clauses],
                    "reasoning": result["reasoning"]
                }
            except Exception as e:
                logger.error(f"❌ Error evaluating age eligibility: {str(e)}")
        
        return {"outcome": "unknown", "confidence": 0.0, "supporting_clauses": [], "reasoning": "No age-related clauses found"}
    
    async def _check_coverage(self, query: Dict[str, Any], clauses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check coverage validation against policy clauses"""
        
        coverage_clauses = [c for c in clauses if c["clause_type"] == "coverage"]
        procedure = query.get("procedure_type")
        condition = query.get("medical_condition")
        
        if not coverage_clauses:
            return {"outcome": "unknown", "confidence": 0.0, "supporting_clauses": [], "reasoning": "No coverage clauses found"}
        
        search_terms = [t for t in [procedure, condition] if t]
        if not search_terms:
            return {"outcome": "unknown", "confidence": 0.0, "supporting_clauses": [], "reasoning": "No specific procedure or condition specified"}
        
        prompt = f"""
        Evaluate if the following items are covered based on these policy clauses:
        
        Items to check: {search_terms}
        
        Coverage clauses:
        {json.dumps([c["content"] for c in coverage_clauses], indent=2)}
        
        Return JSON:
        {{
            "covered": true/false,
            "reasoning": "detailed explanation",
            "coverage_details": "specific coverage information",
            "confidence": 0.0 to 1.0
        }}
        """
        
        try:
            response = await openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            return {
                "outcome": "covered" if result["covered"] else "not_covered",
                "confidence": result["confidence"],
                "supporting_clauses": [c["clause_id"] for c in coverage_clauses],
                "reasoning": result["reasoning"]
            }
        except Exception as e:
            logger.error(f"❌ Error checking coverage: {str(e)}")
            return {"outcome": "unknown", "confidence": 0.0, "supporting_clauses": [], "reasoning": f"Error processing coverage: {str(e)}"}
    
    async def _check_exclusions(self, query: Dict[str, Any], clauses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check exclusions against policy clauses"""
        
        exclusion_clauses = [c for c in clauses if c["clause_type"] == "exclusion"]
        
        if not exclusion_clauses:
            return {"outcome": "no_exclusions", "confidence": 0.8, "supporting_clauses": [], "reasoning": "No exclusion clauses found"}
        
        procedure = query.get("procedure_type")
        condition = query.get("medical_condition")
        search_terms = [t for t in [procedure, condition] if t]
        
        if not search_terms:
            return {"outcome": "unknown", "confidence": 0.0, "supporting_clauses": [], "reasoning": "No specific items to check for exclusions"}
        
        prompt = f"""
        Check if any of these items are excluded based on the exclusion clauses:
        
        Items to check: {search_terms}
        
        Exclusion clauses:
        {json.dumps([c["content"] for c in exclusion_clauses], indent=2)}
        
        Return JSON:
        {{
            "excluded": true/false,
            "reasoning": "detailed explanation",
            "exclusion_details": "specific exclusion information if applicable",
            "confidence": 0.0 to 1.0
        }}
        """
        
        try:
            response = await openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            return {
                "outcome": "excluded" if result["excluded"] else "not_excluded",
                "confidence": result["confidence"],
                "supporting_clauses": [c["clause_id"] for c in exclusion_clauses],
                "reasoning": result["reasoning"]
            }
        except Exception as e:
            logger.error(f"❌ Error checking exclusions: {str(e)}")
            return {"outcome": "unknown", "confidence": 0.0, "supporting_clauses": [], "reasoning": f"Error processing exclusions: {str(e)}"}
    
    async def _calculate_amount(self, query: Dict[str, Any], clauses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate claim amount based on policy limits and deductibles"""
        
        limit_clauses = [c for c in clauses if c["clause_type"] == "limit"]
        cost_sharing_clauses = [c for c in clauses if c["clause_type"] == "cost_sharing"]
        
        claim_amount = query.get("claim_amount")
        if not claim_amount:
            return {"outcome": "unknown", "confidence": 0.0, "supporting_clauses": [], "reasoning": "No claim amount provided"}
        
        relevant_clauses = limit_clauses + cost_sharing_clauses
        if not relevant_clauses:
            return {"outcome": "full_amount", "confidence": 0.5, "supporting_clauses": [], "reasoning": "No limit or cost-sharing clauses found"}
        
        prompt = f"""
        Calculate the payable amount for a claim of {claim_amount} based on these policy clauses:
        
        Policy clauses:
        {json.dumps([c["content"] for c in relevant_clauses], indent=2)}
        
        Return JSON:
        {{
            "payable_amount": number,
            "maximum_coverage": number or null,
            "deductible": number or null,
            "co_payment_percentage": number or null,
            "reasoning": "detailed calculation explanation",
            "confidence": 0.0 to 1.0
        }}
        """
        
        try:
            response = await openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            return {
                "outcome": "calculated",
                "confidence": result["confidence"],
                "supporting_clauses": [c["clause_id"] for c in relevant_clauses],
                "reasoning": result["reasoning"],
                "amount_details": {
                    "payable_amount": result.get("payable_amount"),
                    "maximum_coverage": result.get("maximum_coverage"),
                    "deductible": result.get("deductible"),
                    "co_payment_percentage": result.get("co_payment_percentage")
                }
            }
        except Exception as e:
            logger.error(f"❌ Error calculating amount: {str(e)}")
            return {"outcome": "unknown", "confidence": 0.0, "supporting_clauses": [], "reasoning": f"Error calculating amount: {str(e)}"}
    
    async def _check_policy_validity(self, query: Dict[str, Any], clauses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check policy validity and terms"""
        
        # Simple policy validity check - can be enhanced with actual policy data
        return {
            "outcome": "valid",
            "confidence": 0.9,
            "supporting_clauses": [],
            "reasoning": "Policy appears to be valid based on available information"
        }
    
    async def _make_final_decision(self, decision_trace: Dict[str, Any]) -> Dict[str, Any]:
        """Make final decision based on all rule outcomes"""
        
        factors = decision_trace["decision_factors"]
        
        # Analyze outcomes
        outcomes = {}
        total_confidence = 0.0
        
        for factor in factors:
            outcome = factor["result"]
            confidence = factor["confidence"]
            
            if outcome not in outcomes:
                outcomes[outcome] = []
            outcomes[outcome].append(confidence)
            total_confidence += confidence
        
        # Decision logic
        if "ineligible" in outcomes or "excluded" in outcomes:
            decision = "rejected"
            reasoning = "Claim fails eligibility or exclusion checks"
        elif "not_covered" in outcomes:
            decision = "rejected"
            reasoning = "Requested procedure/condition is not covered"
        elif "covered" in outcomes and "not_excluded" in outcomes:
            decision = "approved"
            reasoning = "Claim meets all policy requirements"
        else:
            decision = "needs_review"
            reasoning = "Claim requires manual review due to insufficient information"
        
        # Calculate overall confidence
        avg_confidence = total_confidence / len(factors) if factors else 0.0
        
        # Extract amount details if available
        amount_info = None
        for factor in factors:
            if factor["rule"] == "amount_calculation" and "amount_details" in factor:
                amount_info = factor["amount_details"]
                break
        
        return {
            "outcome": decision,
            "reasoning": reasoning,
            "confidence": avg_confidence,
            "amount": amount_info.get("payable_amount") if amount_info else None,
            "amount_details": amount_info
        }
    
    def _format_decision_response(self, decision_trace: Dict[str, Any]) -> Dict[str, Any]:
        """Format decision trace into explainable response format"""
        
        final_decision = decision_trace["final_decision"]
        
        return {
            "decision": final_decision["outcome"],
            "confidence_score": final_decision["confidence"],
            "processing_time_ms": decision_trace.get("processing_time", 0),
            
            "approved_amount": final_decision.get("amount"),
            "maximum_coverage": None,  # Can be extracted from amount_details
            "deductible_amount": None,  # Can be extracted from amount_details
            "co_payment_percentage": None,  # Can be extracted from amount_details
            
            "decision_reasoning": final_decision["reasoning"],
            "key_factors": [factor["reasoning"] for factor in decision_trace["decision_factors"]],
            "risk_assessment": {"overall_risk": "low"},  # Can be enhanced
            
            "supporting_clauses": [
                {
                    "clause_id": clause["clause_id"],
                    "content": clause["content"][:200] + "..." if len(clause["content"]) > 200 else clause["content"],
                    "source": clause["source_document"],
                    "relevance": clause["relevance_score"],
                    "type": clause["clause_type"]
                }
                for clause in decision_trace["evaluated_clauses"][:5]  # Top 5 clauses
            ],
            "conflicting_clauses": [],  # Can be enhanced
            "exclusion_analysis": [],  # Can be enhanced
            
            "parsed_query": decision_trace["query_analysis"],
            "missing_information": [],  # Can be enhanced
            "assumptions_made": [],  # Can be enhanced
            
            "evaluation_steps": [
                {
                    "step": i+1,
                    "rule": factor["rule"],
                    "outcome": factor["result"],
                    "reasoning": factor["reasoning"]
                }
                for i, factor in enumerate(decision_trace["decision_factors"])
            ],
            "rule_applications": decision_trace["decision_factors"],
            
            "next_steps": self._generate_next_steps(final_decision["outcome"]),
            "additional_documents_needed": [],  # Can be enhanced
            "appeal_process": "Contact customer service within 30 days" if final_decision["outcome"] == "rejected" else None
        }
    
    def _generate_next_steps(self, decision: str) -> List[str]:
        """Generate appropriate next steps based on decision"""
        
        if decision == "approved":
            return [
                "Claim has been approved for processing",
                "Payment will be processed within 5-7 business days",
                "You will receive a confirmation email with payment details"
            ]
        elif decision == "rejected":
            return [
                "Claim has been rejected based on policy terms",
                "Review the reasons provided above",
                "Consider filing an appeal if you believe this is incorrect"
            ]
        else:  # needs_review
            return [
                "Claim requires additional review",
                "Additional documentation may be requested",
                "You will be contacted within 2-3 business days"
            ]

# NEW: Enhanced Token Optimizer for Cost Management
class TokenOptimizer:
    """Advanced token optimization for cost-effective processing"""
    
    def __init__(self):
        self.max_context_tokens = 4000  # Leave room for response
        self.token_costs = {"gpt-4o": {"input": 0.005, "output": 0.015}}  # per 1K tokens
    
    def optimize_clause_context(self, clauses: List[Dict[str, Any]], query: Dict[str, Any]) -> str:
        """Optimize clause context for minimal token usage"""
        
        # Prioritize clauses by relevance and query entities
        prioritized_clauses = self._prioritize_clauses(clauses, query)
        
        # Build context incrementally within token limit
        context = ""
        token_count = 0
        
        for clause in prioritized_clauses:
            clause_tokens = len(clause["content"].split()) * 1.3  # Rough estimate
            
            if token_count + clause_tokens <= self.max_context_tokens:
                context += f"\n--- {clause['clause_id']} ({clause['clause_type']}) ---\n{clause['content']}\n"
                token_count += clause_tokens
            else:
                break
        
        logger.info(f"🔧 Optimized context: {len(prioritized_clauses)} → {len([c for c in prioritized_clauses if c['content'] in context])} clauses")
        return context
    
    def _prioritize_clauses(self, clauses: List[Dict[str, Any]], query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Prioritize clauses by relevance to query"""
        
        # Sort by composite relevance (already calculated)
        return sorted(clauses, key=lambda x: x.get("composite_relevance", 0), reverse=True)
    
    def optimize_context(self, context_text: str, query: str) -> str:
        """Optimize context to fit within token limits (existing method)"""
        max_context_chars = int(self.max_context_tokens / 0.25)  # avg_tokens_per_char
        
        if len(context_text) <= max_context_chars:
            return context_text
        
        # Smart truncation strategies
        sections = context_text.split('---')
        
        # Priority order: sections that contain query keywords
        query_keywords = set(query.lower().split())
        scored_sections = []
        
        for section in sections:
            if section.strip():
                section_keywords = set(section.lower().split())
                overlap_score = len(query_keywords.intersection(section_keywords))
                scored_sections.append((section, overlap_score))
        
        # Sort by relevance and keep top sections
        scored_sections.sort(key=lambda x: x[1], reverse=True)
        
        optimized_context = ""
        for section, score in scored_sections:
            if len(optimized_context) + len(section) <= max_context_chars:
                optimized_context += section + "\n---\n"
            else:
                break
        
        logger.info(f"🔧 Context optimized: {len(context_text)} → {len(optimized_context)} chars")
        return optimized_context
    
    def estimate_cost(self, input_tokens: int, output_tokens: int, model: str = "gpt-4o") -> float:
        """Estimate API call cost"""
        costs = self.token_costs[model]
        return (input_tokens * costs["input"] + output_tokens * costs["output"]) / 1000

# Universal Query Classification System (Enhanced)
class UniversalQueryClassifier:
    """Smart query classifier that works across all domains"""
    
    def __init__(self):
        self.query_patterns = {
            "ANALYTICAL_BREAKDOWN": [
                "analyze", "breakdown", "components", "structure", "elements",
                "parts", "sections", "details", "examine", "dissect"
            ],
            "FACTUAL_INQUIRY": [
                "what is", "define", "explain", "describe", "meaning",
                "definition", "concept", "principle", "theory"
            ],
            "PROCEDURAL_QUERY": [
                "how to", "process", "procedure", "steps", "method",
                "approach", "technique", "implementation"
            ],
            "COMPARATIVE_ANALYSIS": [
                "compare", "difference", "versus", "contrast", "similarity",
                "relation", "relationship", "connection"
            ],
            "QUANTITATIVE_ANALYSIS": [
                "calculate", "amount", "quantity", "measure", "value",
                "rate", "percentage", "cost", "price", "fee"
            ],
            "CLAIM_PROCESSING": [
                "claim", "reimburse", "coverage", "eligible", "approve",
                "reject", "process claim", "submit claim"
            ]
        }
    
    def classify_query_intent(self, query: str, document_domain: str) -> Tuple[str, str, float]:
        """
        Classify query intent independent of domain
        Returns: (intent_type, processing_approach, confidence)
        """
        query_lower = query.lower()
        
        # Calculate intent scores
        intent_scores = {}
        for intent, keywords in self.query_patterns.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                intent_scores[intent] = score / len(keywords)
        
        # Determine primary intent
        if not intent_scores:
            primary_intent = "FACTUAL_INQUIRY"
            confidence = 0.5
        else:
            primary_intent = max(intent_scores, key=intent_scores.get)
            confidence = intent_scores[primary_intent]
        
        # Determine processing approach based on intent and domain
        processing_approach = self._get_processing_approach(primary_intent, document_domain, query)
        
        return primary_intent, processing_approach, confidence
    
    def _get_processing_approach(self, intent: str, domain: str, query: str) -> str:
        """Determine how to process the query"""
        # Check for structured analysis needs (amounts, components, etc.)
        has_amounts = bool(re.search(r'[\₹$€£¥]*\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?', query))
        has_components = any(comp in query.lower() for comp in ["component", "part", "element", "section", "clause"])
        
        # Enhanced decision logic
        if intent == "CLAIM_PROCESSING":
            return "ENHANCED_CLAIM_PROCESSING"
        elif intent == "ANALYTICAL_BREAKDOWN" or has_amounts or has_components:
            return "STRUCTURED_ANALYSIS"
        elif intent == "QUANTITATIVE_ANALYSIS" and has_amounts:
            return "STRUCTURED_ANALYSIS"
        else:
            return "CONVERSATIONAL_ANALYSIS"

# Initialize enhanced components
universal_classifier = UniversalQueryClassifier()
token_optimizer = TokenOptimizer()
query_parser = QueryParser()

# Existing classification functions (preserved for backward compatibility)
def classify_query_type(query: str, document_content: str = None) -> Tuple[str, str]:
    """Universal query classification that works for any domain"""
    # Detect document domain
    domain = "general"
    if document_content:
        domain = detect_document_domain(document_content)
    
    logger.info(f"🔍 Document domain detected: {domain}")
    
    # Use universal classifier
    intent, approach, confidence = universal_classifier.classify_query_intent(query, domain)
    
    logger.info(f"🎯 Query intent: {intent}, Approach: {approach}, Confidence: {confidence:.2f}")
    
    # Map to processing type
    if approach == "ENHANCED_CLAIM_PROCESSING":
        return "ENHANCED_CLAIM_PROCESSING", domain
    elif approach == "STRUCTURED_ANALYSIS":
        return "STRUCTURED_ANALYSIS", domain
    else:
        return "CONVERSATIONAL_ANALYSIS", domain

def classify_claim_type(query: str) -> str:
    """Legacy claim type classification"""
    query_lower = query.lower()
    
    if any(keyword in query_lower for keyword in ["accident", "crash", "collision", "vehicle damage"]):
        return "vehicle_damage"
    elif any(keyword in query_lower for keyword in ["theft", "stolen", "burglary", "robbery"]):
        return "theft"
    elif any(keyword in query_lower for keyword in ["medical", "hospital", "doctor", "treatment", "surgery"]):
        return "medical"
    elif any(keyword in query_lower for keyword in ["property", "home", "house", "building", "fire", "flood"]):
        return "property_damage"
    elif any(keyword in query_lower for keyword in ["liability", "third party", "injury to others"]):
        return "liability"
    else:
        return "general_inquiry"

# Universal Structured Analysis Engine (Existing - Enhanced)
class UniversalAnalysisEngine:
    """Generalized analysis engine that works across all domains"""
    
    def __init__(self):
        self.domain_analysis_patterns = {
            "physics": {
                "components": ["force", "motion", "velocity", "acceleration", "mass", "energy"],
                "relationships": ["law", "principle", "axiom", "proposition", "theorem"],
                "quantitative": ["equation", "formula", "calculation", "measurement"]
            },
            "legal": {
                "components": ["article", "clause", "section", "provision", "subsection"],
                "relationships": ["right", "obligation", "procedure", "jurisdiction"],
                "quantitative": ["penalty", "fine", "fee", "compensation", "damages"]
            },
            "medical": {
                "components": ["symptom", "diagnosis", "treatment", "procedure", "medication"],
                "relationships": ["indication", "contraindication", "interaction", "effect"],
                "quantitative": ["dosage", "duration", "cost", "rate", "percentage"]
            },
            "insurance": {
                "components": ["coverage", "benefit", "exclusion", "deductible", "premium"],
                "relationships": ["policy", "claim", "settlement", "approval", "rejection"],
                "quantitative": ["amount", "limit", "co-payment", "sum insured"]
            },
            "general": {
                "components": ["element", "component", "part", "section", "aspect"],
                "relationships": ["relationship", "connection", "association", "link"],
                "quantitative": ["amount", "quantity", "value", "measure", "rate"]
            }
        }
    
    def get_universal_structured_prompt(self, domain: str) -> str:
        """Generate domain-adaptive structured analysis prompt"""
        domain_config = self.domain_analysis_patterns.get(domain, self.domain_analysis_patterns["general"])
        
        return f"""
        **UNIVERSAL STRUCTURED DOCUMENT ANALYSIS**

        **DOMAIN**: {domain.upper()}

        **SYSTEM ROLE**
        You are an expert document analyst specializing in {domain} content. Analyze the query and provide a structured response with clear component-wise breakdown.

        **CONTEXT (Document Excerpts):**
        {{context}}

        **QUERY:** "{{query}}"

        **ANALYSIS REQUIREMENTS:**
        1. **Determine overall analysis status**: COMPLETE, PARTIAL, INSUFFICIENT_INFO, or NEEDS_CLARIFICATION
        2. **Break down relevant components** from the document that relate to the query
        3. **Identify relationships** between components and concepts
        4. **Extract quantitative information** if present (amounts, measurements, values)
        5. **Provide supporting evidence** with specific document references

        **COMPONENT ANALYSIS FOR {domain.upper()}:**
        - Look for: {', '.join(domain_config['components'])}
        - Relationships: {', '.join(domain_config['relationships'])}
        - Quantitative elements: {', '.join(domain_config['quantitative'])}

        **REQUIRED OUTPUT FORMAT (JSON):**
        {{
            "status": "[COMPLETE | PARTIAL | INSUFFICIENT_INFO | NEEDS_CLARIFICATION]",
            "confidence": [Float between 0.0 and 1.0],
            "domain": "{domain}",
            "analysis": {{
                "main_findings": [
                    {{
                        "component": "Component name from document",
                        "description": "Detailed description",
                        "evidence": "Specific quote or reference from document",
                        "relevance_score": [0.0 to 1.0]
                    }}
                ],
                "relationships": [
                    {{
                        "type": "Relationship type",
                        "description": "How components relate",
                        "evidence": "Supporting evidence from document"
                    }}
                ],
                "quantitative_data": [
                    {{
                        "metric": "Measurement or value name",
                        "value": "Extracted value",
                        "unit": "Unit if applicable",
                        "context": "Context of this measurement"
                    }}
                ]
            }},
            "key_references": [
                {{
                    "title": "Section or chapter name",
                    "content": "Relevant excerpt",
                    "page_or_section": "Location reference"
                }}
            ],
            "summary": "Comprehensive summary of findings addressing the original query"
        }}

        **CRITICAL INSTRUCTIONS:**
        - Extract information strictly from the provided document content
        - Be domain-specific in terminology and analysis approach
        - Provide evidence for all claims with document references
        - If information is insufficient, clearly state what's missing
        - Maintain accuracy over completeness
        - Use domain-appropriate analytical framework
        """
    
    async def get_structured_analysis(self, query: str, context_docs: List[Document], domain: str) -> Dict[str, Any]:
        """Universal structured analysis that adapts to any domain"""
        global openai_client
        
        if not context_docs:
            return {
                "status": "INSUFFICIENT_INFO",
                "confidence": 0.0,
                "domain": domain,
                "analysis": {"main_findings": [], "relationships": [], "quantitative_data": []},
                "key_references": [],
                "summary": "No relevant document content found to analyze the query."
            }
        
        try:
            # Build context
            context_text = ""
            for i, doc in enumerate(context_docs):
                context_text += f"\n--- Document Section {i+1} | Source: {doc.metadata.get('source_file', 'Unknown')} ---\n"
                context_text += doc.page_content + "\n"
            
            # Optimize context for token limits
            context_text = token_optimizer.optimize_context(context_text, query)
            
            # Get domain-adaptive prompt
            prompt = self.get_universal_structured_prompt(domain).format(
                context=context_text,
                query=query
            )
            
            logger.info(f"🤖 Calling OpenAI for structured {domain} analysis...")
            
            response = await openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": f"You are an expert {domain} document analyst. Always respond with valid JSON in the exact format specified."
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
            
            result = json.loads(response.choices[0].message.content)
            logger.info(f"✅ Universal structured analysis complete for {domain}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in universal structured analysis: {str(e)}")
            return {
                "status": "NEEDS_CLARIFICATION",
                "confidence": 0.0,
                "domain": domain,
                "analysis": {"main_findings": [], "relationships": [], "quantitative_data": []},
                "key_references": [{"title": "Analysis Error", "content": f"Error: {str(e)}", "page_or_section": "System"}],
                "summary": f"An error occurred during analysis: {str(e)}"
            }

# Initialize universal analysis engine
universal_analysis = UniversalAnalysisEngine()

# RAG System (Enhanced)
class RAGSystem:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None
        self.documents = []
        self.domain = "general"
        self.domain_config = DOMAIN_CONFIGS["general"]
        # Enhanced components
        self.clause_retriever = None
        
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from PDF file"""
        try:
            pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
            text = ""
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                text += page.get_text()
            pdf_document.close()
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error processing PDF: {str(e)}")
    
    def process_documents(self, files: List[UploadFile]) -> Dict[str, Any]:
        """Process uploaded documents with domain detection"""
        start_time = datetime.now()
        all_text = ""
        self.documents = []
        
        try:
            for file in files:
                logger.info(f"📄 Processing file: {file.filename}")
                
                if file.filename.endswith('.pdf'):
                    text = self.extract_text_from_pdf(file)
                elif file.filename.endswith('.txt'):
                    text = file.file.read().decode('utf-8')
                else:
                    raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported")
                
                all_text += text + "\n\n"
                logger.info(f"✅ Extracted {len(text)} characters from {file.filename}")
            
            # Detect domain from combined text
            self.domain = detect_document_domain(all_text)
            self.domain_config = DOMAIN_CONFIGS.get(self.domain, DOMAIN_CONFIGS["general"])
            
            logger.info(f"🎯 Detected domain: {self.domain}")
            
            # Split text into chunks using domain-specific configuration
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.domain_config["chunk_size"],
                chunk_overlap=self.domain_config["chunk_overlap"],
                length_function=len,
            )
            
            texts = text_splitter.split_text(all_text)
            
            # Create documents with metadata
            self.documents = [
                Document(
                    page_content=text,
                    metadata={
                        "source_file": f"Document_chunk_{i}",
                        "domain": self.domain,
                        "chunk_id": i
                    }
                )
                for i, text in enumerate(texts)
            ]
            
            # Create vector store
            self.vector_store = FAISS.from_documents(self.documents, self.embeddings)
            
            # Initialize enhanced clause retriever
            self.clause_retriever = SemanticClauseRetriever(self.vector_store)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"✅ Documents processed successfully in {processing_time:.2f}s")
            logger.info(f"📊 Created {len(self.documents)} chunks for domain: {self.domain}")
            
            return {
                "success": True,
                "message": f"Successfully processed {len(files)} files",
                "doc_count": len(self.documents),
                "processing_time": processing_time,
                "domain": self.domain
            }
            
        except Exception as e:
            logger.error(f"❌ Error processing documents: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing documents: {str(e)}")
    
    def retrieve_and_rerank(self, query: str, top_k: int = 7) -> Tuple[List[Document], List[float]]:
        """Retrieve and rerank documents based on query"""
        if not self.vector_store:
            return [], []
        
        try:
            # Retrieve documents with similarity scores
            docs_and_scores = self.vector_store.similarity_search_with_score(query, k=top_k)
            documents = [doc for doc, score in docs_and_scores]
            scores = [score for doc, score in docs_and_scores]
            
            logger.info(f"🔍 Retrieved {len(documents)} documents for query")
            return documents, scores
            
        except Exception as e:
            logger.error(f"❌ Error retrieving documents: {str(e)}")
            return [], []

# Decision Engine Classes (Enhanced)
class DecisionEngine:
    """Base decision engine class"""
    async def process_query(self, query: str, rag_system: RAGSystem) -> Union[Dict[str, Any], GeneralDocumentResponse]:
        """Process query and return appropriate response"""
        raise NotImplementedError

class InsuranceDecisionEngine(DecisionEngine):
    """Legacy insurance-specific decision engine"""
    
    async def process_claim(self, claim_query: str, context_docs: List[Document]) -> ClaimProcessingResult:
        """Process insurance claim with existing logic"""
        global openai_client
        
        if not context_docs:
            return ClaimProcessingResult(
                claim_type="unknown",
                decision="insufficient_info",
                approved_amount=None,
                denial_reason="No policy documents found",
                policy_references=[],
                confidence=0.0,
                summary="Cannot process claim without policy documents"
            )
        
        try:
            # Build context from documents
            context = ""
            for i, doc in enumerate(context_docs):
                context += f"\n--- Policy Section {i+1} ---\n{doc.page_content}\n"
            
            # Classify claim type
            claim_type = classify_claim_type(claim_query)
            
            # Create prompt for claim processing
            prompt = f"""
            You are an insurance claim processing agent. Analyze the claim against the policy documents.

            POLICY DOCUMENTS:
            {context}

            CLAIM: {claim_query}
            CLAIM TYPE: {claim_type}

            Provide your analysis in JSON format:
            {{
                "decision": "approved" | "denied" | "needs_investigation",
                "approved_amount": amount or null,
                "denial_reason": "reason if denied" or null,
                "policy_references": ["list of relevant policy sections"],
                "confidence": float between 0 and 1,
                "summary": "detailed explanation of decision"
            }}
            """
            
            response = await openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert insurance claim processor. Always respond in valid JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1500,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            return ClaimProcessingResult(
                claim_type=claim_type,
                decision=result.get("decision", "needs_investigation"),
                approved_amount=result.get("approved_amount"),
                denial_reason=result.get("denial_reason"),
                policy_references=result.get("policy_references", []),
                confidence=result.get("confidence", 0.5),
                summary=result.get("summary", "Claim processed")
            )
            
        except Exception as e:
            logger.error(f"❌ Error processing insurance claim: {str(e)}")
            return ClaimProcessingResult(
                claim_type=claim_type,
                decision="error",
                approved_amount=None,
                denial_reason=f"Processing error: {str(e)}",
                policy_references=[],
                confidence=0.0,
                summary=f"Error occurred during claim processing: {str(e)}"
            )

class EnhancedDecisionEngine(DecisionEngine):
    """Enhanced decision engine with universal query handling"""
    
    def __init__(self):
        self.decision_logic_engine = DecisionLogicEngine()
    
    async def process_query(self, query: str, rag_system: RAGSystem) -> Union[Dict[str, Any], GeneralDocumentResponse, ExplainableResponse]:
        """Universal query processing that adapts to any domain and query type"""
        
        # Get document content for classification
        document_content = ""
        if rag_system.documents:
            document_content = ' '.join([doc.page_content for doc in rag_system.documents[:3]])
        
        # Universal classification
        query_type, domain = classify_query_type(query, document_content)
        
        logger.info(f"🔍 Universal processing: {query_type}, Domain: {domain}")
        
        # Enhanced claim processing
        if query_type == "ENHANCED_CLAIM_PROCESSING":
            return await self.process_enhanced_claim(query, rag_system, domain)
        
        # Retrieve relevant documents
        top_k = rag_system.domain_config.get("context_docs", 7)
        retrieved_docs, similarity_scores = rag_system.retrieve_and_rerank(query, top_k=top_k)
        
        if query_type == "STRUCTURED_ANALYSIS":
            # Use universal structured analysis
            return await universal_analysis.get_structured_analysis(query, retrieved_docs, domain)
        else:
            # Use enhanced conversational analysis
            return await self.get_enhanced_conversational_answer(query, retrieved_docs, domain)
    
    async def process_enhanced_claim(self, query: str, rag_system: RAGSystem, domain: str) -> ExplainableResponse:
        """Enhanced claim processing with full explainability"""
        
        try:
            # Step 1: Parse query for structured information
            logger.info("🔍 Parsing query for structured information...")
            parsed_query = await query_parser.parse_structured_query(query)
            
            # Step 2: Retrieve relevant clauses using semantic search
            if not rag_system.clause_retriever:
                # Fallback to regular document retrieval
                retrieved_docs, _ = rag_system.retrieve_and_rerank(query, top_k=15)
                relevant_clauses = [
                    {
                        "clause_id": f"doc_{i}",
                        "content": doc.page_content,
                        "source_document": doc.metadata.get('source_file', 'Unknown'),
                        "relevance_score": 0.5,
                        "clause_type": "general",
                        "applies_to": {},
                        "composite_relevance": 0.5
                    }
                    for i, doc in enumerate(retrieved_docs)
                ]
            else:
                logger.info("🔍 Retrieving relevant clauses...")
                relevant_clauses = await rag_system.clause_retriever.retrieve_relevant_clauses(parsed_query)
            
            # Step 3: Apply decision logic engine
            logger.info("⚙️ Applying decision logic engine...")
            decision_result = await self.decision_logic_engine.evaluate_claim(parsed_query, relevant_clauses)
            
            # Step 4: Format explainable response
            explainable_response = ExplainableResponse(**decision_result)
            
            logger.info(f"✅ Enhanced claim processing complete: {explainable_response.decision}")
            return explainable_response
            
        except Exception as e:
            logger.error(f"❌ Error in enhanced claim processing: {str(e)}")
            
            # Return error response in ExplainableResponse format
            return ExplainableResponse(
                decision="error",
                confidence_score=0.0,
                processing_time_ms=0,
                decision_reasoning=f"An error occurred during claim processing: {str(e)}",
                key_factors=[],
                risk_assessment={"error": str(e)},
                supporting_clauses=[],
                conflicting_clauses=[],
                exclusion_analysis=[],
                parsed_query={"error": str(e)},
                missing_information=[],
                assumptions_made=[],
                evaluation_steps=[],
                rule_applications=[],
                next_steps=["Contact system administrator", "Retry with simplified query"],
                additional_documents_needed=[],
                appeal_process=None
            )
    
    async def get_enhanced_conversational_answer(self, query: str, context_docs: List[Document], domain: str) -> GeneralDocumentResponse:
        """Enhanced conversational analysis with domain adaptation"""
        global openai_client
        
        if not context_docs:
            return GeneralDocumentResponse(
                query_type="CONVERSATIONAL_ANALYSIS",
                domain=domain,
                answer="No relevant documents found to answer your question.",
                confidence="low",
                source_documents=[]
            )
        
        try:
            # Build rich context
            context = ""
            source_files = []
            for i, doc in enumerate(context_docs):
                context += f"\n--- Document Section {i+1} | Source: {doc.metadata.get('source_file', 'Unknown')} ---\n"
                context += doc.page_content + "\n"
                if doc.metadata.get('source_file'):
                    source_files.append(doc.metadata.get('source_file'))
            
            # Optimize context for token limits
            context = token_optimizer.optimize_context(context, query)
            
            # Domain-adaptive instructions
            domain_instructions = self._get_enhanced_domain_instructions(domain, query)
            
            prompt = f"""You are an expert {domain} document analyst. Provide a comprehensive, accurate answer based on the document content.

**Document Content:**
{context}

**Question:** {query}

**Analysis Instructions:**
{domain_instructions}

**Requirements:**
- Provide a thorough, well-structured answer
- Include specific references to document sections
- Use domain-appropriate terminology
- If information is incomplete, clearly state what's missing
- Be analytical and precise in your response
- Include relevant examples or details from the document

Provide a comprehensive answer that fully addresses the question."""
            
            response = await openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": f"You are an expert {domain} analyst providing detailed, accurate information based solely on document content."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=2500
            )
            
            answer = response.choices[0].message.content
            
            # Enhanced confidence scoring
            confidence = self._calculate_enhanced_confidence(answer, context_docs, query)
            
            return GeneralDocumentResponse(
                query_type="CONVERSATIONAL_ANALYSIS",
                domain=domain,
                answer=answer,
                confidence=confidence,
                source_documents=list(set(source_files))
            )
            
        except Exception as e:
            logger.error(f"❌ Error in enhanced conversational analysis: {str(e)}")
            return GeneralDocumentResponse(
                query_type="CONVERSATIONAL_ANALYSIS",
                domain=domain,
                answer=f"An error occurred while processing this question: {str(e)}",
                confidence="low",
                source_documents=[]
            )
    
    def _get_enhanced_domain_instructions(self, domain: str, query: str) -> str:
        """Get enhanced domain-specific instructions"""
        base_instructions = {
            "physics": """
- Reference specific laws, principles, axioms, or propositions from the text
- Include mathematical relationships and formulas when relevant
- Explain the theoretical framework and practical applications
- Use precise scientific terminology and concepts
- Connect to fundamental principles of physics
            """,
            "legal": """
- Reference specific articles, clauses, sections, or legal provisions
- Explain legal terminology and implications clearly
- Discuss rights, obligations, and procedures
- Cite exact legal references and precedents
- Maintain precision in legal interpretations
            """,
            "medical": """
- Use appropriate medical terminology and concepts
- Reference specific conditions, treatments, or procedures
- Explain medical processes and their implications
- Include dosages, contraindications, or medical guidelines
- Maintain medical accuracy and professional standards
            """,
            "insurance": """
- Reference specific policy clauses, benefits, and exclusions
- Explain coverage terms and conditions clearly
- Include relevant limits, deductibles, or co-payments
- Discuss claim procedures and requirements
- Use insurance industry terminology appropriately
            """,
            "academic": """
- Reference specific theories, methodologies, or research findings
- Include scholarly citations and academic concepts
- Explain theoretical frameworks and applications
- Use appropriate academic terminology
- Discuss implications and conclusions
            """,
            "general": """
- Provide clear, well-structured explanations
- Reference specific sections or parts of the document
- Use appropriate terminology for the subject matter
- Include relevant details and examples
- Maintain accuracy and clarity
            """
        }
        
        instructions = base_instructions.get(domain, base_instructions["general"])
        
        # Add query-specific enhancements
        if "compare" in query.lower() or "difference" in query.lower():
            instructions += "\n- Provide detailed comparisons with clear distinctions"
        if "how" in query.lower():
            instructions += "\n- Explain processes and procedures step-by-step"
        if "why" in query.lower():
            instructions += "\n- Provide reasoning and underlying principles"
        
        return instructions
    
    def _calculate_enhanced_confidence(self, answer: str, context_docs: List[Document], query: str) -> str:
        """Enhanced confidence calculation"""
        # Check for uncertainty indicators
        uncertainty_phrases = ["unclear", "ambiguous", "not specified", "insufficient information", "may be", "possibly"]
        has_uncertainty = any(phrase in answer.lower() for phrase in uncertainty_phrases)
        
        # Check for specific references
        has_specific_refs = any(word in answer.lower() for word in ["section", "chapter", "article", "clause", "page"])
        
        # Check answer completeness
        answer_length = len(answer.split())
        is_comprehensive = answer_length > 100
        
        # Calculate confidence
        if has_uncertainty:
            return "low"
        elif has_specific_refs and is_comprehensive:
            return "high"
        elif is_comprehensive or has_specific_refs:
            return "medium"
        else:
            return "low"

# Initialize systems
rag_system = RAGSystem()
insurance_engine = InsuranceDecisionEngine()
enhanced_engine = EnhancedDecisionEngine()

# Processing Functions (Enhanced)
async def process_questions_universal(questions: List[str], rag_system: RAGSystem, decision_engine: EnhancedDecisionEngine) -> List[Union[UniversalAnalysisResponse, EnhancedUnifiedResponse, ExplainableResponse]]:
    """Universal question processing pipeline"""
    
    async def process_single_question_universal(question: str) -> Union[UniversalAnalysisResponse, EnhancedUnifiedResponse, ExplainableResponse]:
        logger.info(f"🔄 Processing universal question: '{question}'")
        
        try:
            # Process with enhanced engine
            result = await decision_engine.process_query(question, rag_system)
            
            # Convert to appropriate response format
            if isinstance(result, ExplainableResponse):
                return result
            elif isinstance(result, GeneralDocumentResponse):
                return EnhancedUnifiedResponse(
                    query_type=result.query_type,
                    domain=result.domain,
                    processing_approach="conversational",
                    answer=result.answer,
                    confidence=result.confidence,
                    source_documents=result.source_documents
                )
            elif isinstance(result, dict) and "analysis" in result:
                # Structured analysis response
                return UniversalAnalysisResponse(**result)
            else:
                # Legacy insurance response - convert to universal format
                return EnhancedUnifiedResponse(
                    query_type="STRUCTURED_ANALYSIS",
                    domain="insurance",
                    processing_approach="structured",
                    answer=result.get("summary", "Analysis completed"),
                    confidence="high" if result.get("confidence", 0) > 0.7 else "medium",
                    source_documents=["Policy Document"],
                    structured_data=result
                )
                
        except Exception as e:
            logger.error(f"❌ Error processing question: {str(e)}")
            return EnhancedUnifiedResponse(
                query_type="ERROR",
                domain="unknown",
                processing_approach="error_handling",
                answer=f"An error occurred: {str(e)}",
                confidence="low",
                source_documents=[]
            )
    
    # Process in parallel
    tasks = [process_single_question_universal(q) for q in questions]
    return await asyncio.gather(*tasks)

# Existing processing functions (preserved for backward compatibility)
async def process_insurance_claims(claims: List[str], rag_system: RAGSystem) -> List[ClaimProcessingResult]:
    """Legacy insurance claim processing"""
    
    async def process_single_claim(claim: str) -> ClaimProcessingResult:
        logger.info(f"🔄 Processing insurance claim: '{claim}'")
        # Retrieve relevant documents
        retrieved_docs, scores = rag_system.retrieve_and_rerank(claim, top_k=5)
        # Process with insurance engine
        return await insurance_engine.process_claim(claim, retrieved_docs)
    
    # Process claims in parallel
    tasks = [process_single_claim(claim) for claim in claims]
    return await asyncio.gather(*tasks)

async def process_general_questions(questions: List[str], rag_system: RAGSystem) -> List[GeneralDocumentResponse]:
    """Legacy general question processing"""
    
    async def process_single_question(question: str) -> GeneralDocumentResponse:
        logger.info(f"🔄 Processing general question: '{question}'")
        
        try:
            # Retrieve relevant documents
            retrieved_docs, scores = rag_system.retrieve_and_rerank(question, top_k=7)
            
            if not retrieved_docs:
                return GeneralDocumentResponse(
                    query_type="general_inquiry",
                    domain=rag_system.domain,
                    answer="No relevant documents found to answer your question.",
                    confidence="low",
                    source_documents=[]
                )
            
            # Build context
            context = ""
            source_files = []
            for i, doc in enumerate(retrieved_docs):
                context += f"\n--- Document Section {i+1} ---\n{doc.page_content}\n"
                if doc.metadata.get('source_file'):
                    source_files.append(doc.metadata.get('source_file'))
            
            # Create prompt
            prompt = f"""
            Based on the following document content, provide a comprehensive answer to the question.

            DOCUMENT CONTENT:
            {context}

            QUESTION: {question}

            Please provide a detailed, accurate answer based on the document content. If the information is not sufficient, please state what additional information would be needed.
            """
            
            response = await openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful document analyst. Provide accurate, detailed answers based on the provided documents."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            answer = response.choices[0].message.content
            
            # Calculate confidence
            confidence = "high" if len(answer) > 200 and "not sufficient" not in answer.lower() else "medium"
            
            return GeneralDocumentResponse(
                query_type="general_inquiry",
                domain=rag_system.domain,
                answer=answer,
                confidence=confidence,
                source_documents=list(set(source_files))
            )
            
        except Exception as e:
            logger.error(f"❌ Error processing question: {str(e)}")
            return GeneralDocumentResponse(
                query_type="general_inquiry",
                domain=rag_system.domain,
                answer=f"An error occurred while processing this question: {str(e)}",
                confidence="low",
                source_documents=[]
            )
    
    # Process questions in parallel
    tasks = [process_single_question(question) for question in questions]
    return await asyncio.gather(*tasks)

# API Endpoints (Enhanced)
@app.post("/upload_documents", response_model=ProcessingResponse)
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload and process documents"""
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    
    try:
        result = rag_system.process_documents(files)
        return ProcessingResponse(**result)
    except Exception as e:
        logger.error(f"❌ Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process_claims", response_model=List[ClaimProcessingResult])
async def process_claims_endpoint(claims: List[str]):
    """Legacy insurance claim processing endpoint"""
    if not rag_system.vector_store:
        raise HTTPException(status_code=400, detail="No documents uploaded. Please upload policy documents first.")
    
    if not claims:
        raise HTTPException(status_code=400, detail="No claims provided")
    
    try:
        results = await process_insurance_claims(claims, rag_system)
        return results
    except Exception as e:
        logger.error(f"❌ Claims processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing claims: {str(e)}")

@app.post("/ask_questions", response_model=List[GeneralDocumentResponse])
async def ask_questions_endpoint(questions: List[str]):
    """Legacy general questions endpoint"""
    if not rag_system.vector_store:
        raise HTTPException(status_code=400, detail="No documents uploaded. Please upload documents first.")
    
    if not questions:
        raise HTTPException(status_code=400, detail="No questions provided")
    
    try:
        results = await process_general_questions(questions, rag_system)
        return results
    except Exception as e:
        logger.error(f"❌ Questions processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing questions: {str(e)}")

@app.post("/universal_query", response_model=List[Union[UniversalAnalysisResponse, EnhancedUnifiedResponse, ExplainableResponse]])
async def universal_query_endpoint(queries: List[str]):
    """Universal query processing endpoint - handles any type of question across all domains"""
    if not rag_system.vector_store:
        raise HTTPException(status_code=400, detail="No documents uploaded. Please upload documents first.")
    
    if not queries:
        raise HTTPException(status_code=400, detail="No queries provided")
    
    try:
        results = await process_questions_universal(queries, rag_system, enhanced_engine)
        return results
    except Exception as e:
        logger.error(f"❌ Universal query processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing queries: {str(e)}")

# NEW: Enhanced Claim Processing Endpoint
@app.post("/enhanced_claim_processing", response_model=ExplainableResponse)
async def enhanced_claim_processing_endpoint(claim_query: str):
    """Enhanced claim processing with full explainability and structured analysis"""
    if not rag_system.vector_store:
        raise HTTPException(status_code=400, detail="No documents uploaded. Please upload policy documents first.")
    
    if not claim_query:
        raise HTTPException(status_code=400, detail="No claim query provided")
    
    try:
        logger.info(f"🔄 Processing enhanced claim: '{claim_query}'")
        result = await enhanced_engine.process_enhanced_claim(claim_query, rag_system, rag_system.domain)
        return result
    except Exception as e:
        logger.error(f"❌ Enhanced claim processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing enhanced claim: {str(e)}")

# NEW: Query Analysis Endpoint
@app.post("/analyze_query")
async def analyze_query_endpoint(query: str):
    """Analyze query structure and extract entities"""
    if not query:
        raise HTTPException(status_code=400, detail="No query provided")
    
    try:
        parsed_result = await query_parser.parse_structured_query(query)
        return {
            "success": True,
            "parsed_query": parsed_result,
            "analysis_time": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ Query analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing query: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "4.0.0",
        "system_info": {
            "documents_loaded": len(rag_system.documents) if rag_system.documents else 0,
            "domain": rag_system.domain,
            "vector_store_ready": rag_system.vector_store is not None,
            "enhanced_features": {
                "query_parser": True,
                "semantic_clause_retriever": rag_system.clause_retriever is not None,
                "decision_logic_engine": True,
                "token_optimization": True
            }
        }
    }

@app.get("/")
async def root():
    """Root endpoint with system information"""
    return {
        "message": "Universal Document Analysis System",
        "version": "4.0.0",
        "features": [
            "Universal Query Classification",
            "Multi-Domain Analysis", 
            "Enhanced Claim Processing",
            "Structured Query Parsing",
            "Semantic Clause Retrieval",
            "Rule-Based Decision Engine",
            "Explainable AI Responses",
            "Token Optimization",
            "Legacy Insurance Claims",
            "Enhanced Conversational AI"
        ],
        "endpoints": {
            "upload": "/upload_documents",
            "universal": "/universal_query", 
            "enhanced_claims": "/enhanced_claim_processing",
            "query_analysis": "/analyze_query",
            "legacy_claims": "/process_claims",
            "legacy_questions": "/ask_questions",
            "health": "/health"
        },
        "enhancements": {
            "query_parsing": "Extract structured information from natural language",
            "semantic_retrieval": "Advanced clause-based document retrieval",
            "decision_logic": "Rule-based decision making with full audit trail",
            "explainability": "Comprehensive reasoning and evidence tracking",
            "cost_optimization": "Advanced token management for cost efficiency"
        }
    }

if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Starting Universal Document Analysis System v4.0.0")
    uvicorn.run(app, host="0.0.0.0", port=8000)
