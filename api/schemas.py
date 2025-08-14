from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class QueryRequest(BaseModel):
    question: str = Field(..., description="Question to ask the RAG system")
    prompt_name: Optional[str] = Field(None, description="Name of the prompt to use")
    prompt_technique: Optional[str] = Field(None, description="Prompting technique (standard, cot, react, self_ask)")
    session_id: Optional[str] = Field(None, description="User session ID")
    language: Optional[str] = Field("en", description="Response language (en, fr, mg, it, es)")
    k: Optional[int] = Field(5, description="Number of documents to retrieve")

class QueryResponse(BaseModel):
    answer: str = Field(..., description="Generated answer")
    sources: List[int] = Field(..., description="Page numbers of source documents")
    prompt_used: str = Field(..., description="Name of the prompt used")
    session_id: str = Field(..., description="User session ID")
    language: str = Field(..., description="Response language")

class SimilarityRequest(BaseModel):
    query: str = Field(..., description="Query to test similarity")
    k: Optional[int] = Field(5, description="Number of results to return")

class SimilarityResponse(BaseModel):
    results: List[Dict[str, Any]] = Field(..., description="Similarity results")
    query: str = Field(..., description="Original query")

class EvaluationRequest(BaseModel):
    query: str = Field(..., description="Query to evaluate")
    expected_pages: List[int] = Field(..., description="Expected page numbers")
    threshold: Optional[float] = Field(0.7, description="Similarity threshold")
    top_k: Optional[int] = Field(10, description="Number of documents to retrieve")

class EvaluationResponse(BaseModel):
    precision: float = Field(..., description="Precision score")
    recall: float = Field(..., description="Recall score")
    f1_score: float = Field(..., description="F1 score")
    retrieved_docs: List[Dict[str, Any]] = Field(..., description="Retrieved documents with scores")
    relevant_docs: List[Dict[str, Any]] = Field(..., description="Relevant documents with scores")

class PromptListResponse(BaseModel):
    available_prompts: List[str] = Field(..., description="List of available prompts")
    current_prompt: str = Field(..., description="Currently active prompt")

class PromptSetRequest(BaseModel):
    prompt_name: str = Field(..., description="Name of the prompt to set as active")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Health status")
    version: str = Field(..., description="API version")
    components: Dict[str, str] = Field(..., description="Component statuses")

class SessionCreateRequest(BaseModel):
    user_id: Optional[str] = Field(None, description="User identifier")

class SessionCreateResponse(BaseModel):
    session_id: str = Field(..., description="New session ID")
    message: str = Field(..., description="Confirmation message")

class SessionHistoryResponse(BaseModel):
    session_id: str = Field(..., description="Session ID")
    history: List[Dict[str, Any]] = Field(..., description="Conversation history")