from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
import uvicorn
from pathlib import Path
import sys
import os
import uuid
from datetime import datetime

# Add project root to path
# project_root = Path(__file__).parent.parent.parent
# sys.path.append(str(project_root))
# sys.path.append(str(project_root / "Module1_project"))
# sys.path.append(str(Path(__file__).parent.parent / "src"))

# Now import the modules
from src.rag_system import RAGSystem
from src.config_loader import ConfigLoader
from src.vector_store import VectorStore
from src.embedding_manager import EmbeddingManager
from src.prompt_manager import PromptManager
from tests.similarity_tester import SimilarityTester
# ...existing code...

# Add src to path
# sys.path.append(str(Path(__file__).parent.parent / "src"))
from src.prompt_manager import PromptManager
# from Module1_project.api.schemas import (
#     QueryRequest, QueryResponse, SimilarityRequest, SimilarityResponse,
#     EvaluationRequest, EvaluationResponse, PromptListResponse,
#     PromptSetRequest, HealthResponse
# )
from api.schemas import (
    QueryRequest, QueryResponse, SimilarityRequest, SimilarityResponse,
    EvaluationRequest, EvaluationResponse, PromptListResponse,
    PromptSetRequest, HealthResponse, SessionCreateRequest, 
    SessionCreateResponse, SessionHistoryResponse
)
app = FastAPI(
    title="Customs Code RAG API",
    description="API for querying Malagasy customs regulations using RAG",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session storage (in production, use a database)
session_storage = {}

# Initialize components
config_loader = ConfigLoader()
embedding_manager = EmbeddingManager(config_loader.embedding_config)
vector_store = VectorStore(config_loader.vector_store_config, embedding_manager)
prompt_manager = PromptManager()
rag_system = RAGSystem(config_loader, vector_store, prompt_manager)
similarity_tester = SimilarityTester(embedding_manager, vector_store)

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the system on startup using lifespan event handler."""
    # Load and process documents
    from Module1_project.src.document_processor import DocumentProcessor
    doc_processor = DocumentProcessor(config_loader.data_directory)
    
    documents = doc_processor.load_documents()
    text_splitter_config = config_loader.text_splitter_config
    chunks = doc_processor.split_documents(
        documents, 
        text_splitter_config.get("chunk_size", 700),
        text_splitter_config.get("chunk_overlap", 200)
    )
    
    # Add to vector store
    vector_store.add_documents(chunks)
    print("System initialized successfully")
    yield

app = FastAPI(
    title="Customs Code RAG API",
    description="API for querying Malagasy customs regulations using RAG",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check system health."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        components={
            "rag_system": "operational",
            "vector_store": "operational",
            "embedding_manager": "operational",
            "prompt_manager": "operational"
        }
    )

@app.post("/session", response_model=SessionCreateResponse)
async def create_session(request: SessionCreateRequest):
    """Create a new user session."""
    session_id = str(uuid.uuid4())
    session_storage[session_id] = {
        "user_id": request.user_id,
        "created_at": datetime.now().isoformat(),
        "history": []
    }
    return SessionCreateResponse(
        session_id=session_id,
        message="Session created successfully"
    )

@app.get("/session/{session_id}/history", response_model=SessionHistoryResponse)
async def get_session_history(session_id: str):
    """Get conversation history for a session."""
    if session_id not in session_storage:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return SessionHistoryResponse(
        session_id=session_id,
        history=session_storage[session_id]["history"]
    )


@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """Query the RAG system."""
    try:
        # Create session if not provided
        if not request.session_id:
            session_id = str(uuid.uuid4())
            session_storage[session_id] = {
                "user_id": None,
                "created_at": datetime.now().isoformat(),
                "history": []
            }
        else:
            session_id = request.session_id
            if session_id not in session_storage:
                raise HTTPException(status_code=404, detail="Session not found")
        
        # Combine prompt name and technique
        prompt_name = request.prompt_name
        # if request.prompt_technique and request.prompt_technique != "standard":
        #     if prompt_name:
        #         prompt_name = f"{prompt_name}_{request.prompt_technique}"
        #     else:
        #         prompt_name = request.prompt_technique
        
        # Get answer from RAG system
        answer = rag_system.query(request.question, prompt_name)
        
        # Extract source pages
        retriever = vector_store.get_retriever({"k": request.k})
        sources = retriever.invoke(request.question)
        source_pages = sorted(list(set(doc.metadata['page'] for doc in sources)))
        
        # Store in session history
        session_storage[session_id]["history"].append({
            "timestamp": datetime.now().isoformat(),
            "question": request.question,
            "answer": answer,
            "sources": source_pages,
            "prompt_used": prompt_name or "default"
        })
        
        return QueryResponse(
            answer=answer,
            sources=source_pages,
            prompt_used=prompt_name or "default",
            session_id=session_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/similarity", response_model=SimilarityResponse)
async def test_similarity(request: SimilarityRequest):
    """Test similarity search."""
    try:
        results = similarity_tester.test_query_similarity(request.query, request.k if request.k is not None else 5)
        formatted_results = [
            {
                "page": doc.metadata.get('page'),
                "content": doc.page_content[:200] + "...",
                "similarity": score
            }
            for doc, score in results
        ]
        
        return SimilarityResponse(
            results=formatted_results,
            query=request.query
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_retrieval(request: EvaluationRequest):
    """Evaluate retrieval quality."""
    try:
        evaluation = similarity_tester.evaluate_retrieval(
            request.query,
            request.expected_pages,
            request.threshold if request.threshold is not None else 0.5,
            request.top_k if request.top_k is not None else 5
        )
        
        return EvaluationResponse(
            precision=evaluation["precision"],
            recall=evaluation["recall"],
            f1_score=evaluation["f1_score"],
            retrieved_docs=evaluation["retrieved_docs"],
            relevant_docs=evaluation["relevant_docs"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/prompts", response_model=PromptListResponse)
async def list_prompts():
    """List available prompts."""
    return PromptListResponse(
        available_prompts=prompt_manager.available_prompts,
        current_prompt=prompt_manager.current_prompt_name or "default"
    )

@app.post("/prompts/set")
async def set_prompt(request: PromptSetRequest):
    """Set the active prompt."""
    try:
        prompt_manager.set_prompt(request.prompt_name)
        return {"message": f"Prompt set to {request.prompt_name}"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)