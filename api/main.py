import os
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uuid
from datetime import datetime
from typing import List, Dict, Any
import uvicorn
import yaml
import os

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config_loader import ConfigLoader
from src.vector_store import VectorStore
from src.embedding_manager import EmbeddingManager
from src.prompt_manager import PromptManager
from src.similarity_tester import SimilarityTester
from src.rag_system import LegalRAGSystem as RAGSystem
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

# Load configuration
config_loader = ConfigLoader()

# Initialize components
embedding_manager = EmbeddingManager(config_loader._config)
vector_store = VectorStore(config_loader.vector_store_config, embedding_manager)
prompt_manager = PromptManager()
# Initialize RAG system with default parameters
rag_system = RAGSystem(config_loader.data_directory)
similarity_tester = SimilarityTester(embedding_manager, vector_store)

@app.on_event("startup")
async def startup_event():
    """Initialize the system on startup."""
    # Load and process documents
    from src.document_processing import LegalDocumentProcessor
    from src.chunker import LegalChunker
    from langchain_core.documents import Document
    
    # Process the document
    # Find the PDF file in the data directory
    data_dir = config_loader.data_directory
    pdf_files = [f for f in os.listdir(data_dir) if f.endswith('.pdf')]
    
    if pdf_files:
        # Use the first PDF file found
        pdf_path = os.path.join(data_dir, pdf_files[0])
        print(f"Processing document: {pdf_path}")
        
        doc_processor = LegalDocumentProcessor(pdf_path)
        articles = doc_processor.extract_articles()
        
        # Create chunks
        chunker = LegalChunker(articles)
        chunks = chunker.create_chunks()
        
        # Add to vector store
        # Convert chunks to Document format for vector store
        documents = []
        for chunk in chunks:
            doc = Document(
                page_content=chunk['text'],
                metadata={
                    'article': chunk['metadata'].get('article', ''),
                    'section': chunk['metadata'].get('section', ''),
                    'page': chunk['metadata'].get('page', 1),
                    'source': pdf_path
                }
            )
            documents.append(doc)
        
        vector_store.add_documents(documents)
        print(f"Added {len(documents)} documents to vector store")
        print(f"Processed {len(chunks)} chunks")
    else:
        print("No PDF files found in data directory")
    
    print("System initialized successfully")

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
        if request.prompt_technique and request.prompt_technique != "standard":
            if prompt_name:
                prompt_name = f"{prompt_name}_{request.prompt_technique}"
            else:
                prompt_name = request.prompt_technique
        
        # Get answer from RAG system
        rag_response = rag_system.query(request.question, prompt_type=prompt_name or "default")
        answer = rag_response.get("answer", "No answer found") if isinstance(rag_response, dict) else "No answer found"
        
        # Extract source pages
        source_pages = []
        if rag_response.get("context"):
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
            sources=source_pages or [],
            prompt_used=prompt_name or "default",
            session_id=session_id or "",
            language=request.language or "en"
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
        current_prompt=prompt_manager.current_prompt_name
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
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)