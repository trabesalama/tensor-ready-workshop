from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
import uvicorn
from pathlib import Path
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "Module1-project"))

# Now import the modules
from Module1_project.src.rag_system import RAGSystem
from Module1_project.src.config_loader import ConfigLoader
from Module1_project.src.vector_store import VectorStore
from Module1_project.src.embedding_manager import EmbeddingManager
from Module1_project.src.prompt_manager import PromptManager
from Module1_project.tests.similarity_tester import SimilarityTester
# ...existing code...

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))
from Module1_project.src.prompt_manager import PromptManager
# from src.rag_system import RAGSystem
# from src.config_loader import ConfigLoader
# from src.vector_store import VectorStore
# from src.embedding_manager import EmbeddingManager
# from src.prompt_manager import PromptManager
# from tests.similarity_tester import SimilarityTester
from Module1_project.api.schemas import (
    QueryRequest, QueryResponse, SimilarityRequest, SimilarityResponse,
    EvaluationRequest, EvaluationResponse, PromptListResponse,
    PromptSetRequest, HealthResponse
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

# Initialize components
config_loader = ConfigLoader()
embedding_manager = EmbeddingManager(config_loader.embedding_config)
vector_store = VectorStore(config_loader.vector_store_config, embedding_manager)
prompt_manager = PromptManager()
rag_system = RAGSystem(config_loader, vector_store, prompt_manager)
similarity_tester = SimilarityTester(embedding_manager, vector_store)

@app.on_event("startup")
async def startup_event():
    """Initialize the system on startup."""
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

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """Query the RAG system."""
    try:
        answer = rag_system.query(request.question, request.prompt_name)
        
        # Extract source pages
        retriever = vector_store.get_retriever({"k": request.k})
        sources = retriever.invoke(request.question)
        source_pages = sorted(list(set(doc.metadata['page'] for doc in sources)))
        
        return QueryResponse(
            answer=answer,
            sources=source_pages,
            prompt_used=prompt_manager.current_prompt_name
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