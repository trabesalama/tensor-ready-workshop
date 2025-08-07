from langchain_chroma import Chroma
from langchain_core.documents import Document
from typing import List, Dict, Any, Optional
from pathlib import Path

class VectorStore:
    """Manages the vector database for document retrieval."""
    
    def __init__(self, vector_store_config: Dict[str, Any], embedding_manager):
        self.config = vector_store_config
        self.embedding_manager = embedding_manager
        self._vectorstore = None
    
    @property
    def vectorstore(self) -> Chroma:
        """Get or create vector store instance."""
        if self._vectorstore is None:
            persist_dir = Path(self.config.get("persist_directory", "./Module1-project/chroma_db"))
            persist_dir.mkdir(exist_ok=True)
            
            self._vectorstore = Chroma(
                collection_name=self.config.get("collection_name", "default_collection"),
                embedding_function=self.embedding_manager.embeddings,
                persist_directory=str(persist_dir)
            )
        return self._vectorstore
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store."""
        self.vectorstore.add_documents(documents)
    
    def get_retriever(self, search_kwargs: Optional[Dict[str, Any]] = None):
        """Get a retriever for the vector store."""
        if search_kwargs is None:
            search_kwargs = self.config.get("search_kwargs", {"k": 10})
        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Perform similarity search."""
        return self.vectorstore.similarity_search(query, k=k)
    
    def similarity_search_with_score(self, query: str, k: int = 5) -> List[tuple]:
        """Perform similarity search with scores."""
        return self.vectorstore.similarity_search_with_score(query, k=k)
    
    def persist(self) -> None:
        """Persist the vector store to disk."""
        # Chroma automatically persists to disk if persist_directory is set.
        # No explicit persist method is required.
        pass