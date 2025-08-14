"""
Embedding and vector database management module.
Handles document embedding, storage, and retrieval using ChromaDB.
"""

import os
from typing import List, Optional, Dict, Any
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever

class EmbeddingManager:
    """
    Manages document embeddings and vector storage using ChromaDB.
    
    Attributes:
        collection_name (str): Name of the ChromaDB collection
        persist_directory (str): Directory to persist the vector database
        embedding_model (str): Name of the embedding model to use
        model_kwargs (dict): Additional arguments for the embedding model
        embeddings (Embeddings): Initialized embedding model
        vectorstore (Optional[Chroma]): ChromaDB vector store instance
    """
    
    def __init__(self, collection_name: str, persist_directory: str, 
                 embedding_model: str, model_kwargs: Dict[str, Any]):
        """
        Initialize the EmbeddingManager with configuration parameters.
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the vector database
            embedding_model: Name of the embedding model to use
            model_kwargs: Additional arguments for the embedding model
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        self.model_kwargs = model_kwargs
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs=self.model_kwargs
        )
        self.vectorstore: Optional[Chroma] = None

    def load_or_create_vectorstore(self, documents: Optional[List[Document]] = None) -> Chroma:
        """
        Load existing vector store or create a new one if it doesn't exist.
        
        Args:
            documents: List of documents to embed (required if creating new store)
            
        Returns:
            Chroma vector store instance
            
        Raises:
            ValueError: If documents are not provided when creating a new store
        """
        # Check if the vectorstore already exists
        if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
            # Load existing vectorstore
            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
        else:
            # Create new vectorstore
            if documents is None:
                raise ValueError("Documents must be provided to create a new vectorstore.")
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                collection_name=self.collection_name,
                persist_directory=self.persist_directory
            )
        return self.vectorstore

    def get_retriever(self, search_kwargs: Dict[str, Any]) -> VectorStoreRetriever:
        """
        Get a retriever for the vector store.
        
        Args:
            search_kwargs: Arguments for the retriever search
            
        Returns:
            VectorStoreRetriever instance
            
        Raises:
            ValueError: If vectorstore is not loaded
        """
        if self.vectorstore is None:
            raise ValueError("Vectorstore not loaded. Call load_or_create_vectorstore first.")
        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)

    def get_retriever_with_scores(self, search_kwargs: Dict[str, Any]) -> VectorStoreRetriever:
        """
        Get a retriever that returns documents with relevance scores.
        
        Args:
            search_kwargs: Arguments for the retriever search
            
        Returns:
            VectorStoreRetriever instance configured to return scores
        """
        if self.vectorstore is None:
            raise ValueError("Vectorstore not loaded. Call load_or_create_vectorstore first.")
        
        # Configure search with score threshold
        search_type = "similarity_score_threshold"
        return self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )