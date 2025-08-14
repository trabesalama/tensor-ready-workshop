import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from config.settings import settings

class DocumentProcessor:
    """Handles loading, preprocessing, and chunking of customs documents."""
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize the document processor.
        
        Args:
            data_dir: Directory containing PDF documents. Defaults to settings.data_dir.
        """
        self.data_dir = data_dir or settings.data_dir
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?"]
        )
    
    def load_documents(self) -> List[Document]:
        """
        Load all PDF documents from the data directory.
        
        Returns:
            List of loaded documents.
            
        Raises:
            FileNotFoundError: If no PDF files are found in the data directory.
        """
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        pdf_files = [f for f in os.listdir(self.data_dir) if f.endswith('.pdf')]
        
        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found in the directory: {self.data_dir}")
        
        documents = []
        for pdf_file in pdf_files:
            file_path = self.data_dir / pdf_file
            loader = PyPDFLoader(str(file_path))
            documents.extend(loader.load())
        
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks for processing.
        
        Args:
            documents: List of documents to split.
            
        Returns:
            List of chunked documents.
        """
        return self.text_splitter.split_documents(documents)
    
    def process_documents(self) -> List[Document]:
        """
        Load and split documents in one step.
        
        Returns:
            List of processed document chunks.
        """
        documents = self.load_documents()
        return self.split_documents(documents)