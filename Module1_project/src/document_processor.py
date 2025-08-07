import os
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from pathlib import Path

class DocumentProcessor:
    """Handles loading and processing PDF documents."""
    
    def __init__(self, data_directory: str):
        self.data_directory = Path(data_directory)
    
    def load_documents(self) -> List[Document]:
        """Load all PDF documents from the data directory."""
        pdf_files = list(self.data_directory.glob("*.pdf"))
        
        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found in directory: {self.data_directory}")
        
        documents = []
        for pdf_file in pdf_files:
            loader = PyPDFLoader(str(pdf_file))
            documents.extend(loader.load())
        
        return documents
    
    def split_documents(self, documents: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
        """Split documents into chunks for processing."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?"]
        )
        return text_splitter.split_documents(documents)