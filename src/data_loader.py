"""
Data loading and preprocessing module for customs code documents.
Handles PDF loading, text splitting, and document preparation.
"""

import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class DataLoader:
    """
    Handles loading and preprocessing of PDF documents.
    
    Attributes:
        data_directory (str): Path to directory containing PDF files
        chunk_size (int): Size of text chunks for splitting
        chunk_overlap (int): Overlap between chunks
    """
    
    def __init__(self, data_directory: str, chunk_size: int, chunk_overlap: int):
        """
        Initialize the DataLoader with configuration parameters.
        
        Args:
            data_directory: Path to directory containing PDF files
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
        """
        self.data_directory = data_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_and_split(self) -> List[Document]:
        """
        Load PDF files from the data directory and split them into chunks.
        
        Returns:
            List of Document objects containing text chunks and metadata
            
        Raises:
            FileNotFoundError: If no PDF files are found in the directory
        """
        # Find all PDF files in the data directory
        pdf_files = [f for f in os.listdir(self.data_directory) if f.endswith('.pdf')]
        
        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found in the directory: {self.data_directory}")

        documents = []
        # Load each PDF file
        for pdf_file in pdf_files:
            loader = PyPDFLoader(os.path.join(self.data_directory, pdf_file))
            documents.extend(loader.load())

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?"]
        )
        return text_splitter.split_documents(documents)