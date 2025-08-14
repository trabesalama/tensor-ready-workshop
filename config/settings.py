from pathlib import Path
from typing import Dict, Any
import os
from pydantic import BaseSettings, Field

class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # API Keys
    groq_api_key: str = Field(..., env="GROQ_API_KEY")
    
    # Model Configuration
    model_name: str = "llama-3.1-8b-instant"
    temperature: float = 0.1
    
    # Document Processing
    chunk_size: int = 700
    chunk_overlap: int = 200
    
    # Vector Database
    collection_name: str = "french_customs_code"
    persist_directory: str = "./chroma_db"
    
    # Data Directory
    data_dir: Path = Field(default_factory=lambda: Path("data"))
    
    # UI Configuration
    page_title: str = "French Customs Code Assistant"
    page_icon: str = "🏛️"
    
    class Config:
        env_file = ".env"

settings = Settings()