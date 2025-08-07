from langchain_huggingface import HuggingFaceEmbeddings
from typing import Dict, Any

class EmbeddingManager:
    """Manages embedding models for document vectorization."""
    
    def __init__(self, embedding_config: Dict[str, Any]):
        self.config = embedding_config
        self._embeddings = None
    
    @property
    def embeddings(self):
        """Get or create embeddings instance."""
        if self._embeddings is None:
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.config.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
                model_kwargs={'device': self.config.get("device", "cpu")}
            )
        return self._embeddings
    
    def embed_query(self, text: str) -> list:
        """Embed a single query text."""
        return self.embeddings.embed_query(text)
    
    def embed_documents(self, texts: list) -> list:
        """Embed a list of documents."""
        return self.embeddings.embed_documents(texts)