import requests
import streamlit as st
from typing import Dict, Any, List
import pandas as pd
import plotly.express as px

class APIClient:
    """Client for interacting with the RAG API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    def check_health(self) -> bool:
        """Check if API is healthy."""
        try:
            response = requests.get(f"{self.base_url}/health")
            return response.status_code == 200
        except:
            return False
    
    def get_prompts(self) -> Dict[str, Any]:
        """Get available prompts."""
        response = requests.get(f"{self.base_url}/prompts")
        if response.status_code == 200:
            return response.json()
        return {"available_prompts": [], "current_prompt": ""}
    
    def set_prompt(self, prompt_name: str) -> bool:
        """Set active prompt."""
        payload = {"prompt_name": prompt_name}
        response = requests.post(f"{self.base_url}/prompts/set", json=payload)
        return response.status_code == 200
    
    def query(self, question: str, prompt_name: str = None, k: int = 5) -> Dict[str, Any]:
        """Query the RAG system."""
        payload = {"question": question, "prompt_name": prompt_name, "k": k}
        response = requests.post(f"{self.base_url}/query", json=payload)
        if response.status_code == 200:
            return response.json()
        return None
    
    def test_similarity(self, query: str, k: int = 5) -> Dict[str, Any]:
        """Test similarity search."""
        payload = {"query": query, "k": k}
        response = requests.post(f"{self.base_url}/similarity", json=payload)
        if response.status_code == 200:
            return response.json()
        return None
    
    def evaluate_retrieval(
        self, 
        query: str, 
        expected_pages: List[int], 
        threshold: float = 0.7, 
        top_k: int = 10
    ) -> Dict[str, Any]:
        """Evaluate retrieval quality."""
        payload = {
            "query": query,
            "expected_pages": expected_pages,
            "threshold": threshold,
            "top_k": top_k
        }
        response = requests.post(f"{self.base_url}/evaluate", json=payload)
        if response.status_code == 200:
            return response.json()
        return None

def plot_similarity_scores(results: List[Dict[str, Any]], threshold: float = 0.7):
    """Create a bar chart of similarity scores."""
    df = pd.DataFrame(results)
    fig = px.bar(
        df, 
        x="page", 
        y="similarity",
        title="Similarity Scores by Page",
        color="similarity",
        color_continuous_scale="RdYlGn"
    )
    fig.add_hline(y=threshold, line_dash="dash", line_color="red")
    return fig

def display_metrics(precision: float, recall: float, f1_score: float):
    """Display evaluation metrics in columns."""
    col1, col2, col3 = st.columns(3)
    col1.metric("Precision", f"{precision:.2%}")
    col2.metric("Recall", f"{recall:.2%}")
    col3.metric("F1 Score", f"{f1_score:.2%}")