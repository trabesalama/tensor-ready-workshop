import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from typing import List, Dict, Any
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))
from paths import DATA_DIR

# API configuration
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Customs Code Assistant",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="expanded"
)

def check_api_health():
    """Check if API is running."""
    try:
        response = requests.get(f"{API_URL}/health")
        return response.status_code == 200
    except:
        return False

def get_available_prompts():
    """Get list of available prompts."""
    response = requests.get(f"{API_URL}/prompts")
    if response.status_code == 200:
        return response.json()
    return {"available_prompts": [], "current_prompt": ""}

def query_rag(question: str, prompt_name: str = None, k: int = 5):
    """Query the RAG system."""
    payload = {"question": question, "prompt_name": prompt_name, "k": k}
    response = requests.post(f"{API_URL}/query", json=payload)
    if response.status_code == 200:
        return response.json()
    return None

def test_similarity(query: str, k: int = 5):
    """Test similarity search."""
    payload = {"query": query, "k": k}
    response = requests.post(f"{API_URL}/similarity", json=payload)
    if response.status_code == 200:
        return response.json()
    return None

def evaluate_retrieval(query: str, expected_pages: List[int], threshold: float = 0.7, top_k: int = 10):
    """Evaluate retrieval quality."""
    payload = {
        "query": query,
        "expected_pages": expected_pages,
        "threshold": threshold,
        "top_k": top_k
    }
    response = requests.post(f"{API_URL}/evaluate", json=payload)
    if response.status_code == 200:
        return response.json()
    return None

def set_prompt(prompt_name: str):
    """Set the active prompt."""
    payload = {"prompt_name": prompt_name}
    response = requests.post(f"{API_URL}/prompts/set", json=payload)
    return response.status_code == 200

def main():
    st.title("📜 Customs Code Assistant")
    st.markdown("AI-powered system for interpreting Malagasy customs regulations")
    
    # Check API health
    if not check_api_health():
        st.error("⚠️ API server is not running. Please start the API server first.")
        st.stop()
    
    # Sidebar
    st.sidebar.title("Settings")
    
    # Get available prompts
    prompts_info = get_available_prompts()
    available_prompts = prompts_info.get("available_prompts", [])
    current_prompt = prompts_info.get("current_prompt", "")
    
    # Prompt selection
    selected_prompt = st.sidebar.selectbox(
        "Select Prompt Style",
        available_prompts,
        index=available_prompts.index(current_prompt) if current_prompt in available_prompts else 0
    )
    
    if selected_prompt != current_prompt:
        if set_prompt(selected_prompt):
            st.sidebar.success(f"Prompt set to {selected_prompt}")
            st.rerun()
    
    # Number of documents to retrieve
    k_docs = st.sidebar.slider("Number of documents to retrieve", 1, 20, 5)
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["Query", "Similarity Test", "Evaluation"])
    
    with tab1:
        st.header("Query Customs Code")
        
        # Question input
        question = st.text_area(
            "Enter your question about Malagasy customs regulations:",
            height=100,
            placeholder="e.g., What are the customs duties for importing vehicles?"
        )
        
        if st.button("Get Answer", type="primary"):
            if question:
                with st.spinner("Searching for answer..."):
                    result = query_rag(question, selected_prompt, k_docs)
                    
                    if result:
                        st.subheader("Answer")
                        st.markdown(result["answer"])
                        
                        st.subheader("Sources")
                        sources_df = pd.DataFrame({
                            "Page": result["sources"],
                            "Document": [f"Document {i+1}" for i in range(len(result["sources"]))]
                        })
                        st.dataframe(sources_df, use_container_width=True)
                        
                        st.caption(f"Prompt used: {result['prompt_used']}")
                    else:
                        st.error("Failed to get answer from API")
            else:
                st.warning("Please enter a question")
    
    with tab2:
        st.header("Similarity Search Test")
        
        test_query = st.text_input(
            "Enter query to test similarity:",
            placeholder="e.g., customs duties for vehicles"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            k_sim = st.number_input("Number of results", min_value=1, max_value=20, value=5)
        with col2:
            threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.5, 0.05)
        
        if st.button("Test Similarity"):
            if test_query:
                with st.spinner("Testing similarity..."):
                    result = test_similarity(test_query, k_sim)
                    
                    if result:
                        st.subheader("Similarity Results")
                        
                        # Create dataframe
                        df = pd.DataFrame(result["results"])
                        df["similarity"] = df["similarity"].round(4)
                        
                        # Color code similarity
                        def highlight_similarity(val):
                            color = 'green' if val >= threshold else 'red'
                            return f'background-color: {color}'
                        
                        styled_df = df.style.applymap(highlight_similarity, subset=['similarity'])
                        st.dataframe(styled_df, use_container_width=True)
                        
                        # Plot similarity scores
                        fig = px.bar(
                            df, 
                            x="page", 
                            y="similarity",
                            title="Similarity Scores by Page",
                            color="similarity",
                            color_continuous_scale="RdYlGn"
                        )
                        fig.add_hline(y=threshold, line_dash="dash", line_color="red")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("Failed to test similarity")
            else:
                st.warning("Please enter a query")
    
    with tab3:
        st.header("Retrieval Evaluation")
        
        eval_query = st.text_input(
            "Enter query to evaluate:",
            placeholder="e.g., penalties for false declaration"
        )
        
        expected_pages = st.text_input(
            "Expected page numbers (comma-separated):",
            placeholder="e.g., 45, 67, 89"
        )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            eval_threshold = st.slider("Threshold", 0.0, 1.0, 0.7, 0.05)
        with col2:
            eval_top_k = st.number_input("Top K", min_value=1, max_value=20, value=10)
        with col3:
            st.empty()
        
        if st.button("Evaluate Retrieval"):
            if eval_query and expected_pages:
                try:
                    pages = [int(p.strip()) for p in expected_pages.split(",")]
                    
                    with st.spinner("Evaluating retrieval..."):
                        result = evaluate_retrieval(eval_query, pages, eval_threshold, eval_top_k)
                        
                        if result:
                            # Metrics
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Precision", f"{result['precision']:.2%}")
                            col2.metric("Recall", f"{result['recall']:.2%}")
                            col3.metric("F1 Score", f"{result['f1_score']:.2%}")
                            
                            # Retrieved documents
                            st.subheader("Retrieved Documents")
                            retrieved_df = pd.DataFrame(result["retrieved_docs"], columns=["Page", "Score"])
                            retrieved_df["Relevant"] = retrieved_df["Page"].isin(pages)
                            st.dataframe(retrieved_df, use_container_width=True)
                            
                            # Relevant documents
                            st.subheader("Relevant Documents")
                            relevant_df = pd.DataFrame(result["relevant_docs"], columns=["Page", "Score"])
                            st.dataframe(relevant_df, use_container_width=True)
                        else:
                            st.error("Failed to evaluate retrieval")
                except ValueError:
                    st.error("Please enter valid page numbers (comma-separated integers)")
            else:
                st.warning("Please enter both query and expected page numbers")

if __name__ == "__main__":
    main()