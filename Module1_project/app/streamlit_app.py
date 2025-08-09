# import streamlit as st
# import requests
# import pandas as pd
# import plotly.express as px
# from typing import List, Dict, Any
# import os
# from pathlib import Path
# import sys

# # Add src to path
# sys.path.append(str(Path(__file__).parent.parent / "src"))
# from Module1_project.paths import DATA_DIR

# # API configuration
# API_URL = "http://127.0.0.1:8000"
# st.set_page_config(
#     page_title="Customs Code Assistant",
#     page_icon="icons/logo.png",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# def check_api_health():
#     """Check if API is running."""
#     try:
#         response = requests.get(f"{API_URL}/health")
#         return response.status_code == 200
#     except:
#         return False

# def get_available_prompts():
#     """Get list of available prompts."""
#     response = requests.get(f"{API_URL}/prompts")
#     if response.status_code == 200:
#         return response.json()
#     return {"available_prompts": [], "current_prompt": ""}

# from typing import Optional

# def query_rag(question: str, prompt_name: Optional[str] = None, k: int = 5):
#     """Query the RAG system."""
#     payload = {"question": question, "prompt_name": prompt_name, "k": k}
#     response = requests.post(f"{API_URL}/query", json=payload)
#     if response.status_code == 200:
#         return response.json()
#     return None

# # def test_similarity(query: str, k: int = 5):
# #     """Test similarity search."""
# #     payload = {"query": query, "k": k}
# #     response = requests.post(f"{API_URL}/similarity", json=payload)
# #     if response.status_code == 200:
# #         return response.json()
# #     return None

# # def evaluate_retrieval(query: str, expected_pages: List[int], threshold: float = 0.7, top_k: int = 10):
# #     """Evaluate retrieval quality."""
# #     payload = {
# #         "query": query,
# #         "expected_pages": expected_pages,
# #         "threshold": threshold,
# #         "top_k": top_k
# #     }
# #     response = requests.post(f"{API_URL}/evaluate", json=payload)
# #     if response.status_code == 200:
# #         return response.json()
# #     return None

# def set_prompt(prompt_name: str):
#     """Set the active prompt."""
#     payload = {"prompt_name": prompt_name}
#     response = requests.post(f"{API_URL}/prompts/set", json=payload)
#     return response.status_code == 200

# def main():
#     st.title("Customs Code Assistant")
#     st.markdown("AI-powered system for interpreting Malagasy customs regulations")
    
#     # Check API health
#     if not check_api_health():
#         st.error("⚠️ API server is not running. Please start the API server first.")
#         st.stop()
    
#     # Sidebar
#     st.sidebar.title("Settings")
    
#     # Get available prompts
#     prompts_info = get_available_prompts()
#     available_prompts = prompts_info.get("available_prompts", [])
#     current_prompt = prompts_info.get("current_prompt", "")
    
#     # Prompt selection
#     selected_prompt = st.sidebar.selectbox(
#         "Select Prompt Style",
#         available_prompts,
#         index=available_prompts.index(current_prompt) if current_prompt in available_prompts else 0
#     )
    
#     if selected_prompt != current_prompt:
#         if set_prompt(selected_prompt):
#             st.sidebar.success(f"Prompt set to {selected_prompt}")
#             st.rerun()
    
#     # Number of documents to retrieve
#     k_docs = st.sidebar.slider("Number of documents to retrieve", 1, 20, 5)
    
#     # Main content
#     tab1, tab2, tab3 = st.tabs(["Query", "Similarity Test", "Evaluation"])
    
#     with tab1:
#         st.header("Query Customs Code")
        
#         # Question input
#         question = st.text_area(
#             "Enter your question about Malagasy customs regulations:",
#             height=100,
#             placeholder="e.g., What are the customs duties for importing vehicles?"
#         )
        
#         if st.button("Get Answer", type="primary"):
#             if question:
#                 with st.spinner("Searching for answer..."):
#                     result = query_rag(question, selected_prompt, k_docs)
                    
#                     if result:
#                         st.subheader("Answer")
#                         st.markdown(result["answer"])
                        
#                         st.subheader("Sources")
#                         sources_df = pd.DataFrame({
#                             "Page": result["sources"],
#                             "Document": [f"Document {i+1}" for i in range(len(result["sources"]))]
#                         })
#                         st.dataframe(sources_df, use_container_width=True)
                        
#                         st.caption(f"Prompt used: {result['prompt_used']}")
#                     else:
#                         st.error("Failed to get answer from API")
#             else:
#                 st.warning("Please enter a question")
    
#     # with tab2:
#     #     st.header("Similarity Search Test")
        
#     #     test_query = st.text_input(
#     #         "Enter query to test similarity:",
#     #         placeholder="e.g., customs duties for vehicles"
#     #     )
        
#     #     col1, col2 = st.columns(2)
#     #     with col1:
#     #         k_sim = st.number_input("Number of results", min_value=1, max_value=20, value=5)
#     #     with col2:
#     #         threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.5, 0.05)
        
#     #     if st.button("Test Similarity"):
#     #         if test_query:
#     #             with st.spinner("Testing similarity..."):
#     #                 result = test_similarity(test_query, k_sim)
                    
#     #                 if result:
#     #                     st.subheader("Similarity Results")
                        
#     #                     # Create dataframe
#     #                     df = pd.DataFrame(result["results"])
#     #                     df["similarity"] = df["similarity"].round(4)
                        
#     #                     # Color code similarity
#     #                     def highlight_similarity(col):
#     #                         return ['background-color: green' if val >= threshold else 'background-color: red' for val in col]
                        
#     #                     styled_df = df.style.apply(highlight_similarity, subset=['similarity'])
#     #                     st.dataframe(styled_df, use_container_width=True)
                        
#     #                     # Plot similarity scores
#     #                     fig = px.bar(
#     #                         df, 
#     #                         x="page", 
#     #                         y="similarity",
#     #                         title="Similarity Scores by Page",
#     #                         color="similarity",
#     #                         color_continuous_scale="RdYlGn"
#     #                     )
#     #                     fig.add_hline(y=threshold, line_dash="dash", line_color="red")
#     #                     st.plotly_chart(fig, use_container_width=True)
#     #                 else:
#     #                     st.error("Failed to test similarity")
#     #         else:
#     #             st.warning("Please enter a query")
    
#     # with tab3:
#     #     st.header("Retrieval Evaluation")
        
#     #     eval_query = st.text_input(
#     #         "Enter query to evaluate:",
#     #         placeholder="e.g., penalties for false declaration"
#     #     )
        
#     #     expected_pages = st.text_input(
#     #         "Expected page numbers (comma-separated):",
#     #         placeholder="e.g., 45, 67, 89"
#     #     )
        
#     #     col1, col2, col3 = st.columns(3)
#     #     with col1:
#     #         eval_threshold = st.slider("Threshold", 0.0, 1.0, 0.7, 0.05)
#     #     with col2:
#     #         eval_top_k = st.number_input("Top K", min_value=1, max_value=20, value=10)
#     #     with col3:
#     #         st.empty()
        
#     #     if st.button("Evaluate Retrieval"):
#     #         if eval_query and expected_pages:
#     #             try:
#     #                 pages = [int(p.strip()) for p in expected_pages.split(",")]
                    
#     #                 with st.spinner("Evaluating retrieval..."):
#     #                     result = evaluate_retrieval(eval_query, pages, eval_threshold, eval_top_k)
                        
#     #                     if result:
#     #                         # Metrics
#     #                         col1, col2, col3 = st.columns(3)
#     #                         col1.metric("Precision", f"{result['precision']:.2%}")
#     #                         col2.metric("Recall", f"{result['recall']:.2%}")
#     #                         col3.metric("F1 Score", f"{result['f1_score']:.2%}")
                            
#     #                         # Retrieved documents
#     #                         st.subheader("Retrieved Documents")
#     #                         retrieved_df = pd.DataFrame(result["retrieved_docs"], columns=["Page", "Score"])
#     #                         retrieved_df["Relevant"] = retrieved_df["Page"].isin(pages)
#     #                         st.dataframe(retrieved_df, use_container_width=True)
                            
#     #                         # Relevant documents
#     #                         st.subheader("Relevant Documents")
#     #                         relevant_df = pd.DataFrame(result["relevant_docs"], columns=["Page", "Score"])
#     #                         st.dataframe(relevant_df, use_container_width=True)
#     #                     else:
#     #                         st.error("Failed to evaluate retrieval")
#     #             except ValueError:
#     #                 st.error("Please enter valid page numbers (comma-separated integers)")
#     #         else:
#     #             st.warning("Please enter both query and expected page numbers")

# if __name__ == "__main__":
#     main()



import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from typing import List, Dict, Any
import os
from pathlib import Path
import sys
import time
import uuid
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))
from  Module1_project.app.paths import DATA_DIR

# API configuration
API_URL = "http://127.0.0.1:8000"  # Utiliser 127.0.0.1 au lieu de localhost

st.set_page_config(
    page_title="Customs Code Assistant",
    page_icon="./icons/logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

def check_api_health():
    """Check if API is running."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_available_prompts():
    """Get list of available prompts."""
    response = requests.get(f"{API_URL}/prompts")
    if response.status_code == 200:
        return response.json()
    return {"available_prompts": [], "current_prompt": ""}

def create_session(user_id=None):
    """Create a new session."""
    payload = {"user_id": user_id}
    response = requests.post(f"{API_URL}/session", json=payload)
    if response.status_code == 200:
        return response.json()
    return None

def get_session_history(session_id):
    """Get session history."""
    response = requests.get(f"{API_URL}/session/{session_id}/history")
    if response.status_code == 200:
        return response.json()
    return None

from typing import Optional

def query_rag(question: str, prompt_name: Optional[str] = None, prompt_technique: Optional[str] = None, session_id: Optional[str] = None, k: int = 5):
    """Query the RAG system."""
    payload = {
        "question": question, 
        "prompt_name": prompt_name,
        "prompt_technique": prompt_technique,
        "session_id": session_id,
        "k": k
    }
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

def initialize_session_state():
    """Initialize session state variables."""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = None
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'current_prompt' not in st.session_state:
        st.session_state.current_prompt = None
    if 'current_technique' not in st.session_state:
        st.session_state.current_technique = "standard"
    if 'page' not in st.session_state:
        st.session_state.page = "intro"

def intro_page():
    """Display the introduction page."""
    st.title("Customs Code Assistant")
    st.markdown("### AI-powered system for interpreting Malagasy customs regulations")
    
    st.markdown("""
    ## Welcome to the Customs Code Assistant
    
    This advanced AI system helps professionals navigate the complex Madagascar Customs Code (LFI 2025) using state-of-the-art retrieval-augmented generation (RAG) technology.
    
    ### Key Features:
    - **Accurate Answers**: Get precise information from official customs documents
    - **Source Citations**: Every answer includes references to the original documents
    - **Multiple Prompting Techniques**: Choose from different reasoning approaches
    - **Session Management**: Your conversation history is saved for continuity
    - **Similarity Testing**: Evaluate how well the system retrieves relevant documents
    
    ### Prompting Techniques:
    
    1. **Standard**:
       - Direct question-answering approach
       - Best for straightforward factual questions
    
    2. **Chain of Thought (CoT)**:
       - Step-by-step reasoning before answering
       - Better for complex questions requiring logical deduction
    
    3. **ReAct**:
       - Combines reasoning and acting
       - Useful for procedural questions requiring multiple steps
    
    4. **Self-Ask**:
       - Breaks down complex questions into simpler sub-questions
       - Ideal for multi-faceted regulatory questions
    
    ### How to Use:
    1. Click "Start New Session" below to begin
    2. Select your preferred prompting technique
    3. Ask questions about Malagasy customs regulations
    4. Review the answers with source citations
    5. Continue the conversation as needed
    
    Your session history will be saved automatically for future reference.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Start New Session", type="primary"):
            # Create new session
            session_data = create_session()
            if session_data:
                st.session_state.session_id = session_data["session_id"]
                st.session_state.page = "chat"
                st.rerun()
            else:
                st.error("Failed to create session")
    
    with col2:
        if st.button("Continue Previous Session"):
            st.session_state.page = "sessions"
            st.rerun()

def sessions_page():
    """Display session management page."""
    st.title("Session Management")
    
    st.markdown("### Your Sessions")
    st.markdown("Select a session to continue your conversation:")
    
    # In a real app, you would fetch sessions from a database
    # For demo, we'll show a placeholder
    st.info("Session management would be implemented here in a production environment")
    
    if st.button("Back to Introduction"):
        st.session_state.page = "intro"
        st.rerun()

def chat_page():
    """Display the chat interface."""
    st.title("💬 Chat with Customs Code Assistant")
    
    # Sidebar settings
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
    
    # Prompting technique selection
    technique_options = ["standard", "cot", "react", "self_ask"]
    selected_technique = st.sidebar.selectbox(
        "Select Prompting Technique",
        technique_options,
        index=technique_options.index(st.session_state.current_technique)
    )
    
    # Update current technique
    if selected_technique != st.session_state.current_technique:
        st.session_state.current_technique = selected_technique
    
    # Number of documents to retrieve
    k_docs = st.sidebar.slider("Number of documents to retrieve", 1, 20, 5)
    
    # Session info
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Session ID**: `{st.session_state.session_id[:8]}...`")
    
    if st.sidebar.button("End Session"):
        st.session_state.page = "intro"
        st.rerun()
    
    # Load session history
    if st.session_state.session_id:
        history_data = get_session_history(st.session_state.session_id)
        if history_data:
            st.session_state.chat_history = history_data["history"]
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for i, msg in enumerate(st.session_state.chat_history):
            with st.chat_message("user", avatar="👤"):
                st.write(msg["question"])
            
            with st.chat_message("assistant", avatar="🤖"):
                st.write(msg["answer"])
                with st.expander("Sources"):
                    st.write(f"Pages: {', '.join(map(str, msg['sources']))}")
                    st.caption(f"Prompt used: {msg['prompt_used']}")
    
    # Input for new question
    question = st.chat_input("Ask a question about customs regulations...")
    
    if question:
        # Add user message to chat history
        with st.chat_message("user", avatar="👤"):
            st.write(question)
        
        # Get answer from API
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Searching for answer..."):
                result = query_rag(
                    question, 
                    selected_prompt, 
                    selected_technique,
                    st.session_state.session_id,
                    k_docs
                )
                
                if result:
                    st.write(result["answer"])
                    with st.expander("Sources"):
                        sources_df = pd.DataFrame({
                            "Page": result["sources"],
                            "Document": [f"Document {i+1}" for i in range(len(result["sources"]))]
                        })
                        st.dataframe(sources_df, use_container_width=True)
                    
                    # Add to session history
                    st.session_state.chat_history.append({
                        "timestamp": datetime.now().isoformat(),
                        "question": question,
                        "answer": result["answer"],
                        "sources": result["sources"],
                        "prompt_used": result["prompt_used"]
                    })
                else:
                    st.error("Failed to get answer from API")

def similarity_page():
    """Display the similarity test page."""
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
                    def highlight_similarity(col):
                        return [f'background-color: {"green" if val >= threshold else "red"}' for val in col]
                    
                    styled_df = df.style.apply(highlight_similarity, subset=['similarity'])
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

def evaluation_page():
    """Display the evaluation page."""
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

def main():
    # Initialize session state
    initialize_session_state()
    
    # Check API health
    if not check_api_health():
        st.error("⚠️ API server is not running. Please start the API server first.")
        st.stop()
    
    # Navigation based on current page
    if st.session_state.page == "intro":
        intro_page()
    elif st.session_state.page == "sessions":
        sessions_page()
    elif st.session_state.page == "chat":
        chat_page()
    elif st.session_state.page == "similarity":
        similarity_page()
    elif st.session_state.page == "evaluation":
        evaluation_page()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    
    if st.session_state.page != "intro":
        if st.sidebar.button("Introduction"):
            st.session_state.page = "intro"
            st.rerun()
    
    if st.session_state.page != "chat" and st.session_state.session_id:
        if st.sidebar.button("Chat"):
            st.session_state.page = "chat"
            st.rerun()
    
    if st.session_state.page != "similarity":
        if st.sidebar.button("Similarity Test"):
            st.session_state.page = "similarity"
            st.rerun()
    
    if st.session_state.page != "evaluation":
        if st.sidebar.button("Evaluation"):
            st.session_state.page = "evaluation"
            st.rerun()

if __name__ == "__main__":
    main()