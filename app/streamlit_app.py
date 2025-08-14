import streamlit as st
import os
import sys
from typing import Dict, List, Optional
import pandas as pd
import plotly.express as px
from PIL import Image

# Add src to path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
# import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.rag_system import LegalRAGSystem
from src.utils import load_yaml_config

# Load configuration
config = load_yaml_config("config/config.yaml")

# Set page configuration
st.set_page_config(
    page_title=config['app']['name'],
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    """Initialize Streamlit session state."""
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'current_prompt' not in st.session_state:
        st.session_state.current_prompt = "default"
    if 'page' not in st.session_state:
        st.session_state.page = "home"

def home_page():
    """Home page with document upload and system initialization."""
    st.title("📜 Customs Code RAG System")
    st.markdown("### AI-powered system for interpreting Malagasy customs regulations")
    
    st.markdown("""
    #### Features:
    - **Advanced Document Processing**: Extract and structure legal articles from PDFs
    - **Semantic Search**: Find relevant information using natural language queries
    - **Contextual Answers**: Generate precise responses with legal references
    - **Multiple Prompt Types**: Choose from different response styles
    - **Source Attribution**: All answers include citations to original documents
    """)
    
    st.markdown("---")
    
    # Document upload
    st.subheader("1. Upload Customs Code Document")
    # uploaded_file = st.file_uploader(
    #     "Upload PDF document",
    #     type="pdf",
    #     help="Upload the Malagasy Customs Code PDF document"
    # )
    
    # Initialize system button
    if st.button("Initialize System", type="primary"):
        # if uploaded_file is not None:
            # Save uploaded file
            # upload_dir = "data"
            # os.makedirs(upload_dir, exist_ok=True)
            # file_path = os.path.join(upload_dir, uploaded_file.name)
            
            # with open(file_path, "wb") as f:
                # f.write(uploaded_file.getbuffer())
            
            # Initialize RAG system
        with st.spinner("Initializing system... This may take a few minutes."):
            try:
                st.session_state.rag_system = LegalRAGSystem()
                st.success("System initialized successfully!")
                st.session_state.page = "chat"
                st.rerun()
            except Exception as e:
                st.error(f"Error initializing system: {e}")
        # else:
        #     st.error("Please upload a PDF document first.")
    
    st.markdown("---")
    
    # System information
    st.subheader("System Information")
    st.markdown(f"""
    - **Embedding Model**: {config['embedding']['model_name']}
    - **LLM**: {config['groq']['model']}
    - **Chunk Size**: {config['chunking']['max_chunk_size']} words
    - **Database**: ChromaDB
    """)

def chat_page():
    """Chat interface for querying the RAG system."""
    st.title("💬 Query Customs Code")
    
    # Sidebar settings
    st.sidebar.title("Settings")
    
    # Prompt type selection
    prompt_types = {
        "Default": "default",
        "Technical": "technical",
        "Simplified": "simplified"
    }
    
    selected_prompt = st.sidebar.selectbox(
        "Response Style",
        options=list(prompt_types.keys()),
        format_func=lambda x: x,
        index=list(prompt_types.values()).index(st.session_state.current_prompt)
    )
    
    if selected_prompt != list(prompt_types.keys())[list(prompt_types.values()).index(st.session_state.current_prompt)]:
        st.session_state.current_prompt = prompt_types[selected_prompt]
    
    # Search filters
    st.sidebar.subheader("Search Filters")
    theme_filter = st.sidebar.selectbox(
        "Filter by Theme",
        options=["All", "General Principles", "Customs Tariffs", "Economic Regimes", 
                "Customs Procedures", "Disputes and Sanctions", "Miscellaneous Taxes"],
        index=0
    )
    
    # Apply theme filter
    filters = None if theme_filter == "All" else {"theme": theme_filter}
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for i, msg in enumerate(st.session_state.chat_history):
            with st.chat_message("user", avatar="👤"):
                st.write(msg["question"])
            
            with st.chat_message("assistant", avatar="🤖"):
                st.write(msg["answer"])
                
                with st.expander("Sources"):
                    st.write(f"**Prompt Type**: {msg['prompt_type']}")
                    st.write(f"**Filters Used**: {msg['filters']}")
                    
                    # Display context sources
                    for j, context in enumerate(msg['context'][:3], 1):
                        st.write(f"**Source {j}**:")
                        st.write(f"Article: {context['metadata'].get('article', 'N/A')}")
                        st.write(f"Theme: {context['metadata'].get('theme', 'N/A')}")
                        st.write(f"Text: {context['text'][:200]}...")
    
    # Input for new question
    question = st.chat_input("Ask a question about customs regulations...")
    
    if question:
        # Add user message to chat history
        with st.chat_message("user", avatar="👤"):
            st.write(question)
        
        # Get answer from RAG system
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Searching for answer..."):
                result = st.session_state.rag_system.query(
                    question, 
                    filters=filters,
                    prompt_type=st.session_state.current_prompt
                )
                
                st.write(result["answer"])
                
                with st.expander("Sources"):
                    st.write(f"**Prompt Type**: {result['prompt_type']}")
                    st.write(f"**Filters Used**: {result['filters']}")
                    
                    # Display context sources
                    for j, context in enumerate(result['context'][:3], 1):
                        st.write(f"**Source {j}**:")
                        st.write(f"Article: {context['metadata'].get('article', 'N/A')}")
                        st.write(f"Theme: {context['metadata'].get('theme', 'N/A')}")
                        st.write(f"Text: {context['text'][:200]}...")
                
                # Add to chat history
                st.session_state.chat_history.append({
                    "question": question,
                    "answer": result["answer"],
                    "prompt_type": result["prompt_type"],
                    "filters": result["filters"],
                    "context": result["context"]
                })

def main():
    """Main application function."""
    initialize_session_state()
    
    # Navigation
    if st.session_state.page == "home":
        home_page()
    elif st.session_state.page == "chat":
        if st.session_state.rag_system is None:
            st.session_state.page = "home"
            st.rerun()
        else:
            chat_page()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    
    if st.session_state.page != "home":
        if st.sidebar.button("Home"):
            st.session_state.page = "home"
            st.rerun()
    
    if st.session_state.page != "chat" and st.session_state.rag_system is not None:
        if st.sidebar.button("Chat"):
            st.session_state.page = "chat"
            st.rerun()

if __name__ == "__main__":
    main()