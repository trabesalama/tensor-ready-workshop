"""
Streamlit application for the Customs Code Assistant.
Provides a user interface for interacting with the RAG system.
"""

import os
import streamlit as st
from dotenv import load_dotenv
from typing import Dict, Any, Optional

# Load environment variables
load_dotenv()

# Import custom modules
from src.data_loader import DataLoader
from src.embedding_manager import EmbeddingManager
from src.prompt_manager import PromptManager
from src.rag_system import RAGSystem
from src.memory import ConversationMemory
from src.utils import load_yaml, ensure_directory_exists
from paths import DATA_DIR, CONFIG_DIR

# Load configurations
settings = load_yaml(os.path.join(CONFIG_DIR, "settings.yaml"))
prompts_config = load_yaml(os.path.join(CONFIG_DIR, "prompts.yaml"))

# Initialize session state
def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationMemory()
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    if 'initialization_error' not in st.session_state:
        st.session_state.initialization_error = None

def initialize_rag_system() -> RAGSystem:
    """
    Initialize the RAG system with data and configurations.
    
    Returns:
        Initialized RAG system
        
    Raises:
        ValueError: If required environment variables are not set
    """
    # Ensure directories exist
    ensure_directory_exists(settings['chroma']['persist_directory'])
    
    # Load data
    data_loader = DataLoader(
        data_directory=settings['data_directory'],
        chunk_size=settings['chunking']['chunk_size'],
        chunk_overlap=settings['chunking']['chunk_overlap']
    )
    documents = data_loader.load_and_split()
    
    # Initialize embedding manager
    embedding_manager = EmbeddingManager(
        collection_name=settings['chroma']['collection_name'],
        persist_directory=settings['chroma']['persist_directory'],
        embedding_model=settings['embedding']['model_name'],
        model_kwargs=settings['embedding']['model_kwargs']
    )
    vectorstore = embedding_manager.load_or_create_vectorstore(documents)
    
    # Get retriever with scores
    retriever = embedding_manager.get_retriever_with_scores(
        search_kwargs=settings['retrieval']['search_kwargs']
    )
    
    # Initialize prompt manager
    prompt_manager = PromptManager(os.path.join(CONFIG_DIR, "prompts.yaml"))
    
    # Initialize LLM
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set.")
    
    llm = RAGSystem.create_llm(
        api_key=groq_api_key,
        model_name=settings['model']['name'],
        temperature=settings['model']['temperature']
    )
    
    # Create RAG system
    return RAGSystem(retriever, prompt_manager, llm, vectorstore)

def display_response(response_data: Dict[str, Any]) -> None:
    """
    Display the RAG system response in the Streamlit UI.
    
    Args:
        response_data: Dictionary containing response and metadata
    """
    # Display main response
    st.write("### Assistant Response")
    st.write(response_data["response"])
    
    # Display sources
    st.write("### Sources")
    st.write(f"Pages consulted: {', '.join(map(str, response_data['sources']))}")
    
    # Display relevance scores
    st.write("### Relevance Scores")
    for i, score in enumerate(response_data['scores']):
        st.write(f"Document {i+1}: {score:.2f}")
    
    # Display prompt type
    st.write("### Prompt Strategy")
    st.write(f"Used: {response_data['prompt_type']}")

def main():
    """Main function for the Streamlit application."""
    # Initialize session state
    initialize_session_state()
    
    # Page configuration
    st.set_page_config(
        page_title="Customs Code Assistant",
        page_icon="📜",
        layout="wide"
    )
    
    # Header
    st.title("📜 Customs Code Assistant")
    st.markdown("Expert system for interpreting Malagasy customs regulations")
    
    # Initialize RAG system if not already done
    if not st.session_state.initialized and st.session_state.initialization_error is None:
        try:
            with st.spinner("Initializing RAG system..."):
                st.session_state.rag_system = initialize_rag_system()
                st.session_state.initialized = True
            st.success("RAG system initialized successfully!")
        except Exception as e:
            st.session_state.initialization_error = str(e)
            st.error(f"Error initializing RAG system: {st.session_state.initialization_error}")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Prompt strategy selection
        prompt_type = st.selectbox(
            "Select Prompt Strategy",
            options=["system_prompt", "react_prompt", "chain_of_thought_prompt", "self_ask_prompt"],
            index=0
        )
        
        # Memory settings
        st.subheader("Memory Settings")
        max_short_term = st.slider("Short-term Memory Size", 1, 20, 10)
        importance_threshold = st.slider("Importance Threshold", 0.0, 1.0, 0.8)
        
        # Update memory settings
        st.session_state.memory.max_short_term = max_short_term
        st.session_state.memory.importance_threshold = importance_threshold
        
        # Clear conversation button
        if st.button("Clear Conversation"):
            st.session_state.memory.clear()
            st.session_state.conversation_history = []
            st.success("Conversation cleared!")
        
        # Reinitialize button
        if st.button("Reinitialize System"):
            st.session_state.initialized = False
            st.session_state.initialization_error = None
            st.rerun()
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # User input
        user_input = st.text_input("Ask a question about Malagasy customs codes:", key="user_input")
        
        # Submit button
        if st.button("Submit", key="submit_button"):
            if user_input:
                if not st.session_state.initialized:
                    st.error("System not initialized. Please check the configuration.")
                else:
                    try:
                        # Get response from RAG system
                        response_data = st.session_state.rag_system.invoke(user_input, prompt_type)
                        
                        # Save to memory
                        st.session_state.memory.save_context(
                            {"question": user_input},
                            {"response": response_data["response"]}
                        )
                        
                        # Update conversation history
                        st.session_state.conversation_history.append({
                            "user": user_input,
                            "assistant": response_data["response"],
                            "sources": response_data["sources"],
                            "scores": response_data["scores"],
                            "prompt_type": response_data["prompt_type"]
                        })
                        
                        # Display response
                        display_response(response_data)
                        
                    except Exception as e:
                        st.error(f"Error processing your question: {str(e)}")
    
    with col2:
        # Conversation history
        st.subheader("Conversation History")
        
        if st.session_state.conversation_history:
            for i, exchange in enumerate(st.session_state.conversation_history):
                with st.expander(f"Q&A {i+1}"):
                    st.write(f"**User:** {exchange['user']}")
                    st.write(f"**Assistant:** {exchange['assistant']}")
                    st.write(f"**Sources:** {', '.join(map(str, exchange['sources']))}")
                    st.write(f"**Strategy:** {exchange['prompt_type']}")
        else:
            st.write("No conversation history yet.")

if __name__ == "__main__":
    main()