# French Customs Code RAG System

A professional Retrieval-Augmented Generation (RAG) system for interpreting French customs regulations with advanced reasoning capabilities and memory management.

## Features

- **Advanced Reasoning**: Supports ReAct, Chain of Thought, and Self-Ask reasoning strategies
- **Memory Management**: Implements both short-term and long-term conversation memory
- **Optimized Embeddings**: Uses French-optimized sentence embeddings for better retrieval
- **Response Scoring**: Evaluates response quality with multiple criteria
- **Professional UI**: Streamlit-based interface with conversation history
- **Source Attribution**: Provides precise references to customs code articles and pages
- **Privacy-Focused**: Processes documents locally without external data sharing

## Architecture

The system is built with a modular architecture:
customs_rag_system/
├── config/ # Configuration management
├── data/ # PDF documents (French customs code)
├── src/ # Core modules
│ ├── document_processing.py # Document loading and chunking
│ ├── embedding_manager.py # Vector store management
│ ├── rag_chain.py # Main RAG chain implementation
│ ├── memory.py # Conversation memory
│ └── utils.py # Utility functions
├── app.py # Streamlit application
└── requirements.txt # Dependencies


## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/customs-rag-system.git
cd customs-rag-system

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# Edit .env file with your GROQ_API_KEY

Usage
1. Initialize the system:
python -c "from src.utils import initialize_system; initialize_system()"
2. Run Streamlit app:
python -c "from src.utils import initialize_system; initialize_system()"


Configuration
The system can be configured through:

Environment variables: Set in .env file
YAML configuration: In config/settings.py
UI settings: Through the Streamlit sidebar
Reasoning Strategies
The system supports three reasoning strategies:

ReAct: Combines reasoning and acting for complex problem-solving
Chain of Thought: Provides step-by-step reasoning process
Self-Ask: Uses self-questioning to break down complex queries
Memory Management
The system implements two types of memory:

Short-term memory: Maintains the last 10 conversation exchanges
Long-term memory: Stores older conversations with relevance-based retrieval
Scoring System
Responses are evaluated on four criteria:

Relevance: How well the answer addresses the question
Accuracy: Factual correctness of the information
Completeness: Coverage of all aspects of the question
Clarity: Understandability of the response
Privacy and Security
All document processing happens locally
No data is sent to external services except for API calls to Groq
Conversation memory is stored locally in JSON format
No personal or sensitive information is collected
License
This project is licensed under the MIT License - see the LICENSE file for details.