# Customs Code RAG System

AI System for interpreting Malagasy customs codes using Retrieval-Augmented Generation (RAG) architecture with FastAPI backend and Streamlit frontend.

## Features

- Dynamic prompt management
- Similarity search testing
- FastAPI backend with REST API
- Interactive Streamlit frontend

## Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Create a `.env` file based on `.env.example`
4. Add your PDF documents to the `data/` directory

## Running the Application

### Start the API Server
```bash
cd api
python main.py