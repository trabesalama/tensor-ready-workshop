from src.config_loader import ConfigLoader
from src.document_processor import DocumentProcessor
from src.embedding_manager import EmbeddingManager
from src.vector_store import VectorStore
from src.rag_system import RAGSystem
from src.similarity_tester import SimilarityTester
from src.prompt_manager import PromptManager

def main():
    """Main function to run the RAG system."""
    # Initialize components
    config_loader = ConfigLoader()
    doc_processor = DocumentProcessor(config_loader.data_directory)
    embedding_manager = EmbeddingManager(config_loader.embedding_config)
    vector_store = VectorStore(config_loader.vector_store_config, embedding_manager)
    prompt_manager = PromptManager()  # Initialize prompt manager
    
    # Load and process documents
    documents = doc_processor.load_documents()
    text_splitter_config = config_loader.text_splitter_config
    chunks = doc_processor.split_documents(
        documents, 
        text_splitter_config.get("chunk_size", 700),
        text_splitter_config.get("chunk_overlap", 200)
    )
    
    # Add to vector store
    vector_store.add_documents(chunks)
    
    # Initialize RAG system with prompt manager
    rag_system = RAGSystem(config_loader, vector_store, prompt_manager)
    
    # Show available prompts
    print("Available prompts:", prompt_manager.available_prompts)
    print("Current prompt:", prompt_manager.current_prompt_name)
    
    # Example queries with different prompts
    queries = [
        "Quel est le taux de la prestation de Gasy Net pour les marchandises importées?",
        "How to declare pharmaceutical products under temporary admission regime?",
        "What penalties apply for false declaration of origin?"
    ]
    
    # Process queries with default prompt
    print("\n--- Using default prompt ---")
    for query in queries:
        print(f"\nQuery: {query}")
        response = rag_system.query(query)
        print(response)
    
    # Switch to technical prompt and reprocess
    if 'technical' in prompt_manager.available_prompts:
        print("\n--- Switching to technical prompt ---")
        for query in queries:
            print(f"\nQuery: {query}")
            response = rag_system.query(query, prompt_name='technical')
            print(response)
    
    # Switch to simplified prompt and reprocess
    if 'simplified' in prompt_manager.available_prompts:
        print("\n--- Switching to simplified prompt ---")
        for query in queries:
            print(f"\nQuery: {query}")
            response = rag_system.query(query, prompt_name='simplified')
            print(response)
    
    # Test similarity
    similarity_tester = SimilarityTester(embedding_manager, vector_store)
    test_query = "What are the customs duties for importing vehicles?"
    results = similarity_tester.test_query_similarity(test_query, top_k=3)
    
    print("\nSimilarity Test Results:")
    for doc, score in results:
        print(f"Page {doc.metadata.get('page')}: Similarity = {score:.4f}")
        print(f"Content: {doc.page_content[:100]}...\n")

if __name__ == "__main__":
    main()
