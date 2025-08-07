import unittest
from src.config_loader import ConfigLoader
from src.document_processor import DocumentProcessor
from src.embedding_manager import EmbeddingManager
from src.vector_store import VectorStore
from src.similarity_tester import SimilarityTester

class TestSimilarity(unittest.TestCase):
    """Test cases for similarity search functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.config_loader = ConfigLoader()
        cls.doc_processor = DocumentProcessor(cls.config_loader.data_directory)
        cls.embedding_manager = EmbeddingManager(cls.config_loader.embedding_config)
        cls.vector_store = VectorStore(cls.config_loader.vector_store_config, cls.embedding_manager)
        cls.similarity_tester = SimilarityTester(cls.embedding_manager, cls.vector_store)
        
        # Load and process documents
        documents = cls.doc_processor.load_documents()
        text_splitter_config = cls.config_loader.text_splitter_config
        chunks = cls.doc_processor.split_documents(
            documents, 
            text_splitter_config.get("chunk_size", 700),
            text_splitter_config.get("chunk_overlap", 200)
        )
        
        # Add to vector store
        cls.vector_store.add_documents(chunks)
    
    def test_similarity_calculation(self):
        """Test similarity calculation between query and document."""
        query = "What are the customs duties for importing vehicles?"
        docs = self.vector_store.vectorstore.similarity_search(query, k=1)
        
        if docs:
            similarity = self.similarity_tester.calculate_similarity(query, docs[0])
            self.assertGreater(similarity, 0.5, "Similarity score should be greater than 0.5")
    
    def test_retrieval_evaluation(self):
        """Test retrieval evaluation with expected results."""
        query = "What is the Gasy Net fee for imported goods?"
        expected_pages = [45, 67]  # Replace with actual expected pages
        
        evaluation = self.similarity_tester.evaluate_retrieval(query, expected_pages)
        
        self.assertGreaterEqual(evaluation["precision"], 0.5, "Precision should be at least 0.5")
        self.assertGreaterEqual(evaluation["recall"], 0.5, "Recall should be at least 0.5")
        self.assertGreaterEqual(evaluation["f1_score"], 0.5, "F1 score should be at least 0.5")
    
    def test_top_k_retrieval(self):
        """Test retrieval of top-k similar documents."""
        query = "Penalties for false declaration of origin"
        results = self.similarity_tester.test_query_similarity(query, top_k=3)
        
        self.assertEqual(len(results), 3, "Should retrieve exactly 3 documents")
        
        # Check that results are sorted by similarity (descending)
        for i in range(1, len(results)):
            self.assertGreaterEqual(
                results[i-1][1], 
                results[i][1], 
                "Results should be sorted by similarity score"
            )