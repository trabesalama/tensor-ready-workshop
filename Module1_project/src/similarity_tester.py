import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from langchain_core.documents import Document
import numpy as np
from typing import Union, List, Any
import numpy.typing as npt

class SimilarityTester:
    """Tests similarity between queries and document chunks."""
    
    def __init__(self, embedding_manager, vector_store):
        self.embedding_manager = embedding_manager
        self.vector_store = vector_store

    def calculate_cosine_similarity(self, vec1: Union[List[Any], npt.NDArray[Any]], 
                                    vec2: Union[List[Any], npt.NDArray[Any]]) -> float:
        """Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector (list or numpy array)
            vec2: Second vector (list or numpy array)
        
        Returns:
            float: Cosine similarity between the two vectors (range [-1, 1])
        """
        try:
            # Convertir en tableaux numpy de manière sécurisée
            vec1_array = np.asarray(vec1, dtype=np.float64)
            vec2_array = np.asarray(vec2, dtype=np.float64)
            
            # Vérifier que les vecteurs ont la même dimension
            if vec1_array.shape != vec2_array.shape:
                raise ValueError(f"Vectors must have the same shape. Got {vec1_array.shape} and {vec2_array.shape}")
            
            # Calculer le produit scalaire et les normes
            dot_product = np.dot(vec1_array, vec2_array)
            norm_vec1 = np.linalg.norm(vec1_array)
            norm_vec2 = np.linalg.norm(vec2_array)
            
            # Gérer le cas des vecteurs nuls
            if norm_vec1 == 0 or norm_vec2 == 0:
                return 0.0
            
            # Calculer et retourner la similarité cosinus
            cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
            
            # S'assurer que le résultat est dans la plage valide [-1, 1]
            return np.clip(cosine_similarity, -1.0, 1.0)
            
        except Exception as e:
            print(f"Error calculating cosine similarity: {e}")
            return 0.0
    
    def calculate_similarity(self, query: str, document: Document) -> float:
        """Calculate cosine similarity between query and document."""
        query_embedding = self.embedding_manager.embed_query(query)
        doc_embedding = self.embedding_manager.embed_query(document.page_content)
        
        return self.calculate_cosine_similarity(query_embedding, doc_embedding)
    
    def test_query_similarity(self, query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
        """Test similarity between query and top document chunks."""
        docs = self.vector_store.similarity_search(query, k=top_k)
        
        results = []
        for doc in docs:
            similarity = self.calculate_similarity(query, doc)
            results.append((doc, similarity))
        
        # Sort by similarity score (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def evaluate_retrieval(
        self, 
        query: str, 
        expected_pages: List[int], 
        threshold: float = 0.7,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """Evaluate retrieval quality against expected results."""
        results = self.test_query_similarity(query, top_k=top_k)
        
        relevant_docs = [
            (doc, score) for doc, score in results 
            if doc.metadata.get('page') in expected_pages and score >= threshold
        ]
        
        precision = len(relevant_docs) / len(results) if results else 0
        recall = len(relevant_docs) / len(expected_pages) if expected_pages else 0
        
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "query": query,
            "expected_pages": expected_pages,
            "retrieved_docs": [(doc.metadata.get('page'), score) for doc, score in results],
            "relevant_docs": [(doc.metadata.get('page'), score) for doc, score in relevant_docs],
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "threshold": threshold,
            "top_k": top_k
        }
    
    def batch_evaluate(
        self, 
        test_queries: List[Dict[str, Any]], 
        threshold: float = 0.7,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """Evaluate multiple test queries."""
        results = []
        total_precision = 0
        total_recall = 0
        total_f1 = 0
        
        for test_case in test_queries:
            evaluation = self.evaluate_retrieval(
                query=test_case["query"],
                expected_pages=test_case["expected_pages"],
                threshold=threshold,
                top_k=top_k
            )
            results.append(evaluation)
            total_precision += evaluation["precision"]
            total_recall += evaluation["recall"]
            total_f1 += evaluation["f1_score"]
        
        num_queries = len(test_queries)
        avg_precision = total_precision / num_queries if num_queries > 0 else 0
        avg_recall = total_recall / num_queries if num_queries > 0 else 0
        avg_f1 = total_f1 / num_queries if num_queries > 0 else 0
        
        return {
            "individual_results": results,
            "average_precision": avg_precision,
            "average_recall": avg_recall,
            "average_f1_score": avg_f1,
            "threshold": threshold,
            "top_k": top_k,
            "num_queries": num_queries
        }
    
    def find_optimal_threshold(
        self, 
        test_queries: List[Dict[str, Any]], 
        threshold_range: tuple = (0.5, 0.9, 0.05),
        top_k: int = 10
    ) -> Dict[str, Any]:
        """Find optimal similarity threshold for retrieval."""
        thresholds = np.arange(*threshold_range)
        best_threshold = 0.5
        best_f1 = 0
        
        results = {}
        
        for threshold in thresholds:
            batch_result = self.batch_evaluate(test_queries, threshold, top_k)
            results[threshold] = batch_result
            
            if batch_result["average_f1_score"] > best_f1:
                best_f1 = batch_result["average_f1_score"]
                best_threshold = threshold
        
        return {
            "best_threshold": best_threshold,
            "best_f1_score": best_f1,
            "all_results": results
        }