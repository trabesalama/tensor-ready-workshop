"""
RAG (Retrieval-Augmented Generation) system module.
Combines document retrieval with language model generation.
"""

from typing import Dict, Any, List, Tuple
from langchain_core.runnables import RunnablePassthrough, RunnableSerializable
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from pydantic import SecretStr

class RAGSystem:
    """
    Retrieval-Augmented Generation system for customs code interpretation.
    
    Attributes:
        retriever: Document retriever
        prompt_manager: Prompt manager for different strategies
        llm: Language model for generation
        vectorstore: Chroma vectorstore for direct access
        chain: RAG chain
    """
    
    def __init__(self, retriever, prompt_manager, llm, vectorstore):
        """
        Initialize the RAG system.
        
        Args:
            retriever: Document retriever
            prompt_manager: Prompt manager instance
            llm: Language model
            vectorstore: Chroma vectorstore for direct access
        """
        self.retriever = retriever
        self.prompt_manager = prompt_manager
        self.llm = llm
        self.vectorstore = vectorstore  # Stocker le vectorstore
        self.chain = self._build_chain()

    def _build_chain(self) -> RunnableSerializable[Dict[str, Any], str]:
        """
        Build the RAG chain using the retriever and language model.
        
        Returns:
            RAG chain runnable
        """
        rag_chain = (
            {"context": self.retriever | self._format_docs, "question": RunnablePassthrough()}
            | self.prompt_manager.get_prompt("system_prompt")
            | self.llm
            | StrOutputParser()
        )
        return rag_chain

    def _format_docs(self, docs: List[Document]) -> str:
        """
        Format documents for inclusion in the prompt.
        
        Args:
            docs: List of documents
            
        Returns:
            Formatted string with document content
        """
        formatted_docs = []
        for doc in docs:
            formatted_docs.append(
                f"Page {doc.metadata['page']}: {doc.page_content}"
            )
        return "\n\n".join(formatted_docs)

    def _format_docs_with_scores(self, docs_with_scores: List[Tuple[Document, float]]) -> str:
        """
        Format documents with their relevance scores for inclusion in the prompt.
        
        Args:
            docs_with_scores: List of (document, score) tuples
            
        Returns:
            Formatted string with document content and scores
        """
        formatted_docs = []
        for doc, score in docs_with_scores:
            formatted_docs.append(
                f"Page {doc.metadata['page']} (Score: {score:.2f}): {doc.page_content}"
            )
        return "\n\n".join(formatted_docs)

    def invoke(self, question: str, prompt_type: str = "system_prompt") -> Dict[str, Any]:
        """
        Invoke the RAG system with a question.
        
        Args:
            question: Question to answer
            prompt_type: Type of prompt to use
            
        Returns:
            Dictionary with response and metadata
        """
        # Get relevant documents with scores directly from vectorstore
        search_kwargs = self.retriever.search_kwargs
        k = search_kwargs.get('k', 10)
        score_threshold = search_kwargs.get('score_threshold', 0.7)
        
        # Récupérer les documents avec leurs scores (sans seuil)
        docs_with_scores = self.vectorstore.similarity_search_with_score(question, k=k)
        
        # Filtrer manuellement les documents selon le seuil
        # Note: Dans ChromaDB, plus le score est bas, plus le document est pertinent
        filtered_docs_with_scores = [
            (doc, score) for doc, score in docs_with_scores 
            if score <= score_threshold
        ]
        
        # Si aucun document ne passe le seuil, on garde tous les documents
        if not filtered_docs_with_scores:
            filtered_docs_with_scores = docs_with_scores
        
        # Séparer les documents et les scores
        docs = [doc for doc, _ in filtered_docs_with_scores]
        scores = [score for _, score in filtered_docs_with_scores]
        
        # Formater le contexte avec les scores
        context = self._format_docs_with_scores(filtered_docs_with_scores)
        
        # Obtenir le prompt formaté
        prompt = self.prompt_manager.get_formatted_prompt(prompt_type, context, question)
        
        # Générer la réponse
        response = self.llm.invoke(prompt)
        
        # Extraire les numéros de page uniques des sources
        source_pages = sorted(list(set(doc.metadata['page'] for doc in docs)))
        
        return {
            "response": response.content,
            "sources": source_pages,
            "scores": scores,
            "prompt_type": prompt_type
        }

    @staticmethod
    def create_llm(api_key: str, model_name: str, temperature: float) -> ChatGroq:
        """
        Create and configure the language model.
        
        Args:
            api_key: API key for the language model service
            model_name: Name of the model to use
            temperature: Temperature parameter for generation
            
        Returns:
            Configured language model instance
        """
        return ChatGroq(
            api_key=SecretStr(api_key),
            model=model_name,
            temperature=temperature
        )