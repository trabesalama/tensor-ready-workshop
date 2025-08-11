import os
from typing import Dict, Any, Optional
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from pydantic import SecretStr
from dotenv import load_dotenv
from .prompt_manager import PromptManager





class RAGSystem:
    """Manages the RAG (Retrieval-Augmented Generation) system."""
    
    def __init__(self, config_loader, vector_store, prompt_manager: PromptManager):
        load_dotenv()
        self.config_loader = config_loader
        self.vector_store = vector_store
        self.prompt_manager = prompt_manager
        self._llm = None
        self._rag_chain = None
    
    @property
    def llm(self):
        """Get or create LLM instance."""
        if self._llm is None:
            groq_api_key = os.getenv("GROQ_API_KEY")
            if not groq_api_key:
                raise ValueError("GROQ_API_KEY environment variable is not set.")
            
            model_config = self.config_loader.model_config
            self._llm = ChatGroq(
                api_key=SecretStr(groq_api_key),
                model=model_config.get("name", "llama-3.1-8b-instant"),
                temperature=model_config.get("temperature", 0.1)
            )
        return self._llm
    
    @property
    def rag_chain(self):
        """Get or create RAG chain."""
        if self._rag_chain is None:
            self._rag_chain = self._create_rag_chain()
        return self._rag_chain
    
    def _create_rag_chain(self):
        """Create the RAG chain with prompt template."""
        system_prompt = self.prompt_manager.get_prompt()
        
        prompt = PromptTemplate(
            template=system_prompt,
            input_variables=["context", "question"]
        )
        
        retriever = self.vector_store.get_retriever()
        
        def format_docs(docs):
            return "\n\n".join(f"Page {doc.metadata['page']}: {doc.page_content}" for doc in docs)
        
        return (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
    
    def query(self, question: str, prompt_name: Optional[str] = None) -> str:
        """Query the RAG system with a question and optional prompt."""
        # Change prompt if specified
        if prompt_name and prompt_name != self.prompt_manager.current_prompt_name:
            self.prompt_manager.set_prompt(prompt_name)
            self._rag_chain = None  # Force recreation of the chain
        
        response = self.prompt_manager.get_prompt()  + " \n" + self.rag_chain.invoke(question)
        
        # Add consulted sources
        retriever = self.vector_store.get_retriever()
        sources = retriever.invoke(question)
        source_pages = [doc.metadata['page'] for doc in sources]
        unique_pages = sorted(list(set(source_pages)))
        
        if unique_pages:
            response += f"\n\n**Sources consulted**: Page {', '.join(map(str, unique_pages))}"
        
        return response