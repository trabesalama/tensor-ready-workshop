"""
Memory management module for conversation history.
Implements both short-term and long-term memory for the RAG system.
"""

from typing import List, Dict, Any, Optional
from langchain_core.memory import BaseMemory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from pydantic import BaseModel, Field

class ConversationMemory(BaseMemory, BaseModel):
    """
    Manages conversation history with short-term and long-term memory.
    """
    
    # Définition explicite des champs Pydantic
    short_term_memory: List[BaseMessage] = Field(default_factory=list)
    long_term_memory: List[BaseMessage] = Field(default_factory=list)
    max_short_term: int = Field(default=10)
    importance_threshold: float = Field(default=0.8)
    
    class Config:
        """Configuration Pydantic"""
        arbitrary_types_allowed = True

    def __init__(self, max_short_term: int = 10, importance_threshold: float = 0.8, **data):
        """
        Initialize the conversation memory.
        
        Args:
            max_short_term: Maximum number of messages in short-term memory
            importance_threshold: Threshold for moving messages to long-term memory
        """
        # Créer un dictionnaire avec les valeurs par défaut
        defaults = {
            "short_term_memory": [],
            "long_term_memory": [],
            "max_short_term": max_short_term,
            "importance_threshold": importance_threshold
        }
        
        # Fusionner avec les données fournies
        defaults.update(data)
        
        # Initialiser avec les données fusionnées
        super().__init__(**defaults)

    def add_message(self, message: BaseMessage, importance: float = 0.0) -> None:
        """
        Add a message to the conversation memory.
        
        Args:
            message: Message to add
            importance: Importance score of the message (0.0 to 1.0)
        """
        # Add to short-term memory
        self.short_term_memory.append(message)
        
        # If importance is high enough, add to long-term memory
        if importance >= self.importance_threshold:
            self.long_term_memory.append(message)
        
        # Maintain short-term memory size
        if len(self.short_term_memory) > self.max_short_term:
            # Remove oldest message
            self.short_term_memory.pop(0)

    def get_memory(self) -> List[BaseMessage]:
        """
        Get the combined conversation history.
        
        Returns:
            List of messages from both short-term and long-term memory
        """
        return self.long_term_memory + self.short_term_memory

    def clear(self) -> None:
        """Clear all conversation history."""
        self.short_term_memory = []
        self.long_term_memory = []

    @property
    def memory_variables(self) -> List[str]:
        """Get the memory variable names."""
        return ["history"]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load memory variables.
        
        Args:
            inputs: Input dictionary
            
        Returns:
            Dictionary with memory variables
        """
        return {"history": self.get_memory()}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """
        Save conversation context.
        
        Args:
            inputs: Input dictionary with user message
            outputs: Output dictionary with AI response
        """
        # Extract messages
        user_input = inputs.get("question", "")
        ai_output = outputs.get("response", "")
        
        # Create message objects
        human_message = HumanMessage(content=user_input)
        ai_message = AIMessage(content=ai_output)
        
        # Add to memory
        self.add_message(human_message)
        self.add_message(ai_message)

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        Get formatted conversation history for display.
        
        Returns:
            List of dictionaries with user and assistant messages
        """
        history = []
        for i in range(0, len(self.short_term_memory), 2):
            if i + 1 < len(self.short_term_memory):
                user_msg = self.short_term_memory[i]
                ai_msg = self.short_term_memory[i + 1]
                history.append({
                    "user": user_msg.content,
                    "assistant": ai_msg.content
                })
        return history