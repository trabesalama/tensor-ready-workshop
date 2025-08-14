"""
Prompt management module for different prompting strategies.
Handles loading and formatting of various prompt templates.
"""

import yaml
from typing import Dict, Any, Optional
from langchain_core.prompts import PromptTemplate

class PromptManager:
    """
    Manages different prompt templates and strategies.
    
    Attributes:
        prompts (Dict[str, str]): Dictionary of prompt templates
    """
    
    def __init__(self, prompts_file: str):
        """
        Initialize the PromptManager by loading prompts from a YAML file.
        
        Args:
            prompts_file: Path to the YAML file containing prompt templates
        """
        self.prompts = self._load_prompts(prompts_file)

    def _load_prompts(self, prompts_file: str) -> Dict[str, str]:
        """
        Load prompt templates from a YAML file.
        
        Args:
            prompts_file: Path to the YAML file
            
        Returns:
            Dictionary of prompt templates
            
        Raises:
            FileNotFoundError: If the prompts file is not found
            yaml.YAMLError: If the YAML file is invalid
        """
        try:
            with open(prompts_file, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompts file not found: {prompts_file}")
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing prompts YAML: {e}")

    def get_prompt(self, prompt_type: str) -> PromptTemplate:
        """
        Get a specific prompt template by type.
        
        Args:
            prompt_type: Type of prompt to retrieve
            
        Returns:
            PromptTemplate instance
            
        Raises:
            ValueError: If the prompt type is not found
        """
        if prompt_type not in self.prompts:
            raise ValueError(f"Prompt type '{prompt_type}' not found in prompts file.")
        return PromptTemplate(
            template=self.prompts[prompt_type],
            input_variables=["context", "question"]
        )

    def get_formatted_prompt(self, prompt_type: str, context: str, question: str) -> str:
        """
        Get a formatted prompt with context and question.
        
        Args:
            prompt_type: Type of prompt to retrieve
            context: Context information to include
            question: Question to answer
            
        Returns:
            Formatted prompt string
        """
        prompt_template = self.get_prompt(prompt_type)
        return prompt_template.format(context=context, question=question)