import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path

class PromptManager:
    """Manages dynamic loading and switching of system prompts."""
    
    def __init__(self, prompts_dir: str = "Module1_project/prompts"):
        self.prompts_dir = Path(prompts_dir)
        self._prompts: Dict[str, str] = {}
        self._current_prompt: Optional[str] = None
        self._load_prompts()
    
    def _load_prompts(self) -> None:
        """Load all prompt files from the prompts directory."""
        if not self.prompts_dir.exists():
            raise FileNotFoundError(f"Prompts directory not found: {self.prompts_dir}")
        
        for prompt_file in self.prompts_dir.glob("*.yaml"):
            with open(prompt_file, 'r') as f:
                prompt_data = yaml.safe_load(f)
                prompt_name = prompt_file.stem
                self._prompts[prompt_name] = prompt_data['system_prompt']
        
        if not self._prompts:
            raise ValueError("No prompt files found in the prompts directory")
        
        # Set default prompt
        self._current_prompt = next(iter(self._prompts))
    
    @property
    def available_prompts(self) -> list:
        """Return list of available prompt names."""
        return list(self._prompts.keys())
    
    @property
    def current_prompt_name(self) -> Optional[str]:
        """Return the name of the current prompt."""
        return self._current_prompt
    
    def get_prompt(self, prompt_name: Optional[str] = None) -> str:
        """Get a prompt by name. If name is None, returns current prompt."""
        if prompt_name is None:
            prompt_name = self._current_prompt
        
        if prompt_name not in self._prompts:
            raise ValueError(f"Prompt '{prompt_name}' not found. Available prompts: {self.available_prompts}")
        
        return self._prompts[prompt_name]
    
    def set_prompt(self, prompt_name: str) -> None:
        """Set the current prompt by name."""
        if prompt_name not in self._prompts:
            raise ValueError(f"Prompt '{prompt_name}' not found. Available prompts: {self.available_prompts}")
        
        self._current_prompt = prompt_name
        print(f"Switched to prompt: {prompt_name}")
    
    def add_prompt(self, name: str, prompt_content: str) -> None:
        """Add a new prompt dynamically."""
        self._prompts[name] = prompt_content
        print(f"Added new prompt: {name}")
    
    def save_prompt(self, name: str) -> None:
        """Save the current prompt to a YAML file."""
        if name not in self._prompts:
            raise ValueError(f"Prompt '{name}' not found")
        
        prompt_file = self.prompts_dir / f"{name}.yaml"
        with open(prompt_file, 'w') as f:
            yaml.dump({"system_prompt": self._prompts[name]}, f)
        
        print(f"Saved prompt to: {prompt_file}")