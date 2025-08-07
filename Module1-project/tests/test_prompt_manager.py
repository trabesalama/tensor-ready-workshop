import unittest
import tempfile
import os
import yaml
from src.prompt_manager import PromptManager

class TestPromptManager(unittest.TestCase):
    """Test cases for PromptManager."""
    
    def setUp(self):
        """Set up test environment with temporary prompt files."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.prompts_dir = os.path.join(self.temp_dir.name, "prompts")
        os.makedirs(self.prompts_dir)
        
        # Create test prompt files
        self.test_prompts = {
            "default": {
                "system_prompt": "Default prompt with {context} and {question}"
            },
            "technical": {
                "system_prompt": "Technical prompt with {context} and {question}"
            }
        }
        
        for name, content in self.test_prompts.items():
            with open(os.path.join(self.prompts_dir, f"{name}.yaml"), 'w') as f:
                yaml.dump(content, f)
        
        self.prompt_manager = PromptManager(self.prompts_dir)
    
    def tearDown(self):
        """Clean up temporary directory."""
        self.temp_dir.cleanup()
    
    def test_load_prompts(self):
        """Test that prompts are loaded correctly."""
        self.assertEqual(len(self.prompt_manager.available_prompts), 2)
        self.assertIn("default", self.prompt_manager.available_prompts)
        self.assertIn("technical", self.prompt_manager.available_prompts)
    
    def test_get_prompt(self):
        """Test getting prompts by name."""
        default_prompt = self.prompt_manager.get_prompt("default")
        self.assertIn("Default prompt", default_prompt)
        
        technical_prompt = self.prompt_manager.get_prompt("technical")
        self.assertIn("Technical prompt", technical_prompt)
    
    def test_set_prompt(self):
        """Test setting the current prompt."""
        self.prompt_manager.set_prompt("technical")
        self.assertEqual(self.prompt_manager.current_prompt_name, "technical")
        
        current_prompt = self.prompt_manager.get_prompt()
        self.assertIn("Technical prompt", current_prompt)
    
    def test_add_prompt(self):
        """Test adding a new prompt dynamically."""
        new_prompt = "New prompt with {context} and {question}"
        self.prompt_manager.add_prompt("new", new_prompt)
        
        self.assertIn("new", self.prompt_manager.available_prompts)
        self.assertEqual(self.prompt_manager.get_prompt("new"), new_prompt)
    
    def test_save_prompt(self):
        """Test saving a prompt to file."""
        self.prompt_manager.save_prompt("default")
        prompt_file = os.path.join(self.prompts_dir, "default.yaml")
        
        self.assertTrue(os.path.exists(prompt_file))
        
        with open(prompt_file, 'r') as f:
            saved_data = yaml.safe_load(f)
            self.assertEqual(saved_data["system_prompt"], self.test_prompts["default"]["system_prompt"])
