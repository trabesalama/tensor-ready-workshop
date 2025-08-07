import yaml
from typing import Dict, Any
from pathlib import Path
from paths import CONFIG_DIR


class ConfigLoader:
    """Handles loading and accessing configuration settings."""

    def __init__(self, config_path: str = f"{CONFIG_DIR}/config.yaml"):
        self.config_path = Path(config_path)
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration: {e}")
    
    @property
    def model_config(self) -> Dict[str, Any]:
        return self._config.get("model", {})
    
    @property
    def text_splitter_config(self) -> Dict[str, Any]:
        return self._config.get("text_splitter", {})
    
    @property
    def vector_store_config(self) -> Dict[str, Any]:
        return self._config.get("vector_store", {})
    
    @property
    def embedding_config(self) -> Dict[str, Any]:
        return self._config.get("embedding", {})
    
    @property
    def data_directory(self) -> str:
        return self._config.get("data_directory", "./data")