from pathlib import Path

# Define project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT  / "data"
CONFIG_DIR = PROJECT_ROOT / "config"
PROMPTS_DIR = PROJECT_ROOT  / "prompts"
CHROMA_DB_DIR = PROJECT_ROOT  / "chroma_db"
TESTS_DIR = PROJECT_ROOT  / "tests"

# Ensure directories exist
for directory in [DATA_DIR, CONFIG_DIR, PROMPTS_DIR, CHROMA_DB_DIR, TESTS_DIR]:
    directory.mkdir(exist_ok=True)