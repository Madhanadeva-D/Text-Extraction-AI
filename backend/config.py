import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional

env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

class Config:
    def __init__(self):
        self.MILVUS_HOST = self._get_env_var("MILVUS_HOST", "localhost")
        self.MILVUS_PORT = self._get_env_var("MILVUS_PORT", "19530")
        self.COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_docs")
        self.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        self.MAX_TOKENS = int(os.getenv("MAX_TOKENS", 200))
        self.EMBEDDING_DIM = 384
        self.TEXT_SPLIT_CHUNK_SIZE = int(os.getenv("TEXT_SPLIT_CHUNK_SIZE", 10000))
        self.TEXT_SPLIT_OVERLAP = int(os.getenv("TEXT_SPLIT_OVERLAP", 500))
        self.DEEPSEEK_API_KEY = self._get_env_var("DEEPSEEK_API_KEY")
        self.DEBUG = os.getenv("DEBUG", "false").lower() == "true"  # Add this line

    def _get_env_var(self, var_name: str, default: Optional[str] = None) -> str:
        value = os.getenv(var_name)
        if not value and default is None:
            raise ValueError(f"{var_name} environment variable not set")
        return value or default

config = Config()