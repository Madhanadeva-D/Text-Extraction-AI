import os
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent/'.env'
load_dotenv(dotenv_path=env_path)

class Config:
    def __init__(self):
        self.MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
        self.MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
        self.COLLECTION_NAME = os.getenv("COLLECTION_NAME", "document_chunks")
        self.HF_API_KEY = os.getenv("HF_API_KEY", "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")
        self.HF_EMBEDDING_MODEL = os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.HF_GENERATION_MODEL = os.getenv("HF_GENERATION_MODEL", "google/flan-t5-base")
        self.CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
        self.CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
        self.BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
        self.EMBEDDING_DIM = 384  # Dimension for all-MiniLM-L6-v2

config = Config()