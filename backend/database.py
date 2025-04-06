from pymilvus import connections, utility, Collection, CollectionSchema, FieldSchema, DataType
from langchain_community.vectorstores import Milvus
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from typing import Optional, List
import logging
import numpy as np
from .config import config

logger = logging.getLogger(__name__)

class EmbeddingWrapper:
    def __init__(self, model):
        self.model = model
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts"""
        embeddings = self.model.encode(texts)
        return [embedding.tolist() for embedding in embeddings]
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        return self.model.encode(text).tolist()

class VectorDatabase:
    def __init__(self):
        self.host = config.MILVUS_HOST
        self.port = config.MILVUS_PORT
        self.collection_name = config.COLLECTION_NAME
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        self.embedding_wrapper = EmbeddingWrapper(self.embedding_model)
        self.vector_store: Optional[Milvus] = None
        
        # Configure text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.TEXT_SPLIT_CHUNK_SIZE,
            chunk_overlap=config.TEXT_SPLIT_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def connect(self) -> None:
        """Connect to Milvus and initialize vector store"""
        try:
            # Clean up existing connections
            try:
                connections.disconnect("default")
            except Exception:
                pass
            
            # Connect to Milvus server
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port
            )
            
            # Initialize vector store with proper schema
            self.vector_store = Milvus(
                embedding_function=self.embedding_wrapper,
                collection_name=self.collection_name,
                connection_args={
                    "host": self.host,
                    "port": self.port
                },
                consistency_level="Strong",
                auto_id=True,
                index_params={
                    "index_type": "IVF_FLAT",
                    "metric_type": "L2",
                    "params": {"nlist": 128}
                },
                search_params={"metric_type": "L2", "params": {"nprobe": 10}},
                drop_old=False  # Important to keep existing data
            )
            logger.info(f"Connected to Milvus collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Connection failed: {str(e)}")
            raise

    def process_content(self, content: str, source: str) -> int:
        """Process and store content with automatic chunking"""
        try:
            if not content or not isinstance(content, str):
                raise ValueError("Content must be a non-empty string")
            
            # Split content into manageable chunks
            chunks = self.text_splitter.split_text(content)
            
            # Prepare metadata for each chunk
            metadatas = [{"source": source, "chunk_idx": i} 
                        for i in range(len(chunks))]
            
            # Store chunks in vector database
            if chunks:
                self.vector_store.add_texts(texts=chunks, metadatas=metadatas)
            
            logger.info(f"Stored {len(chunks)} chunks from source: {source}")
            return len(chunks)
            
        except Exception as e:
            logger.error(f"Content processing failed: {str(e)}")
            raise

    def get_retriever(self, k: int = 3, score_threshold: float = 0.7):
        """Create a retriever with configurable parameters"""
        if not self.vector_store:
            raise RuntimeError("Database not connected. Call connect() first.")
            
        return self.vector_store.as_retriever(
            search_kwargs={
                "k": k,
                "score_threshold": score_threshold,
                "params": {"nprobe": 10}
            }
        )

    def disconnect(self):
        """Clean up connection"""
        try:
            connections.disconnect("default")
            logger.info("Disconnected from Milvus")
        except Exception as e:
            logger.error(f"Disconnection failed: {str(e)}")

    def is_connected(self) -> bool:
        """Check connection status"""
        return self.vector_store is not None