from pymilvus import connections, Collection, utility, FieldSchema, CollectionSchema, DataType
from sentence_transformers import SentenceTransformer
from .config import config
import logging
import os
from typing import List

# Disable TensorFlow
os.environ['USE_TF'] = '0'
os.environ['USE_TORCH'] = '1'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorDB:
    def __init__(self):
        self.connect()
        self.create_collection()
        self.embedding_model = SentenceTransformer(
            config.HF_EMBEDDING_MODEL,
            device='cpu',
            cache_folder='./model_cache'
        )
        
    def connect(self):
        connections.connect("default", 
                          host=config.MILVUS_HOST, 
                          port=config.MILVUS_PORT)
        
    def create_collection(self):
        if utility.has_collection(config.COLLECTION_NAME):
            self.collection = Collection(config.COLLECTION_NAME)
            return
            
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),  # Increased max length
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=config.EMBEDDING_DIM)
        ]
        
        schema = CollectionSchema(fields, "Document chunks with embeddings")
        self.collection = Collection(config.COLLECTION_NAME, schema)
        
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 128}
        }
        self.collection.create_index("embedding", index_params)
    
    def get_embeddings(self, texts: list) -> list:
        return self.embedding_model.encode(texts).tolist()
        
    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks that fit Milvus limits"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 > config.CHUNK_SIZE:  # +1 for space
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0
            current_chunk.append(word)
            current_length += len(word) + 1
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks
        
    def insert_data(self, texts: list, source: str):
        # Process each text to ensure proper chunking
        all_chunks = []
        for text in texts:
            chunks = self.chunk_text(text)
            all_chunks.extend(chunks)
        
        if not all_chunks:
            return
            
        embeddings = self.get_embeddings(all_chunks)
        entities = [all_chunks, [source]*len(all_chunks), embeddings]
        self.collection.insert(entities)
        self.collection.flush()
        
    def search(self, query: str, top_k: int = 5) -> list:
        query_embedding = self.get_embeddings([query])[0]
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["text", "source"]
        )
        return [{"text": hit.entity.get("text"), "source": hit.entity.get("source")} 
               for hits in results for hit in hits]