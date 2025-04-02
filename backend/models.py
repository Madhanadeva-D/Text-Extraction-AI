import os
import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from .config import config
import torch

# Disable TensorFlow completely
os.environ['USE_TF'] = '0'
os.environ['USE_TORCH'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['DISABLE_TENSORBOARD'] = '1'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GenerationModel:
    def __init__(self):
        logger.info(f"Initializing generation model: {config.HF_GENERATION_MODEL}")
        
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {self.device}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.HF_GENERATION_MODEL,
                token=config.HF_API_KEY if config.HF_API_KEY else None
            )
            
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                config.HF_GENERATION_MODEL,
                token=config.HF_API_KEY if config.HF_API_KEY else None,
                torch_dtype=torch.float32
            ).to(self.device)
            
        except Exception as e:
            logger.error(f"Model initialization failed: {str(e)}")
            raise

    def generate_response(self, query: str, context: str) -> str:
        try:
            prompt = f"Answer based on context:\nQuestion: {query}\nContext: {context}\nAnswer:"
            
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding="max_length"
            ).to(self.device)
            
            outputs = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=200,
                temperature=0.7
            )
            
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            return "Error generating response"