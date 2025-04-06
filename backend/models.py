from fastapi import HTTPException
import requests
import logging
from typing import Any, Dict, List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import LLMResult, Generation
from langchain_core.callbacks import CallbackManagerForLLMRun
from .config import config

logger = logging.getLogger(__name__)

class DeepSeekOpenRouterLLM(BaseLLM):
    """Custom LLM class for DeepSeek through OpenRouter API"""
    
    api_key: str
    model: str = "deepseek/deepseek-chat-v3-0324:free"
    temperature: float = 0.7
    max_tokens: int = 200
    timeout: int = 30
    base_url: str = "https://openrouter.ai/api/v1"
    
    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """
        Generate text from prompts using DeepSeek via OpenRouter API
        
        Args:
            prompts: List of prompt strings
            stop: Optional list of stop words
            run_manager: Callback manager for LLM run
            **kwargs: Additional generation parameters
            
        Returns:
            LLMResult containing generated texts
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://yourdomain.com",
            "X-Title": "RAG Application"
        }
        
        generations = []
        for prompt in prompts:
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": kwargs.get('temperature', self.temperature),
                "max_tokens": min(kwargs.get('max_tokens', self.max_tokens), config.MAX_TOKENS),
                "stop": stop
            }
            
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                # Extract content and format for LangChain
                content = response.json()["choices"][0]["message"]["content"]
                generations.append([Generation(text=content)])
                
            except requests.exceptions.HTTPError as e:
                error_msg = f"DeepSeek API request failed: {str(e)}"
                if e.response.status_code == 400:
                    try:
                        error_data = e.response.json()
                        error_msg = f"Model error: {error_data.get('error', {}).get('message', error_msg)}"
                    except:
                        pass
                logger.error(error_msg)
                raise HTTPException(status_code=502, detail=error_msg)
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}", exc_info=True)
                raise
        
        return LLMResult(generations=generations)
    
    def _llm_type(self) -> str:
        """Return type of LLM"""
        return "deepseek-openrouter"

class Generator:
    """Handles RAG chain generation with DeepSeek LLM"""
    
    def __init__(self):
        self.llm = DeepSeekOpenRouterLLM(
            api_key=config.DEEPSEEK_API_KEY,
            max_tokens=config.MAX_TOKENS
        )
        
        # Define the prompt template
        self.prompt = ChatPromptTemplate.from_template(
            """Answer the question based only on the following context:
            {context}
            
            Question: {question}
            
            Guidelines:
            1. Provide a concise answer in 2-3 sentences
            2. If the answer isn't in the context, say:
               "I couldn't find that information in the provided documents"
            3. Always cite sources when available
            4. Be factual and avoid speculation
            
            Answer:"""
        )
        
        self.rag_chain = None

    def init_rag_chain(self, retriever) -> None:
        """
        Initialize the RAG chain with a retriever
        
        Args:
            retriever: Vector store retriever instance
        """
        if retriever is None:
            raise ValueError("Retriever cannot be None")
            
        self.rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | self.prompt 
            | self.llm
            | StrOutputParser()
        )

    def generate(self, query: str) -> str:
        """
        Generate an answer to a query using the RAG chain
        
        Args:
            query: The question to answer
            
        Returns:
            Generated answer string
        """
        if not self.rag_chain:
            raise RuntimeError("RAG chain not initialized. Call init_rag_chain() first.")
        
        try:
            return self.rag_chain.invoke(query)
        except requests.exceptions.HTTPError as e:
            logger.error(f"API request failed: {e.response.text}")
            raise HTTPException(
                status_code=502,
                detail="Failed to communicate with AI service"
            )
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail="Failed to generate answer"
            )