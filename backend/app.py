from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import List, Optional
import logging
import json
import requests
from .config import config
from .models import Generator
from .database import VectorDatabase
from .document_processor import extract_from_url, extract_from_pdf, extract_from_image

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize components
db = VectorDatabase()
generator = Generator()

app = FastAPI(title="RAG Backend API")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class UrlRequest(BaseModel):
    url: HttpUrl

class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3

class DocumentResponse(BaseModel):
    status: str
    message: str
    document_id: str
    chunks: Optional[int] = None

class AnswerResponse(BaseModel):
    answer: str
    sources: List[str]
    confidence: float

# Startup/Shutdown
@app.on_event("startup")
async def startup_event():
    try:
        db.connect()
        generator.init_rag_chain(db.get_retriever())
        logger.info("Application startup completed")
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    db.disconnect()

# Helper Functions
def process_content(content: str, source: str) -> int:
    return db.process_content(content, source)

# API Endpoints
@app.post("/process_url/")
async def process_url(request: UrlRequest) -> DocumentResponse:
    try:
        logger.info(f"Processing URL: {request.url}")
        content = extract_from_url(str(request.url))
        chunks = process_content(content, f"url:{request.url}")
        
        return DocumentResponse(
            status="success",
            message="URL processed successfully",
            document_id=str(request.url),
            chunks=chunks
        )
    except Exception as e:
        logger.error(f"URL processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)) -> DocumentResponse:
    try:
        logger.info(f"Processing file: {file.filename}")
        
        if file.filename.lower().endswith('.pdf'):
            content = extract_from_pdf(file.file)
            source_type = "pdf"
        else:
            content = extract_from_image(file.file)
            source_type = "image"
        
        chunks = process_content(content, f"{source_type}:{file.filename}")
        
        return DocumentResponse(
            status="success",
            message="File processed successfully",
            document_id=file.filename,
            chunks=chunks
        )
    except Exception as e:
        logger.error(f"File upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/")
async def process_query(request: Request):
    try:
        # Parse JSON body
        try:
            body = await request.json()
            question = body.get("question", "").strip()
            if not question:
                raise HTTPException(status_code=400, detail="Question is required")
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON format")
        
        logger.info(f"Processing query: {question}")
        
        try:
            # Process the question
            answer = generator.generate(question)
            retriever = db.get_retriever()
            docs = retriever.invoke(question)  # Use invoke() instead of get_relevant_documents()
            
            sources = list(set([
                doc.metadata.get("source", "unknown") 
                for doc in docs 
                if hasattr(doc, 'metadata')
            ]))
            
            response_data = {
                "answer": answer,
                "sources": sources,
                "confidence": min(0.99, len(docs)/3),
            }
            
            # Only include debug info if DEBUG is True
            if hasattr(config, 'DEBUG') and config.DEBUG:
                response_data["relevant_documents"] = [
                    {
                        "content": doc.page_content,
                        "source": doc.metadata.get("source", "unknown")
                    } 
                    for doc in docs
                ]
            
            return response_data
            
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}", exc_info=True)
            retriever = db.get_retriever()
            docs = retriever.invoke(question)
            sources = list(set([
                doc.metadata.get("source", "unknown") 
                for doc in docs 
                if hasattr(doc, 'metadata')
            ]))
            
            return {
                "answer": "I couldn't generate a response, but here are relevant documents:",
                "sources": sources,
                "confidence": 0.0
            }
            
    except Exception as e:
        logger.error(f"Query processing failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/test_openrouter/")
async def test_openrouter(request: Request):
    try:
        body = await request.json()
        question = body.get("question", "").strip()
        
        headers = {
            "Authorization": f"Bearer {config.DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://yourdomain.com",
            "X-Title": "RAG Application"
        }
        
        payload = {
            "model": "deepseek/deepseek-chat-v3-0324:free",
            "messages": [{"role": "user", "content": question}],
            "temperature": 0.7,
            "max_tokens": config.MAX_TOKENS
        }
        
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload
        )
        
        return response.json()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# python -m uvicorn backend.app:app --reload 