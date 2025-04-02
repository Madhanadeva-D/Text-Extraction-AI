from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from io import BytesIO
from .document_processor import extract_from_url, extract_from_image, extract_from_pdf
from .database import VectorDB
from .models import GenerationModel
from .config import config
import os

# Disable TensorFlow before any other imports
os.environ['USE_TF'] = '0'
os.environ['USE_TORCH'] = '1'

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

db = VectorDB()
generation_model = GenerationModel()

class URLRequest(BaseModel):
    url: str

class QueryRequest(BaseModel):
    query: str

@app.post("/load/url")
async def load_url(request: URLRequest):
    try:
        text = extract_from_url(request.url)
        db.insert_data([text], request.url)
        return {"status": "success", "chunks": 1}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/load/file")
async def load_file(file_type: str, file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        
        # Verify file type
        file_ext = file.filename.split('.')[-1].lower()
        if file_ext not in ['jpg', 'jpeg', 'png', 'pdf']:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
        # Process based on type
        if file_ext in ['jpg', 'jpeg', 'png']:
            try:
                text = extract_from_image(BytesIO(file_bytes))
                if len(text) < 20:  # Minimum text threshold
                    raise HTTPException(
                        status_code=422,
                        detail="Insufficient text extracted from image"
                    )
            except ValueError as e:
                raise HTTPException(
                    status_code=422,
                    detail=f"Image processing failed: {str(e)}"
                )
        elif file_ext == 'pdf':
            text = extract_from_pdf(BytesIO(file_bytes))
        
        # Insert to database
        db.insert_data([text], file.filename)
        
        return {
            "status": "success",
            "chunks": len(text.split('\n')),
            "source": file.filename
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"File processing error: {str(e)}"
        )

@app.post("/query")
async def query(request: QueryRequest):
    try:
        results = db.search(request.query)
        if not results:
            return {"answer": "No relevant information found", "sources": []}
            
        context = "\n\n".join(f"From {r['source']}:\n{r['text']}" for r in results)
        answer = generation_model.generate_response(request.query, context)
        
        return {
            "answer": answer,
            "sources": list(set(r['source'] for r in results))
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def status():
    return {
        "status": "OK",
        "collection": db.collection.name,
        "chunks": db.collection.num_entities
    }

@app.post("/process-image")
async def process_image(file: UploadFile = File(...)):
    try:
        # Validate file type
        if file.content_type not in ['image/jpeg', 'image/png']:
            raise HTTPException(400, "Only JPEG/PNG images allowed")
        
        # Save original for debugging
        file_bytes = await file.read()
        with open("last_upload.jpg", "wb") as f:
            f.write(file_bytes)
        
        # Process image
        text = extract_from_image(BytesIO(file_bytes))
        
        return JSONResponse({
            "status": "success",
            "text": text,
            "length": len(text)
        })
        
    except ValueError as e:
        raise HTTPException(422, detail=str(e))
    except Exception as e:
        raise HTTPException(500, detail=f"Processing error: {str(e)}")

@app.get("/debug-last-image")
async def debug_last_image():
    """Endpoint to view last uploaded image"""
    if not os.path.exists("last_upload.jpg"):
        raise HTTPException(404, "No image uploaded yet")
    return FileResponse("last_upload.jpg")


# python -m uvicorn backend.app:app --reload 
 