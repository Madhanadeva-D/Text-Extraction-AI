import requests
from bs4 import BeautifulSoup
from PIL import Image, UnidentifiedImageError
import pytesseract
import pdfplumber
from io import BytesIO
import re
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_image(img: Image.Image) -> None:
    """Validate image before processing"""
    if min(img.size) < 50:
        raise ValueError("Image too small (min 50x50 pixels required)")
    if img.size[0] * img.size[1] > 10_000_000:  # 10MP
        raise ValueError("Image too large (max 10MP allowed)")

def enhance_image(img: Image.Image) -> Image.Image:
    """Preprocess image for optimal OCR"""
    try:
        # Convert to grayscale
        img = img.convert('L')
        
        # Auto-rotate if needed
        img = ImageOps.exif_transpose(img)
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2.0)
        
        # Denoise and sharpen
        img = img.filter(ImageFilter.MedianFilter(size=3))
        img = img.filter(ImageFilter.SHARPEN)
        
        # Adaptive threshold
        img = img.point(lambda x: 0 if x < 140 else 255)
        
        return img
    except Exception as e:
        logger.error(f"Image enhancement failed: {str(e)}")
        raise ValueError("Image preprocessing error")

def extract_from_image(file: BytesIO, lang: str = 'eng') -> str:
    try:
        file.seek(0)
        img = Image.open(file)
        validate_image(img)
        processed_img = enhance_image(img)
        
        # Try multiple OCR configurations
        configs = [
            r'--oem 3 --psm 6',  # Assume uniform block
            r'--oem 3 --psm 11', # Sparse text
            r'--oem 3 --psm 4'   # Alternate orientations
        ]
        
        best_text = ""
        for config in configs:
            try:
                text = pytesseract.image_to_string(
                    processed_img,
                    config=f"{config} -l {lang}",
                    timeout=10
                )
                text = re.sub(r'\s+', ' ', text).strip()
                if len(text) > len(best_text):
                    best_text = text
            except Exception:
                continue
        
        if not best_text or len(best_text) < 20:
            raise ValueError("No readable text extracted (try clearer image)")
            
        logger.info(f"Extracted {len(best_text)} characters")
        return best_text
        
    except Exception as e:
        logger.error(f"Image processing failed: {str(e)}")
        raise ValueError(f"Image extraction error: {str(e)}")


def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_from_url(url: str) -> str:
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        for element in soup(['script', 'style', 'nav', 'footer', 'iframe']):
            element.decompose()
            
        return clean_text(soup.get_text(separator=' ', strip=True))
    except Exception as e:
        raise Exception(f"URL extraction error: {str(e)}")


def extract_from_pdf(file: BytesIO) -> str:
    try:
        file.seek(0)  # Reset file pointer
        with pdfplumber.open(file) as pdf:
            text = "\n".join(
                page.extract_text() or "" 
                for page in pdf.pages 
            )
        return clean_text(text)
    except Exception as e:
        raise Exception(f"PDF extraction error: {str(e)}")