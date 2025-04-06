import pdfplumber
import pytesseract
import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
import logging
import validators
from typing import Union

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def extract_from_url(url: str) -> str:
    """Extract text content from a webpage URL"""
    try:
        if not validators.url(url):
            raise ValueError("Invalid URL format")
            
        headers = {
            'User-Agent': 'Mozilla/5.0',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'iframe', 'noscript']):
            element.decompose()
            
        # Get text with proper spacing
        text = ' '.join(soup.stripped_strings)
        logger.info(f"Extracted {len(text)} characters from URL: {url}")
        return text
        
    except requests.exceptions.RequestException as e:
        logger.error(f"URL request failed: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"URL extraction failed: {str(e)}")
        raise

def extract_from_image(file_stream: BytesIO) -> str:
    """Extract text from image using OCR"""
    try:
        img = Image.open(file_stream)
        
        # Enhance OCR accuracy with basic image processing
        img = img.convert('L')  # Convert to grayscale
        img = img.point(lambda x: 0 if x < 128 else 255, '1')  # Binarize
        
        text = pytesseract.image_to_string(img)
        text = text.strip()
        logger.info(f"Extracted {len(text)} characters from image")
        return text
        
    except Exception as e:
        logger.error(f"Image extraction failed: {str(e)}")
        raise

def extract_from_pdf(file_stream: BytesIO) -> str:
    """Extract text from PDF document"""
    try:
        with pdfplumber.open(file_stream) as pdf:
            text = "\n".join(
                page.extract_text() or "" 
                for page in pdf.pages
                if page.extract_text() is not None
            )
        text = text.strip()
        logger.info(f"Extracted {len(text)} characters from PDF")
        return text
        
    except Exception as e:
        logger.error(f"PDF extraction failed: {str(e)}")
        raise