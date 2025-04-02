# Text-Extraction-AI

## Overview
A comprehensive AI-powered system for extracting, processing, and querying text content from multiple sources including URLs, images (JPG/PNG), and PDF documents. The system transforms unstructured data into searchable knowledge using vector embeddings and provides intelligent question answering through generative AI.

## Features

### Data Extraction
- **Web Content**: Scrape and clean text from URLs using BeautifulSoup
- **Image OCR**: Extract text from images using Tesseract with advanced preprocessing
- **PDF Processing**: Parse PDF documents while preserving layout structure using pdfplumber

### Intelligent Search & Q&A
- **Vector Embeddings**: Transform text into embeddings using Sentence Transformers
- **Semantic Search**: Milvus vector database for efficient similarity searches
- **AI Responses**: Generate answers using OpenAI GPT or Hugging Face Transformers

### User Interface
- **Interactive Dashboard**: Streamlit-based web interface
- **Multi-format Upload**: Support for URLs, images, and PDFs
- **Query Interface**: Natural language question answering

## System Architecture

```mermaid
graph TD
    subgraph Frontend
        A[Streamlit]
    end

    subgraph Backend
        B[FastAPI]
        C[Config]
        D[Database]
        E[Document Processor]
        F[AI Models]
    end

    subgraph Infrastructure
        G[Milvus]
        H[Model Cache]
        I[Docker]
    end

    A --> B
    B --> C
    B --> D --> G
    B --> E
    B --> F --> H
    I --> G
    I --> H
