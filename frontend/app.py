import streamlit as st
import requests
import os
from io import BytesIO
from PIL import Image

# Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

def main():
    st.set_page_config(page_title="Document RAG System", layout="wide")
    
    if 'documents' not in st.session_state:
        st.session_state.documents = []
    
    st.title("📚 Document RAG System")
    
    with st.sidebar:
        st.header("📥 Load Data")
        source_type = st.radio("Source Type", ["URL", "File Upload"])
        
        if source_type == "URL":
            url = st.text_input("Enter URL:")
            if st.button("Process URL"):
                with st.spinner("Processing URL..."):
                    response = requests.post(
                        f"{BACKEND_URL}/process_url/",
                        json={"url": url}
                    )
                    if response.ok:
                        st.session_state.documents.append(response.json())
                        st.success("URL processed!")
                    else:
                        st.error(f"Error: {response.text}")
        else:
            file = st.file_uploader("Upload document", type=["pdf", "png", "jpg", "jpeg"])
            if st.button("Upload File") and file:
                with st.spinner("Processing file..."):
                    files = {"file": (file.name, file.getvalue())}
                    response = requests.post(
                        f"{BACKEND_URL}/upload/",
                        files=files
                    )
                    if response.ok:
                        st.session_state.documents.append(response.json())
                        st.success("File processed!")
                    else:
                        st.error(f"Error: {response.text}")
        
        if st.session_state.documents:
            st.subheader("Loaded Documents")
            for doc in st.session_state.documents:
                st.write(f"- {doc['document_id']}")

    st.header("❓ Ask Questions")
    question = st.text_area("Enter your question:")
    
    if st.button("Get Answer") and question:
        with st.spinner("Searching documents..."):
            response = requests.post(
                f"{BACKEND_URL}/query/",
                json={"question": question}
            )
            
            if response.ok:
                result = response.json()
                st.subheader("Answer")
                st.write(result["answer"])
                
                if result["sources"]:
                    st.subheader("Sources")
                    for source in result["sources"]:
                        st.write(f"- {source}")
                
                st.write(f"Confidence: {result['confidence']*100:.1f}%")
            else:
                st.error(f"Error: {response.text}")

if __name__ == "__main__":
    main()


# python -m streamlit run app.py