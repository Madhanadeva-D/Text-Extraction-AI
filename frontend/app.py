import streamlit as st
import requests
import os
from io import BytesIO
from dotenv import load_dotenv

load_dotenv('../backend/.env')
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Document AI Query System", layout="wide")

def main():
    st.title("📄 Document AI Query System")
    
    with st.sidebar:
        st.header("📥 Load Data")
        source_type = st.radio("Select source type:", ["URL", "File"])
        
        if source_type == "URL":
            url = st.text_input("Enter URL:", value="https://example.com")
            if st.button("Load URL Data"):
                with st.spinner("Loading data from URL..."):
                    try:
                        response = requests.post(
                            f"{BACKEND_URL}/load/url",
                            json={"url": url},
                            timeout=30
                        )
                        if response.status_code == 200:
                            st.success(f"✅ Successfully loaded data!")
                        else:
                            st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
                    except Exception as e:
                        st.error(f"Connection error: {str(e)}")
        
        elif source_type == "File":
            uploaded_file = st.file_uploader(
                "Upload a file (JPG, PNG, PDF)",
                type=["pdf", "jpg", "jpeg", "png"],
                accept_multiple_files=False
            )
            
            if st.button("Load File Data") and uploaded_file is not None:
                with st.spinner("Processing file..."):
                    try:
                        file_ext = uploaded_file.name.split('.')[-1].lower()
                        
                        if file_ext in ['jpg', 'jpeg', 'png']:
                            # Preview image
                            st.image(uploaded_file, caption="Uploaded Image", width=300)
                            
                            # Add OCR confidence indicator
                            with st.expander("OCR Settings"):
                                lang = st.selectbox(
                                    "Language",
                                    ["eng", "fra", "spa", "deu"],
                                    index=0
                                )
                                contrast = st.slider("Contrast Boost", 1.0, 3.0, 2.0)
                        
                        response = requests.post(
                            f"{BACKEND_URL}/load/file?file_type={file_ext}",
                            files={"file": (uploaded_file.name, uploaded_file.getvalue())},
                            timeout=60
                        )
                        
                        if response.status_code == 200:
                            st.success("File processed successfully!")
                            if file_ext in ['jpg', 'jpeg', 'png']:
                                st.code(f"Extracted Text:\n{response.json().get('preview', '')[:500]}...", 
                                    language='text')
                        else:
                            error_detail = response.json().get('detail', '')
                            if "OCR" in error_detail:
                                st.error("⚠️ OCR failed. Try:")
                                st.markdown("""
                                - Clearer/higher resolution image
                                - Different language setting
                                - Better lighting/contrast
                                """)
                            st.error(f"Error: {error_detail}")
                            
                    except Exception as e:
                        st.error(f"Processing failed: {str(e)}")
    
    st.header("❓ Ask a Question")
    query = st.text_area("Enter your question:", height=100)
    
    if st.button("Get Answer"):
        if not query:
            st.warning("Please enter a question")
            return
            
        with st.spinner("Searching for answers..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/query",
                    json={"query": query},
                    timeout=30
                )
                if response.status_code == 200:
                    result = response.json()
                    
                    st.subheader("💡 Answer")
                    st.write(result["answer"])
                    
                    if result.get("sources"):
                        st.subheader("📚 Reference Sources")
                        for source in result["sources"]:
                            st.write(f"- {source}")
                else:
                    st.error(f"Backend error: {response.text}")
            except Exception as e:
                st.error(f"Connection error: {str(e)}")

if __name__ == "__main__":
    main()