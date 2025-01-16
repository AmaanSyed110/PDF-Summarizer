import streamlit as st
import pdfplumber
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
from pathlib import Path
import tempfile
import os
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Get API key and URL from environment variables
BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")

# Check if the API_KEY and BASE_URL are loaded correctly
if not API_KEY or not BASE_URL:
    raise ValueError("API_KEY or BASE_URL not found in .env file.")

# Initialize session state variables
if 'current_summaries' not in st.session_state:
    st.session_state.current_summaries = {}
if 'display_output' not in st.session_state:
    st.session_state.display_output = True
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
if 'history' not in st.session_state:
    st.session_state.history = []
if 'saved_summaries' not in st.session_state:
    st.session_state.saved_summaries = {}
if 'show_history' not in st.session_state:
    st.session_state.show_history = False
if 'current_view' not in st.session_state:
    st.session_state.current_view = 'main'
if 'settings' not in st.session_state:
    st.session_state.settings = {
        'max_summary_length': 300,
        'chunk_size': 1000
    }

# Configure OpenAI API
client = OpenAI(
    base_url=BASE_URL, 
    api_key=API_KEY
)

# Helper functions
def switch_to_history():
    st.session_state.current_view = 'history'

def switch_to_main():
    st.session_state.current_view = 'main'

def extract_text_from_pdf(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        with pdfplumber.open(tmp_file_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""
        
        os.unlink(tmp_file_path)
        return text.strip()
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return None

def split_text_into_chunks(text, max_chunk_length=1000):
    sentences = text.split('. ')
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length > max_chunk_length:
            chunks.append('. '.join(current_chunk) + '.')
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length
    
    if current_chunk:
        chunks.append('. '.join(current_chunk) + '.')
    return chunks

def generate_summary_with_gpt4_combined(text, max_length=300):
    try:
        # Split the text into chunks
        chunks = split_text_into_chunks(text)

        # Combine all chunks into a single text
        combined_text = ' '.join(chunks)

        # Generate a single summary for the combined text
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes text."},
                {"role": "user", "content": f"Summarize the following text: {combined_text}"}
            ],
            max_tokens=max_length,
            temperature=0.7
        )

        # Extract and return the summary
        summary = response.choices[0].message.content
        return summary
    except Exception as e:
        st.error(f"Error generating summary: {str(e)}")
        return None

def search_relevant_sections(text, query):
    try:
        sentences = text.split('. ')
        sentence_embeddings = st.session_state.embedding_model.encode(sentences, convert_to_tensor=True)
        query_embedding = st.session_state.embedding_model.encode(query, convert_to_tensor=True)
        
        similarities = torch.nn.functional.cosine_similarity(query_embedding.unsqueeze(0), 
                                                          sentence_embeddings)
        
        top_indices = torch.topk(similarities, min(5, len(sentences))).indices
        relevant_text = '. '.join([sentences[idx] for idx in top_indices])
        return relevant_text
    except Exception as e:
        st.error(f"Error searching relevant sections: {str(e)}")
        return None

def save_summary_history(filename, summary):
    st.session_state.history.append({
        'filename': filename,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'summary': summary
    })

def export_summaries_as_text():
    if st.session_state.history:
        export_text = "PDF SUMMARIES EXPORT\n"
        export_text += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        export_text += "=" * 50 + "\n\n"
        
        for entry in st.session_state.history:
            export_text += f"File: {entry['filename']}\n"
            export_text += f"Date: {entry['timestamp']}\n"
            export_text += "-" * 30 + "\n"
            export_text += f"{entry['summary']}\n"
            export_text += "=" * 50 + "\n\n"
        
        return export_text
    return None

def show_main_interface():
    container = st.container()
    with container:
        uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
        search_query = st.text_input("🔍 Enter keywords to focus on specific topics (optional)")
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                st.write(f"### Processing: {uploaded_file.name}")
                
                # Generate a unique key for this file and search query combination
                file_key = f"{uploaded_file.name}_{search_query}"
                
                # Only process if we haven't generated a summary yet
                if file_key not in st.session_state.current_summaries:
                    with st.spinner("Extracting text from PDF..."):
                        text = extract_text_from_pdf(uploaded_file)
                        
                    if text:
                        if search_query:
                            with st.spinner("Searching relevant sections..."):
                                text = search_relevant_sections(text, search_query)
                        
                        with st.spinner("Generating summary..."):
                            # Generate a single summary for the combined chunks
                            summary = generate_summary_with_gpt4_combined(
                                text, 
                                st.session_state.settings['max_summary_length']
                            )
                        
                        if summary:
                            # Store the summary in session state
                            st.session_state.current_summaries[file_key] = summary
                            save_summary_history(uploaded_file.name, summary)
                    else:
                        st.error(f"Failed to process {uploaded_file.name}")
                        continue
                
                # Use the stored summary
                summary = st.session_state.current_summaries[file_key]
                st.success("Summary generated successfully!")
                st.write(summary)

                col1, col2 = st.columns([1, 2])
                with col1:
                    st.write("Download Summary:")
                    
                    # Generate PDF with proper encoding
                    pdf_summary_filename = f"summary_{uploaded_file.name.split('.')[0]}.pdf"
                    try:
                        from fpdf import FPDF
                        
                        class UTF8PDF(FPDF):
                            def __init__(self):
                                super().__init__()
                                self.add_font('DejaVu', '', 'DejaVuSansCondensed.ttf', uni=True)
                            
                            def header(self):
                                pass
                            
                            def footer(self):
                                pass
                        
                        pdf = UTF8PDF()
                        pdf.add_page()
                        pdf.set_font('Arial', size=12)
                        
                        # Handle Unicode text
                        summary_clean = summary.encode('latin-1', 'replace').decode('latin-1')
                        pdf.multi_cell(0, 10, summary_clean)
                        
                        pdf_output = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                        pdf.output(pdf_output.name)
                    except Exception as e:
                        # Fallback to basic ASCII if Unicode fails
                        pdf = FPDF()
                        pdf.add_page()
                        pdf.set_font('Arial', size=12)
                        summary_ascii = ''.join(char if ord(char) < 128 else '?' for char in summary)
                        pdf.multi_cell(0, 10, summary_ascii)
                        pdf_output = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                        pdf.output(pdf_output.name)
                    
                    st.download_button(
                        label="Download as PDF",
                        data=open(pdf_output.name, "rb").read(),
                        file_name=pdf_summary_filename,
                        mime="application/pdf",
                        key=f"pdf_{file_key}"
                    )
                    
                    # Generate DOCX
                    docx_summary_filename = f"summary_{uploaded_file.name.split('.')[0]}.docx"
                    from docx import Document
                    doc = Document()
                    doc.add_heading("Summary", level=1)
                    doc.add_paragraph(summary)  # DOCX handles Unicode correctly
                    docx_output = tempfile.NamedTemporaryFile(delete=False, suffix=".docx")
                    doc.save(docx_output.name)
                    
                    st.download_button(
                        label="Download as DOCX",
                        data=open(docx_output.name, "rb").read(),
                        file_name=docx_summary_filename,
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        key=f"docx_{file_key}"
                    )
                
                # Clean up temporary files
                os.unlink(pdf_output.name)
                os.unlink(docx_output.name)

def show_history_interface():
    st.header("Summary History")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("← Back to Main"):
            switch_to_main()
            st.rerun()
    
    with col2:
        if st.session_state.history:
            export_text = export_summaries_as_text()
            if export_text:
                st.download_button(
                    label="Export All Summaries",
                    data=export_text,
                    file_name="pdf_summaries_export.txt",
                    mime="text/plain"
                )
    
    if st.session_state.history:
        for entry in st.session_state.history:
            with st.expander(f"📄 {entry['filename']} - {entry['timestamp']}", expanded=True):
                st.write(entry['summary'])
    else:
        st.info("No summaries in history yet. Process some PDFs to see them here!")

def main():
    st.set_page_config(
        page_title="Multi-PDF Summarizer",
        layout="centered",  
        initial_sidebar_state="collapsed"  
    )
    
    main_container = st.container()
    with main_container:
        st.title("📄 Multi-PDF Summarizer")
        st.subheader("Upload multiple PDFs to generate concise summaries.", divider="gray")
        
        if st.session_state.current_view == 'main':
            if st.button("📚 View Summary History", type="secondary"):
                switch_to_history()
                st.rerun()
            show_main_interface()
        else:
            show_history_interface()

if __name__ == "__main__":
    main()
