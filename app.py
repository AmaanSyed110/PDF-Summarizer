import streamlit as st
import pdfplumber
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
from pathlib import Path
import tempfile
import os
import pandas as pd
from datetime import datetime

# Add new session state variable for storing current summaries
if 'current_summaries' not in st.session_state:
    st.session_state.current_summaries = {}
if 'display_output' not in st.session_state:
    st.session_state.display_output = True
if 'summarizer' not in st.session_state:
    st.session_state.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
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
        'max_summary_length': 150,
        'min_summary_length': 50,
        'chunk_size': 1000
    }


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

def generate_summary(text, max_length=150, min_length=50):
    try:
        chunks = split_text_into_chunks(text)
        summaries = []
        progress_bar = st.progress(0)
        
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) > 10:
                summary = st.session_state.summarizer(chunk,
                                                    max_length=max_length,
                                                    min_length=min_length,
                                                    do_sample=False)
                summaries.append(summary[0]['summary_text'])
            progress_bar.progress((i + 1) / len(chunks))
        
        progress_bar.empty()
        return ' '.join(summaries)
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
                
                with st.spinner("Extracting text from PDF..."):
                    text = extract_text_from_pdf(uploaded_file)
                    
                if text:
                    if search_query:
                        with st.spinner("Searching relevant sections..."):
                            text = search_relevant_sections(text, search_query)
                    
                    with st.spinner("Generating summary..."):
                        summary = generate_summary(
                            text, 
                            st.session_state.settings['max_summary_length'],
                            st.session_state.settings['min_summary_length']
                        )
                    
                    if summary:
                        st.success("Summary generated successfully!")
                        st.write(summary)
                        save_summary_history(uploaded_file.name, summary)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("Download Summary:")

                            # Generate PDF
                            pdf_summary_filename = f"summary_{uploaded_file.name.split('.')[0]}.pdf"
                            from fpdf import FPDF
                            pdf = FPDF()
                            pdf.add_page()
                            pdf.set_font("Arial", size=12)
                            pdf.multi_cell(0, 10, summary)
                            pdf_output = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                            pdf.output(pdf_output.name)

                            # Generate DOCX
                            docx_summary_filename = f"summary_{uploaded_file.name.split('.')[0]}.docx"
                            from docx import Document
                            doc = Document()
                            doc.add_heading("Summary", level=1)
                            doc.add_paragraph(summary)
                            docx_output = tempfile.NamedTemporaryFile(delete=False, suffix=".docx")
                            doc.save(docx_output.name)

                            # Display download buttons
                            st.download_button(
                                label="Download as PDF",
                                data=open(pdf_output.name, "rb").read(),
                                file_name=pdf_summary_filename,
                                mime="application/pdf"
                            )
                            st.download_button(
                                label="Download as DOCX",
                                data=open(docx_output.name, "rb").read(),
                                file_name=docx_summary_filename,
                                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                            )
                        
                    
                else:
                    st.error(f"Failed to process {uploaded_file.name}")

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
        layout="centered",  # Changed from "wide" to "centered"
        initial_sidebar_state="collapsed"  # Hide the sidebar by default
    )
    
    # Use container to create a more compact layout
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