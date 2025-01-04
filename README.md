# PDF-Summarizer

## Overview
A Streamlit-based web application that allows users to upload multiple PDF files, extract their text, and generate concise summaries. The app uses state-of-the-art NLP models for summarization and enables keyword-based focused summaries for more relevant insights. Summaries can be downloaded in PDF or DOCX format, and a history of generated summaries is maintained.

## Demo Video
[Multi-PDF Summarizer.webm](https://github.com/user-attachments/assets/1c704d42-867f-4214-b699-4bfabd24443b)


## Features
- **Multi-PDF Upload**: Upload and process multiple PDF files simultaneously.
  
- **Advanced Summarization**: Generate concise summaries using the [BART-large-cnn model](https://huggingface.co/facebook/bart-large-cnn).
  
- **Keyword-Based Focus**: Focus on specific sections by entering keywords.
  
- **Download Options**: Export summaries as **PDF** or **DOCX** files.
  
- **History Management**: View and export past summaries for quick reference.
  
- **Cosine Similarity Search**: Identify and prioritize sections most relevant to your query.

## Tech Stack
- **Python**: Core programming language.
  
- **Streamlit**: For creating the interactive web interface.
  
- **Hugging Face Transformers**: For summarization using the `facebook/bart-large-cnn` model.
  
- **Sentence-Transformers**: For embedding-based similarity search.
  
- **pdfplumber**: For text extraction from PDF files.
  
- **FPDF**: To create PDF downloads.
  
- **python-docx**: To create DOCX downloads.

## Steps to Run the MultiPDF-RAG Project on Your Local Machine:
- ### Clone the Repository
Open a terminal and run the following command to clone the repository:

```
git clone https://github.com/AmaanSyed110/PDF-Summarizer.git
```
- ### Set Up a Virtual Environment
It is recommended to use a virtual environment for managing dependencies:

```
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate     # On Windows
```
- ### Install Dependencies
Install the required packages listed in the ```requirements.txt``` file
```
pip install -r requirements.txt
```

- ### Run the Application
Launch the Streamlit app by running the following command:
```
streamlit run app.py
```
- ### Upload PDF Documents
Use the web interface to upload PDF files.

- ### Results
View the generated summaries in the main interface.

- ### Downloads
Download summaries in your preferred format (PDF or DOCX).

- ### History
Access past summaries in the **Summary History** section.
