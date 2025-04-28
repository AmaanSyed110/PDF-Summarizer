# PDF-Summarizer

## Overview
The **Multi-PDF Summarizer** is a Streamlit-based web application that allows users to upload multiple PDF files and generate concise summaries using OpenAI's GPT-4 model. It also supports keyword-based focus to extract relevant sections from the PDFs and uses FAISS for efficient embedding-based search.

## Flow of the Project
![diagram-export-3-29-2025-2_52_28-PM](https://github.com/user-attachments/assets/e8f28cdb-7de8-46f6-9909-3dead8dfad6b)


## Demo Video
[Multi-PDF_Summarizer.webm](https://github.com/user-attachments/assets/c5eb6e94-6701-4d23-b9c3-2bfe59439ea9)



## Features
- **Multi-PDF Support**: Upload and process multiple PDF files simultaneously.

- **AI-Powered Summarization**: Generate summaries using OpenAI's `gpt-4o` model.
  
- **Keyword-Based Focus**: Enter keywords to focus on specific topics within the PDFs.
  
- **Embedding-Based Search**: Use FAISS and OpenAI's `text-embedding-3-large` model for efficient similarity search.
  
- **Download Summaries**: Download summaries as **PDF** or **DOCX** files.
  
- **History Tracking**: View and export previously generated summaries.
  
## Tech Stack
- **Python**: The core programming language used for the entire application, including text extraction, summarization, and embedding generation.

- **streamlit**: For building the web application.

- **openai**: For interacting with OpenAI's GPT-4 and embedding models.

- **pdfplumber**: For extracting text from PDF files.

- **faiss**: For efficient similarity search using embeddings.

- **python-dotenv**: For loading environment variables from a .env file.

- **numpy**: For numerical operations.

- **pandas**: For data handling (optional, if needed for future features).

- **fpdf**: For generating PDF files from summaries.

- **python-docx**: For generating DOCX files from summaries.

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
- ### Creating a env file
Create a ```.env``` file in the root of the project with the following content:
```
OPENAI_API_KEY=<your-openai-api-key>
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


## Contributions
Contributions are welcome! Please fork the repository and create a pull request for any feature enhancements or bug fixes.
