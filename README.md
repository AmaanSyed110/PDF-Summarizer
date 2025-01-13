# PDF-Summarizer

## Overview
A web application that allows users to upload multiple PDF files and generate concise summaries using GPT-4. Users can focus on specific sections of the text based on keywords and download the summaries in PDF or DOCX formats. The summaries are stored in history for later access and export.

## Demo Video
[Multi-PDF Summarizer.webm](https://github.com/user-attachments/assets/1c704d42-867f-4214-b699-4bfabd24443b)


## Features
- **Upload Multiple PDFs**: Users can upload multiple PDF files at once.

- **Embedding Generation**: Generate Embeddings using all-MiniLM-L6-v2 model
  
- **Summarization with GPT-4**: The application uses OpenAI's GPT-4 API to summarize the content of the PDFs.
  
- **Keyword Search**: Focus on specific topics within the PDFs using a search query.
  
- **Download Summaries**: The summaries can be downloaded in both **PDF** and **DOCX** formats.
  
- **Summary History**: View and export previous summaries generated in the app.
  
- **Export Summaries**: All summaries can be exported as a text file for external use.

## Tech Stack
- **Python**: Core programming language.
  
- **Streamlit**: For creating the interactive web interface.
  
- **OpenAI**: For generating summaries using GPT-4 model
  
- **Sentence-Transformers**: For embedding-based similarity search using `all-MiniLM-L6-v2`.
  
- **pdfplumber**: For text extraction from PDF files.
  
- **FPDF**: To create PDF downloads.
  
- **python-docx**: To create DOCX downloads.

- **python-dotenv**: To load environment variables from a ```.env``` file.

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
API_KEY=<your-openai-api-key>
BASE_URL=<your-api-base-url>
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
