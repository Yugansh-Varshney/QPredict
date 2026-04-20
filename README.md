# QPredict

AI-Powered Exam Prediction & Context Retrieval System utilizing RAG.

## Overview
QPredict is a Retrieval-Augmented Generation (RAG) tool designed to assist university students. It processes previous university examination papers to:
- Answer context-aware queries about past exams.
- Aggregate and predict high-frequency topics based on historical data.

## Tech Stack
- **LangChain & HuggingFace Embeddings**: For document chunking and local vectorization (`all-MiniLM-L6-v2`).
- **ChromaDB**: Embedded vector database for document storage.
- **Streamlit**: Interactive web dashboard.
- **Google Gemini API**: Large Language Model logic and analysis.
- **Apple Vision OCR**: Native OCR extraction for scanned PDF documents.

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Initialize the project structure:
   ```bash
   python setup.py
   ```

3. Setup environment variables:
   Provide your API key in a `.env` file at the root.
   ```
   GOOGLE_API_KEY=your_gemini_api_key
   ```

4. Add sample papers:
   Place your PDF materials in the respective `data/Semester_X/Subject_Name` directories.

5. Ingest data:
   ```bash
   python src/ingestion.py
   ```

6. Launch the dashboard:
   ```bash
   streamlit run src/app.py
   ```