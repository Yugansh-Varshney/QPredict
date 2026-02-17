# 🔮 QPredict

> **AI-Powered Exam Prediction & Context Retrieval System**

## 📖 Overview
QPredict uses RAG (Retrieval-Augmented Generation) to help university students:
1. **Chat with Previous Year Questions (PYQs):** Get context-aware answers from past exams.
2. **Predict Exam Trends:** Analyze 5+ years of data to forecast high-probability topics.

## 🚀 Setup
1. `pip install -r requirements.txt`
2. `python setup_folders.py` (Generates the semester folder tree)
3. `python src/ingestion.py` (Reads PDFs and builds the database)
4. `streamlit run src/app.py` (Starts the dashboard)