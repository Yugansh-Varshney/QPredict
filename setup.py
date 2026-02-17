import os

# --- Configuration ---
PROJECT_NAME = "QPredict"

# Define the directory structure
STRUCTURE = {
    "src": ["__init__.py", "ingestion.py", "retrieval.py", "app.py", "utils.py"],
    "notebooks": ["experiment.ipynb"],
    "tests": ["__init__.py", "test_ingestion.py"],
    "data": [],         # Local data storage (GitIgnored)
    "chroma_db": [],    # Local Vector DB storage (GitIgnored)
}

# Define file contents
GITIGNORE_CONTENT = """
# Data & Database (Too large for Git)
data/
chroma_db/
*.pdf
*.png
*.jpg

# Python & Environment
__pycache__/
*.py[cod]
venv/
.env
.DS_Store
"""

README_CONTENT = f"""
# 🔮 {PROJECT_NAME}

> **AI-Powered Exam Prediction & Context Retrieval System**

## 📖 Overview
{PROJECT_NAME} uses RAG (Retrieval-Augmented Generation) to help university students:
1. **Chat with Previous Year Questions (PYQs):** Get context-aware answers from past exams.
2. **Predict Exam Trends:** Analyze 5+ years of data to forecast high-probability topics.

## 🚀 Setup
1. `pip install -r requirements.txt`
2. `python setup_folders.py` (Generates the semester folder tree)
3. `python src/ingestion.py` (Reads PDFs and builds the database)
4. `streamlit run src/app.py` (Starts the dashboard)
"""

REQUIREMENTS_CONTENT = """
langchain
langchain-community
langchain-huggingface
chromadb
pypdf
sentence-transformers
streamlit
python-dotenv
pandas
numpy
"""

def create_qpredict_structure():
    print(f"🚀 Initializing {PROJECT_NAME} Repository...\n")

    # 1. Create Directories & Files
    for folder, files in STRUCTURE.items():
        os.makedirs(folder, exist_ok=True)
        print(f"📁 Created: {folder}/")
        
        for file in files:
            file_path = os.path.join(folder, file)
            if not os.path.exists(file_path):
                with open(file_path, "w") as f:
                    f.write("") # Create empty file
                print(f"   📄 Created: {file}")

    # 2. Create Root Config Files
    
    # .gitignore (Critical for safety)
    with open(".gitignore", "w") as f:
        f.write(GITIGNORE_CONTENT.strip())
    print("🛡️  Created .gitignore")

    # README.md
    with open("README.md", "w") as f:
        f.write(README_CONTENT.strip())
    print("📄 Created README.md")

    # requirements.txt
    with open("requirements.txt", "w") as f:
        f.write(REQUIREMENTS_CONTENT.strip())
    print("📦 Created requirements.txt")
    
    # .env template
    with open(".env.example", "w") as f:
        f.write("GOOGLE_API_KEY=your_api_key_here\nOPENAI_API_KEY=optional")
    print("🔑 Created .env.example")

    print(f"\n✅ {PROJECT_NAME} structure is ready!")

if __name__ == "__main__":
    create_qpredict_structure()