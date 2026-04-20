import os

PROJECT_NAME = "QPredict"

STRUCTURE = {
    "src": ["__init__.py", "ingestion.py", "retrieval.py", "app.py", "utils.py"],
    "notebooks": ["experiment.ipynb"],
    "tests": ["__init__.py", "test_ingestion.py"],
    "data": [],
    "chroma_db": [],
}

GITIGNORE_CONTENT = """
data/
chroma_db/
*.pdf
*.png
*.jpg

__pycache__/
*.py[cod]
venv/
.env
.DS_Store
"""

README_CONTENT = f"""
# {PROJECT_NAME}

AI-Powered Exam Prediction & Context Retrieval System.

## Overview
{PROJECT_NAME} uses RAG (Retrieval-Augmented Generation) to help university students:
- Query past examination papers.
- Predict high-probability topics based on multi-year data.

## Setup
1. `pip install -r requirements.txt`
2. `python setup.py`
3. `python src/ingestion.py`
4. `streamlit run src/app.py`
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
    print(f"Initializing {PROJECT_NAME} Repository...")

    for folder, files in STRUCTURE.items():
        os.makedirs(folder, exist_ok=True)
        print(f"Created directory: {folder}/")
        
        for file in files:
            file_path = os.path.join(folder, file)
            if not os.path.exists(file_path):
                with open(file_path, "w") as f:
                    f.write("")
                print(f"Created file: {file}")

    with open(".gitignore", "w") as f:
        f.write(GITIGNORE_CONTENT.strip())
    print("Created .gitignore")

    with open("README.md", "w") as f:
        f.write(README_CONTENT.strip())
    print("Created README.md")

    with open("requirements.txt", "w") as f:
        f.write(REQUIREMENTS_CONTENT.strip())
    print("Created requirements.txt")
    
    with open(".env.example", "w") as f:
        f.write("GOOGLE_API_KEY=your_api_key_here\nOPENAI_API_KEY=optional")
    print("Created .env.example")

    print(f"Setup complete.")

if __name__ == "__main__":
    create_qpredict_structure()