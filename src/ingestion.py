import os
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import fitz
from ocrmac import ocrmac
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(BASE_DIR, "chroma_db")

PAGE_MAPPINGS = {
    "Semester_5.pdf": {
        "Compiler_Design": (12, 28),
        "Operating_Systems": (29, 51),
    }
}

def ingest_documents():
    print(f"Starting ingestion process from {DATA_PATH}")
    
    documents = []
    
    for root, dirs, files in os.walk(DATA_PATH):
        for file in files:
            if not file.endswith(".pdf"): continue

            full_path = os.path.join(root, file)
            parts = full_path.split(os.sep)
            
            try:
                data_index = parts.index("data")
                semester = parts[data_index + 1]
            except (ValueError, IndexError):
                print(f"Skipping {file}: Invalid directory structure.")
                continue

            print(f"Processing: {file}")

            try:
                loader = PyPDFLoader(full_path)
                file_docs = loader.load()
                
                if file in PAGE_MAPPINGS:
                    mapping = PAGE_MAPPINGS[file]
                    print(f"Applying page mapping rules for {file}")
                    
                    for subject, (start_idx, end_idx) in mapping.items():
                        subject_docs = file_docs[start_idx:end_idx+1]
                        
                        for doc in subject_docs:
                            doc.metadata["semester"] = semester
                            doc.metadata["subject"] = subject
                            doc.metadata["year"] = "Multi-Year"
                            doc.metadata["source"] = f"{file} (Pages {start_idx}-{end_idx})"
                        
                        documents.extend(subject_docs)
                        print(f"Extracted {len(subject_docs)} pages for {subject}")
                
                else:
                    try:
                        subject = parts[data_index + 2]
                        for doc in file_docs:
                            doc.metadata["semester"] = semester
                            doc.metadata["subject"] = subject
                            doc.metadata["year"] = "Multi-Year"
                            doc.metadata["source"] = file
                        
                        documents.extend(file_docs)
                        print(f"Assigned to {subject}")
                    except IndexError:
                        print(f"Failed to identify subject mapping for {file}.")

            except Exception as e:
                print(f"Failed to load {file}: {e}")

    if not documents:
        print("No valid documents located.")
        return

    print(f"Detecting unreadable elements across {len(documents)} pages for OCR processing...")
    
    for i, doc in enumerate(documents):
        if len(doc.page_content.strip()) < 10:
            pdf_path = None
            for root, _, files in os.walk(DATA_PATH):
                for file in files:
                    source_basename = doc.metadata.get('source', '').split(' (Pages')[0]
                    if file == source_basename:
                        pdf_path = os.path.join(root, file)
                        break
                if pdf_path: break
            
            if pdf_path:
                try:
                    page_num = doc.metadata.get("page", 0)
                    print(f"Initializing OCR on {pdf_path.split(os.sep)[-1]} : Page {page_num}")
                    doc_fitz = fitz.open(pdf_path)
                    page_pix = doc_fitz[page_num].get_pixmap(dpi=150)
                    tmp_img = f"/tmp/ocr_{time.time()}.png"
                    page_pix.save(tmp_img)
                    
                    ocr_result = ocrmac.OCR(tmp_img).recognize()
                    ocr_text = "\n".join([t[0] for t in ocr_result])
                    doc.page_content = ocr_text
                    
                    os.remove(tmp_img)
                except Exception as e:
                    print(f"OCR failure: {e}")

    print("Executing document chunking...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    print(f"Formed {len(chunks)} chunks.")

    print(f"Persisting to vector store at {DB_PATH}...")
    
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        persist_directory=DB_PATH
    )
    
    print("Ingestion sequence complete.")

if __name__ == "__main__":
    ingest_documents()