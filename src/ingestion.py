import os
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import fitz # PyMuPDF for rendering PDF images
from ocrmac import ocrmac # Apple Native Vision OCR

# --- CONFIGURATION ---
DATA_PATH = "./data"
DB_PATH = "./chroma_db"

# 📖 PAGE MAPPING CONFIGURATION 📖
# If you have one massive "All Subjects" PDF for a semester, place it in data/Semester_X/
# and configure the page ranges here. The script will automatically split it for you!
# Format: "Filename" : { "Subject_Name": (Start_Page, End_Page) }
# NOTE: Pages are 0-indexed (Page 1 = 0)
PAGE_MAPPINGS = {
    "Semester_5.pdf": {
        "Compiler_Design": (12, 28), # Maps to physical pages 13-29
        "Operating_Systems": (29, 51),
        # Add other subjects here...
    }
}

def ingest_documents():
    print(f"🚀 Starting Ingestion in: {DATA_PATH}")
    
    documents = []
    
    # 1. Walk through the directory tree
    for root, dirs, files in os.walk(DATA_PATH):
        for file in files:
            if not file.endswith(".pdf"): continue

            full_path = os.path.join(root, file)
            parts = full_path.split(os.sep)
            
            try:
                data_index = parts.index("data")
                semester = parts[data_index + 1]  # e.g., "Semester_5"
            except (ValueError, IndexError):
                print(f"⚠️  Skipping {file}: Not inside a semester folder.")
                continue

            print(f"📄 Processing: {file} in {semester}")

            # 2. Load the PDF
            try:
                loader = PyPDFLoader(full_path)
                file_docs = loader.load()
                
                # Check if this file is in our smart PAGE_MAPPINGS dict
                if file in PAGE_MAPPINGS:
                    mapping = PAGE_MAPPINGS[file]
                    print(f"   🧠 Smart Mapping found for {file}! Slicing pages subject-wise...")
                    
                    for subject, (start_idx, end_idx) in mapping.items():
                        subject_docs = file_docs[start_idx:end_idx+1] # Slice the array
                        
                        for doc in subject_docs:
                            # Modify metadata
                            doc.metadata["semester"] = semester
                            doc.metadata["subject"] = subject
                            doc.metadata["year"] = "Multi-Year"
                            doc.metadata["source"] = f"{file} (Pages {start_idx}-{end_idx})"
                        
                        documents.extend(subject_docs)
                        print(f"      -> Extracted {len(subject_docs)} pages for {subject}")
                
                else:
                    # Fallback to old folder-based logic if not mapped
                    try:
                        subject = parts[data_index + 2] # Fallback
                        for doc in file_docs:
                            doc.metadata["semester"] = semester
                            doc.metadata["subject"] = subject
                            doc.metadata["year"] = "Multi-Year"
                            doc.metadata["source"] = file
                        
                        documents.extend(file_docs)
                        print(f"   -> Appended entirely to {subject}")
                    except IndexError:
                        print(f"   ⚠️ Could not determine subject for {file}. Please put it in a Subject folder or add it to PAGE_MAPPINGS.")

            except Exception as e:
                print(f"❌ Error loading {file}: {e}")

    if not documents:
        print("\n⚠️  No valid documents found!")
        return

    # 3. Post-Process via Native MacOS OCR
    # For any pages that PyPDFLoader couldn't read (because they are scanned images), we run OCR natively.
    print(f"\n🔍 Scanning {len(documents)} pages to detect blank image-only pages...")
    import time
    for i, doc in enumerate(documents):
        if len(doc.page_content.strip()) < 10:
            pdf_path = None
            # Find the actual path of the file
            for root, _, files in os.walk(DATA_PATH):
                for file in files:
                    # Strip out '(Pages ...)' from source if needed
                    source_basename = doc.metadata.get('source', '').split(' (Pages')[0]
                    if file == source_basename:
                        pdf_path = os.path.join(root, file)
                        break
                if pdf_path: break
            
            if pdf_path:
                try:
                    page_num = doc.metadata.get("page", 0)
                    print(f"      👀 OCR'ing Image on: {pdf_path.split(os.sep)[-1]} (Page {page_num})")
                    # Render Image using PyMuPDF
                    doc_fitz = fitz.open(pdf_path)
                    page_pix = doc_fitz[page_num].get_pixmap(dpi=150)
                    tmp_img = f"/tmp/ocr_{time.time()}.png"
                    page_pix.save(tmp_img)
                    
                    # Run Apple Vision OCR
                    ocr_result = ocrmac.OCR(tmp_img).recognize()
                    ocr_text = "\n".join([t[0] for t in ocr_result])
                    doc.page_content = ocr_text
                    
                    os.remove(tmp_img)
                except Exception as e:
                    print(f"      ❌ failed OCR on page: {e}")

    # 4. Split Text (Chunking)
    print(f"\n🧩 Splitting {len(documents)} pages into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    print(f"   -> Created {len(chunks)} chunks.")

    # 5. Create/Update Vector Database
    print(f"💾 Saving to ChromaDB at '{DB_PATH}'...")
    
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        persist_directory=DB_PATH
    )
    
    print("✅ Ingestion Complete! Database is ready.")

if __name__ == "__main__":
    ingest_documents()