import torch
# Neutralize the Streamlit + PyTorch macOS Segfault issue
torch.classes.__path__ = [] 
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

import streamlit as st
import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

from utils import get_subjects_structure
from retrieval import get_rag_chain, predict_exam_trends

# 1. Load Environment Variables
load_dotenv()

# --- CONFIGURATION ---
DB_PATH = "./chroma_db"
DATA_PATH = "./data"

st.set_page_config(page_title="QPredict", page_icon="🎓", layout="wide")

# --- FUNCTIONS ---
@st.cache_resource
def load_vector_store():
    """Load the ChromaDB only once to save time."""
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'} # Prevents Apple M-series GPU threading segfaults
    )
    if os.path.exists(DB_PATH):
        vector_store = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
        return vector_store
    return None

# --- UI LAYOUT ---
st.title("🎓 QPredict: University Exam Intelligence")
st.markdown("Chat with your Previous Year Questions (PYQs) and forecast high-probability topics.")

# Sidebar for Context Selection
st.sidebar.header("📂 Select Context")
structure = get_subjects_structure()

if not structure:
    st.error("No data folders found! Please run 'python setup_folders.py' and 'src/ingestion.py'.")
    st.stop()

# 1. Select Semester
selected_sem = st.sidebar.selectbox("Select Semester", list(structure.keys()))

# 2. Select Subject
available_subjects = structure.get(selected_sem, [])
selected_subj = st.sidebar.selectbox("Select Subject", available_subjects)

if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []

# Tabs Setup
tab1, tab2 = st.tabs(["💬 Chat with PYQs", "🔮 Predict Exam Trends"])

with tab1:
    # Initialize Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle User Input
    if prompt := st.chat_input(f"Ask a question about {selected_subj}..."):
        # Display User Message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        try:
            vector_store = load_vector_store()
            if not vector_store:
                st.error("Database not found. Please ingest PDFs first.")
                st.stop()
                
            # Use decoupled retrieval logic
            rag_chain = get_rag_chain(vector_store, selected_subj)

            with st.chat_message("assistant"):
                with st.spinner("Analyzing past papers..."):
                    response = rag_chain.invoke({"input": prompt})
                    answer = response["answer"]
                    st.markdown(answer)
                    
                    # Show Sources
                    with st.expander("View Source Questions"):
                        for i, doc in enumerate(response["context"]):
                            st.markdown(f"**Chunk {i+1} (Page {doc.metadata.get('page','X')}):** {doc.page_content[:200]}...")

            # Save Assistant Message
            st.session_state.messages.append({"role": "assistant", "content": answer})

        except Exception as e:
            st.error(f"Error: {e}")
            st.info("Make sure you have added your GOOGLE_API_KEY to the .env file!")

with tab2:
    st.header(f"📈 Trend Forecast for {selected_subj}")
    st.markdown("We will fetch up to 30 past exam question snippets and ask the AI to find patterns.")
    
    if st.button("Generate Trend Report"):
        try:
            vector_store = load_vector_store()
            if not vector_store:
                st.error("Database not found. Please ingest PDFs first.")
                st.stop()
                
            with st.spinner(f"Reading patterns for {selected_subj}... This might take 10-20 seconds..."):
                report = predict_exam_trends(vector_store, selected_subj)
                st.markdown(report)
                
        except Exception as e:
            st.error(f"Error predicting trends: {e}")
            st.info("Make sure you have added your GOOGLE_API_KEY to the .env file!")