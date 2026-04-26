import torch
import os

torch.classes.__path__ = [] 
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

from utils import get_subjects_structure
from retrieval import get_rag_chain, predict_exam_trends

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "chroma_db")
DATA_PATH = os.path.join(BASE_DIR, "data")

st.set_page_config(page_title="QPredict", page_icon="🎓", layout="wide")

@st.cache_resource
def load_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    if os.path.exists(DB_PATH):
        vector_store = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
        return vector_store
    return None

st.title("🎓 QPredict: Exam Intelligence")
st.markdown("Analyze and interact with past examination data using Retrieval-Augmented Generation.")

st.sidebar.header("Context Selection")
structure = get_subjects_structure()

if not structure:
    st.error("Data directory missing or empty. Please initialize data folders and run ingestion.")
    st.stop()

selected_sem = st.sidebar.selectbox("Semester", list(structure.keys()))
available_subjects = structure.get(selected_sem, [])
selected_subj = st.sidebar.selectbox("Subject", available_subjects)

if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []

tab1, tab2 = st.tabs(["Chat Context", "Trend Prediction"])

with tab1:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input(f"Query {selected_subj}..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        try:
            vector_store = load_vector_store()
            if not vector_store:
                st.error("Vector database unavailable. Ensure data ingestion was successful.")
                st.stop()
                
            rag_chain = get_rag_chain(vector_store, selected_subj)

            with st.chat_message("assistant"):
                with st.spinner("Processing..."):
                    response = rag_chain.invoke({"input": prompt})
                    answer = response["answer"]
                    st.markdown(answer)
                    
                    with st.expander("Source References"):
                        for i, doc in enumerate(response["context"]):
                            st.markdown(f"**Document {i+1} (Page {doc.metadata.get('page','-')}):** {doc.page_content[:200]}...")

            st.session_state.messages.append({"role": "assistant", "content": answer})

        except Exception as e:
            st.error(f"Processing error: {e}")
            st.info("Verify environment configuration and API keys.")

with tab2:
    st.header(f"Trend Analysis: {selected_subj}")
    st.markdown("Generates a topic forecast based on semantic extraction of historical data.")
    
    if st.button("Generate Report"):
        try:
            vector_store = load_vector_store()
            if not vector_store:
                st.error("Vector database unavailable.")
                st.stop()
                
            with st.spinner("Analyzing historical patterns..."):
                report = predict_exam_trends(vector_store, selected_subj)
                st.markdown(report)
                
        except Exception as e:
            st.error(f"Analysis error: {e}")