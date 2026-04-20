import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

def get_rag_chain(vector_store, subject):
    """Creates a RAG chain focused on a specific subject."""
    # Create a Retriever that filters ONLY for the selected subject
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 5, 
            "filter": {"subject": subject}  # Metadata Filter
        }
    )

    # Initialize Google Gemini LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

    # Create the System Prompt
    system_prompt = (
        "You are an expert University Tutor. Use the following context from previous year question papers "
        "to answer the student's question. \n\n"
        "If the answer is found in the context, explicitly mention which years (if available) or "
        "frequency of the topic. If you don't know, just say that the topic hasn't appeared in the uploaded papers.\n\n"
        "Context: {context}"
    )

    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "{input}")]
    )

    # Create the RAG Chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain


def predict_exam_trends(vector_store, subject):
    """Pulls large context from vector store to predict trends for the subject."""
    try:
        # Retrieve a large number of chunks (k=30) for better topic analysis
        # Alternatively, we just search for "exam paper topics" to pull diverse chunks
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 30, 
                "filter": {"subject": subject}
            }
        )
        
        # Invoke retriever directly with a generic broad question
        docs = retriever.invoke(f"What are the most frequent questions and important topics in {subject} exams?")
        
        if not docs:
            return "No previous year questions found for this subject to predict trends."

        # Pass all retrieved docs into the LLM
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)

        system_prompt = (
            "You are an expert Exam Trend Forecaster for a University. "
            f"You have been provided with 30 chunks of text extracted from past exam papers for the subject: {subject}. "
            "Analyze these snippets and identify recurring themes, repeated questions, or highly emphasized concepts. "
            "Output a structured Markdown report that highlights: \n"
            "1. True 'Hot Topics' (High Probability for upcoming exams) \n"
            "2. Essential Definitions to memorize \n"
            "3. Common Long-Answer or Essay questions. \n"
            "Keep the report highly readable, engaging, and specifically grounded in the provided snippets. "
            "If the snippets are too narrow or repetitive, do your best to generalize the core syllabus areas they cover. \n\n"
            "Context (Exam Snippets): {context}"
        )

        prompt_template = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("human", "Please predict the exam trends.")]
        )
        
        chain = create_stuff_documents_chain(llm, prompt_template)
        response = chain.invoke({"context": docs, "input": "Please predict the exam trends."})
        
        return response
    except Exception as e:
        return f"Error analyzing trends: {e}"
