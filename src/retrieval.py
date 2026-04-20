import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

def get_rag_chain(vector_store, subject):
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 5, 
            "filter": {"subject": subject}
        }
    )

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

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

    question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain

def predict_exam_trends(vector_store, subject):
    try:
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 30, 
                "filter": {"subject": subject}
            }
        )
        
        docs = retriever.invoke(f"What are the most frequent questions and important topics in {subject} exams?")
        
        if not docs:
            return "Insufficient historical data available for trend analysis."

        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)

        system_prompt = (
            "You are an expert Exam Trend Forecaster for a University. "
            f"Given these extracts from examination papers related to: {subject}. "
            "Determine recurring patterns, critical concepts, or emphasized topics. "
            "Return a structured Markdown report highlighting: \n"
            "1. High Probability Topics \n"
            "2. Essential Terminology \n"
            "3. Common Essay structures. \n"
            "Ensure the output reflects only the provided context. \n\n"
            "Context: {context}"
        )

        prompt_template = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("human", "Initialize trend prediction sequence.")]
        )
        
        chain = create_stuff_documents_chain(llm, prompt_template)
        response = chain.invoke({"context": docs, "input": "Initialize trend prediction sequence."})
        
        return response
    except Exception as e:
        return f"Trend analysis failure: {e}"
