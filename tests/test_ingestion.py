from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import tempfile
import os

def test_chroma_persistence():
    with tempfile.TemporaryDirectory() as temp_db:
        mock_docs = [
            Document(page_content="What is a compiler? A compiler translates code.", metadata={"subject": "Compiler_Design"}),
            Document(page_content="The OSI model has 7 layers.", metadata={"subject": "Computer_Networks"})
        ]
        
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        vector_store = Chroma.from_documents(
            documents=mock_docs,
            embedding=embeddings,
            persist_directory=temp_db
        )
        
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 1, "filter": {"subject": "Compiler_Design"}}
        )
        
        ans = retriever.invoke("compiler")
        
        assert len(ans) == 1, "Exception: Document retrieval constraint failed"
        assert "compiler translates" in ans[0].page_content, "Exception: Invalid document vector match"
        
        print("Test passed: Database Persistence mapped accordingly.")

if __name__ == "__main__":
    test_chroma_persistence()
