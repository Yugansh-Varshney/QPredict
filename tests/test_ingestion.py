from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import tempfile
import os

def test_chroma_persistence():
    """A basic sanity test to ensure Chroma DB can initialize and save chunks."""
    
    # Create a temporary directory so we don't mess up the real db
    with tempfile.TemporaryDirectory() as temp_db:
        
        # 1. Mock some small documents
        mock_docs = [
            Document(page_content="What is a compiler? A compiler translates code.", metadata={"subject": "Compiler_Design"}),
            Document(page_content="The OSI model has 7 layers.", metadata={"subject": "Computer_Networks"})
        ]
        
        # 2. Init Embedding function
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # 3. Save to chroma
        vector_store = Chroma.from_documents(
            documents=mock_docs,
            embedding=embeddings,
            persist_directory=temp_db
        )
        
        # 4. Verify they were saved by retrieving
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 1, "filter": {"subject": "Compiler_Design"}}
        )
        
        ans = retriever.invoke("compiler")
        
        assert len(ans) == 1, "Failed to retrieve the document correctly."
        assert "compiler translates" in ans[0].page_content, "Wrong document retrieved!"
        
        print("✅ Database Persistence and Metadata filtering works perfectly!")

if __name__ == "__main__":
    test_chroma_persistence()
