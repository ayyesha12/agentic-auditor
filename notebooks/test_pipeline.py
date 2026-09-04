"""
Manual end-to-end test. Run this before opening your PR.
This is NOT production code — it's a development verification script.
"""
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from agents.graph import build_graph

INDEX_DIR = "data/faiss_index"
EMBEDDING_MODEL = "nomic-embed-text"

def test_pipeline():
    print("Loading FAISS index...")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    db = FAISS.load_local(
        INDEX_DIR, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    
    print("Building graph...")
    graph = build_graph(db)
    
    # Pick a question you KNOW is in your corpus
    test_question = "What name did Dmitri Mendeleev use to refer to the predicted element that was later identified as germanium?"
    
    print(f"\nQuestion: {test_question}")
    print("Running pipeline (this takes ~30-60 seconds on CPU)...")
    
    result = graph.invoke({"question": test_question})
    
    print("\n--- RESULT ---")
    print(f"Plan: {result['plan']}")
    print(f"\nRetrieved docs ({len(result['retrieved_docs'])} chunks):")
    for i, doc in enumerate(result['retrieved_docs']):
        print(f"  Chunk {i+1}: {doc[:100]}...")
    print(f"\nAnswer: {result['answer']}")
    
    # Verify the output shape matches the interface contract
    assert "answer" in result, "FAIL: 'answer' key missing from result"
    assert "retrieved_docs" in result, "FAIL: 'retrieved_docs' key missing from result"
    assert isinstance(result["retrieved_docs"], list), "FAIL: retrieved_docs must be a list"
    print("\nInterface contract check: PASSED")

if __name__ == "__main__":
    test_pipeline()