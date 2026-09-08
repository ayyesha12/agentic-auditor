from langchain_community.vectorstores import FAISS
from agents.state import AgentState

TOP_K = 3  # number of chunks to retrieve

def retriever_node(state: AgentState, db: FAISS) -> AgentState:
    # Search using the original question (not the plan steps — simpler and usually works better)
    results = db.similarity_search(state["question"], k=TOP_K)
    # Extract the raw text content from each result
    state["retrieved_docs"] = [result.page_content for result in results]
    return state