from langgraph.graph import StateGraph, END
from langchain_community.vectorstores import FAISS
from agents.state import AgentState
from agents.planner import planner_node
from agents.retriever import retriever_node
from agents.critic import critic_node

def build_graph(db: FAISS):
    """
    Build and compile the 3-agent LangGraph pipeline.
    
    Args:
        db: Pre-loaded FAISS vector store. Passed in at startup so it loads once.
    
    Returns:
        Compiled LangGraph runnable. Call with graph.invoke({"question": "..."})
    """
    graph = StateGraph(AgentState)
    
    # Add all three nodes
    graph.add_node("planner", planner_node)
    graph.add_node("retriever", lambda state: retriever_node(state, db))
    graph.add_node("critic", critic_node)
    
    # Set the entry point
    graph.set_entry_point("planner")
    
    # Wire edges: planner -> retriever -> critic -> END
    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "critic")
    graph.add_edge("critic", END)
    
    return graph.compile()