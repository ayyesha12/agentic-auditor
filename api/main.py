from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from contextlib import asynccontextmanager
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from agents.graph import build_graph
import subprocess
import os

# Global state — loaded once at startup
_graph = None
_db = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the FAISS index and compile the graph at startup."""
    global _graph, _db
    print("Loading FAISS index...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    _db = FAISS.load_local(
        "data/faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    print("Building agent graph...")
    _graph = build_graph(_db)
    print("API ready.")
    yield
    # Cleanup (if needed) goes here

app = FastAPI(
    title="Agent Auditor API",
    description="Multi-agent QA pipeline with eval harness",
    lifespan=lifespan
)


class Query(BaseModel):
    question: str


class AskResponse(BaseModel):
    question: str
    answer: str
    plan: list[str]
    retrieved_docs: list[str]


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok", "graph_loaded": _graph is not None}


@app.post("/ask", response_model=AskResponse)
def ask(query: Query):
    """
    Run a single question through the agent pipeline.
    Returns the answer, the Planner's steps, and the retrieved chunks.
    """
    if _graph is None:
        raise HTTPException(status_code=503, detail="Graph not loaded yet")
    
    result = _graph.invoke({"question": query.question})
    
    return AskResponse(
        question=query.question,
        answer=result["answer"],
        plan=result["plan"],
        retrieved_docs=result["retrieved_docs"]
    )


@app.post("/run-eval")
def run_eval_endpoint(version: str = "v1", background_tasks: BackgroundTasks = None):
    """
    Trigger a full eval run.
    Runs in the background so the endpoint returns immediately.
    The eval results are written to logs/runs.db.
    """
    def _run():
        subprocess.run(
            ["python", "eval/run_eval.py", "--version", version],
            check=True
        )
    
    background_tasks.add_task(_run)
    return {"status": "eval started", "version": version}