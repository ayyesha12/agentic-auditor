import json
import sqlite3
import time
import argparse
import os
from eval.scorer_rules import keyword_score, retrieval_hit, is_insufficient_context_response
from eval.scorer_judge import judge_score

DATASET_PATH = "eval/dataset.json"
DB_PATH = "logs/runs.db"

# ─────────────────────────────────────────────────────────────
# MOCK PIPELINE — replace with real graph.invoke() after Sprint 1 merge
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# REAL PIPELINE — uncomment after feature/agent-pipeline merges
# ─────────────────────────────────────────────────────────────
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from agents.graph import build_graph

_embeddings = OllamaEmbeddings(model="qwen3:1.7b")
_db = FAISS.load_local("data/faiss_index", _embeddings, allow_dangerous_deserialization=True)
_graph = build_graph(_db)

def run_pipeline(question: str) -> dict:
    result = _graph.invoke({"question": question})
    return {"answer": result["answer"], "retrieved_docs": result["retrieved_docs"]}

def setup_database(conn: sqlite3.Connection):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            id TEXT,
            question TEXT,
            answer TEXT,
            rule_pass INTEGER,
            judge_pass INTEGER,
            retrieval_hit INTEGER,
            insufficient_context INTEGER,
            latency_seconds REAL,
            timestamp REAL,
            version TEXT
        )
    """)
    conn.commit()


def run_eval(version: str):
    print(f"Starting eval run — version: {version}")
    
    # Load dataset
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset)} eval items")
    
    # Setup DB
    os.makedirs("logs", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    setup_database(conn)
    
    results = []
    for i, item in enumerate(dataset):
        print(f"  [{i+1}/{len(dataset)}] {item['id']}: {item['question'][:60]}...")
        
        start = time.time()
        
        # Run the pipeline (swap mock → real after Sprint 1 merge)
        pipeline_result = run_pipeline(item["question"])
        # Real call (uncomment after merge):
        # pipeline_result = run_pipeline(item["question"])
        
        latency = time.time() - start
        
        answer = pipeline_result["answer"]
        retrieved_docs = pipeline_result["retrieved_docs"]
        
        # Score: rule-based first (fast, deterministic)
        r_pass = keyword_score(answer, item["expected_keywords"])
        r_hit = retrieval_hit(retrieved_docs, item["expected_source_doc"])
        r_insufficient = is_insufficient_context_response(answer)
        
        # Score: LLM-judge only for items that failed rule-based
        # This saves time and avoids running weak-model-judge unnecessarily
        if r_pass:
            j_pass = True  # rule-based already confirmed it — no need to call judge
        else:
            j_pass = judge_score(item["question"], answer, retrieved_docs)
        
        # Log to DB
        conn.execute(
            "INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                item["id"],
                item["question"],
                answer,
                int(r_pass),
                int(j_pass),
                int(r_hit),
                int(r_insufficient),
                latency,
                time.time(),
                version
            )
        )
        conn.commit()
        
        results.append({
            "id": item["id"],
            "rule_pass": r_pass,
            "judge_pass": j_pass,
            "retrieval_hit": r_hit,
            "insufficient": r_insufficient,
            "latency": round(latency, 2)
        })
        
        status = "✓" if r_pass else ("≈" if j_pass else "✗")
        print(f"    {status} rule={r_pass} judge={j_pass} retrieval_hit={r_hit} latency={latency:.1f}s")
    
    conn.close()
    
    # Print summary
    total = len(results)
    rule_passes = sum(r["rule_pass"] for r in results)
    judge_passes = sum(r["judge_pass"] for r in results)
    
    print(f"\n=== EVAL SUMMARY (version={version}) ===")
    print(f"Rule-based pass rate:  {rule_passes}/{total} = {rule_passes/total*100:.1f}%")
    print(f"Judge pass rate:       {judge_passes}/{total} = {judge_passes/total*100:.1f}%")
    print(f"Results saved to: {DB_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v1", help="Version tag for this eval run (e.g. v1, v2)")
    args = parser.parse_args()
    run_eval(args.version)