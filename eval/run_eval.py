import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import sqlite3
import time
import argparse
import subprocess
from eval.scorer_rules import keyword_score, retrieval_hit, is_insufficient_context_response
from eval.scorer_judge import judge_score
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from agents.graph import build_graph

DATASET_PATH = "eval/dataset_small.json"
DB_PATH = "logs/runs.db"

def wait_for_ollama(max_wait: int = 60):
    """Wait until Ollama is responsive before making calls."""
    import httpx
    print("    Checking Ollama is alive...")
    for i in range(max_wait):
        try:
            r = httpx.get("http://localhost:11434", timeout=3)
            if r.status_code == 200:
                print("    Ollama is ready.")
                return True
        except Exception:
            pass
        time.sleep(1)
    print("    ERROR: Ollama did not respond after waiting.")
    return False

def restart_ollama():
    """Kill and restart Ollama."""
    print("    Restarting Ollama...")
    subprocess.run(["taskkill", "/F", "/IM", "ollama.exe"],
                  capture_output=True)
    time.sleep(5)
    subprocess.Popen(["ollama", "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL)
    time.sleep(10)
    wait_for_ollama()

def load_pipeline():
    """Load FAISS index and build graph."""
    print("Loading FAISS index...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    db = FAISS.load_local(
        "data/faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    graph = build_graph(db)
    print("Pipeline ready.")
    return graph

def run_pipeline(graph, question: str, retries: int = 3) -> dict:
    """Run pipeline with retry and Ollama restart on failure."""
    for attempt in range(retries):
        try:
            result = graph.invoke({"question": question})
            return {
                "answer": result["answer"],
                "retrieved_docs": result["retrieved_docs"]
            }
        except Exception as e:
            print(f"    WARNING: attempt {attempt + 1} failed — {str(e)[:60]}")
            if attempt < retries - 1:
                print(f"    Waiting 20 seconds then retrying...")
                time.sleep(20)
                restart_ollama()
            else:
                print("    All retries failed. Logging as PIPELINE_ERROR.")
                return {
                    "answer": "PIPELINE_ERROR",
                    "retrieved_docs": []
                }

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
    print(f"\nStarting eval run — version: {version}")

    # Load dataset
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset)} eval items")

    # Setup DB
    os.makedirs("logs", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    setup_database(conn)

    # Clear existing rows for this version
    conn.execute("DELETE FROM runs WHERE version = ?", (version,))
    conn.commit()
    print(f"Cleared existing rows for version='{version}'")

    # Load pipeline once
    graph = load_pipeline()

    results = []
    for i, item in enumerate(dataset):
        print(f"\n  [{i+1}/{len(dataset)}] {item['id']}: {item['question'][:60]}...")

        # Wait between items to let Ollama breathe
        if i > 0:
            print("    Cooling down for 30 seconds...")
            time.sleep(30)

        start = time.time()
        pipeline_result = run_pipeline(graph, item["question"])
        latency = time.time() - start

        answer = pipeline_result["answer"]
        retrieved_docs = pipeline_result["retrieved_docs"]

        # Rule-based scoring
        r_pass = keyword_score(answer, item["expected_keywords"])
        r_hit = retrieval_hit(retrieved_docs, item["expected_source_doc"])
        r_insufficient = is_insufficient_context_response(answer)

        # LLM judge only when rule-based fails
        if r_pass:
            j_pass = True
        elif answer == "PIPELINE_ERROR":
            j_pass = False
        else:
            try:
                time.sleep(3)  # small pause before judge call
                j_pass = judge_score(item["question"], answer, retrieved_docs)
            except Exception as e:
                print(f"    Judge failed: {str(e)[:60]}")
                j_pass = False

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
        print(f"    {status} rule={r_pass} judge={j_pass} "
              f"retrieval_hit={r_hit} latency={latency:.1f}s")

    conn.close()

    total = len(results)
    rule_passes = sum(r["rule_pass"] for r in results)
    judge_passes = sum(r["judge_pass"] for r in results)

    print(f"\n=== EVAL SUMMARY (version={version}) ===")
    print(f"Rule-based pass rate:  {rule_passes}/{total} = "
          f"{rule_passes/total*100:.1f}%")
    print(f"Judge pass rate:       {judge_passes}/{total} = "
          f"{judge_passes/total*100:.1f}%")
    print(f"Results saved to: {DB_PATH}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v1",
                       help="Version tag (e.g. v1, v3)")
    args = parser.parse_args()
    run_eval(args.version)