
import sqlite3
import pandas as pd

conn = sqlite3.connect("logs/runs.db")
df = pd.read_sql("SELECT * FROM runs WHERE version='v1'", conn)
conn.close()

print(f"Total items: {len(df)}")
print(f"Rule-based pass rate: {df['rule_pass'].mean()*100:.1f}%")
print(f"Judge pass rate: {df['judge_pass'].mean()*100:.1f}%")
print(f"Retrieval hit rate: {df['retrieval_hit'].mean()*100:.1f}%")
print(f"Insufficient context rate: {df['insufficient_context'].mean()*100:.1f}%")

print("\n--- FAILURES (rule_pass=0 AND judge_pass=0) ---")
failures = df[(df['rule_pass'] == 0) & (df['judge_pass'] == 0)]
for _, row in failures.iterrows():
    print(f"\nID: {row['id']}")
    print(f"Q: {row['question']}")
    print(f"A: {row['answer'][:200]}")
    print(f"Retrieval hit: {row['retrieval_hit']}")
    print(f"Insufficient context: {row['insufficient_context']}")