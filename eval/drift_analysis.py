# Behavioral drift and anomaly detection using IsolationForest.

# Reads all eval runs from logs/runs.db and identifies runs that behave
# unusually in terms of latency, rule_pass, or judge_pass.

# Output: logs/drift_report.csv with an 'anomaly' column.
#   1  = normal run
#  -1  = anomalous run (unusual behavior)

import pandas as pd
import sqlite3
from sklearn.ensemble import IsolationForest
import os

DB_PATH = "logs/runs.db"
OUTPUT_PATH = "logs/drift_report.csv"

def run_drift_analysis():
    if not os.path.exists(DB_PATH):
        print(f"ERROR: Database not found at {DB_PATH}")
        print("Run eval/run_eval.py first to generate data.")
        return
    
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM runs", conn)
    conn.close()
    
    print(f"Loaded {len(df)} runs from database")
    
    if len(df) < 10:
        print("WARNING: Fewer than 10 runs — anomaly detection may not be meaningful.")
    
    # Features for anomaly detection
    features = df[["latency_seconds", "rule_pass", "judge_pass", "retrieval_hit"]].copy()
    
    # Fit IsolationForest
    model = IsolationForest(
        contamination=0.1,   # expect ~10% anomalous runs
        random_state=42,
        n_estimators=100
    )
    df["anomaly"] = model.fit_predict(features)
    
    # Add human-readable anomaly reason
    def label_anomaly(row):
        if row["anomaly"] == 1:
            return "normal"
        reasons = []
        if row["latency_seconds"] > df["latency_seconds"].mean() + 2 * df["latency_seconds"].std():
            reasons.append("slow")
        if row["rule_pass"] == 0 and row["judge_pass"] == 0:
            reasons.append("both_failed")
        if row["rule_pass"] == 1 and row["judge_pass"] == 0:
            reasons.append("rule_judge_disagree")
        return "|".join(reasons) if reasons else "outlier"
    
    df["anomaly_reason"] = df.apply(label_anomaly, axis=1)
    
    # Save
    os.makedirs("logs", exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    
    anomalous = df[df["anomaly"] == -1]
    print(f"\nDrift analysis complete.")
    print(f"Total runs: {len(df)}")
    print(f"Anomalous runs: {len(anomalous)} ({len(anomalous)/len(df)*100:.1f}%)")
    print(f"Output saved to: {OUTPUT_PATH}")
    
    print("\nAnomalous run IDs:")
    for _, row in anomalous.iterrows():
        print(f"  {row['id']} (v{row['version']}) — {row['anomaly_reason']}")

if __name__ == "__main__":
    run_drift_analysis()

