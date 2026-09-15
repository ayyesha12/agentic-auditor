import sqlite3, pandas as pd
conn = sqlite3.connect('logs/runs.db')
df = pd.read_sql('''
    SELECT 
        version,
        COUNT(*) as total,
        SUM(rule_pass) as rule_pass,
        SUM(judge_pass) as judge_pass,
        ROUND(AVG(rule_pass)*100, 1) as rule_pct,
        ROUND(AVG(judge_pass)*100, 1) as judge_pct,
        ROUND(AVG(retrieval_hit)*100, 1) as retrieval_pct
    FROM runs 
    GROUP BY version
    ORDER BY version
''', conn)
conn.close()
print(df.to_string(index=False))
