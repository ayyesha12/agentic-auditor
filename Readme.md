## Hardware Benchmarks

| Machine | Model | tok/s |
|---------|-------|-------|
| Member Ay (Intel(R) Core(TM) i5-8300H CPU @ 2.30GHz (2.30 GHz), 16.0 GB) | qwen3:15.45 | 13.65 |
| Member Ay | phi4-mini | 8.83 |

| Member Bp (Intel(R) Core(TM) i7-6600U CPU @ 2.60GHz (2.80 GHz), 8.0 GB) | qwen3:11.01 | 11.01 |
| Member Bp | phi4-mini | 4.85 |

## Evaluation Methodology

### Dataset
- 45 hand-verified Q&A pairs, manually checked against the fixed 20-document corpus
- Never auto-generated — all expected_keywords confirmed to exist literally in source documents
- Covers: simple factual lookups (30), phrase-level answers (10), expected failures (5)

### Hybrid scoring
Scoring runs in two stages to balance speed, reliability, and coverage:

1. **Rule-based (always runs first):** `keyword_score()` checks if any expected keyword appears
   in the answer. `retrieval_hit()` checks if the correct source document was retrieved.
   These are deterministic and fast — no LLM needed.

2. **LLM-judge (only when rule-based fails):** `phi4-mini` evaluates whether the answer is
   correct and grounded in the retrieved context. Called only on items that fail rule-based
   scoring, to save time and limit noise.

3. **Both scores are logged separately.** `rule_pass` and `judge_pass` are never merged into one score. Their disagreement rate is itself a reported metric, and acknowledges the known limitation of a small model judging another small model's output.

### Data-centric experiment
- **v1:** baseline Planner prompt with no few-shot examples
- **v2:** Planner prompt with 3 curated few-shot examples targeting v1 failure patterns
- Result: rule-based pass rate improved from __% (v1) to __% (v2)

### Drift detection
IsolationForest (`contamination=0.1`) over `latency_seconds`, `rule_pass`, `judge_pass`,
`retrieval_hit` per run. Flags runs that behave unusually. Output: `logs/drift_report.csv`.

### Limitations & Future Work

### Known limitations (honest, not excuses)

- **CPU-only inference:** 15–25 tok/s on Phi-4-mini. Eval batches are limited to 45 items
  by design, not oversight. Running statistically significant trials would require GPU or
  significantly more time.

- **Small model judge:** `judge_score()` uses `phi4-mini` to evaluate `phi4-mini`'s outputs.
  This is a known methodological weakness — a weak model judging another weak model introduces
  noise in the LLM-judge scores. Mitigated (not eliminated) by running rule-based scoring first and only calling the judge on ambiguous cases.

- **Fixed 20-document corpus:** Findings are specific to this corpus and task domain.
  Results do not generalize to open-domain or multi-domain QA.

- **Small eval set (45 items):** Statistical power is limited. A 5% improvement may not be
  meaningful at this sample size. Stated results should be interpreted as directional,
  not definitive.

### Future work
- GPU-accelerated inference for larger eval sets (500+ items)
- Replace LLM-judge with a fine-tuned evaluation model
- Extend to multiple task domains
- Human-in-the-loop review of anomalous runs
- Continuous eval with dataset versioning (DVC)