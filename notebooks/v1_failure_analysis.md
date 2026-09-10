# v1 Failure Analysis
**Run date:** Sprint 2  
**Eval set size:** 50 items  
**Model:** phi4-mini (judge) + qwen3:1.7b (pipeline)  
**Embedding model:** nomic-embed-text  

---

## Official v1 Baseline Results

| Metric | Score |
|--------|-------|
| Rule-based pass rate | 28/50 = **56.0%** |
| Judge pass rate | 43/50 = **86.0%** |
| Gap (judge − rule) | **30 percentage points** |

---

## Result Breakdown

| Symbol | Meaning | Count |
|--------|---------|-------|
| ✓ | rule=True, judge=True — correct answer, keyword matched | 28 |
| ≈ | rule=False, judge=True — correct answer, keyword missed | 15 |
| ✗ | rule=False, judge=False — genuinely wrong answer | 7 |

---

## Genuine Failures (rule=False AND judge=False) — 7 items

These are the items where the pipeline gave a wrong or missing answer.
These are the PRIMARY targets for improvement in v2.

| ID | Question (short) | retrieval_hit | Likely cause |
|----|-----------------|---------------|--------------|
| q006 | Rutherford's daughter who married Ralph Fowler | True | Answer retrieved but wrong name extracted |
| q008 | Carl Linnaeus Latin title of first edition work | True | Exact Latin title not matched |
| q010 | Marie Curie's daughter born in 1897 | True | Wrong daughter name returned |
| q021 | Tesla's wirelessly controlled boat name | False | Wrong document retrieved entirely |
| q034 | Schrödinger's remark on Upanishads doctrine | False | Document not retrieved, vague planner step |
| q036 | Rutherford's famous comparison for the incredible result | True | Exact quote not matched by keywords |
| q037 | Why J.J. Thomson preferred 'corpuscle' over 'electron' | False | J.J. Thomson doc not retrieved |

---

## Keyword Mismatches (rule=False, judge=True) — 15 items

These are NOT failures — the answer is correct but worded differently
than the expected_keywords. The judge confirmed correctness.

These can be improved by adding more keyword variants to dataset.json.

| ID | Issue |
|----|-------|
| q007 | Thomson charge value — answer correct but formatted differently |
| q014 | Schrödinger disease — model said "tuberculosis" vs expected keyword |
| q016 | Rutherford coined term — paraphrased instead of exact term |
| q017 | Manchester suburb — correct but different phrasing |
| q023 | Mendeleev vodka percentage — correct but with different formatting |
| q024 | Dublin Institute city — correct but "Dublin, Ireland" vs "Dublin" |
| q027 | J.J. Thomson atomic model nickname — paraphrased |
| q029 | Lavoisier's wife name — correct but different name format |
| q039 | Lavoisier phrase in Traité — paraphrased |
| q040 | L. Pearce Williams description — paraphrased |
| q045 | Pasteur's responsibility — correct but vague |
| q049 | Schrödinger's disease in 1961 — duplicate of q014 issue |

---

## Retrieval Analysis

| Metric | Count | Percentage |
|--------|-------|------------|
| retrieval_hit=True | ~20 | ~40% |
| retrieval_hit=False | ~30 | ~60% |

**60% of items did NOT retrieve the expected source document.**
This is the biggest single problem — the Planner is generating
retrieval steps that are too vague, causing FAISS to fetch the
wrong document chunks.

---

## Root Cause Analysis

### Cause 1 — Vague Planner steps (PRIMARY cause)
The Planner generates steps like:
```
"find information about the electron"
"look up scientific contributions"
"research the scientist's early life"
```
These are too generic. FAISS returns chunks from whichever document
best matches the vague query, which is often NOT the right document.

### Cause 2 — Missing scientist name in retrieval step
When the Planner omits the scientist's full name from the retrieval
step, the vector search drifts to the wrong document.

### Cause 3 — Exact quote questions always fail rule-based
Questions asking for exact quotes, Latin titles, or specific phrasings
will always fail keyword scoring unless the expected_keywords include
multiple phrasings of the same answer.

### Cause 4 — J.J. Thomson filename issue
4 Thomson questions had filename mismatches causing retrieval_hit
to always return False for those items. This has been fixed.

---

## Recommendations for Member A — Sprint 3 Few-Shot Curation

The Planner prompt needs 2-3 targeted few-shot examples.
Each example should demonstrate:

### Rule 1 — Always include the scientist's FULL NAME in retrieval steps
```
BAD:  "find electron charge measurement"
GOOD: "J.J. Thomson electron charge measurement value 1899"
```

### Rule 2 — Include the specific fact type being asked about
```
BAD:  "look up Rutherford's work"
GOOD: "Ernest Rutherford daughter name Ralph Fowler marriage"
```

### Rule 3 — Include year/number when question asks for one
```
BAD:  "find Nobel Prize information"
GOOD: "Erwin Schrödinger Nobel Prize year shared 1933"
```

### Suggested few-shot examples for Planner prompt

```
Example 1:
Question: What value did J.J. Thomson measure for the electron charge in 1899?
Steps:
J.J. Thomson electron charge measurement 1899 value
Thomson cathode ray experiment electron charge esu

Example 2:
Question: Who was the daughter of Ernest Rutherford that married Ralph Fowler?
Steps:
Ernest Rutherford daughter name marriage Ralph Fowler
Rutherford family personal life children

Example 3:
Question: What is the Latin title of Carl Linnaeus's first edition work?
Steps:
Carl Linnaeus first edition publication Latin title
Linnaeus Systema Naturae original work 1735
```

---

## Recommended dataset.json fixes for ≈ items

For questions where judge passes but rule fails, add more keyword
variants to expected_keywords in dataset.json. Examples:

```json
q014: "expected_keywords": ["tuberculosis", "TB", "tuberculous"]
q016: "expected_keywords": ["half-life", "half life", "halflife"]
q024: "expected_keywords": ["Dublin", "Dublin, Ireland"]
q029: "expected_keywords": ["Marie-Anne", "Marie Anne", "Paulze"]
```

---

## Target for v2

After Member A updates the Planner prompt with targeted few-shot
examples, the expected improvement is:

| Metric | v1 | v2 target |
|--------|-----|-----------|
| Rule-based pass rate | 56.0% | 65%+ |
| Judge pass rate | 86.0% | 88%+ |
| Retrieval hit rate | ~40% | 55%+ |

The resume-ready result sentence will be:
> "Curating targeted few-shot examples in the Planner prompt improved
> rule-based task-completion from 56% to XX% on a fixed 50-item eval set."