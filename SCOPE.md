# Project Scope — Agent Auditor

Locked on: [9/3/2026]
Members: [Praih Alias Faiza and Ayesha ]

## Fixed decisions

- **Agents:** exactly 3 — Planner, Retriever, Critic
- **Corpus:** [agree on a topic here, e.g. "20 Wikipedia articles about the Solar System"]
- **Eval set size:** 40–50 hand-verified Q&A pairs
- **Models:** qwen3:1.7b for dev/testing, phi4-mini for final eval runs
- **Task domain:** question answering over the fixed 20-doc corpus only

## What is OUT of scope

- Web search or external APIs
- More than 3 agents
- Auto-generating eval pairs with an LLM
- Fine-tuning any model
- Building a product UI (Streamlit dashboard is for eval monitoring only)

## Interface contract (NEVER change without notifying both members)

Member A's graph.invoke() must always accept:
  {"question": str}

And must always return a dict containing at least:
  {"answer": str, "retrieved_docs": list[str]}