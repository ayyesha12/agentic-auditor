import ollama
from agents.state import AgentState

PLANNER_MODEL = "qwen3:1.7b"  # use faster model during development

# v1 planner prompt
# PLANNER_SYSTEM_PROMPT = """You are a research planner. Given a question, output 1-2 retrieval steps.
# Each step should be a short phrase describing what information to look for.
# Output one step per line. Do not number the steps. Do not explain."""

# v2 planner prompt
PLANNER_SYSTEM_PROMPT = """You are a research planner. Your only job is to write search phrases.

Given a question, write exactly 2 search phrases.
Each phrase must contain the person's full name.
Each phrase must be specific to the exact fact asked about.
Write one phrase per line. Nothing else.

Question: What value did J.J. Thomson measure for the electron charge in 1899?
J.J. Thomson electron charge value 1899
J.J. Thomson cathode ray experiment measurement esu

Question: Who was the daughter of Ernest Rutherford that married Ralph Fowler?
Ernest Rutherford daughter Eileen marriage Ralph Fowler
Ernest Rutherford family children personal life

Question: What is the Latin title of Carl Linnaeus first edition work published in Netherlands?
Carl Linnaeus Systema Naturae first edition Latin title
Carl Linnaeus 1735 publication Netherlands taxonomy

Question: What name did Tesla give to his wirelessly controlled boat?
Nikola Tesla wirelessly controlled boat telautomaton name
Nikola Tesla remote control boat invention 1898

Question: What disease did Erwin Schrödinger die of in Vienna in 1961?
Erwin Schrödinger death disease Vienna 1961
Erwin Schrödinger tuberculosis illness death cause"""

def planner_node(state: AgentState) -> AgentState:
    response = ollama.chat(
        model=PLANNER_MODEL,
        messages=[
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {state['question']}"}
        ]
    )
    raw_plan = response["message"]["content"]
    # Split by newline, strip whitespace, remove empty lines
    steps = [line.strip() for line in raw_plan.split("\n") if line.strip()]
    state["plan"] = steps[:2]  # cap at 2 steps
    return state