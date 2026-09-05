import ollama
from agents.state import AgentState

PLANNER_MODEL = "qwen3:1.7b"  # use faster model during development

PLANNER_SYSTEM_PROMPT = """You are a research planner. Given a question, output 1-2 specific retrieval steps.
Each step should be a short phrase describing what information to look for.
Output one step per line. Do not number the steps. Do not explain."""

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