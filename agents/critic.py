import ollama
from agents.state import AgentState

CRITIC_MODEL = "qwen3:1.7b"

CRITIC_SYSTEM_PROMPT = """You are a precise answering agent. You will be given:
1. A CONTEXT section containing retrieved text chunks
2. A QUESTION

Rules:
- Answer using ONLY information present in the CONTEXT
- Be specific and concise — include exact numbers, names, dates when present
- If the context does not contain enough information to answer, respond with exactly: INSUFFICIENT CONTEXT
- Do not make up information. Do not use outside knowledge."""

def critic_node(state: AgentState) -> AgentState:
    context = "\n\n---\n\n".join(state["retrieved_docs"])
    
    user_message = f"""CONTEXT:
{context}

QUESTION: {state['question']}"""

    response = ollama.chat(
        model=CRITIC_MODEL,
        messages=[
            {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ]
    )
    state["answer"] = response["message"]["content"].strip()
    state["critique"] = ""  # reserved for future use
    return state