import ollama

JUDGE_MODEL = "phi4-mini"

JUDGE_PROMPT_TEMPLATE = """You are a strict evaluation judge. Your job is to decide if an answer is correct and grounded in the provided context.

QUESTION: {question}

CONTEXT (what the agent retrieved):
{context}

ANSWER (what the agent said):
{answer}

Evaluate: Is this answer CORRECT and SUPPORTED by the context above?
- Correct means it accurately answers the question
- Supported means the information comes from the context, not outside knowledge

Reply with ONLY one word: YES or NO"""


def judge_score(question: str, answer: str, retrieved_docs: list[str]) -> bool:
    """
    Uses phi4-mini as an LLM judge to score an answer.
    
    IMPORTANT: This is a weak model judging another weak model's output.
    This introduces noise. Always log judge_pass separately from rule_pass.
    Never use this as the only scoring mechanism.
    
    Args:
        question: The original question
        answer: The agent's generated answer
        retrieved_docs: The chunks the agent retrieved (used as context for the judge)
    
    Returns:
        True if the judge says YES, False otherwise
    """
    context = "\n\n---\n\n".join(retrieved_docs)
    
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        context=context,
        answer=answer
    )
    
    response = ollama.chat(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    
    reply = response["message"]["content"].strip().upper()
    # Be generous with matching — model sometimes says "YES." or "YES, ..."
    return reply.startswith("YES")