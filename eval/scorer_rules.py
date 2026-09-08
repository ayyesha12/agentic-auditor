from typing import List, Dict
import json
import os

def keyword_score(answer: str, expected_keywords: List[str]) -> bool:
    """
    Returns True if ANY of the expected keywords appear in the answer (case-insensitive).
    
    Args:
        answer: The agent's generated answer string
        expected_keywords: List of strings, any one of which counts as correct
    
    Returns:
        True if at least one keyword is found, False otherwise
    
    Note: This is an OR check, not AND. If expected_keywords = ["6779", "6,779"],
    the answer only needs to contain one of those two forms.
    """
    answer_lower = answer.lower()
    return any(kw.lower() in answer_lower for kw in expected_keywords)


def retrieval_hit(retrieved_docs: List[str], expected_source_doc: str) -> bool:
    """
    Returns True if the expected source document appears to be in the retrieved chunks.
    
    This checks if any retrieved chunk contains a substring match against the
    first 300 characters of the expected source document.
    
    Args:
        retrieved_docs: List of raw text chunks returned by the Retriever
        expected_source_doc: Filename of the document that should have been retrieved
                            (e.g., "doc_01_mars.txt")
    
    Returns:
        True if the expected document appears to be represented in retrieved chunks
    """
    corpus_path = os.path.join("data", "corpus", expected_source_doc)
    
    if not os.path.exists(corpus_path):
        print(f"WARNING: Expected source doc not found: {corpus_path}")
        return False
    
    with open(corpus_path, "r", encoding="utf-8") as f:
        expected_text = f.read()
    
    # Use first 300 chars of the doc as a fingerprint
    # This should appear in at least one retrieved chunk if the right doc was retrieved
    fingerprint = expected_text[:300].strip()
    
    return any(fingerprint[:100] in doc for doc in retrieved_docs)


def is_insufficient_context_response(answer: str) -> bool:
    """
    Checks if the agent returned the canonical failure signal.
    Useful for tracking how often retrieval fails to find relevant context.
    """
    return "INSUFFICIENT CONTEXT" in answer.upper()