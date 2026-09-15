"""Manual unit tests for the scoring functions. Run before opening PR."""
from eval.scorer_rules import keyword_score, retrieval_hit, is_insufficient_context_response

def test_keyword_score():
    # Should pass
    assert keyword_score("The diameter of Mars is 6779 km", ["6779", "6,779"]) == True
    assert keyword_score("MARS diameter: 6,779 kilometers", ["6779", "6,779"]) == True
    
    # Should fail
    assert keyword_score("Mars is a red planet", ["6779", "6,779"]) == False
    assert keyword_score("", ["6779"]) == False
    assert keyword_score("INSUFFICIENT CONTEXT", ["6779"]) == False
    
    print("keyword_score: all tests passed")

def test_insufficient_context():
    assert is_insufficient_context_response("INSUFFICIENT CONTEXT") == True
    assert is_insufficient_context_response("insufficient context") == True
    assert is_insufficient_context_response("I don't know") == False
    assert is_insufficient_context_response("The answer is 42") == False
    
    print("is_insufficient_context_response: all tests passed")

if __name__ == "__main__":
    test_keyword_score()
    test_insufficient_context()
    print("\nAll scorer unit tests passed.")