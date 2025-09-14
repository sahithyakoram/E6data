from typing import Dict

def score_length_penalty(response: str) -> Dict:
    """
    Penalizes overly short or long responses.
    Returns a score (0–10) and explanation.
    """
    word_count = len(response.split())
    if word_count < 5:
        score = 3.0
        explanation = "Too short"
    elif word_count > 50:
        score = 5.0
        explanation = "Too long"
    else:
        score = 10.0
        explanation = "Optimal length"
    return {"score": score, "explanation": f"{word_count} words – {explanation}"}
