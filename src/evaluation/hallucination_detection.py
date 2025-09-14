import re
from typing import Dict

# List of speculative or vague phrases
HALLUCINATION_CUES = [
    r"\bclearly\b", r"\bobviously\b", r"\beveryone knows\b",
    r"\bas is well known\b", r"\bit is evident\b", r"\bwithout a doubt\b",
    r"\bscientists agree\b", r"\bno one disputes\b", r"\bthe fact is\b"
]

def score_hallucination(response: str) -> Dict:
    """
    Heuristic-based hallucination detection.
    Flags speculative or unverifiable phrases.
    Returns a score (0–10, higher is better) and explanation.
    """
    lowered = response.lower()
    hits = sum(bool(re.search(pattern, lowered)) for pattern in HALLUCINATION_CUES)
    word_count = max(len(response.split()), 1)

    # Deduct 2 points per cue, capped at 10
    deduction = min(hits * 2, 10)
    score = round(10.0 - deduction, 2)
    explanation = f"{hits} speculative cue{'s' if hits != 1 else ''} found in {word_count} words"

    return {"score": score, "explanation": explanation}
