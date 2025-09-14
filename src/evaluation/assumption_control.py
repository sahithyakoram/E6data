import re
from typing import Dict

# Speculative language cues
SPECULATIVE_CUES = [
    r"\bprobably\b", r"\bmight\b", r"\bcould be\b", r"\bperhaps\b",
    r"\bseems like\b", r"\bmay have\b", r"\bpossibly\b"
]

def score_assumption_control(response: str) -> Dict:
    """
    Flags speculative or assumptive language.
    Returns a score (0–10) and explanation.
    """
    lowered = response.lower()
    hits = sum(bool(re.search(p, lowered)) for p in SPECULATIVE_CUES)
    deduction = min(hits * 2, 10)
    score = round(10.0 - deduction, 2)
    explanation = f"{hits} speculative cue{'s' if hits != 1 else ''} found"
    return {"score": score, "explanation": explanation}
