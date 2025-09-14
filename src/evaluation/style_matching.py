import re
from typing import Dict

# Informal cues to penalize
INFORMAL_PHRASES = [
    r"\byou know\b", r"\blike\b", r"\bbasically\b", r"\bkinda\b",
    r"\bwhatever\b", r"\bjust saying\b", r"\buh\b", r"\bum\b"
]

def score_style_matching(response: str) -> Dict:
    """
    Penalizes informal or casual language.
    Returns a score (0–10) and explanation.
    """
    lowered = response.lower()
    hits = sum(bool(re.search(p, lowered)) for p in INFORMAL_PHRASES)
    deduction = min(hits * 2, 10)
    score = round(10.0 - deduction, 2)
    explanation = f"{hits} informal phrase{'s' if hits != 1 else ''} detected"
    return {"score": score, "explanation": explanation}
