import language_tool_python
from typing import Dict

# Initialize the LanguageTool client (English)
tool = language_tool_python.LanguageTool('en-US')

def score_coherence_accuracy(response: str) -> Dict:
    """
    Checks grammar and spelling errors in the response.
    Returns a score 0–10 (higher is better) and an explanation.
    """
    matches = tool.check(response)
    error_count = len(matches)
    word_count = max(len(response.split()), 1)

    # Scale errors to a 0–10 score: fewer errors → higher score
    deduction = (error_count / word_count) * 10
    raw_score = max(0.0, 10.0 - deduction)
    score = round(raw_score, 2)

    explanation = (
        f"{error_count} issue{'s' if error_count!=1 else ''} "
        f"detected in {word_count} words"
    )
    return {"score": score, "explanation": explanation}
