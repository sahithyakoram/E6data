from typing import Dict
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")  # Fast, lightweight

def score_instruction_following(prompt: str, response: str) -> Dict:
    similarity = util.cos_sim(model.encode(prompt), model.encode(response)).item()
    score = round(similarity * 10, 2)  # Scale to 0–10

    explanation = "High semantic match" if score > 7 else "Partial or weak alignment"
    return {
        "score": score,
        "explanation": explanation
    }
