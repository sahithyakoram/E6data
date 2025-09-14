from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

def score_reference_alignment(response: str, reference: str, mode: str = "semantic") -> dict:
    if not reference.strip():
        return {
            "score": 10.0,
            "explanation": "No reference provided; full score by default."
        }

    if mode == "token":
        ratio = SequenceMatcher(None, response.strip(), reference.strip()).ratio()
        score = round(ratio * 10, 2)
        return {
            "score": score,
            "explanation": f"Token overlap similarity: {score}/10"
        }

    # Default: semantic
    embeddings = model.encode([response.strip(), reference.strip()], convert_to_tensor=True)
    similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
    score = round(similarity * 10, 2)
    return {
        "score": score,
        "explanation": f"Semantic similarity: {score}/10"
    }
