import json
import random

SOURCE_PATH = "src/evaluation/data/real_responses.json"
OUTPUT_PATH = "src/evaluation/data/real_responses_with_reference.json"

def inject_references(n=50):
    with open(SOURCE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Select n entries to annotate
    annotated = random.sample(data, n)

    for entry in annotated:
        entry["reference"] = generate_reference(entry["prompt"])

    # Merge back into full dataset
    agent_ids_with_ref = {entry["agent_id"] for entry in annotated}
    for entry in data:
        if entry["agent_id"] in agent_ids_with_ref:
            entry["reference"] = next(a["reference"] for a in annotated if a["agent_id"] == entry["agent_id"])

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def generate_reference(prompt: str) -> str:
    # 🔧 Replace this with real references or use an LLM to generate them
    return f"A clear, factual answer to: {prompt}"

if __name__ == "__main__":
    inject_references(n=50)
