import json
from pathlib import Path
import random

PROMPTS = [
    "Summarize the benefits of exercise.",
    "Explain how photosynthesis works.",
    "Describe the impact of climate change.",
    "What is the capital of France?",
    "List three programming languages."
]

RESPONSES = [
    "Clearly, exercise is the best thing ever.",
    "Photosynthesis is obviously how plants eat sunlight.",
    "Everyone knows climate change is fake.",
    "Paris is the capital of France.",
    "Python, JavaScript, and C++ are popular languages."
]

def generate_batch(n=1000, path="src/evaluation/data/large_batch.json"):
    data = []
    for i in range(n):
        prompt = random.choice(PROMPTS)
        response = random.choice(RESPONSES)
        data.append({
            "agent_id": f"Agent_{i:04d}",
            "prompt": prompt,
            "response": response
        })
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"✅ Generated {n} responses at {path}")

if __name__ == "__main__":
    generate_batch()
