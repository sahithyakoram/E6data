from datasets import load_dataset
from pathlib import Path
import json
import html
import random

# 📦 Load and slice Hugging Face dataset
ds = load_dataset("Dahoas/sft-gptj-synthetic-prompt-responses")
subset = ds["train"].select(range(200))

# 🧠 Optional reference generator stub
def generate_reference(prompt: str) -> str:
    # Placeholder for future model-based reference generation
    return ""

# 🔄 Convert to internal format
converted = []
skipped = 0

for i, row in enumerate(subset):
    prompt = row.get("prompt", "").strip()
    response = row.get("response", "").strip()

    if not prompt or not response:
        skipped += 1
        continue

    converted.append({
        "agent_id": f"agent_{i}",
        "prompt": prompt,
        "response": response
    })

print(f"✅ Converted {len(converted)} entries, skipped {skipped} malformed rows")
domains = ["QA", "summarization", "reasoning"]

converted.append({
    "agent_id": f"agent_{i}",
    "prompt": prompt,
    "response": response,
    "domain": random.choice(domains)
})

# 💾 Save to real_responses.json
output_path = Path("src/evaluation/data/real_responses_1000.json")
try:
    json_str = json.dumps(converted, indent=2)
    with output_path.open("w", encoding="utf-8") as f:
        f.write(json_str)
    print(f"📁 Saved to {output_path}")
except Exception as e:
    print(f"❌ JSON serialization failed: {e}")
