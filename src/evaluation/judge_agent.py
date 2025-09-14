import os
from groq import Groq
import re

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def evaluate_with_llm(prompt: str, response: str, model: str = "llama-3.3-70b-versatile") -> dict:
    system_prompt = (
        "You are an expert evaluator. Score the response on a scale of 0–10 across these dimensions:\n"
        "- Instruction Following\n"
        "- Coherence & Accuracy\n"
        "- Hallucination Detection\n"
        "- Style Matching\n"
        "- Length Penalty\n"
        "- Assumption Control\n"
        "Also provide a brief explanation for each score."
    )

    user_input = f"Prompt: {prompt}\nResponse: {response}"

    try:
        chat_completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            temperature=0.3
        )

        reply = chat_completion.choices[0].message.content
        print("Raw LLM reply:\n", reply)
        return parse_llm_output(reply)

    except Exception as e:
        return {
            "scores": {},
            "explanations": {"error": f"Groq evaluation failed: {e}"}
        }


def parse_llm_output(text: str) -> dict:
    scores = {}
    explanations = {}

    # Match lines like: * **Instruction Following: 8**
    pattern = r"\*\s+\*\*(.+?):\s*(\d+)\*\*\s*\n(.*?)(?=\n\*|\Z)"

    matches = re.findall(pattern, text, re.DOTALL)

    for dim, score_str, explanation in matches:
        key = dim.strip().lower().replace(" ", "_")
        try:
            score = float(score_str.strip())
            scores[key] = score
            explanations[key] = explanation.strip()
        except ValueError:
            continue

    return {
        "scores": scores,
        "explanations": explanations
    }

