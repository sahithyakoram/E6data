import requests
import os
import json
import logging
import re
import time

def evaluate_with_llm(prompt: str, response: str, model: str = "llama-3.1-8b-instant") -> dict:
    scoring_prompt = """
You are an evaluation engine. Score the following response across these dimensions (0–10 scale):

1. instruction_following
2. coherence_accuracy
3. hallucination_detection
4. style_matching
5. length_penalty
6. assumption_control

Also provide a brief explanation for each dimension.
"""

    scoring_prompt += f"\nPrompt: {prompt}\nResponse: {response}\nReturn a JSON object with 'scores' and 'explanations'."

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        logging.error("❌ GROQ_API_KEY not set in environment.")
        return {
            "scores": {},
            "explanations": {}
        }

    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": scoring_prompt}
        ],
        "temperature": 0.3
    }

    logging.debug(f"🔧 Payload:\n{json.dumps(payload, indent=2)}")

    def parse_response_content(content: str) -> dict:
        content = re.sub(r"^```json|```$", "", content).strip()
        try:
            first_json = re.search(r"\{.*\}", content, re.DOTALL).group()
            parsed = json.loads(first_json)
            return {
                "scores": parsed.get("scores", {}),
                "explanations": parsed.get("explanations", {})
            }
        except Exception as e:
            logging.error(f"❌ Failed to parse JSON from Groq response: {e}")
            logging.debug(f"📩 Raw output:\n{content}")
            return {
                "scores": {},
                "explanations": {}
            }

    try:
        res = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
        res.raise_for_status()
        output = res.json()["choices"][0]["message"]["content"].strip()
        return parse_response_content(output)

    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 429:
            try:
                error_data = e.response.json()
                msg = error_data.get("error", {}).get("message", "")
                wait_time = float(re.search(r"try again in ([\d.]+)s", msg).group(1))
                logging.warning(f"⏳ Rate limit hit. Waiting {wait_time:.2f}s before retry...")
                time.sleep(wait_time + 0.5)
                return evaluate_with_llm(prompt, response, model=model)  # retry once
            except Exception as parse_err:
                logging.error(f"❌ Failed to parse wait time: {parse_err}")
        logging.error(f"❌ Groq scoring failed: {e}")
        logging.error(f"📩 Response content: {e.response.text}")
        return {
            "scores": {},
            "explanations": {}
        }

    except Exception as e:
        logging.error(f"❌ Unexpected error: {e}")
        return {
            "scores": {},
            "explanations": {}
        }
