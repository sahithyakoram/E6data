# Agentic Evaluation Framework

[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Overview
The **Agentic Evaluation Framework** provides a scalable and interpretable solution to evaluate AI agents across multiple dimensions, including instruction-following, coherence & accuracy, hallucination detection, assumption control, style, and length. It supports **batch processing**, integrates **LLM-based AI judges**, and generates **leaderboards, reports, and visualizations** for large-scale agent evaluation.

## Features
- Hybrid scoring: heuristic rules + LLM-based evaluation
- Supports thousands of agent responses in batch
- Detailed scoring across 6 key dimensions
- Explainable feedback for low/high scores
- Streamlit dashboard for visualization of trends and leaderboards

## Directory Structure
```yaml
src/evaluation/
  assumption_control.py: Flags speculative language
  coherence_accuracy.py: Grammar & spelling checks
  hallucination_detection.py: Detects unverifiable or exaggerated claims
  instruction_following.py: Semantic similarity to prompt
  style_matching.py: Penalizes informal/casual language
  length_penalty.py: Penalizes overly short or long responses
  evaluator.py: Runs all evaluators on a response
  evaluate_with_llm.py: Integrates LLM-based AI judge
  batch_runner.py: Processes batches, outputs leaderboard
  data_loader.py: Converts dataset to internal format
  generate_batch.py: Generates synthetic agent prompts/responses
  tests/: Unit tests for evaluators
```

## Quickstart

### 1. Setup Environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

### 2. Generate or Load Data
```bash
python src/evaluation/generate_batch.py  # generate synthetic batch
# or use your own dataset with data_loader.py
```

### 3. Run Evaluation
```bash
python src/evaluation/batch_runner.py --input src/evaluation/data/large_batch.json \
                                      --output src/evaluation/data/eval_report.json \
                                      --weights src/evaluation/data/weights.json \
                                      --use_llm
```

### 4. Streamlit Dashboard
```bash
streamlit run src/dashboard/app.py
```
- Upload evaluation report JSON
- View leaderboard and explanations
- Analyze trends via charts and heatmaps

## Key Technical Approach
- **Hybrid Evaluation**: Combines rule-based heuristics with LLM-based scoring for robust evaluation.
- **Batch Processing**: Efficiently scores thousands of agent responses and produces interpretable leaderboards.
- **Explainable Scores**: Provides textual explanations for each dimension to enhance transparency.

## Tools, Technologies & Libraries
- Python 3.10+, Streamlit, `sentence-transformers`, `language-tool-python`
- Requests, JSON, re, Groq API for LLM evaluation
- HuggingFace datasets for sample prompts/responses
- Logging, pathlib, argparse, random for batch processing and reproducibility

## Stretch Goals Implemented
- **Explainability**: Each score includes a textual explanation of why it was assigned.
- **LLM Integration**: Uses Groq-powered AI judge to provide advanced evaluation beyond heuristics.
- **Domain Support**: Example dataset includes multiple domains (QA, summarization, reasoning).
## Add free api keys from Groq into the Groq_API_keys.json file under src
## Example Usage
```python
from evaluation.evaluator import evaluate_agent_response

agent_id = "Agent_001"
prompt = "Summarize the benefits of exercise."
response = "Exercise improves health and mental well-being."

result = evaluate_agent_response(agent_id, prompt, response)
print(result)
```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
For any questions, reach out to the author via GitHub or email.

---

**Note:** Ensure you have set your `GROQ_API_KEY` in environment variables for LLM-based scoring to work.

