# Agentic Evaluation Framework

A scalable framework for automated evaluation of AI agent responses across multiple dimensions, combining rule-based heuristics and LLM-based scoring for explainable results.

---

## 🔹 Overview

As AI agents become more powerful, assessing their outputs is critical. This framework scores agent responses across six dimensions:

- Instruction-following
- Coherence & accuracy
- Hallucination detection
- Style matching
- Length penalty
- Assumption control

Supports batch processing, leaderboard generation, and interpretable evaluation reports.

---

## 🔹 Features

- **Hybrid Scoring**: Combines heuristic rules and LLM-based evaluation.
- **Batch Evaluation**: Process thousands of responses across multiple agents.
- **Explainable Scores**: Each dimension includes an explanation for the score.
- **Leaderboard**: Ranks agents based on weighted scoring.
- **Domain Support**: Works for QA, summarization, and reasoning tasks.

---

## 🔹 Project Structure

src/evaluation/
├── assumption_control.py        # Flags speculative language
├── coherence_accuracy.py        # Grammar & spelling checks
├── hallucination_detection.py   # Detects unverifiable or exaggerated claims
├── instruction_following.py     # Semantic similarity to prompt
├── style_matching.py            # Penalizes informal/casual language
├── length_penalty.py            # Penalizes overly short or long responses
├── evaluator.py                 # Runs all evaluators on a response
├── evaluate_with_llm.py         # Integrates LLM-based AI judge
├── batch_runner.py              # Processes batches, outputs leaderboard
├── data_loader.py               # Converts dataset to internal format
├── generate_batch.py            # Generates synthetic agent prompts/responses
└── tests/                       # Unit tests for evaluators


---

## 🔹 Installation

1. Clone the repository:

```bash
git clone https://github.com/<your-username>/agentic-evaluation-framework.git
cd agentic-evaluation-framework

python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate    # Windows

pip install -r requirements.txt

export GROQ_API_KEY="your_groq_api_key"   # Linux/Mac
# set GROQ_API_KEY=your_groq_api_key      # Windows

