from evaluation.hallucination_detection import score_hallucination

responses = [
    "Clearly, this is the best solution.",
    "The algorithm performs well on benchmarks.",
    "Everyone knows this method is superior."
]

for r in responses:
    result = score_hallucination(r)
    print(f"Response: {r}\nResult: {result}\n")
