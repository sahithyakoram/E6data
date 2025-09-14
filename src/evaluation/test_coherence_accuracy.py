from evaluation.coherence_accuracy import score_coherence_accuracy

responses = [
    "This is a well-formed sentence.",
    "This sentence have a error and bad grammar!"
]

for r in responses:
    result = score_coherence_accuracy(r)
    print(f"Response: {r}\nResult: {result}\n")
