from judge_agent import evaluate_with_llm

prompt = "Explain the concept of recursion in programming."
response = "Recursion is when a function calls itself to solve smaller instances of a problem."

result = evaluate_with_llm(prompt, response)

print("Scores:")
for k, v in result["scores"].items():
    print(f"  {k}: {v}")

print("\nExplanations:")
for k, v in result["explanations"].items():
    print(f"  {k}: {v}")
