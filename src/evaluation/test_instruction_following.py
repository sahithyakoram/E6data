from evaluation.instruction_following import score_instruction_following

prompt = "Summarize the article on climate change."
response = "The article discusses rising temperatures and global policies."

result = score_instruction_following(prompt, response)
print(result)
