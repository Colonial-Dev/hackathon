from t5_summarizer import T5QASummarizer
import torch

document = """
Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think, learn, and problem-solve. These machines or systems are designed to perform tasks that typically require human intelligence, such as understanding natural language, recognizing images, making decisions, and solving complex problems.
"""
question = "What is Artificial Intelligence?"

qa_summarizer = T5QASummarizer(model_name="t5-large", device="cuda" if torch.cuda.is_available() else "cpu")

answer = qa_summarizer.answer_question(document, question)

print(f"Question: {question}")
print(f"Answer: {answer}")