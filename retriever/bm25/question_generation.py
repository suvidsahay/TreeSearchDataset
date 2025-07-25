import os
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

def load_openai_key():
    if not os.getenv("OPENAI_API_KEY"):
        raise Exception("Set your OPENAI_API_KEY!")

def generate_questions(doc1, doc2, num_questions=3):
    # This new prompt now requires the model to generate the ground truth answer as well.
    prompt = f"""You are given two documents.

Your task is to generate {num_questions} clear, natural-sounding, fact-based questions that require combining information from BOTH Document 1 and Document 2 to answer.

Requirements:
· The question must not be answerable by reading only one document.
· The question must not be unanswerable.
· The question must not be a yes/no question.
· The answer must be a single, objective fact (e.g., a name, number, date, or location) explicitly stated in the documents.
· The question must sound natural, as if written by a human.

Critical Test: Before generating each question, you must verify internally:
1. What fact comes only from Document 1.
2. What fact comes only from Document 2.
3. Why both are required to answer. If the answer can still be found without using a unique fact from one of the documents, revise your question.

Example of a GOOD question: Document 1: Paris is the capital of France. Document 2: The Eiffel Tower was completed in 1889. Question: Which famous landmark in the capital of France was completed in 1889?

Examples of BAD questions:
· What is the capital of France? (answerable from one document)
· In the year of 1889 completion, what is the landmark that located inside Paris capital of France? (unnatural phrasing)
· Is the Eiffel Tower located in Paris? (yes/no question)

Important:
· Do not mention or imply where each fact comes from.
· Do not assume any relationship between facts unless it is explicitly stated.
· Ensure that named entities are referred to clearly and unambiguously.

For each question you generate, you MUST provide three things:
1.  The question itself.
2.  A brief, one-sentence explanation of the bridge connecting the two documents.
3.  The ground truth answer to the question, synthesized ONLY from the provided documents.
Use the following format exactly, with "---" as a separator between questions:
Question: [Your question here]
Explanation: [Your one-sentence explanation here]
Answer: [The ground truth answer to your question]
---

Question: [Your next question here]

Explanation: [Your next explanation here]

Answer: [The ground truth answer to your next question]

---

...


Here are the documents to use:
---

Document 1:
{doc1}

---

Document 2:
{doc2}

---  """

    # Using gpt-4o for the best quality generation
    chat = ChatOpenAI(temperature=0.5, model_name="gpt-4o-mini")
    response = chat.invoke([HumanMessage(content=prompt)])
    
    print("\nGenerated Questions, Explanations, and Ground Truth Answers:")
    print(f"Document 1 snippet: {doc1[:100]}...")
    print(f"Document 2 snippet: {doc2[:100]}...")

    return response.content
