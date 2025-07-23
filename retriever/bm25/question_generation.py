import os
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

def load_openai_key():
    if not os.getenv("OPENAI_API_KEY"):
        raise Exception("Set your OPENAI_API_KEY!")

def generate_questions(doc1, doc2, num_questions=3):
    # This new prompt now requires the model to generate the ground truth answer as well.
    prompt = f"""You are an expert at generating insightful, natural-sounding, multi-hop questions.
Your task is to generate {num_questions} questions that can ONLY be answered by synthesizing information from BOTH Document 1 and Document 2.
A high-quality question must be coherent, find a logical "bridge" between the documents, and sound natural.
**CRITICAL INSTRUCTIONS**:
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
    chat = ChatOpenAI(temperature=0.5, model_name="gpt-4o")
    response = chat.invoke([HumanMessage(content=prompt)])
    
    print("\nGenerated Questions, Explanations, and Ground Truth Answers:")
    print(f"Document 1 snippet: {doc1[:100]}...")
    print(f"Document 2 snippet: {doc2[:100]}...")

    return response.content
