import os
import re
from langchain_openai import ChatOpenAI
from langchain_core.messages.human import HumanMessage, HumanMessageChunk
#from langchain_core.messages import HumanMesaage
#from langchain.schema import HumanMessage

def load_openai_key():
    if not os.getenv("OPENAI_API_KEY"):
        raise Exception("Set your OPENAI_API_KEY!")

def generate_seed_questions(doc1, num_questions=3):
    prompt = f"""You are an expert at creating the first step of a multi-hop reasoning chain.
Your task is to generate {num_questions} "seed" questions from the document provided.

A good "seed" question identifies a key entity, event, or concept from the document that can serve as a logical **"hook"** to connect to information in other, unknown documents. Avoid questions that are too specific or self-contained.

**Good Seed Question Example:**
- Document Excerpt: "...he played Jaime Lannister in the HBO series Game of Thrones..."
- Good Question: "What major television series, known for its fantasy elements, featured Nikolaj Coster-Waldau in a prominent role?"
- Why it's good: The answer, "Game of Thrones," is a major entity that can easily link to other articles about HBO, other actors, or the fantasy genre.

**Bad Seed Question Example:**
- Document Excerpt: "...Coster-Waldau was born in Rudkøbing, Denmark..."
- Bad Question: "In what Danish town was Coster-Waldau born?"
- Why it's bad: The answer, "Rudkøbing," is too specific and unlikely to be a bridge to other diverse topics.

Here is the document to use:
---
Document 1:
{doc1}
---

For each question you generate, you MUST provide the question, a brief explanation of the "hook" entity, and the ground truth answer. Use the following format exactly:
Question: [Your question here]
Explanation: [Identify the key entity in your question that can link to other topics.]
Answer: [The ground truth answer to your question]
---
"""

    # Using gpt-4o for the best quality generation
    chat = ChatOpenAI(temperature=0.7, model_name="gpt-4o")
    response = chat.invoke([HumanMessage(content=prompt)])

    print("\nGenerated Seed Questions, Explanations, and Ground Truth Answers:")
    print(f"Document 1 snippet: {doc1[:100]}...")

    return response.content

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
    chat = ChatOpenAI(temperature=0.5, model_name="gpt-4o")
    response = chat.invoke([HumanMessage(content=prompt)])
    
    print("\nGenerated Questions, Explanations, and Ground Truth Answers:")
    print(f"Document 1 snippet: {doc1[:100]}...")
    print(f"Document 2 snippet: {doc2[:100]}...")

    return response.content


def generate_multihop_questions(documents: list, num_questions=1): # Reduced default to 1 for quality
    """
    Generates multi-hop questions that require combining information from all provided passages.
    This function is generalized to handle n passages.
    """
    num_docs = len(documents)
    if num_docs < 2:
        print("Warning: At least two documents are required to generate multi-hop questions.")
        return ""

    formatted_docs = []
    for i, doc in enumerate(documents):
        formatted_docs.append(f"Document {i + 1}:\n{doc}")
    documents_section = "\n\n---\n\n".join(formatted_docs)

    prompt = f"""You are an expert at creating complex reasoning questions that require a logical bridge between facts.
You are given {num_docs} documents. Your task is to generate {num_questions} clear, fact-based questions that require a **true reasoning chain** across all {num_docs} documents.

**CRITICAL REQUIREMENT: The question must form a single, coherent thought.** It must NOT be two separate questions stitched together with "and". The facts must be logically dependent on each other.

**Good Question Example (Logical Bridge):**
- Doc 1: "Nikolaj Coster-Waldau starred in the 2011 film Headhunters."
- Doc 2: "The Fox Network was the most-watched network in the 2007-08 season. Nikolaj Coster-Waldau starred in the Fox series 'New Amsterdam'."
- GOOD Question: "Which television network, the most-watched in the U.S. during the 2007-08 season, aired a series starring an actor from the 2011 film Headhunters?"
- Why it's good: You must use the fact from Doc 1 (the film) to identify the actor, then use that actor's name to find the network in Doc 2, and finally link that network to the "most-watched" fact. This is a true reasoning chain.

**Bad Question Example (Simple Mashup):**
- BAD Question: "What film did Nikolaj Coster-Waldau star in in 2011, and which network was the most-watched in 2007-08?"
- Why it's bad: This is just two unrelated questions. There is no logical dependency. DO NOT generate questions like this.

---
**Further Requirements:**
- The final answer must be a single, objective fact (e.g., a name, number, date).
- The question must not be a yes/no question.
- A unique fact must be required from EACH document.

For each question you generate, you MUST provide the question, a brief explanation of the logical bridge, and the ground truth answer.

Use the following format exactly:
Question: [Your question here]
Explanation: [Your one-sentence explanation of the logical reasoning chain]
Answer: [The ground truth answer to your question]
---

Here are the documents to use:
---
{documents_section}
---
"""

    chat = ChatOpenAI(temperature=0.7, model_name="gpt-4o")
    response = chat.invoke([HumanMessage(content=prompt)])

    print(f"\nGenerated Questions from {num_docs} documents:")
    for i, doc in enumerate(documents):
        print(f"Document {i + 1} snippet: {doc[:100]}...")

    return response.content
