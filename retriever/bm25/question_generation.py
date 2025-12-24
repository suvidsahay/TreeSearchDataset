import os
import re
from typing import Dict, List, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.messages.human import HumanMessage, HumanMessageChunk

def load_openai_key():
    if not os.getenv("OPENAI_API_KEY"):
        raise Exception("Set your OPENAI_API_KEY!")

def generate_seed_questions(doc1, num_questions=3):
    """
    Generates initial 1-hop seed questions from a single document.
    Enforces strict answer length (< 5 words) and prohibits 'bridge' answers.
    """
    prompt = f"""You are an expert at creating the first step of a multi-hop reasoning chain.
Your task is to generate {num_questions} "seed" questions from the document provided.

**CRITICAL CONSTRAINTS:**
1. **Answer Length:** The Ground Truth Answer MUST be **less than 5 words** (e.g., a specific name, date, number, or location).
2. **Complexity:** Avoid simple "what is" questions. Focus on specific details (roles, awards, stats).
3. **Bridge Prohibition:** The answer MUST NOT be the name of the entity that connects this document to others. The answer must be a specific fact *about* that entity or related to it.

**Good Seed Question Example:**
- Document Excerpt: "...he played Jaime Lannister in the HBO series Game of Thrones..."
- Good Question: "What major television series, known for its fantasy elements, featured Nikolaj Coster-Waldau in a prominent role?"
- Answer: "Game of Thrones" (< 5 words, specific entity).

**Bad Seed Question Example:**
- Document Excerpt: "...Coster-Waldau was born in Rudkøbing, Denmark..."
- Bad Question: "In what Danish town was Coster-Waldau born?"
- Answer: "Rudkøbing" (Too simple/direct).

Here is the document to use:
---
Document 1:
{doc1}
---

For each question you generate, you MUST provide the question, a brief explanation of the "hook" entity, and the ground truth answer. Use the following format exactly and separate each question explanation answer by \"---\":
Question: [Your question here]
Explanation: [Identify the key entity in your question that can link to other topics.]
Answer: [The ground truth answer to your question]
---
"""
    chat = ChatOpenAI(temperature=0.7, model_name="gpt-4o")
    response = chat.invoke([HumanMessage(content=prompt)])
    # print("\nGenerated Seed Questions...") # Silenced to keep output clean
    return response.content

def generate_questions(doc1, doc2, num_questions=3):
    # Kept for compatibility, but updated constraints just in case
    prompt = f"""Generate {num_questions} questions combining both documents.
    Constraint: Answers must be LESS than 5 words.
    Constraint: Do not ask for the name of the shared entity.
    """
    chat = ChatOpenAI(temperature=0.5, model_name="gpt-4o")
    response = chat.invoke([HumanMessage(content=prompt)])
    return response.content

def generate_multihop_questions(documents: list, num_questions=3):
    """
    Generates multi-hop questions that require combining information from all provided passages.
    Enforces strict < 5 word answers and reasoning chains.
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

**STRICT CONSTRAINTS:**
1. **Answer Length:** The final answer MUST be **less than 5 words** (e.g., a specific date, name, number).
2. **No Bridge Answers:** The answer CANNOT be the name of the entity that bridges the documents. The answer must be a fact *derived* from the reasoning.
3. **Dependency:** The question must NOT be answerable by looking at a single document. It must require synthesis.
4. **Naturalness:** Phrasing must sound natural. Avoid "In Document 1..." phrasing.

**Good Question Example (Logical Bridge):**
- Doc 1: "Nikolaj Coster-Waldau starred in the 2011 film Headhunters."
- Doc 2: "The Fox Network was the most-watched network in the 2007-08 season. Nikolaj Coster-Waldau starred in the Fox series 'New Amsterdam'."
- GOOD Question: "Which television network, the most-watched in the U.S. during the 2007-08 season, aired a series starring an actor from the 2011 film Headhunters?"
- Answer: "Fox Network" (Specific fact, < 5 words).
- Why it's good: Uses Doc 1 (Headhunters -> Actor) -> Doc 2 (Actor -> Network -> Stat).

**Bad Question Example (Simple Mashup):**
- BAD Question: "What film did Nikolaj Coster-Waldau star in in 2011, and which network was the most-watched in 2007-08?"
- Why it's bad: Two unrelated questions stitched together.

---
**Documents to use:**
{documents_section}
---

For each question you generate, you MUST provide the question, a brief explanation of the logical bridge, and the ground truth answer.

Use the following format exactly:
Question: [Your question here]
Explanation: [Your one-sentence explanation of the logical reasoning chain]
Answer: [The ground truth answer (< 5 words)]
--- (Use this to split '---' each question, explanation, answer triplet in between)
"""
    chat = ChatOpenAI(temperature=0.7, model_name="gpt-4o")
    response = chat.invoke([HumanMessage(content=prompt)])
    print(f"\nGenerated Questions from {num_docs} documents:")
    return response.content

def generate_multihop_questions_v2(original_question, new_passage, num_questions=3):
    # Compatibility stub
    return ""

def revise_question(documents: list, failed_question_data: Dict[str, str], minimal_passages_used: List[Tuple[str, str]], naturalness_details: Dict) -> str:
    """
    Revises a question that failed verification (because it didn't use all docs).
    """
    num_docs = len(documents)
    formatted_docs = [f"Document {i + 1}:\n{doc}" for i, doc in enumerate(documents)]
    documents_section = "\n\n---\n\n".join(formatted_docs)

    original_q = failed_question_data.get("question", "N/A")
    used_titles = [title for title, _ in minimal_passages_used]
    used_passages_section = "\n".join([f"- '{title}'" for title in used_titles])
    ld_score = naturalness_details.get("logical_dependency_score", "N/A")
    combines_score = naturalness_details.get("combines_passages_score", "N/A")

    naturalness_feedback = f"""
**PREVIOUS ATTEMPT QUALITY SCORES (Out of 5.0):**
- Logical Dependency Score: {ld_score}
- Combines Passages Score: {combines_score}
"""
    prompt = f"""You previously attempted to generate a complex question using the {num_docs} documents provided below.
The generated question failed verification because it did NOT require all {num_docs} documents to answer.

{naturalness_feedback}

**CRITICAL FEEDBACK FROM REVIEWER (Hop Count Failure):**
The original question, "{original_q}", only required {len(minimal_passages_used)} passages to answer.
The required passages were from the documents titled:
{used_passages_section}

**YOUR TASK:** REVISE and generate a **NEW, single, high-quality question** that **guarantees** a logical dependency across **ALL {num_docs} documents** (i.e., {num_docs} distinct passages).

**Further Requirements:**
- **Answer Length:** Must be **less than 5 words**.
- **No Bridge Answers:** Do not ask for the linking entity's name.
- **Dependency:** Ensure removing ANY document makes the question unanswerable.

---
**DOCUMENTS (Targeted for use):**
{documents_section}
---

Use the following format exactly for your new attempt:
Question: [Your NEW, revised question here]
Explanation: [Your one-sentence explanation of the new logical reasoning chain]
Answer: [The ground truth answer (< 5 words)]
"""
    chat = ChatOpenAI(temperature=0.7, model_name="gpt-4o")
    response = chat.invoke([HumanMessage(content=prompt)])
    return response.content

def generate_simplified_question(original_question: str, original_answer: str, kept_passages: List[str], num_hops: int) -> str:
    """
    Generates a simpler question using only the kept passages (Top-Down Phase 2).
    Includes strong constraints to prevent 1-hop generation (lazy simplification).
    """
    formatted_docs = []
    for i, doc in enumerate(kept_passages):
        formatted_docs.append(f"Document {i + 1}:\n{doc}")
    documents_section = "\n\n---\n\n".join(formatted_docs)

    prompt = f"""You are a question simplification expert.
We are simplifying a complex question chain by removing one document.

**Original Context:**
- Original Question: "{original_question}"
- Original Answer: "{original_answer}"
- Previously used {num_hops + 1} documents. We have removed one.

**Your Task:**
Create a NEW question that relies **ONLY** on the {num_hops} documents provided below. 
The new question should retain the core topic of the original but must be simpler (requiring exactly {num_hops} hops).

**Documents to Use:**
{documents_section}

**CRITICAL CONSTRAINT:**
The new question must **Bridge** facts from ALL {num_hops} documents provided above.
- If you write a question that can be answered by Document 1 alone, YOU FAIL.
- If you write a question that can be answered by Document 2 alone, YOU FAIL.
- You MUST combine information (e.g., "The actor from [Doc 1] starred in which movie described in [Doc 2]?").

**Requirements:**
- The question MUST be answerable using only these {num_hops} documents.
- The question MUST NOT imply knowledge from the removed document.
- The answer must be a single fact found in these texts (**less than 5 words**).

Use this format:
Question: [Your new simplified question]
Explanation: [Brief explanation of how it connects the {num_hops} docs]
Answer: [New Answer (< 5 words)]
---
"""
    chat = ChatOpenAI(temperature=0.7, model_name="gpt-4o")
    response = chat.invoke([HumanMessage(content=prompt)])
    return response.content
