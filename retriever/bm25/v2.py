import json
import os
import itertools
from typing import List, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.messages.human import HumanMessage

def _generate_answer_from_context(context, question, chat_model):
    """
    A generic helper to generate an answer given a context and question.
    If context is None, it asks for general knowledge.
    """
    if context:
        prompt = f"""Using ONLY the information from the provided 'Context', answer the 'Question'.
If the information is not available in the context, you MUST respond with the exact phrase: "Not answerable from passages."
The answer should be concise (less than 5 words) if possible.

Context:
{context}

Question: {question}

Answer:"""
    else: 
        prompt = f"""Answer the following question from your general knowledge, without using any external documents.
If you do not know the answer, respond with "I don't know".
Keep the answer concise.

Question: {question}

Answer:"""

    response = chat_model.invoke([HumanMessage(content=prompt)])
    return response.content

def evaluate_answer_objectivity(answer1, answer2, chat_model):
    """
    Asks the LLM to compare two answers for semantic similarity (equivalence).
    Returns a dictionary with 'is_equivalent' (bool) and 'similarity_score' (1-5).
    """
    eval_prompt = f"""You are an expert semantic evaluator. Your task is to compare two answers to the same question and determine if they are semantically equivalent.

Answer 1: "{answer1}"
Answer 2: "{answer2}"

Are these two answers saying the same thing? 
- "1994" and "The year 1994" -> Equivalent.
- "Brad Pitt" and "He" (if context implies Brad Pitt) -> Equivalent.
- "Paris" and "London" -> NOT Equivalent.

Evaluate their semantic similarity on a scale of 1 to 5 (1 = Very Dissimilar, 5 = Identical in meaning).
Also provide a boolean "is_equivalent".

Respond ONLY with a valid JSON object in the format:
{{"is_equivalent": <true/false>, "similarity_score": <score_integer>, "justification": "<brief justification>"}}
"""
    response = chat_model.invoke([HumanMessage(content=eval_prompt)])
    try:
        json_start = response.content.find('{')
        json_end = response.content.rfind('}') + 1
        json_data = response.content[json_start:json_end]
        return json.loads(json_data)
    except Exception:
        return {"is_equivalent": False, "similarity_score": 0, "justification": "Evaluation parsing error."}

def evaluate_question_naturalness_dynamic(question: str, passages: List[str], chat_model):
    """
    Asks an LLM to dynamically score a question's naturalness based on N passages.
    """
    num_passages = len(passages)
    combines_passages_text = f"Combines Facts from All {num_passages} Passages"
    requires_all_text = f"No Subset of Passages Is Enough"

    eval_prompt = f"""You are an expert in evaluating multi-hop question quality. Evaluate the following question based on the {num_passages} passages it was generated from.

CRITERIA (Score 1-5):
1.  **One Clear Question:** Is this a single, clear question, not multiple questions joined by "and"?
2.  **{combines_passages_text}:** Does the question meaningfully require information from ALL {num_passages} passages?
3.  **{requires_all_text}:** Can the question be answered using a smaller subset of the passages? (If yes, score lower).
4.  **Logical Dependency:** Are the facts from the passages logically chained together in a reasoning path?
5.  **HotpotQA-Style Reasoning:** Does the question require complex reasoning across multiple facts, typical of the HotpotQA benchmark?
6.  **Objectivity:** Is the question fact-based and answerable without speculation?

Question: "{question}"

Return your answer as a valid JSON object in this exact format, using the dynamic key names:
{{
  "clear_single_question_score": <1-5>,
  "combines_passages_score": <1-5>,
  "requires_all_passages_score": <1-5>,
  "logical_dependency_score": <1-5>,
  "hotpot_style_score": <1-5>,
  "objectivity_score": <1-5>,
  "justification": "<one short explanation for your scores>"
}}
"""

    response = chat_model.invoke([HumanMessage(content=eval_prompt)])
    try:
        json_start = response.content.find('{')
        json_end = response.content.rfind('}') + 1
        json_data = response.content[json_start:json_end]
        return json.loads(json_data)
    except Exception as e:
        # print(f"CRITICAL PARSING ERROR in dynamic naturalness evaluation: {e}")
        return {
            "clear_single_question_score": 0, "combines_passages_score": 0,
            "requires_all_passages_score": 0, "logical_dependency_score": 0,
            "hotpot_style_score": 0, "objectivity_score": 0,
            "justification": "Evaluation parsing error."
        }

def verify_question_N_docs(passages: List[str], question: str, ground_truth_answer: str) -> dict:
    """
    OPTIMIZED VERIFIER FOR PHASE 1:
    1.  **General Knowledge Check (Fail Fast):** If the question can be answered without ANY docs, it fails immediately.
    2.  **Subset Check (N-1):** Checks if any subset of size N-1 can answer the question. If yes, it fails "Requires All".
    3.  **Full Set Check:** Confirms the full set CAN answer the question.
    """
    if not passages:
        return {"verification_error": "No passages provided"}

    num_passages = len(passages)
    chat = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", openai_api_key=os.getenv("OPENAI_API_KEY"))

    # --- 1. GENERAL KNOWLEDGE CHECK (FAIL FAST) ---
    # Ask LLM to answer from general knowledge (no context)
    answer_gen_knowledge = _generate_answer_from_context(None, question, chat)
    
    # Compare with Ground Truth
    # If the LLM produces the correct answer from thin air, the question is not "multi-hop document dependent".
    obj_eval_gen = evaluate_answer_objectivity(ground_truth_answer, answer_gen_knowledge, chat)
    
    if obj_eval_gen.get("is_equivalent", False):
        print(f"    [Verifier] ❌ Failed: Answerable from General Knowledge. (Ans: '{answer_gen_knowledge}')")
        return {
            "requires_all_passages": False,
            "answerable_with_subset": False, # Technically answerable with 0 subset
            "answerable_from_general_knowledge": True,
            "not_answerable": False,
            "verification_details": {"general_knowledge": answer_gen_knowledge}
        }

    # --- 2. SUBSET CHECK (Optimization: Only size N-1) ---
    # We generate all subsets of size N-1. If ANY of them works, the question fails verification.
    indices = range(num_passages)
    subset_indices = list(itertools.combinations(indices, num_passages - 1))
    
    can_answer_with_subset = False
    verification_details = {"general_knowledge_check": "passed"}

    print(f"    [Verifier] 🔍 Checking {len(subset_indices)} subsets of size {num_passages-1}...")

    for subset in subset_indices:
        subset_passages = [passages[i] for i in subset]
        subset_text = "\n\n".join([f"Passage {i+1}: {p}" for i, p in enumerate(subset_passages)])
        
        # Ask LLM to answer using only subset
        generated_ans = _generate_answer_from_context(subset_text, question, chat)
        
        # Check if answerable (not "Not answerable") AND matches Ground Truth
        if "not answerable" not in generated_ans.lower():
            is_equiv = evaluate_answer_objectivity(ground_truth_answer, generated_ans, chat).get("is_equivalent", False)
            verification_details[f"subset_{subset}"] = {"answer": generated_ans, "matches_gt": is_equiv}
            
            if is_equiv:
                can_answer_with_subset = True
                print(f"    [Verifier] ⚠️ Answerable with subset {subset}. Answer: '{generated_ans}'")
                # Optimization: We could break here, but we'll let it finish for complete logging if needed.
                # break 
        else:
             verification_details[f"subset_{subset}"] = {"answer": "Not answerable", "matches_gt": False}


    # --- 3. FULL SET CHECK ---
    # Finally, confirm that the FULL set DOES work.
    full_text = "\n\n".join([f"Passage {i+1}: {p}" for i, p in enumerate(passages)])
    generated_full = _generate_answer_from_context(full_text, question, chat)
    
    is_equiv_full = False
    if "not answerable" not in generated_full.lower():
        is_equiv_full = evaluate_answer_objectivity(ground_truth_answer, generated_full, chat).get("is_equivalent", False)

    return {
        "requires_all_passages": is_equiv_full and not can_answer_with_subset,
        "answerable_with_subset": can_answer_with_subset,
        "answerable_from_general_knowledge": False,
        "not_answerable": not is_equiv_full,
        "verification_details": verification_details
    }


def verify_question_phase2(passages: List[str], question: str, ground_truth_answer: str) -> dict:
    """
    STRICT VERIFIER FOR PHASE 2 (Simplification).
    
    This uses the EXACT SAME logic as Phase 1 (verify_question_N_docs).
    
    Why? Because Phase 2 takes a question that was 4 hops (and used 4 docs) and removes 1 doc.
    We then generate a NEW question using the remaining 3 docs.
    
    We need to verify that this NEW question:
    1. Is NOT answerable by General Knowledge.
    2. Is NOT answerable by any subset of the 3 docs (e.g., just 2 docs).
    3. IS answerable by the full set of 3 docs.
    
    If it passes this, it is a valid "3-hop" question derived via simplification.
    """
    return verify_question_N_docs(passages, question, ground_truth_answer)


def get_required_passages(question: str, answer: str, candidate_passages: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """
    Directly asks an LLM to identify the minimal set of passages required to answer a question.
    This is used in the "Revision" step to diagnose why a question failed.
    """
    if not candidate_passages:
        return []

    chat = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=os.getenv("OPENAI_API_KEY"))

    # Create labeled passages and a map to retrieve them later
    passage_map = {}
    formatted_passages = []
    for i, (title, text) in enumerate(candidate_passages):
        label = chr(65 + i)  # A, B, C...
        passage_map[label] = (title, text)
        formatted_passages.append(f"Passage {label} (from '{title}'):\n{text}")

    all_passages_text = "\n\n---\n\n".join(formatted_passages)

    prompt = f"""
You are an expert analyst. Your task is to identify the minimum set of passages required to construct the Ground Truth Answer for the given Question.

**Question:**
{question}

**Ground Truth Answer:**
{answer}

**Candidate Passages:**
{all_passages_text}

---
**Instructions:**
Analyze all passages. Identify the set of passages that are essential to answer the question.

- If a single passage is sufficient, return only its label (e.g., "B").
- If multiple passages are required, return their labels separated by commas (e.g., "A, C").
- If the answer is general knowledge and requires no passages, return the word "None".
- If the answer cannot be formed from the passages, also return "None".

Your response must contain *only* the labels (e.g., "A, C") or the word "None".

**Required Passage Labels:**
"""

    # Call the LLM and parse the response
    response = chat.invoke(prompt)
    response_text = response.content.strip()

    # Build the final list of passages based on the LLM's response
    required_passages = []
    if response_text.upper() != 'NONE':
        required_labels = [label.strip() for label in response_text.split(',')]
        for label in required_labels:
            if label in passage_map:
                required_passages.append(passage_map[label])
            else:
                pass
                # print(f"Warning: Model returned an unknown passage label '{label}'.")

    return required_passages
