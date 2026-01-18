import json
import re
import itertools
from typing import List, Tuple, Optional
from prompt_loader import render_prompt
from langchain_core.messages import HumanMessage

# =========================
# Chat helper
# =========================
def _get_chat(chat_model: Optional[object] = None):
    if chat_model is None:
        raise RuntimeError("VERIFICATION/EVALUATION LLM must be injected (chat_model is None).")
    return chat_model

# =========================
# Core helpers
# =========================
def _generate_answer_from_context(context: Optional[str], question: str, chat_model: Optional[object]):
    """
    A generic helper to generate an answer given a context and question.
    If context is None, it asks for general knowledge.
    """
    if context:
        prompt = render_prompt(
    "verification/_generate_answer_from_context.j2",
    context=context,
    question=question,
    )
    else: 
        prompt = render_prompt("verification/_generate_answer_from_context2.j2", context=context, question=question)

    chat = _get_chat(chat_model)
    response = chat.invoke([HumanMessage(content=prompt)])
    return response.content

def evaluate_answer_objectivity(answer1: str, answer2: str, chat_model: Optional[object]):
    """
    Asks the LLM to compare two answers for semantic similarity (equivalence).
    Returns a dictionary with 'is_equivalent' (bool) and 'similarity_score' (1-5).
    """
    eval_prompt = render_prompt(
    "verification/evaluate_answer_objectivity.j2",
    answer1=answer1,
    answer2=answer2,)
    
    chat = _get_chat(chat_model)
    response = chat.invoke([HumanMessage(content=eval_prompt)])
    try:
        json_start = response.content.find('{')
        json_end = response.content.rfind('}') + 1
        json_data = response.content[json_start:json_end]
        return json.loads(json_data)
    except Exception:
        return {"is_equivalent": False, "similarity_score": 0, "justification": "Evaluation parsing error."}

# =========================
# Naturalness evaluators
# =========================
def evaluate_question_naturalness_dynamic(question: str, passages: List[str], chat_model: Optional[object]):
    """
    Asks an LLM to dynamically score a question's naturalness based on N passages.
    """
    num_passages = len(passages)
    combines_passages_text = f"Combines Facts from All {num_passages} Passages"
    requires_all_text = f"No Subset of Passages Is Enough"

    eval_prompt = render_prompt(
    "verification/evaluate_question_naturalness_dynamic.j2",
    num_passages=num_passages,
    combines_passages_text=combines_passages_text,
    requires_all_text=requires_all_text,
    question=question,)

    chat = _get_chat(chat_model)
    response = chat.invoke([HumanMessage(content=eval_prompt)])
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

# =========================
# Verifiers
# =========================
def verify_question_v3(documents: List[str], question: str, ground_truth_answer: str, chat_model: Optional[object] = None):
    """
    Verifies question necessity and evaluates answer objectivity.
    """
    # ... (The rest of this function remains exactly the same as before) ...
    doc_A = documents[0]
    doc_B = documents[1]
    chat = _get_chat(chat_model)

    answer_A = _generate_answer_from_context(doc_A, question, chat)
    answer_B = _generate_answer_from_context(doc_B, question, chat)
    answer_both = _generate_answer_from_context(f"Passage A: {doc_A}\n\nPassage B: {doc_B}", question, chat)
    answer_none = _generate_answer_from_context(None, question, chat)

    objectivity_details = evaluate_answer_objectivity(ground_truth_answer, answer_both, chat)

    comparison_prompt = render_prompt(
    "verification/verify_question_v3.j2",
    question=question,
    ground_truth_answer=ground_truth_answer,
    answer_A=answer_A,
    answer_B=answer_B,
    answer_both=answer_both,
    answer_none=answer_none,
    )
    final_response = chat.invoke([HumanMessage(content=comparison_prompt)])

    try:
        json_start = final_response.content.find('{')
        json_end = final_response.content.rfind('}') + 1
        json_data = final_response.content[json_start:json_end]
        result = json.loads(json_data)
        result['generated_answer_A'] = answer_A
        result['generated_answer_B'] = answer_B
        result['generated_answer_both'] = answer_both
        result['generated_answer_none'] = answer_none
        result.update(objectivity_details)
        return result
    except Exception as e:
        print(f"CRITICAL PARSING ERROR in verification3: {e}")
        return {"answer": f"Comparison Parsing Error: {final_response.content}"}

def verify_question_N_docs(passages: List[str], question: str, ground_truth_answer: str, chat_model: Optional[object] = None) -> dict:
    if not passages: return {"verification_error": "No passages provided"}
    num_passages = len(passages)
    chat = _get_chat(chat_model)

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

    print(f"[Verifier] 🔍 Checking {len(subset_indices)} subsets of size {num_passages-1}...")

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

def verify_question_phase2(passages: List[str], question: str, ground_truth_answer: str, chat_model: Optional[object] = None) -> dict:
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
    return verify_question_N_docs(passages, question, ground_truth_answer, chat_model=None)


def get_required_passages(question: str, answer: str, candidate_passages: List[Tuple[str, str]], chat_model: Optional[object] = None) -> List[Tuple[str, str]]:
    if not candidate_passages:
        return []

    chat = _get_chat(chat_model)

    passage_map = {}
    formatted_passages = []
    for i, (title, text) in enumerate(candidate_passages):
        label = chr(65 + i)  # A, B, C...
        passage_map[label] = (title, text)
        formatted_passages.append(f"Passage {label} (from '{title}'):\n{text}")

    all_passages_text = "\n\n---\n\n".join(formatted_passages)

    prompt = render_prompt(
        "verification/get_required_passages.j2",
        question=question,
        answer=answer,
        all_passages_text=all_passages_text,
    )

    response = chat.invoke([HumanMessage(content=prompt)])
    response_text = (response.content or "").strip()

    # Treat any standalone NONE as None
    if re.search(r"\bNONE\b", response_text.upper()):
        return []

    # Extract standalone single-letter labels only (A, B, C...)
    labels = re.findall(r"\b[A-Z]\b", response_text.upper())

    # De-duplicate while preserving order
    seen = set()
    labels = [x for x in labels if not (x in seen or seen.add(x))]

    required_passages = []
    for label in labels:
        if label in passage_map:
            required_passages.append(passage_map[label])
        else:
            print(f"Warning: Model returned an unknown passage label '{label}'.")

    return required_passages