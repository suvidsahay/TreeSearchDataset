# verification3.py

import json
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
import os

# --- HELPER FUNCTIONS ---

def _generate_answer_from_context(context, question, chat_model):
    """A generic helper to generate an answer given a context and question."""
    if context:
        prompt = f"""Using ONLY the information from the provided 'Context', answer the 'Question'.
If the information is not available in the context, you MUST respond with the exact phrase: "Not answerable from passages."

Context:
{context}

Question: {question}

Answer:"""
    else: # No context provided, test for general knowledge
        prompt = f"""Answer the following question from your general knowledge, without using any external documents.

Question: {question}

Answer:"""
    
    response = chat_model.invoke([HumanMessage(content=prompt)])
    return response.content

def evaluate_answer_objectivity(answer1, answer2, chat_model):
    """Asks the LLM to compare two answers for semantic similarity to score objectivity."""
    print("   - Evaluating answer objectivity...")
    eval_prompt = f"""You are an expert semantic evaluator. Your task is to compare two answers to the same question and score their similarity. A high similarity score suggests the question is objective and fact-based.
    
Answer 1: "{answer1}"
Answer 2: "{answer2}"
    
Evaluate their semantic similarity on a scale of 1 to 5 (1 = Very Dissimilar, 5 = Identical in meaning).
    
Respond ONLY with a valid JSON object in the format:
{{"objectivity_score": <score_integer>, "objectivity_justification": "<brief justification>"}}
"""
    response = chat_model.invoke([HumanMessage(content=eval_prompt)])
    try:
        json_start = response.content.find('{')
        json_end = response.content.rfind('}') + 1
        json_data = response.content[json_start:json_end]
        return json.loads(json_data)
    except Exception:
        return {"objectivity_score": None, "objectivity_justification": "Objectivity parsing error."}

# --- CONSOLIDATED EVALUATION FUNCTIONS ---

def evaluate_question_naturalness(question, chat_model):
    """Asks the LLM to rate the naturalness and coherence of a question."""
    print("   - Evaluating question naturalness...")
    eval_prompt = f"""You are a linguistic expert evaluating a question's quality.
A 'natural' question is one a human might plausibly ask. It flows well.
A 'coherent' question connects its parts logically around a central theme, not just joining unrelated facts.

Evaluate the following question on a scale of 1 to 5 (1=Very Unnatural/Incoherent, 5=Very Natural/Coherent).

Question: "{question}"

Respond ONLY with a valid JSON object in the format:
{{"naturalness_score": <score_integer>, "justification": "<brief justification>"}}
"""
    response = chat_model.invoke([HumanMessage(content=eval_prompt)])
    try:
        json_start = response.content.find('{')
        json_end = response.content.rfind('}') + 1
        json_data = response.content[json_start:json_end]
        return json.loads(json_data)
    except Exception:
        return {"naturalness_score": None, "justification": "Evaluation parsing error."}

def verify_question_v3(documents, question, ground_truth_answer):
    """
    Verifies question necessity and evaluates answer objectivity.
    """
    # ... (The rest of this function remains exactly the same as before) ...
    doc_A = documents[0]
    doc_B = documents[1]
    chat = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", openai_api_key=os.getenv("OPENAI_API_KEY"))

    answer_A = _generate_answer_from_context(doc_A, question, chat)
    answer_B = _generate_answer_from_context(doc_B, question, chat)
    answer_both = _generate_answer_from_context(f"Passage A: {doc_A}\n\nPassage B: {doc_B}", question, chat)
    answer_none = _generate_answer_from_context(None, question, chat)
    
    objectivity_details = evaluate_answer_objectivity(ground_truth_answer, answer_both, chat)
    
    comparison_prompt = f"""You are an expert evaluator. Your task is to see which of the generated answers ('Answer A', 'Answer B', 'Answer Both', 'Answer None') is the closest match to the 'Ground Truth Answer'.

Question: "{question}"

Ground Truth Answer:
"{ground_truth_answer}"

---
Generated Answers to Compare:
1. Answer A (from Passage A only): "{answer_A}"
2. Answer B (from Passage B only): "{answer_B}"
3. Answer Both (from both passages): "{answer_both}"
4. Answer None (from general knowledge): "{answer_none}"
---

Analyze the answers and determine which context was sufficient to produce the Ground Truth Answer. Respond ONLY with a single valid JSON object.
- Set "Correct_A_passage" to true if "Answer A" is the best and closest match to the Ground Truth.
- Set "Correct_B_passage" to true if "Answer B" is the best and closest match.
- Set "Correct_2_passage" to true if "Answer Both" is the best and closest match, AND both "Answer A" and "Answer B" are clearly inferior or incomplete.
- Set "Correct_no_passage" to true if "Answer None" is the best and closest match.
- If none of the generated answers are a good match for the Ground Truth, set all flags to false.

Your JSON response (e.g., {{"Correct_2_passage": true, "Correct_A_passage": false, "Correct_B_passage": false, "Correct_no_passage": false}}):
"""
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
        print(f"Comparision Prompt: \n{comparison_prompt}")
        print(f"final_response: \n{final_response.content}")
        print(result)
        return result
    except Exception as e:
        print(f"CRITICAL PARSING ERROR in verification3: {e}")
        return {"answer": f"Comparison Parsing Error: {final_response.content}"}
