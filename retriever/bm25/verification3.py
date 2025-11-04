# verification3.py

import json
from langchain_openai import ChatOpenAI
from langchain_core.messages.human import HumanMessage, HumanMessageChunk
#from langchain.schema import HumanMessage
import os
from typing import List, Tuple

# --- HELPER FUNCTIONS ---

# prompt = f"""Based strictly on the 'Context' provided, answer the 'Question'.
#You must synthesize information across sentences if necessary, but you **cannot** use any external knowledge or make assumptions not supported by the text.
#If the answer cannot be confidently inferred from the text, you MUST respond with the exact phrase: "Not answerable from passages."""

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


def evaluate_question_naturalness_dynamic(question: str, passages: List[str], chat_model):
    """
    Asks an LLM to dynamically score a question's naturalness based on N passages.
    """
    print(f"   - Evaluating question naturalness for {len(passages)}-hops...")
    num_passages = len(passages)

    # Dynamically change the criteria text based on the number of passages
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

EXAMPLE OF A GOOD 3-HOP QUESTION:
- Passages: [About Paris], [About the Louvre], [About the Mona Lisa]
- Question: "In which country is the museum that houses the Mona Lisa located?"
- Justification: This is excellent. It requires finding the Mona Lisa's location (Louvre) from P3, finding the Louvre's city (Paris) from P2, and finding Paris's country (France) from P1. This is a perfect A->B->C logical chain.

---
Now, evaluate the following question using the criteria above.

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
        print(f"CRITICAL PARSING ERROR in dynamic naturalness evaluation: {e}")
        # Return a structure with None values on error
        return {
            "clear_single_question_score": None, "combines_passages_score": None,
            "requires_all_passages_score": None, "logical_dependency_score": None,
            "hotpot_style_score": None, "objectivity_score": None,
            "justification": "Evaluation parsing error."
        }







def evaluate_question_naturalness(question, chat_model):
    """Asks the LLM to score the question across 6 task-specific dimensions."""
    print("   - Evaluating question naturalness (multi-dimension)...")

    eval_prompt = f"""You are an expert in evaluating multi-hop question quality.

Evaluate the following question using the SIX criteria below. For each one, give a score from 1 to 5 and one short explanation at the end.

Before you score the question, here are FIVE EXAMPLES with their scores and justifications to guide you:

---

✅ Strong Example  
Question: What roles did Peter Dinklage and Nikolaj Coster-Waldau play in Game of Thrones, and what awards did they receive?  
Scores:  
  - clear_single_question_score: 5  
  - combines_passages_score: 5  
  - requires_both_score: 5  
  - logical_dependency_score: 5  
  - hotpot_style_score: 5  
  - objectivity_score: 5  
Justification: This question is clearly phrased, meaningfully combines facts about both actors and their awards, and requires reasoning across passages to construct the answer. It is grounded in the texts and structured in HotpotQA style.

---

🟡 Moderate Example  
Question: How did the visual branding of Fox during the late 1990s coincide with Nikolaj Coster-Waldau's early career in the United States?  
Scores:  
  - clear_single_question_score: 5  
  - combines_passages_score: 5  
  - requires_both_score: 5  
  - logical_dependency_score: 4  
  - hotpot_style_score: 4  
  - objectivity_score: 5  
Justification: This question links two timelines and references both documents. However, the passages do not contain any explicit or inferred connection between branding and career, making the logical link weaker. It’s clear and well-structured, but lacks answerability.

---

❌ Weak Example (Irrelevant Passage Reference)  
Question: How does Peter Dinklage’s role in “Game of Thrones” relate to his role in “X-Men: Days of Future Past”?  
Scores:  
  - clear_single_question_score: 5  
  - combines_passages_score: 2  
  - requires_both_score: 2  
  - logical_dependency_score: 2  
  - hotpot_style_score: 1  
  - objectivity_score: 5  
Justification: Only one passage discusses “Game of Thrones”; the other does not mention “X-Men” at all. While the question is clear, it lacks grounding in both passages and fails to require reasoning or comparison across them.

---

⚠️ Weak Example (Side-by-Side Facts Without Connection)  
Question: Which major mountain range in Slovenia is part of the country's seismic zone, and what year did Slovenia join the Eurozone?  
Scores:  
  - clear_single_question_score: 5  
  - combines_passages_score: 3  
  - requires_both_score: 3  
  - logical_dependency_score: 2  
  - hotpot_style_score: 2  
  - objectivity_score: 5  
Justification: While this question refers to both passages, the two facts are not logically related. They are placed side by side without reasoning or conceptual connection. This weakens both the dependency and HotpotQA style.

---

⚠️ Weak Example (Superficial Passage Mention)  
Question: What is the population of Chad's capital, N'Djamena, which is relevant to understanding the scale of social issues that Ryan Gosling addressed during his work in Chad?  
Scores:  
  - clear_single_question_score: 5  
  - combines_passages_score: 2  
  - requires_both_score: 2  
  - logical_dependency_score: 2  
  - hotpot_style_score: 1  
  - objectivity_score: 5  
Justification: This question tries to force a connection between Chad’s demographics and Gosling’s charity work, but the Gosling passage contains no meaningful data about Chad’s population or issues. This results in shallow cross-passage use.

---

Now evaluate the following question using the same criteria:

Score Guide:  
1 = Very Poor  
2 = Poor  
3 = Average  
4 = Good  
5 = Excellent

Criteria:  
1. One Clear Question (No “and”): Is this a single clear question, or does it combine two separate questions with “and”?  
2. Combines Facts from Both Passages: Does the question require information from BOTH passages?  
3. Neither Passage Alone Is Enough: Can the question be answered using just one passage? (If yes, score lower.)  
4. Logical Dependency: Are the two facts logically connected, or just listed side-by-side?  
5. HotpotQA-Style Reasoning: Does the question require reasoning across multiple facts like HotpotQA?  
6. Objectivity and Speculation: Is the question based on facts from the documents, or does it ask for speculation?

Return your answer as a valid JSON object in this exact format:
{{
  "clear_single_question_score": <1-5>,
  "combines_passages_score": <1-5>,
  "requires_both_score": <1-5>,
  "logical_dependency_score": <1-5>,
  "hotpot_style_score": <1-5>,
  "objectivity_score": <1-5>,
  "justification": "<one short explanation>"
}}

Question: "{question}" """

    response = chat_model.invoke([HumanMessage(content=eval_prompt)])

    try:
        json_start = response.content.find('{')
        json_end = response.content.rfind('}') + 1
        json_data = response.content[json_start:json_end]
        return json.loads(json_data)
    except Exception:
       return {
            "clear_single_question_score": None,
            "combines_passages_score": None,
            "requires_both_score": None,
            "logical_dependency_score": None,
            "hotpot_style_score": None,
            "objectivity_score": None,
            "justification": "Evaluation parsing error."
        }

def verify_question_v3(documents, question, ground_truth_answer):
    """
    Verifies question necessity and evaluates answer objectivity.
    """
    # ... (The rest of this function remains exactly the same as before) ...
    doc_A = documents[0]
    doc_B = documents[1]
    chat = ChatOpenAI(temperature=0, model_name="gpt-4o", openai_api_key=os.getenv("OPENAI_API_KEY"))

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
        return result
    except Exception as e:
        print(f"CRITICAL PARSING ERROR in verification3: {e}")
        return {"answer": f"Comparison Parsing Error: {final_response.content}"}

def verify_question_3docs(documents, question, ground_truth_answer):
    if len(documents) != 3:
        raise ValueError("Expected exactly 3 documents.")

    doc_A = documents[0]
    doc_B = documents[1]
    doc_C = documents[2]
    chat = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", openai_api_key=os.getenv("OPENAI_API_KEY"))

    answer_A = _generate_answer_from_context(doc_A, question, chat)
    answer_B = _generate_answer_from_context(doc_B, question, chat)
    answer_C = _generate_answer_from_context(doc_C, question, chat)
    answer_all = _generate_answer_from_context(f"Passage A:\n{documents[0]}\n\n"
        f"Passage B:\n{documents[1]}\n\n"
        f"Passage C:\n{documents[2]}\n", question, chat)
    answer_none = _generate_answer_from_context(None, question, chat)

    objectivity_details = evaluate_answer_objectivity(ground_truth_answer, answer_all, chat)


    comparison_prompt = f"""You are an expert evaluator. Your task is to determine which context produces the answer closest to the 'Ground Truth Answer' with the following priority: 
1. Answer A (from Passage A only), 
2. Answer B (from Passage B only), 
2. Answer C (from Passage C only), 
3. Answer All (from All passages), 
4. Answer None (from general knowledge).

Question: "{question}"

Ground Truth Answer:
"{ground_truth_answer}"

---
Generated Answers to Compare:
1. Answer A (from Passage A only): "{answer_A}"
2. Answer B (from Passage B only): "{answer_B}"
2. Answer C (from Passage C only): "{answer_B}"
3. Answer All (from All passages): "{answer_all}"
4. Answer None (from general knowledge): "{answer_none}"
---

Analyze the answers and return a single string indicating which context was sufficient to produce the closest match to the Ground Truth Answer. 
The output must be one of: "Correct_A_passage", "Correct_B_passage", "Correct_C_passage", "Correct_all_passage", "Correct_no_passage". If none are a good match, return "Correct_no_passage".

Your response (e.g., "Correct_C_passage"):
"""

    final_response = chat.invoke([HumanMessage(content=comparison_prompt)])

    try:
        result = {
            "Correct_A_passage": False,
            "Correct_B_passage": False,
            "Correct_C_passage": False,
            "Correct_all_passage": False,
            "Correct_no_passage": False,
            "generated_answer_A": answer_A,
            "generated_answer_B": answer_B,
            "generated_answer_C": answer_C,
            "generated_answer_all": answer_all,
            "generated_answer_none": answer_none
        }
        result.update(objectivity_details)

        response_text = final_response.content.strip()
        if response_text in ["Correct_A_passage", "Correct_B_passage", "Correct_C_passage", "Correct_all_passage", "Correct_no_passage"]:
            result[response_text] = True
        else:
            result["Correct_no_passage"] = True

        print(f"Comparison Prompt: \n{comparison_prompt}")
        print(f"final_response: \n{final_response.content}")
        print(result)
        return result
    except Exception as e:
        print(f"CRITICAL PARSING ERROR in verification3: {e}")
        return {"answer": f"Comparison Parsing Error: {final_response.content}"}

def verify_question_N_docs(passages: List[str], question: str, ground_truth_answer: str) -> dict:
  """Verifies if a question is answerable using subsets of the provided N passages."""
  if not passages:
    return {"verification_error": "No passages provided"}

  passage_labels = [f"Passage_{i + 1}" for i in range(len(passages))]
  passage_text = "\n\n".join([f"{label}: \"{text}\"" for label, text in zip(passage_labels, passages)])
  num_passages = len(passages)
  all_subsets = [list(itertools.combinations(range(num_passages), i)) for i in range(1, num_passages + 1)]
  subset_prompts = [f"Subset {i + 1} ({', '.join([passage_labels[j] for j in subset])})" for i, subsets_at_level in
           enumerate(all_subsets) for subset in subsets_at_level]

  prompt = f"""
  You are given a question, a ground truth answer, and {num_passages} passages.
  For each of the following subsets of passages, determine if you can fully answer the question.
  Respond with only "Yes" or "No".

  Question: "{question}"
  Ground Truth Answer: "{ground_truth_answer}"

  {passage_text}
  ---
  Analysis Tasks:
  {chr(10).join(subset_prompts)}
  """
  response = chat_for_eval.invoke(prompt)
  results = response.content.strip().split('\n')
  verification_details = {f"answerable_with_{prompt}": "yes" in res.lower() for prompt, res in
              zip(subset_prompts, results) if res}

  answerable_full_set = list(verification_details.values())[-1] if verification_details else False
  answerable_smaller_set = any(list(verification_details.values())[:-1]) if len(verification_details) > 1 else False

  final_verdict = {
    "requires_all_passages": answerable_full_set and not answerable_smaller_set,
    "answerable_with_subset": answerable_full_set and answerable_smaller_set,
    "not_answerable": not answerable_full_set,
    "verification_details": verification_details
  }
  return final_verdict


def verify_question_final(documents: list[str], question: str, ground_truth_answer: str) -> dict:
    """
    Verifies a question by generating answers from individual documents, all documents combined,
    and general knowledge, then asks an LLM to determine which context is sufficient.
    (This is the function you provided).
    """
    # ... (Paste your full verify_question_final function here) ...
    if not documents:
        raise ValueError("The documents list cannot be empty.")

    num_docs = len(documents)
    chat = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", openai_api_key=os.getenv("OPENAI_API_KEY"))

    # --- 1. Generate answers from all sources ---
    generated_answers = {}

    # Answers from individual passages
    for i, doc in enumerate(documents):
        doc_label = chr(65 + i)  # A, B, C, ...
        generated_answers[f'answer_{doc_label}'] = _generate_answer_from_context(doc, question, chat)

    # Answer from all passages combined
    all_docs_context = "\n\n".join(
        [f"Passage {chr(65 + i)}:\n{doc}" for i, doc in enumerate(documents)]
    )
    generated_answers['answer_all'] = _generate_answer_from_context(all_docs_context, question, chat)

    # Answer from no context (general knowledge)
    generated_answers['answer_none'] = _generate_answer_from_context(None, question, chat)

    # --- 2. Perform objectivity evaluation on the 'all passages' answer ---
    objectivity_details = evaluate_answer_objectivity(
        ground_truth_answer, generated_answers['answer_all'], chat
    )

    # --- 3. Build the dynamic comparison prompt ---
    prompt_parts = [
        "You are an expert evaluator. Your task is to determine which context produces the answer closest to the 'Ground Truth Answer'.",
        f"Question: \"{question}\"",
        f"Ground Truth Answer:\n\"{ground_truth_answer}\"",
        "---",
        "Generated Answers to Compare:"
    ]

    possible_choices = []
    # Add individual passage answers to the prompt
    for i in range(num_docs):
        doc_label = chr(65 + i)
        prompt_parts.append(
            f"{i + 1}. Answer {doc_label} (from Passage {doc_label} only): \"{generated_answers[f'answer_{doc_label}']}\"")
        possible_choices.append(f"Correct_{doc_label}_passage")

    # Add combined and no-context answers to the prompt
    prompt_parts.append(f"{num_docs + 1}. Answer All (from All passages): \"{generated_answers['answer_all']}\"")
    prompt_parts.append(f"{num_docs + 2}. Answer None (from general knowledge): \"{generated_answers['answer_none']}\"")
    possible_choices.extend(["Correct_all_passage", "Correct_no_passage"])

    prompt_parts.extend([
        "---",
        f"Analyze the answers and return a single string indicating which context was sufficient to produce the closest match. The output must be one of: {', '.join(possible_choices)}.",
        "If none are a good match, return \"Correct_no_passage\".",
        "Your response:"
    ])

    comparison_prompt = "\n".join(prompt_parts)

    # --- 4. Get the final evaluation from the model ---
    final_response = chat.invoke([HumanMessage(content=comparison_prompt)])
    response_text = final_response.content.strip()

    # --- 5. Structure and return the results ---
    try:
        result = {}
        for choice in possible_choices:
            result[choice] = False
        if response_text in result:
            result[response_text] = True
        else:
            result["Correct_no_passage"] = True
        # ... (add other generated answers and details if needed) ...
        return result
    except Exception as e:
        return {"error": f"Comparison Parsing Error: {e}"}


def get_required_passages(question: str, answer: str, candidate_passages: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """
    Directly asks an LLM to identify the minimal set of passages required to answer a question.
    """
    if not candidate_passages:
        return []

    chat = ChatOpenAI(model_name="o4-mini", openai_api_key=os.getenv("OPENAI_API_KEY"))

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
                print(f"Warning: Model returned an unknown passage label '{label}'.")

    return required_passages
