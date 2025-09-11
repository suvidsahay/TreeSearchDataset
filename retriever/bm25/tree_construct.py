import json
import re
import os
import heapq
import itertools
from ast import parse
from dataclasses import dataclass, field
from typing import List, Tuple
from itertools import islice

from retriever import retrieve_best_passage, fetch_wikipedia_page, get_doc_score_from_passages
from question_generation import load_openai_key, generate_seed_questions, generate_multihop_questions
from verification3 import evaluate_question_naturalness, get_required_passages
from tqdm import tqdm
from langchain_openai import ChatOpenAI
from sentence_transformers import CrossEncoder

# --- Configuration ---
load_openai_key()
FILE1 = "filtered_fever_with_wiki_updated.jsonl"
FILE2 = "reranked_output_5.jsonl"
OUTPUT_FILE = "results_iterative.jsonl"
K_ITERATIONS = 10  # Max number of hops (passages) for a question
MAX_CANDIDATE_DOCS = 6  # Process top N documents for each claim

# --- Initialize Models ---
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')
chat_for_eval = ChatOpenAI(temperature=0, model_name="gpt-4o", openai_api_key=os.getenv("OPENAI_API_KEY"))
# This model is used for the placeholder generation functions
chat_for_generation = ChatOpenAI(temperature=0.7, model_name="gpt-4o", openai_api_key=os.getenv("OPENAI_API_KEY"))
tie_breaker = itertools.count()  # Used for stable sorting in the priority queue


# --- New Data Structure for Iterative Process ---
@dataclass
class QuestionState:
    """Holds the state of a question during the iterative generation process."""
    question: str
    explanation: str
    answer: str
    passages_used: List[Tuple[str, str]] = field(default_factory=list)  # List of (title, text)


def print_pqs_debug(pqs: List[List[tuple]], candidate_passages: List[Tuple[str, str]]):
    """Visually prints the state of all priority queues for debugging."""
    print("\n" + "=" * 25 + " DEBUG: PRIORITY QUEUE STATE " + "=" * 25)
    for i, pq in enumerate(pqs):
        print(f"\n--- PQ {i} (Contains {i + 1}-hop questions")
        if not pq:
            print("[EMPTY]")
            continue

        # For display, create a sorted copy to show items in their actual priority order
        # We sort by the first element of the tuple (the negative score)
        sorted_pq = sorted(pq, key=lambda x: x[0], reverse=False)

        for neg_score, _, state, passage_to_add in sorted_pq:
            score = -neg_score
            question_preview = state.question
            passages_used = state.passages_used

            print(f"  - Candidate Score: {score:.4f}")
            print(f"    - Current Question: \"{question_preview}\"")
            print(f"    - Passages Used: {passages_used}")
            print(f"    - -> Next Passage to Add: '{passage_to_add[0]}'")

    print("=" * 75 + "\n")


# --- New Verification Function (Generalized) ---
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
    verification_details = {f"answerable_with_{subset_prompts[i]}": "yes" in res.lower() for i, res in
                            enumerate(results) if res}

    answerable_full_set = list(verification_details.values())[-1] if verification_details else False
    answerable_smaller_set = any(list(verification_details.values())[:-1]) if len(verification_details) > 1 else False

    final_verdict = {
        "requires_all_passages": answerable_full_set and not answerable_smaller_set,
        "answerable_with_subset": answerable_full_set and answerable_smaller_set,
        "not_answerable": not answerable_full_set,
        "verification_details": verification_details
    }
    return final_verdict


# --- Helper Functions ---
def find_next_best_passage(question: str, remaining_passages: List[Tuple[str, str]]) -> Tuple[float, Tuple[str, str]]:
    """Finds the most relevant next passage from a list using CrossEncoder."""
    if not remaining_passages:
        return -1.0, None
    pairs = [[question, p_text] for _, p_text in remaining_passages]
    scores = cross_encoder.predict(pairs)
    best_idx = scores.argmax()
    return scores[best_idx], remaining_passages[best_idx]


def parse_generated_text(text: str) -> List[dict]:
    """
    Parses text containing Q/E/A blocks separated by '---', using a single, non-greedy regex.
    """
    # This single pattern finds all three parts at once.
    # (.*?) is a non-greedy match that stops before the next keyword.
    pattern = re.compile(
        r"Question:\s*(.*?)\s*Explanation:\s*(.*?)\s*Answer:\s*(.*)",
        re.DOTALL
    )

    parsed_data = []
    blocks = text.strip().split('---')

    for block in blocks:
        if not block.strip():
            continue

        match = pattern.search(block)
        if match:
            # group(1) is the question, group(2) is the explanation, group(3) is the answer
            q = match.group(1).strip()
            e = match.group(2).strip()
            a = match.group(3).strip()

            if q and e and a:
                parsed_data.append({"question": q, "explanation": e, "answer": a})

    return parsed_data

# --- Main Processing Logic ---
def main():
    print(f"Clearing old results from {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        pass

    # --- Load and Merge Data ---
    fever_data = {}
    with open(FILE1, 'r') as f1:
        for line in f1:
            if not line.strip(): continue
            rec = json.loads(line)
            claim = rec.get("claim")
            urls = [title.replace("_", " ") for title in rec.get("wiki_urls", [])]
            if claim:
                fever_data[claim] = {"claim": claim, "wiki_urls": list(dict.fromkeys(urls))}

    with open(FILE2, 'r') as f2:
        for line in f2:
            if not line.strip(): continue
            rec = json.loads(line)
            query = rec.get("query")
            if not query or query not in fever_data: continue
            existing = set(fever_data[query]["wiki_urls"])
            for d in rec.get("docs", []):
                title = d.get("title").replace("_", " ")
                if title and title not in existing:
                    fever_data[query]["wiki_urls"].append(title)
                    existing.add(title)

    fever_data = dict(islice(fever_data.items(), 1))

    print(f"Processing {len(fever_data)} FEVER entries...\n")

    for record in tqdm(list(fever_data.values())):
        claim = record['claim']
        wiki_titles = record.get('wiki_urls', [])

        # --- Step 1: Fetch and Rank All Candidate Documents ---
        cache = {title: fetch_wikipedia_page(title) or "" for title in wiki_titles}
        doc_scores = [(title, get_doc_score_from_passages(claim, title, cache)) for title in wiki_titles]
        doc_scores = [ds for ds in doc_scores if ds[1] is not None]
        ranked_docs = sorted(doc_scores, key=lambda x: x[1], reverse=True)[:MAX_CANDIDATE_DOCS]

        print(f"\n\nProcessing claim: {claim}")
        print(f"Top candidate docs: {[title for title, _ in ranked_docs]}")

        # --- Step 2: Get Best Passage from Each Candidate Doc ---
        candidate_passages = []
        for title, _ in ranked_docs:
            passage_text = retrieve_best_passage(title, claim, method='cross-encoder')
            if passage_text:
                candidate_passages.append((title, passage_text))
        if not candidate_passages: continue

        # --- Step 3: Initialize Priority Queues ---
        pqs = [[] for _ in range(MAX_CANDIDATE_DOCS)]

        # --- Step 4: Seed the first PQ ---
        all_generated_questions = []

        initial_passage_tuple = candidate_passages[0]
        remaining_passages_list = candidate_passages[1:]

        generated_initial = generate_seed_questions(initial_passage_tuple[1])
        parsed_initial = parse_generated_text(generated_initial)

        for item in parsed_initial:
            initial_state = QuestionState(
                question=item["question"],
                explanation=item["explanation"],
                answer=item["answer"],
                passages_used=[initial_passage_tuple]
            )
            all_generated_questions.append(initial_state)

            score, next_best_passage = find_next_best_passage(initial_state.question, remaining_passages_list)
            if next_best_passage:
                heapq.heappush(pqs[0], (-score, next(tie_breaker), initial_state, next_best_passage))

        # --- Step 5: Run Iterative Generation for K hops ---
        for iteration_num in range(K_ITERATIONS):
            # 5a: Find the best candidate across the top of all PQs
            best_candidate_peek = None
            best_pq_index = -1
            for i, pq in enumerate(pqs):
                if not pq: continue

                current_top_score = pq[0][0]  # This is neg_score
                if best_candidate_peek is None or current_top_score < best_candidate_peek[0]:
                    best_candidate_peek = pq[0]
                    best_pq_index = i


            # 5b: If no candidates are left, break
            if best_pq_index == -1:
                print("All priority queues are empty. Stopping expansions.")
                break

            print(f"\n--- Iteration {iteration_num + 1}/{K_ITERATIONS} ---")
            print_pqs_debug(pqs, candidate_passages)
            print(f"Selected best candidate from PQ for {best_pq_index + 1}-hop questions.")

            # 5c: Pop the globally best candidate from its PQ
            neg_score, _, prev_state, passage_to_add = heapq.heappop(pqs[best_pq_index])
            print(f"Expanding with score {-neg_score:.4f} using passage from '{passage_to_add[0]}'")

            # 5d: Generate the new, more complex question
            all_passage_tuples = prev_state.passages_used + [passage_to_add]

            passage_texts = [text for title, text in all_passage_tuples]
            multihop_text = generate_multihop_questions(passage_texts)
            parsed_multihop = parse_generated_text(multihop_text)
            if not parsed_multihop: continue

            for item in parsed_multihop:
                minimal_passages_used = get_required_passages(item["question"], item["answer"], all_passage_tuples)

                new_state = QuestionState(
                    question=item["question"], explanation=item["explanation"],
                    answer=item["answer"], passages_used=minimal_passages_used
                )
                all_generated_questions.append(new_state)  # Add the new question to our list

                # 5e: Find next best passage and push to the correct PQ
                # Calculate index based on the ACTUAL number of passages in the new state
                next_pq_index = len(new_state.passages_used) - 1

                # Ensure the calculated index is valid
                if 0 <= next_pq_index < MAX_CANDIDATE_DOCS:
                    current_titles_used = {title for title, _ in new_state.passages_used}
                    new_remaining = [p for p in candidate_passages if p[0] not in current_titles_used]

                    if new_remaining:
                        score, next_best_passage = find_next_best_passage(new_state.question, new_remaining)
                        if next_best_passage:
                            # The state is a candidate to be expanded to a (hop+1) question
                            # So we push it into the queue for its current hop level
                            heapq.heappush(pqs[next_pq_index],
                                           (-score, next(tie_breaker), new_state, next_best_passage))

        # --- Step 6: Perform Final Analysis ---
        print("\n" + "=" * 20 + " SUMMARY OF GENERATED QUESTIONS " + "=" * 20)
        if not all_generated_questions:
            print("No multi-hop questions were generated in the expansions.")
            print("=" * 68)
        else:
            # Display all the questions that were created
            for i, state in enumerate(all_generated_questions):
                print(f"  - Q{i + 1} ({len(state.passages_used)} hops): {state}")
            print("=" * 68)

            # Now, perform the final analysis ONLY on the last question from the list
            print("\n--- Verifying the final generated question ---")
            final_question_to_verify = all_generated_questions[-1]

            print(
                f"\nFinal Question ({len(final_question_to_verify.passages_used)} hops): {final_question_to_verify.question}")
            final_passages_text = [p_text for _, p_text in final_question_to_verify.passages_used]
            verification_details = verify_question_N_docs(final_passages_text, final_question_to_verify.question,
                                                          final_question_to_verify.answer)
            naturalness_details = evaluate_question_naturalness(final_question_to_verify.question, chat_for_eval)

            log_entry = {
                "claim": claim, "final_question": final_question_to_verify.question,
                "explanation": final_question_to_verify.explanation, "answer": final_question_to_verify.answer,
                "num_hops": len(final_question_to_verify.passages_used),
                "passages": [{"title": t, "text": txt} for t, txt in final_question_to_verify.passages_used],
                **verification_details, **naturalness_details
            }
            with open(OUTPUT_FILE, 'a') as f_out:
                json.dump(log_entry, f_out)
                f_out.write('\n')


# --- Metrics Calculation Section ---
def calculate_metrics():
    # This function is the same as the previous version and remains compatible.
    # It reads the generalized output format and calculates metrics.
    total_questions, hop_counts = 0, {}
    verdict_counts = {"requires_all": 0, "subset": 0, "not_answerable": 0, "error": 0}
    naturalness_keys = ["clear_single_question_score", "combines_passages_score", "requires_both_score",
                        "logical_dependency_score", "hotpot_style_score", "objectivity_score"]
    naturalness_totals = {k: 0 for k in naturalness_keys}
    naturalness_counts = {k: 0 for k in naturalness_keys}

    if not os.path.exists(OUTPUT_FILE):
        print("Output file not found. Cannot calculate metrics.")
        return

    with open(OUTPUT_FILE, 'r') as f:
        for line in f:
            total_questions += 1
            entry = json.loads(line)
            hops = entry.get("num_hops", 0)
            hop_counts[hops] = hop_counts.get(hops, 0) + 1

            if entry.get("requires_all_passages"):
                verdict_counts["requires_all"] += 1
            elif entry.get("answerable_with_subset"):
                verdict_counts["subset"] += 1
            elif entry.get("not_answerable"):
                verdict_counts["not_answerable"] += 1
            else:
                verdict_counts["error"] += 1

            for key in naturalness_keys:
                if key in entry and entry[key] is not None:
                    naturalness_totals[key] += entry[key]
                    naturalness_counts[key] += 1

    if total_questions == 0:
        print("\nNo questions were processed to calculate metrics!")
        return

    print("\n\n--- FINAL METRICS ---")
    print(f"Total Questions Processed: {total_questions}")

    print("\n📊 Question Hops Distribution:")
    for hops, count in sorted(hop_counts.items()):
        print(f"  - {hops}-Hop Questions: {count} ({(count / total_questions) * 100:.2f}%)")

    print("\n✅ Verification Verdicts:")
    print(
        f"  - Requires All Passages: {verdict_counts['requires_all']} ({(verdict_counts['requires_all'] / total_questions) * 100:.2f}%)")
    print(
        f"  - Answerable by Subset: {verdict_counts['subset']} ({(verdict_counts['subset'] / total_questions) * 100:.2f}%)")
    print(
        f"  - Not Answerable: {verdict_counts['not_answerable']} ({(verdict_counts['not_answerable'] / total_questions) * 100:.2f}%)")

    print("\n🌿 Average Naturalness Scores by Dimension:")
    for key in naturalness_keys:
        avg = naturalness_totals[key] / naturalness_counts[key] if naturalness_counts[key] > 0 else 0
        print(f"  - {key}: {avg:.2f} / 5.0")


if __name__ == "__main__":
    main()
    calculate_metrics()