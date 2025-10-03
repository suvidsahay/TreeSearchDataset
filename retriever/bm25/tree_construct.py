import json
import re
import os
import heapq
import itertools
import argparse
from dataclasses import dataclass, field
from typing import List, Tuple, Set
from itertools import islice

from retriever import retrieve_best_passage, fetch_wikipedia_page, get_doc_score_from_passages, \
    llm_select_next_passage_with_score
from elasticsearch import Elasticsearch, ConnectionError
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
ES_INDEX_NAME = "fever"

# --- Initialize Models ---
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')
chat_for_eval = ChatOpenAI(temperature=0, model_name="gpt-4o", openai_api_key=os.getenv("OPENAI_API_KEY"))
chat_for_generation = ChatOpenAI(temperature=0.7, model_name="gpt-4o", openai_api_key=os.getenv("OPENAI_API_KEY"))
tie_breaker = itertools.count()

# --- Initialize Elasticsearch Client ---
try:
    es = Elasticsearch("http://localhost:9200")
    if not es.ping():
        raise ConnectionError("Could not connect to Elasticsearch")
    print("Successfully connected to Elasticsearch.")
except ConnectionError as e:
    print(f"Elasticsearch connection failed: {e}")
    print("Please ensure Elasticsearch is running and accessible at http://localhost:9200")
    es = None


# --- NEW: Argument Parser ---
def get_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Run the iterative question generation process.")
    parser.add_argument(
        '--retrieval_method',
        type=str,
        choices=['llm', 'bm25'],
        default='bm25',
        help="The method to use for retrieving the next best passage ('llm' or 'bm25')."
    )
    return parser.parse_args()


@dataclass
class QuestionState:
    """Holds the state of a question during the iterative generation process."""
    question: str
    explanation: str
    answer: str
    passages_used: List[Tuple[str, str]] = field(default_factory=list)


def print_pqs_debug(pqs: List[List[tuple]], candidate_passages: List[Tuple[str, str]]):
    """Visually prints the state of all priority queues for debugging."""
    print("\n" + "=" * 25 + " DEBUG: PRIORITY QUEUE STATE " + "=" * 25)
    for i, pq in enumerate(pqs):
        print(f"\n--- PQ {i} (Contains {i + 1}-hop questions")
        if not pq:
            print("[EMPTY]")
            continue
        sorted_pq = sorted(pq, key=lambda x: x[0], reverse=False)
        for neg_score, _, state, passage_to_add in sorted_pq:
            score = -neg_score
            question_preview = state.question
            print(f"  - Candidate Score: {score:.4f}")
            print(f"    - Current Question: \"{question_preview}\"")
            print(f"    - Passages Used: {[p[0] for p in state.passages_used]}")
            print(f"    - -> Next Passage to Add: '{passage_to_add[0]}'")
    print("=" * 75 + "\n")


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

    prompt = f'''
    You are given a question, a ground truth answer, and {num_passages} passages.
    For each of the following subsets of passages, determine if you can fully answer the question.
    Respond with only "Yes" or "No".

    Question: "{question}"
    Ground Truth Answer: "{ground_truth_answer}"

    {passage_text}
    ---
    Analysis Tasks:
    {chr(10).join(subset_prompts)}
    '''

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


def retrieve_passages_with_bm25(query: str, es_client: Elasticsearch, index_name: str, size: int = 5) -> List[
    Tuple[str, str]]:
    """Retrieves top passages from Elasticsearch using BM25."""
    if not es_client:
        print("Elasticsearch client not available. Skipping retrieval.")
        return []

    request = ([
        {"index": index_name, "search_type": "dfs_query_then_fetch"},
        {
            "_source": True,
            "query": {
                "multi_match": {
                    "query": query, "type": "best_fields",
                    "fields": ["txt"], "tie_breaker": 0.5
                }
            },
            "size": size
        }
    ])
    try:
        resp = es_client.msearch(body=request)
    except Exception as e:
        print(f"Error during Elasticsearch msearch: {e}")
        return []

    docs = []
    for r in resp.get("responses", []):
        if "hits" in r and "hits" in r["hits"]:
            for hit in r["hits"]["hits"]:
                title = hit["_source"].get("title", "")
                full_text = fetch_wikipedia_page(title) or ""
                if title and full_text:
                    docs.append((title, full_text))
    return docs


def find_next_best_passage(question: str, candidate_passages: List[Tuple[str, str]]) -> Tuple[float, Tuple[str, str]]:
    """Finds the most relevant next passage from a list using CrossEncoder."""
    if not candidate_passages:
        return -1.0, None
    pairs = [[question, p_text] for _, p_text in candidate_passages]
    scores = cross_encoder.predict(pairs)
    best_idx = scores.argmax()
    return scores[best_idx].item(), candidate_passages[best_idx]


def find_next_best_passage_bm25_and_rerank(question: str, current_titles_used: Set[str]) -> Tuple[
    float, Tuple[str, str]]:
    """Orchestrates a three-stage process using BM25 to find the next best passage."""
    retrieved_docs = retrieve_passages_with_bm25(question, es, ES_INDEX_NAME, size=5)
    unique_titles = list(dict.fromkeys([title for title, _ in retrieved_docs]))
    candidate_titles = [title for title in unique_titles if title not in current_titles_used]

    if not candidate_titles:
        return -1.0, None

    best_passages_from_docs = []
    print(f"\n[INFO] Extracting best passage from candidate docs: {candidate_titles}")
    for title in candidate_titles:
        passage_text = retrieve_best_passage(title, question, method='cross-encoder')
        if passage_text:
            best_passages_from_docs.append((title, passage_text))

    if not best_passages_from_docs:
        return -1.0, None

    return find_next_best_passage(question, best_passages_from_docs)


def parse_generated_text(text: str) -> List[dict]:
    """Parses text containing Q/E/A blocks separated by '---'."""
    pattern = re.compile(r"Question:\s*(.*?)\s*Explanation:\s*(.*?)\s*Answer:\s*(.*)", re.DOTALL)
    parsed_data = []
    blocks = text.strip().split('---')
    for block in blocks:
        if not block.strip():
            continue
        match = pattern.search(block)
        if match:
            q, e, a = match.group(1).strip(), match.group(2).strip(), match.group(3).strip()
            if q and e and a:
                parsed_data.append({"question": q, "explanation": e, "answer": a})
    return parsed_data


def main():
    args = get_args()
    print(f"Using retrieval method: {args.retrieval_method}")

    if not es and args.retrieval_method == 'bm25':
        print("BM25 retrieval method selected, but cannot connect to Elasticsearch. Exiting.")
        return

    print(f"Clearing old results from {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        pass

    fever_data = {}
    with open(FILE1, 'r') as f1:
        for line in f1:
            if not line.strip(): continue
            rec = json.loads(line)
            claim = rec.get("claim")
            urls = [title.replace("_", " ") for title in rec.get("wiki_urls", [])]
            if claim: fever_data[claim] = {"claim": claim, "wiki_urls": list(dict.fromkeys(urls))}
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

        cache = {title: fetch_wikipedia_page(title) or "" for title in wiki_titles}
        doc_scores = [(title, get_doc_score_from_passages(claim, title, cache)) for title in wiki_titles]
        doc_scores = [ds for ds in doc_scores if ds[1] is not None]
        ranked_docs = sorted(doc_scores, key=lambda x: x[1], reverse=True)[:MAX_CANDIDATE_DOCS]
        print(f"\n\nProcessing claim: {claim}")
        print(f"Top candidate docs: {[title for title, _ in ranked_docs]}")
        candidate_passages = []
        for title, _ in ranked_docs:
            passage_text = retrieve_best_passage(title, claim, method='cross-encoder')
            if passage_text: candidate_passages.append((title, passage_text))
        if not candidate_passages: continue

        pqs = [[] for _ in range(MAX_CANDIDATE_DOCS)]
        all_generated_questions = []
        initial_passage_tuple = candidate_passages[0]
        generated_initial = generate_seed_questions(initial_passage_tuple[1])
        parsed_initial = parse_generated_text(generated_initial)

        for item in parsed_initial:
            initial_state = QuestionState(
                question=item["question"], explanation=item["explanation"],
                answer=item["answer"], passages_used=[initial_passage_tuple]
            )
            all_generated_questions.append(initial_state)

            score, next_best_passage = -1.0, None
            current_titles_used = {initial_passage_tuple[0]}
            if args.retrieval_method == 'llm':
                new_remaining = [p for p in candidate_passages if p[0] not in current_titles_used]
                if new_remaining:
                    score, next_best_passage = llm_select_next_passage_with_score(initial_state, new_remaining,
                                                                                  chat_for_eval)
            else:  # bm25
                score, next_best_passage = find_next_best_passage_bm25_and_rerank(initial_state.question,
                                                                                  current_titles_used)

            if next_best_passage:
                heapq.heappush(pqs[0], (-score, next(tie_breaker), initial_state, next_best_passage))

        for iteration_num in range(K_ITERATIONS):
            best_candidate_peek, best_pq_index = None, -1
            for i, pq in enumerate(pqs):
                if not pq: continue
                current_top_score = pq[0][0]
                if best_candidate_peek is None or current_top_score < best_candidate_peek[0]:
                    best_candidate_peek, best_pq_index = pq[0], i
            if best_pq_index == -1:
                print("All priority queues are empty. Stopping expansions.")
                break

            print(f"\n--- Iteration {iteration_num + 1}/{K_ITERATIONS} ---")
            print_pqs_debug(pqs, candidate_passages)
            print(f"Selected best candidate from PQ for {best_pq_index + 1}-hop questions.")

            neg_score, _, prev_state, passage_to_add = heapq.heappop(pqs[best_pq_index])
            print(f"Expanding with score {-neg_score:.4f} using passage from '{passage_to_add[0]}'")
            all_passage_tuples = prev_state.passages_used + [passage_to_add]
            passage_texts = [text for title, text in all_passage_tuples]
            multihop_text = generate_multihop_questions(passage_texts)
            parsed_multihop = parse_generated_text(multihop_text)
            if not parsed_multihop: continue

            for item in parsed_multihop:
                print("   - Evaluating naturalness of new candidate question...")
                naturalness_details = evaluate_question_naturalness(item["question"], chat_for_eval)
                LOGICAL_DEPENDENCY_THRESHOLD = 3
                if naturalness_details.get("logical_dependency_score", 0) <= LOGICAL_DEPENDENCY_THRESHOLD:
                    print(f"❌ Quality gate failed. Logical dependency score was too low. Discarding.")
                    continue
                print(f"✅ Quality gate passed.")

                minimal_passages_used = get_required_passages(item["question"], item["answer"], all_passage_tuples)
                new_state = QuestionState(
                    question=item["question"], explanation=item["explanation"],
                    answer=item["answer"], passages_used=minimal_passages_used
                )
                all_generated_questions.append(new_state)

                previous_passages_set = {title for title, _ in prev_state.passages_used}
                new_passages_set = {title for title, _ in new_state.passages_used}
                new_hop_count = len(new_passages_set)
                previous_hop_count = len(previous_passages_set)

                is_successful_expansion = new_hop_count > previous_hop_count
                is_successful_transformation = (
                            new_hop_count == previous_hop_count and new_passages_set != previous_passages_set)

                if is_successful_expansion or is_successful_transformation:
                    if is_successful_expansion:
                        print(f"✅ Expansion successful! Hops increased from {previous_hop_count} to {new_hop_count}.")
                    else:
                        print(f"🔄 Transformation successful! Passages shifted. Re-queuing at same hop level.")

                    next_pq_index = new_hop_count - 1
                    if next_pq_index < MAX_CANDIDATE_DOCS:
                        score, next_best_passage = -1.0, None
                        current_titles_used = {title for title, _ in new_state.passages_used}

                        if args.retrieval_method == 'llm':
                            new_remaining = [p for p in candidate_passages if p[0] not in current_titles_used]
                            if new_remaining:
                                score, next_best_passage = llm_select_next_passage_with_score(new_state, new_remaining,
                                                                                              chat_for_eval)
                        else:  # bm25
                            score, next_best_passage = find_next_best_passage_bm25_and_rerank(new_state.question,
                                                                                              current_titles_used)

                        if next_best_passage:
                            heapq.heappush(pqs[next_pq_index],
                                           (-score, next(tie_breaker), new_state, next_best_passage))
                else:
                    print(f"❌ Expansion failed. Question complexity did not increase or change. Discarding.")

        print("\n" + "=" * 20 + " SUMMARY OF GENERATED QUESTIONS " + "=" * 20)
        if not all_generated_questions:
            print("No multi-hop questions were generated in the expansions.")
        else:
            for i, state in enumerate(all_generated_questions):
                print(f"  - Q{i + 1} ({len(state.passages_used)} hops): {state.question}")
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
        print("=" * 68)


def calculate_metrics():
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
