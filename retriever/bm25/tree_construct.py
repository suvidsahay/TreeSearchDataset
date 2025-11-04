import json
import re
import os
import heapq
import itertools
import argparse
from dataclasses import dataclass, field
from typing import List, Tuple, Set
from itertools import islice

# --- External Library Imports ---
from tqdm import tqdm
from elasticsearch import Elasticsearch, ConnectionError
#from langchain.chat_models.openai import ChatOpenAI
from langchain_openai import ChatOpenAI
#from langchain.chat_models import ChatOpenAI
from sentence_transformers import CrossEncoder

# --- Local Module Imports ---
# These are kept as imports from separate files as requested.
from retrieverS import (
    retrieve_best_passage,
    fetch_wikipedia_page,
    get_doc_score_from_passages,
    llm_select_next_passage_with_score
)
from question_generationS import (
    load_openai_key,
    generate_seed_questions,
    generate_multihop_questions
)
from verification3 import (
    evaluate_question_naturalness_dynamic,
    get_required_passages,
    verify_question_N_docs
)

# ==============================================================================
# --- MERGED SCRIPT: tree_construct.py ---
# This file combines the features from both of your provided scripts.
# ==============================================================================

# --- FEATURE: PrettyPrinter Class for clean logging ---
class PrettyPrinter:
    PURPLE = '\033[95m'; CYAN = '\033[96m'; BLUE = '\033[94m'; GREEN = '\033[92m'
    YELLOW = '\033[93m'; RED = '\033[91m'; ENDC = '\033[0m'; BOLD = '\033[1m'
    def header(self, text): print(f"\n{self.PURPLE}{self.BOLD}{'=' * 25} {text} {'=' * 25}{self.ENDC}")
    def info(self, text, indent=0): print(f"{' ' * indent}{self.BLUE}INFO: {text}{self.ENDC}")
    def success(self, text, indent=0): print(f"{' ' * indent}{self.GREEN}✅ {text}{self.ENDC}")
    def warning(self, text, indent=0): print(f"{' ' * indent}{self.YELLOW}⚠️ {text}{self.ENDC}")
    def step(self, text, indent=0): print(f"{' ' * indent}{self.CYAN}➡️  {text}{self.ENDC}")
    def print_question_state(self, state, indent=0):
        passage_titles = [f"'{title}'" for title, _ in state.passages_used]
        print(f"{' ' * indent}❓ {self.BOLD}Q:{self.ENDC} \"{state.question}\"")
        print(f"{' ' * (indent+2)}{self.YELLOW}Hops:{self.ENDC} {len(state.passages_used)} | {self.YELLOW}Passages:{self.ENDC} {', '.join(passage_titles)}")

pp = PrettyPrinter()

# --- Configuration & Global Initializations ---
load_openai_key()
FILE1 = "filtered_fever_with_wiki_updated.jsonl"
FILE2 = "reranked_output_5.jsonl"
OUTPUT_FILE = "results_iterative.jsonl"
K_ITERATIONS = 4  # Max expansion attempts per claim
MAX_CANDIDATE_DOCS = 6
ES_INDEX_NAME = "fever"

cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')
chat_for_eval = ChatOpenAI(temperature=0, model_name="gpt-4o", openai_api_key=os.getenv("OPENAI_API_KEY"))
tie_breaker = itertools.count()

# --- FEATURE: Elasticsearch Client (from file 2) ---
try:
    es = Elasticsearch("http://localhost:9200", request_timeout=30)
    if not es.ping():
        raise ConnectionError("Could not connect to Elasticsearch")
    print("Successfully connected to Elasticsearch.")
except ConnectionError as e:
    print(f"Elasticsearch connection failed: {e}")
    es = None

# --- FEATURE: Argument Parser (from file 2) ---
def get_args():
    parser = argparse.ArgumentParser(description="Run the iterative question generation process.")
    parser.add_argument(
        '--retrieval_method', type=str, choices=['llm', 'bm25'], default='llm',
        help="The method for finding the next best passage for expansion ('llm' or 'bm25')."
    )
    return parser.parse_args()

# --- Data Structures ---
@dataclass
class QuestionState:
    question: str; explanation: str; answer: str
    passages_used: List[Tuple[str, str]] = field(default_factory=list)
    def __str__(self):
        passage_titles = [f"'{title}'" for title, _ in self.passages_used]
        return f"Q: \"{self.question}\" (Hops: {len(self.passages_used)}, Passages: {', '.join(passage_titles)})"

# ==============================================================================
# --- HELPER FUNCTIONS ---
# ==============================================================================

# --- PROMPT CHANGE: New Anchor/Bridge Logic using Common Attributes ---
def extract_common_attributes_with_llm(document_text: str, chat_model, num_attributes=5) -> List[str]:
    """
    Uses an LLM to extract high-level common themes/attributes from a text
    to encourage diverse question generation.
    """
    prompt = f"""You are a creative analyst. Your task is to read the following document and identify {num_attributes} high-level attributes, themes, or concepts that could inspire a creative, diverse question when combined with another document.

Instead of just listing specific named entities, focus on broader concepts that describe the text.

Here are some examples of the kind of attributes to identify:
- The main subject's profession (e.g., "actor's career", "political history")
- The genre of a work (e.g., "fantasy series", "comedy-drama film")
- A key event or theme (e.g., "time travel plot", "legal estate battle", "espionage and intelligence")
- A significant achievement (e.g., "award nominations", "box office success")

DOCUMENT:
{document_text[:3000]} # Truncate for efficiency

---
Return a single, comma-separated string of the most promising common attributes.
For example: "actor's career, fantasy series, award nominations, US film debut, television roles"

COMMON ATTRIBUTES:
"""
    try:
        response = chat_model.invoke(prompt)
        attributes = [attr.strip() for attr in response.content.split(',') if attr.strip()]
        return attributes
    except Exception as e:
        print(f"Error during attribute extraction: {e}")
        return []

def find_anchor_and_bridge_documents(claim: str, wiki_titles: List[str], cache: dict, chat_model, cross_encoder):
    if len(wiki_titles) < 2: return []
    pp.step("Finding Anchor Document...", indent=2)
    doc_scores = [(title, get_doc_score_from_passages(claim, title, cache)) for title in wiki_titles]
    doc_scores = [ds for ds in doc_scores if ds[1] is not None]
    if not doc_scores: return []
    ranked_docs = sorted(doc_scores, key=lambda x: x[1], reverse=True)
    anchor_title, _ = ranked_docs[0]
    pp.success(f"Anchor found: '{anchor_title}'", indent=4)
    pp.step("Extracting Common Attributes from Anchor...", indent=2)
    anchor_text = cache.get(anchor_title, "")
    common_attributes = extract_common_attributes_with_llm(anchor_text, chat_model)
    if not common_attributes:
        pp.warning("Could not extract common attributes. Aborting.", indent=4)
        return []
    pp.info(f"Extracted attributes: {common_attributes}", indent=4)
    pp.step("Finding Bridge Document using attributes...", indent=2)
    new_query = claim + " " + " ".join(common_attributes)
    bridge_candidates = [ds for ds in ranked_docs if ds[0] != anchor_title]
    if not bridge_candidates:
        pp.warning("No other documents available to serve as a bridge.", indent=4)
        return [ranked_docs[0]]
    bridge_scores = [(title, get_doc_score_from_passages(new_query, title, cache)) for title, _ in bridge_candidates]
    bridge_scores = [ds for ds in bridge_scores if ds[1] is not None]
    if not bridge_scores: return [ranked_docs[0]]
    best_bridge_title, _ = sorted(bridge_scores, key=lambda x: x[1], reverse=True)[0]
    pp.success(f"Bridge found: '{best_bridge_title}'", indent=4)
    final_docs = [(t, s) for t, s in ranked_docs if t == anchor_title or t == best_bridge_title]
    return final_docs

# --- FEATURE: Elasticsearch BM25 Retrieval (from file 2) ---
def retrieve_passages_with_bm25(query: str, es_client: Elasticsearch, index_name: str, size: int = 5) -> List[Tuple[str, str]]:
    if not es_client:
        pp.warning("Elasticsearch client not available. Skipping retrieval.")
        return []
    request = [{"index": index_name}, {"query": {"match": {"txt": query}}, "size": size}]
    try:
        resp = es_client.msearch(body=request)
    except Exception as e:
        pp.warning(f"Error during Elasticsearch msearch: {e}")
        return []
    docs = []
    for r in resp.get("responses", []):
        for hit in r.get("hits", {}).get("hits", []):
            title = hit["_source"].get("title", "")
            if title: docs.append((title, fetch_wikipedia_page(title) or ""))
    return docs

def find_next_best_passage_bm25_and_rerank(question: str, current_titles_used: Set[str]) -> Tuple[float, Tuple[str, str]]:
    retrieved_docs = retrieve_passages_with_bm25(question, es, ES_INDEX_NAME, size=5)
    unique_titles = list(dict.fromkeys([title for title, _ in retrieved_docs]))
    candidate_titles = [title for title in unique_titles if title not in current_titles_used]
    if not candidate_titles: return -1.0, None

    best_passages_from_docs = []
    pp.info(f"BM25 found candidates: {candidate_titles}. Reranking passages...", indent=8)
    for title in candidate_titles:
        passage_text = retrieve_best_passage(title, question, method='cross-encoder')
        if passage_text:
            best_passages_from_docs.append((title, passage_text))
    if not best_passages_from_docs: return -1.0, None
    
    pairs = [[question, p_text] for _, p_text in best_passages_from_docs]
    scores = cross_encoder.predict(pairs)
    best_idx = scores.argmax()
    return scores[best_idx].item(), best_passages_from_docs[best_idx]


# --- DEBUGGING CHANGE: Added logging to the parser ---
def parse_generated_text(text: str) -> List[dict]:
    pp.header("PARSING LLM OUTPUT")
    print(text)
    pattern = re.compile(r"Question:\s*(.*?)\s*Explanation:\s*(.*?)\s*Answer:\s*(.*)", re.DOTALL)
    parsed_data = []
    blocks = text.strip().split('---')
    pp.info(f"Found {len(blocks)} blocks separated by '---'.")
    for block in blocks:
        if not block.strip(): continue
        match = pattern.search(block)
        if match:
            q, e, a = match.group(1).strip(), match.group(2).strip(), match.group(3).strip()
            if q and e and a:
                parsed_data.append({"question": q, "explanation": e, "answer": a})
    pp.success(f"Successfully parsed {len(parsed_data)} questions.")
    return parsed_data

# ==============================================================================
# --- MAIN LOGIC (Fully Merged & Upgraded) ---
# ==============================================================================
def main():
    args = get_args()
    pp.header(f"INITIALIZING PIPELINE (Retrieval: {args.retrieval_method.upper()})")
    if not es and args.retrieval_method == 'bm25':
        pp.warning("BM25 retrieval method selected, but cannot connect to Elasticsearch. Exiting.")
        return

    # --- Load and Merge Data ---
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
    
    fever_data = dict(islice(fever_data.items(), 7)) # Limit for testing
    print(f"Processing {len(fever_data)} FEVER entries...\n")

    for record in tqdm(list(fever_data.values())):
        claim = record['claim']
        wiki_titles = record.get('wiki_urls', [])
        pp.header(f"PROCESSING CLAIM: {claim}")

        pp.step("Step 1: Finding Anchor and Bridge documents...")
        cache = {title: fetch_wikipedia_page(title) or "" for title in wiki_titles}
        ranked_docs = find_anchor_and_bridge_documents(claim, wiki_titles, cache, chat_for_eval, cross_encoder)
        if len(ranked_docs) < 2:
           pp.warning("Could not find a valid Anchor/Bridge pair. Skipping claim.", indent=2)
           continue
        doc_titles = [title for title, _ in ranked_docs]
        pp.info(f"Using document pair: {doc_titles}", indent=2)

        pp.step("Step 2: Retrieving best passage from each document...")
        candidate_passages = []
        for title, _ in ranked_docs:
            passage_text = retrieve_best_passage(title, claim, method='cross-encoder')
            if passage_text: candidate_passages.append((title, passage_text))
        if not candidate_passages: continue

        pqs = [[] for _ in range(MAX_CANDIDATE_DOCS)]
        all_generated_questions = []

        pp.step("Step 4: Generating seed questions from the top passage...")
        initial_passage_tuple = candidate_passages[0]
        generated_initial = generate_seed_questions(initial_passage_tuple[1], num_questions=3)
        parsed_initial = parse_generated_text(generated_initial)

        for item in parsed_initial:
            initial_state = QuestionState(
                question=item["question"], explanation=item["explanation"],
                answer=item["answer"], passages_used=[initial_passage_tuple]
            )
            all_generated_questions.append(initial_state)
            pp.success(f"Generated 1-hop Seed -> {initial_state.question}", indent=2)
            
            score, next_best_passage = -1.0, None
            current_titles_used = {initial_passage_tuple[0]}
            if args.retrieval_method == 'llm':
                new_remaining = [p for p in candidate_passages if p[0] not in current_titles_used]
                if new_remaining:
                    score, next_best_passage = llm_select_next_passage_with_score(initial_state, new_remaining, chat_for_eval)
            else: # bm25
                score, next_best_passage = find_next_best_passage_bm25_and_rerank(initial_state.question, current_titles_used)

            if next_best_passage:
                heapq.heappush(pqs[0], (-score, next(tie_breaker), initial_state, next_best_passage))

        pp.step("Step 5: Starting iterative expansion process...")
        for iteration_num in range(K_ITERATIONS):
            print("\n" + "─" * 30 + f" Iteration {iteration_num + 1}/{K_ITERATIONS} " + "─" * 30)

            best_candidate_peek, best_pq_index = None, -1
            for i, pq in enumerate(pqs):
                if not pq: continue
                current_top_score = pq[0][0]
                if best_candidate_peek is None or current_top_score < best_candidate_peek[0]:
                    best_candidate_peek, best_pq_index = pq[0], i
            if best_pq_index == -1:
                pp.warning("All priority queues are empty. Stopping expansions.", indent=2)
                break

            neg_score, _, prev_state, passage_to_add = heapq.heappop(pqs[best_pq_index])
            pp.info(f"Expanding best candidate from PQ-{best_pq_index} (Score: {-neg_score:.4f})", indent=2)
            pp.print_question_state(prev_state, indent=4)
            pp.info(f"Adding passage from '{passage_to_add[0]}'", indent=4)

            all_passage_tuples = prev_state.passages_used + [passage_to_add]
            passage_texts = [text for _, text in all_passage_tuples]
            multihop_text = generate_multihop_questions(passage_texts)
            parsed_multihop = parse_generated_text(multihop_text)
            if not parsed_multihop: continue

            for item in parsed_multihop:
                pp.step("Evaluating naturalness of new candidate question...", indent=4)
                naturalness_details = evaluate_question_naturalness_dynamic(item["question"], passage_texts, chat_for_eval)
                LOGICAL_DEPENDENCY_THRESHOLD = 3
                if naturalness_details.get("logical_dependency_score", 0) <= LOGICAL_DEPENDENCY_THRESHOLD:
                    pp.warning(f"Quality gate failed. Logical dependency score was too low. Discarding.", indent=6)
                    continue
                pp.success(f"Quality gate passed.", indent=6)

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

                if new_hop_count > previous_hop_count:
                    pp.success(f"Expansion successful! Hops increased from {previous_hop_count} to {new_hop_count}.", indent=6)
                    next_pq_index = new_hop_count - 1
                    if next_pq_index < MAX_CANDIDATE_DOCS:
                        current_titles_used = {title for title, _ in new_state.passages_used}
                        if args.retrieval_method == 'llm':
                            new_remaining = [p for p in candidate_passages if p[0] not in current_titles_used]
                            if new_remaining:
                                score, next_best_passage = llm_select_next_passage_with_score(new_state, new_remaining, chat_for_eval)
                                if next_best_passage:
                                    pp.info(f"Queuing for next expansion (Confidence: {score:.1f}/5)", indent=8)
                                    heapq.heappush(pqs[next_pq_index], (-score, next(tie_breaker), new_state, next_best_passage))
                        else: # bm25
                            score, next_best_passage = find_next_best_passage_bm25_and_rerank(new_state.question, current_titles_used)
                            if next_best_passage:
                                pp.info(f"Queuing for next expansion (BM25 Score: {score:.4f})", indent=8)
                                heapq.heappush(pqs[next_pq_index], (-score, next(tie_breaker), new_state, next_best_passage))
                
                elif new_hop_count == previous_hop_count and new_passages_set != previous_passages_set:
                    pp.success(f"Transformation successful! Passages shifted. Re-queuing at {previous_hop_count}-hops.", indent=6)
                    current_pq_index = previous_hop_count - 1
                    if current_pq_index < MAX_CANDIDATE_DOCS:
                        current_titles_used = {title for title, _ in new_state.passages_used}
                        if args.retrieval_method == 'llm':
                            new_remaining = [p for p in candidate_passages if p[0] not in current_titles_used]
                            if new_remaining:
                                score, next_best_passage = llm_select_next_passage_with_score(new_state, new_remaining, chat_for_eval)
                                if next_best_passage:
                                    pp.info(f"Re-queuing for next expansion (Confidence: {score:.1f}/5)", indent=8)
                                    heapq.heappush(pqs[current_pq_index], (-score, next(tie_breaker), new_state, next_best_passage))
                        else: # bm25
                            score, next_best_passage = find_next_best_passage_bm25_and_rerank(new_state.question, current_titles_used)
                            if next_best_passage:
                                pp.info(f"Re-queuing for next expansion (BM25 Score: {score:.4f})", indent=8)
                                heapq.heappush(pqs[current_pq_index], (-score, next(tie_breaker), new_state, next_best_passage))
                else:
                    pp.warning(f"Expansion failed. Complexity did not increase or change. Discarding.", indent=6)

        # --- Final Analysis ---
        pp.header("FINAL ANALYSIS")
        if not all_generated_questions:
            pp.warning("No questions were generated.", indent=2)
            continue
        best_question = max(all_generated_questions, key=lambda q: len(q.passages_used), default=None)
        if best_question and len(best_question.passages_used) > 1:
            pp.success(f"Pipeline finished. Best question found has {len(best_question.passages_used)} hops.", indent=2)
            pp.step("Final Question Details:")
            pp.print_question_state(best_question, indent=2)
            pp.step("Performing Final Verification...")
            final_passages_text = [p_text for _, p_text in best_question.passages_used]
            verification_details = verify_question_N_docs(final_passages_text, best_question.question, best_question.answer)
            verdict = "Requires All Passages" if verification_details.get("requires_all_passages") else "Answerable by Subset" if verification_details.get("answerable_with_subset") else "Not Answerable"
            pp.success(f"Verdict: {verdict}", indent=2)
            pp.step("Performing Final Naturalness Evaluation...")
            naturalness_details = evaluate_question_naturalness_dynamic(best_question.question, final_passages_text, chat_for_eval)
            if naturalness_details:
                for key, value in naturalness_details.items():
                    if key != "justification": print(f"{' ' * 4}{key:<30} {value or 'N/A'}/5.0")
                print(f"{' ' * 4}{'justification':<30} {naturalness_details.get('justification')}")
            log_entry = { "status": "success", "claim": claim, "final_question": best_question.question, "num_hops": len(best_question.passages_used), **verification_details, **naturalness_details }
            with open(OUTPUT_FILE, 'a') as f_out: json.dump(log_entry, f_out); f_out.write('\n')
        else:
            pp.warning("Pipeline finished but failed to generate a valid multi-hop question.", indent=2)
            if best_question:
                pp.info("The process ended with this 1-hop question:", indent=4)
                pp.print_question_state(best_question, indent=4)
            log_entry = { "status": "failure", "claim": claim, "final_question": best_question.question if best_question else "N/A", "num_hops": len(best_question.passages_used) if best_question else 0 }
            with open(OUTPUT_FILE, 'a') as f_out: json.dump(log_entry, f_out); f_out.write('\n')



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
