import argparse
import json
import re
import os
import heapq
import itertools
from dataclasses import dataclass, field
from typing import List, Tuple, Set
from itertools import islice

# --- External Library Imports ---
from tqdm import tqdm
from elasticsearch import Elasticsearch, ConnectionError
from sentence_transformers import CrossEncoder

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# --- Local Module Imports ---
from retriever import (
    retrieve_best_passage,
    fetch_wikipedia_page,
    get_doc_score_from_passages,
    llm_select_next_passage_with_score
)
from question_generation import (
    load_openai_key,
    generate_seed_questions,
    generate_multihop_questions
)
from verification3 import (
    evaluate_question_naturalness_dynamic,
    get_required_passages,
    verify_question_N_docs
)

# =========================
# Pretty logging
# =========================
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

# =========================
# Config & Globals
# =========================
load_openai_key() 

FILE1 = "filtered_fever_with_wiki_updated.jsonl"
FILE2 = "reranked_output_5.jsonl"
OUTPUT_FILE = "results_iterative.jsonl"
K_ITERATIONS = 10
MAX_CANDIDATE_DOCS = 6
ES_INDEX_NAME = "fever"

cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')
tie_breaker = itertools.count()

# =========================
# Elasticsearch
# =========================
try:
    es = Elasticsearch("http://localhost:9200", request_timeout=30)
    if not es.ping():
        raise ConnectionError("Could not connect to Elasticsearch")
    print("Successfully connected to Elasticsearch.")
except ConnectionError as e:
    print(f"Elasticsearch connection failed: {e}")
    es = None

# =========================
# Args & Chat factory
# =========================
parser = argparse.ArgumentParser(
    description="Iterative multi-hop question generation pipeline"
)
parser.add_argument(
    "--llm",
    default="gpt4o",
    choices=["gpt4o", "gpt5", "claude", "llama", "gemma"],
    help="Which LLM to use (alias).",
)

# Fixed temperature for all models (override via env if you really want)
DEFAULT_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))

# vLLM endpoint for open-source models
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")

MODEL_ALIASES = {
    # OpenAI cloud (must have OPENAI_API_KEY set, and OPENAI_BASE_URL **unset**)
    "gpt4o":  {"provider": "openai",    "model": "gpt-4o"},
    "gpt5":   {"provider": "openai",    "model": "gpt-5"},

    # Anthropic cloud
    "claude": {"provider": "anthropic", "model": "claude-3-5-haiku-latest"},

    # Open-source via vLLM OpenAI-compatible server
    "llama":  {"provider": "openai",    "model": "meta-llama/Llama-3.1-8B-Instruct"},
    "gemma":  {"provider": "openai",    "model": "google/medgemma-4b-it"},
}

def build_chat(llm_alias: str):
    alias = MODEL_ALIASES.get(llm_alias)
    if alias is None:
        raise ValueError(f"Unknown llm alias: {llm_alias}")

    provider = alias["provider"]
    model_id = alias["model"]

    is_open_source = model_id in (
        MODEL_ALIASES["llama"]["model"],
        MODEL_ALIASES["gemma"]["model"],
    )

    # ---------- OpenAI / vLLM ----------
    if provider == "openai":
        if is_open_source:
            # Always go through vLLM backend for llama/gemma
            base_url = VLLM_BASE_URL
            api_key = os.getenv("OPENAI_API_KEY", "dummy-vllm-key")
        else:
            # Must hit real OpenAI cloud for gpt4o/gpt5
            base_url = None  # critical: do NOT use OPENAI_BASE_URL
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY is required for gpt4o/gpt5.")

        return ChatOpenAI(
            model=model_id,
            temperature=DEFAULT_TEMPERATURE,
            base_url=base_url,  # None = OpenAI cloud; URL = vLLM
            api_key=api_key,
        )

    # ---------- Anthropic ----------
    if provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            pp.warning("ANTHROPIC_API_KEY not set; Claude calls will fail.", indent=2)
        return ChatAnthropic(
            model=model_id,
            temperature=DEFAULT_TEMPERATURE,
            api_key=api_key,
        )

    raise ValueError(f"Unknown provider: {provider}")

# =========================
# Data structures
# =========================
@dataclass
class QuestionState:
    question: str
    explanation: str
    answer: str
    passages_used: List[Tuple[str, str]] = field(default_factory=list)
    def __str__(self):
        passage_titles = [f"'{title}'" for title, _ in self.passages_used]
        return f"Q: \"{self.question}\" (Hops: {len(self.passages_used)}, Passages: {', '.join(passage_titles)})"

# ==============================================================================
# Helpers
# ==============================================================================

def extract_common_attributes_with_llm(document_text: str, chat_model, num_attributes=5) -> List[str]:
    """
    Use an LLM to extract high-level attributes/themes.
    """
    prompt = f"""You are a creative analyst. Identify {num_attributes} high-level attributes, themes, or concepts
that would inspire diverse questions when combined with another document.

Prefer broad concepts over specific named entities.

DOCUMENT:
{document_text[:3000]}

Return a single, comma-separated list only.
"""
    try:
        response = chat_model.invoke(prompt)
        text = getattr(response, "content", response)
        attributes = [attr.strip() for attr in str(text).split(',') if attr.strip()]
        return attributes
    except Exception as e:
        print(f"Error during attribute extraction: {e}")
        return []

def retrieve_passages_with_bm25(query: str, es_client: Elasticsearch, index_name: str, size: int = 5) -> List[Tuple[str, str]]:
    """Retrieves top passages from Elasticsearch using BM25."""
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
            if title:
                docs.append((title, fetch_wikipedia_page(title) or ""))
    return docs

def find_anchor_and_bridge_documents(claim: str, wiki_titles: List[str], cache: dict, chat_model, cross_encoder):
    if len(wiki_titles) < 2:
        return []
    pp.step("Finding Anchor Document...", indent=2)
    doc_scores = [(title, get_doc_score_from_passages(claim, title, cache)) for title in wiki_titles]
    doc_scores = [ds for ds in doc_scores if ds[1] is not None]
    if not doc_scores:
        return []
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
    if not bridge_scores:
        return [ranked_docs[0]]
    best_bridge_title, _ = sorted(bridge_scores, key=lambda x: x[1], reverse=True)[0]
    pp.success(f"Bridge found: '{best_bridge_title}'", indent=4)
    final_docs = [(t, s) for t, s in ranked_docs if t == anchor_title or t == best_bridge_title]
    return final_docs

def rerank_passages(question: str, passages_to_rerank: List[Tuple[str, str]], top_k: int) -> List[Tuple[str, str]]:
    """Rerank passages with CrossEncoder and return the top_k."""
    if not passages_to_rerank:
        return []
    print(f"[INFO] Reranking {len(passages_to_rerank)} passages to select top {top_k}...")
    pairs = [[question, p_text] for _, p_text in passages_to_rerank]
    scores = cross_encoder.predict(pairs)
    scored_passages = list(zip(passages_to_rerank, scores))
    sorted_passages = sorted(scored_passages, key=lambda x: x[1], reverse=True)
    return [passage for passage, score in sorted_passages[:top_k]]

def find_next_passage_multistage(current_state: 'QuestionState',
                                 current_titles_used: Set[str],
                                 chat_model) -> Tuple[float, Tuple[str, str]]:
    """
    1) BM25 retrieve many docs
    2) Extract best passage per doc + rerank → top-10
    3) LLM picks the single best from top-10 (returns score, passage)
    """
    question = current_state.question

    print(f"\n[INFO] Step 1: BM25 retrieval for: \"{question[:100]}...\"")
    retrieved_docs = retrieve_passages_with_bm25(question, es, ES_INDEX_NAME, size=100)

    unique_titles = list(dict.fromkeys([title for title, _ in retrieved_docs]))
    candidate_titles = [title for title in unique_titles if title not in current_titles_used]

    if not candidate_titles:
        print("[INFO] Step 1: BM25 found no new documents.")
        return -1.0, None

    print(f"[INFO] Extracting best passage from {len(candidate_titles)} candidate documents...")
    best_passages_from_docs = []
    for title in tqdm(candidate_titles, desc="Extracting Passages"):
        passage_text = retrieve_best_passage(title, question, method='cross-encoder')
        if passage_text:
            best_passages_from_docs.append((title, passage_text))

    if not best_passages_from_docs:
        print("[INFO] Step 2: No relevant passages found.")
        return -1.0, None

    top_10_passages = rerank_passages(question, best_passages_from_docs, top_k=10)
    if not top_10_passages:
        print("[INFO] Step 2: Reranking yielded no passages.")
        return -1.0, None

    print("[INFO] Step 3: LLM selecting the final best passage from top-10.")
    # Be compatible with older signature that may not accept chat_model
    try:
        return llm_select_next_passage_with_score(current_state, top_10_passages, chat_model)
    except TypeError:
        return llm_select_next_passage_with_score(current_state, top_10_passages)

# Robust wrappers in case other files haven't yet been refactored to accept `chat=...`
def _gen_seed_questions(doc_text: str, num_questions: int, chat):
    try:
        return generate_seed_questions(doc_text, num_questions=num_questions, chat=chat)
    except TypeError:
        return generate_seed_questions(doc_text, num_questions=num_questions)

def _gen_multihop_questions(passages_texts: List[str], num_questions: int, chat):
    try:
        return generate_multihop_questions(passages_texts, num_questions=num_questions, chat=chat)
    except TypeError:
        return generate_multihop_questions(passages_texts, num_questions=num_questions)

def _verify_question(passages_texts: List[str], q: str, a: str, chat):
    try:
        return verify_question_N_docs(passages_texts, q, a, chat_model=chat)
    except TypeError:
        return verify_question_N_docs(passages_texts, q, a)

def _eval_naturalness(q: str, passages_texts: List[str], chat):
    try:
        return evaluate_question_naturalness_dynamic(q, passages_texts, chat_model=chat)
    except TypeError:
        return evaluate_question_naturalness_dynamic(q, passages_texts)

# =========================
# Parsing of model outputs
# =========================
def parse_generated_text(text: str) -> List[dict]:
    pp.header("PARSING LLM OUTPUT")
    print(text)
    pattern = re.compile(r"Question:\s*(.*?)\s*Explanation:\s*(.*?)\s*Answer:\s*(.*)", re.DOTALL)
    parsed_data = []
    blocks = str(text).strip().split('---')
    pp.info(f"Found {len(blocks)} blocks separated by '---'.")
    for block in blocks:
        if not block.strip():
            continue
        match = pattern.search(block)
        if match:
            q, e, a = match.group(1).strip(), match.group(2).strip(), match.group(3).strip()
            if q and e and a:
                parsed_data.append({"question": q, "explanation": e, "answer": a})
    pp.success(f"Successfully parsed {len(parsed_data)} questions.")
    return parsed_data

# ==============================================================================
# Main
# ==============================================================================
def main():
    if not es:
        print("Cannot proceed without an Elasticsearch connection. Exiting.")
        return

    args = parser.parse_args()
    chat = build_chat(args.llm)

    # --- Load and Merge Data ---
    fever_data = {}
    with open(FILE1, 'r') as f1:
        for line in f1:
            if not line.strip():
                continue
            rec = json.loads(line)
            claim = rec.get("claim")
            urls = [title.replace("_", " ") for title in rec.get("wiki_urls", [])]
            if claim:
                fever_data[claim] = {"claim": claim, "wiki_urls": list(dict.fromkeys(urls))}
    with open(FILE2, 'r') as f2:
        for line in f2:
            if not line.strip():
                continue
            rec = json.loads(line)
            query = rec.get("query")
            if not query or query not in fever_data:
                continue
            existing = set(fever_data[query]["wiki_urls"])
            for d in rec.get("docs", []):
                title = d.get("title").replace("_", " ")
                if title and title not in existing:
                    fever_data[query]["wiki_urls"].append(title)
                    existing.add(title)

    # Keep a small slice for demo/fast runs (same as before)
    fever_data = dict(islice(fever_data.items(), 2))
    print(f"Processing {len(fever_data)} FEVER entries...\n")

    for record in tqdm(list(fever_data.values())):
        claim = record['claim']
        wiki_titles = record.get('wiki_urls', [])
        pp.header(f"PROCESSING CLAIM: {claim}")

        pp.step("Step 1: Finding Anchor and Bridge documents...")
        cache = {title: fetch_wikipedia_page(title) or "" for title in wiki_titles}
        ranked_docs = find_anchor_and_bridge_documents(claim, wiki_titles, cache, chat, cross_encoder)
        if len(ranked_docs) < 2:
            pp.warning("Could not find a valid Anchor/Bridge pair. Skipping claim.", indent=2)
            continue
        doc_titles = [title for title, _ in ranked_docs]
        pp.info(f"Using document pair: {doc_titles}", indent=2)

        pp.step("Step 2: Retrieving best passage from each document...")
        candidate_passages = []
        for title, _ in ranked_docs:
            passage_text = retrieve_best_passage(title, claim, method='cross-encoder')
            if passage_text:
                candidate_passages.append((title, passage_text))
        if not candidate_passages:
            continue

        pqs = [[] for _ in range(MAX_CANDIDATE_DOCS)]
        all_generated_questions = []

        pp.step("Step 4: Generating seed questions from the top passage...")
        initial_passage_tuple = candidate_passages[0]
        generated_initial = _gen_seed_questions(initial_passage_tuple[1], num_questions=3, chat=chat)
        parsed_initial = parse_generated_text(generated_initial)

        for item in parsed_initial:
            initial_state = QuestionState(
                question=item["question"], explanation=item["explanation"],
                answer=item["answer"], passages_used=[initial_passage_tuple]
            )
            all_generated_questions.append(initial_state)
            pp.success(f"Generated 1-hop Seed -> {initial_state.question}", indent=2)

            current_titles_used = {initial_passage_tuple[0]}
            score, next_best_passage = find_next_passage_multistage(initial_state, current_titles_used, chat)
            if next_best_passage:
                heapq.heappush(pqs[0], (-score, next(tie_breaker), initial_state, next_best_passage))

        pp.step("Step 5: Starting iterative expansion process...")
        for iteration_num in range(K_ITERATIONS):
            print("\n" + "─" * 30 + f" Iteration {iteration_num + 1}/{K_ITERATIONS} " + "─" * 30)

            best_candidate_peek, best_pq_index = None, -1
            for i, pq in enumerate(pqs):
                if not pq:
                    continue
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
            multihop_text = _gen_multihop_questions(passage_texts, num_questions=3, chat=chat)
            parsed_multihop = parse_generated_text(multihop_text)
            if not parsed_multihop:
                continue

            for item in parsed_multihop:
                pp.step("Evaluating naturalness of new candidate question...", indent=4)
                naturalness_details = evaluate_question_naturalness_dynamic(item["question"], passage_texts, chat)
                LOGICAL_DEPENDENCY_THRESHOLD = 3
                logical_dep = naturalness_details.get("logical_dependency_score")
                try:
                    logical_dep = int(logical_dep) if logical_dep is not None else 0
                except Exception:
                    logical_dep = 0
                
                if logical_dep <= LOGICAL_DEPENDENCY_THRESHOLD:
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
                        score, next_best_passage = find_next_passage_multistage(new_state, current_titles_used, chat)
                        if next_best_passage:
                            pp.info(f"Queuing for next expansion (Confidence: {score:.1f}/5)", indent=8)
                            heapq.heappush(pqs[next_pq_index],
                                           (-score, next(tie_breaker), new_state, next_best_passage))

                elif new_hop_count == previous_hop_count and new_passages_set != previous_passages_set:
                    pp.success(f"Transformation successful! Passages shifted. Re-queuing at {previous_hop_count}-hops.", indent=6)
                    current_pq_index = previous_hop_count - 1
                    if current_pq_index < MAX_CANDIDATE_DOCS:
                        current_titles_used = {title for title, _ in new_state.passages_used}
                        score, next_best_passage = find_next_passage_multistage(new_state, current_titles_used, chat)
                        if next_best_passage:
                            pp.info(f"Queuing for next expansion (Confidence: {score:.1f}/5)", indent=8)
                            heapq.heappush(pqs[current_pq_index],
                                           (-score, next(tie_breaker), new_state, next_best_passage))
                else:
                    pp.warning("Expansion failed. Complexity did not increase or change. Discarding.", indent=6)

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
            verification_details = _verify_question(final_passages_text, best_question.question, best_question.answer, chat)
            verdict = ("Requires All Passages" if verification_details.get("requires_all_passages")
                       else "Answerable by Subset" if verification_details.get("answerable_with_subset")
                       else "Not Answerable")
            pp.success(f"Verdict: {verdict}", indent=2)

            pp.step("Performing Final Naturalness Evaluation...")
            naturalness_details = _eval_naturalness(best_question.question, final_passages_text, chat)
            if naturalness_details:
                for key, value in naturalness_details.items():
                    if key != "justification":
                        print(f"{' ' * 4}{key:<30} {value or 'N/A'}/5.0")
                print(f"{' ' * 4}{'justification':<30} {naturalness_details.get('justification')}")

            log_entry = {
                "status": "success",
                "claim": claim,
                "final_question": best_question.question,
                "num_hops": len(best_question.passages_used),
                **verification_details,
                **naturalness_details
            }
            with open(OUTPUT_FILE, 'a') as f_out:
                json.dump(log_entry, f_out); f_out.write('\n')
        else:
            pp.warning("Pipeline finished but failed to generate a valid multi-hop question.", indent=2)
            if best_question:
                pp.info("The process ended with this 1-hop question:", indent=4)
                pp.print_question_state(best_question, indent=4)
            log_entry = {
                "status": "failure",
                "claim": claim,
                "final_question": best_question.question if best_question else "N/A",
                "num_hops": len(best_question.passages_used) if best_question else 0
            }
            with open(OUTPUT_FILE, 'a') as f_out:
                json.dump(log_entry, f_out); f_out.write('\n')

# =========================
# Metrics
# =========================
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
    print(f"  - Requires All Passages: {verdict_counts['requires_all']} ({(verdict_counts['requires_all'] / total_questions) * 100:.2f}%)")
    print(f"  - Answerable by Subset: {verdict_counts['subset']} ({(verdict_counts['subset'] / total_questions) * 100:.2f}%)")
    print(f"  - Not Answerable: {verdict_counts['not_answerable']} ({(verdict_counts['not_answerable'] / total_questions) * 100:.2f}%)")

    print("\n🌿 Average Naturalness Scores by Dimension:")
    for key in naturalness_keys:
        avg = naturalness_totals[key] / naturalness_counts[key] if naturalness_counts[key] > 0 else 0
        print(f"  - {key}: {avg:.2f} / 5.0")


if __name__ == "__main__":
    main()
    calculate_metrics()
