import json
import re
import os
import heapq
import itertools
from dataclasses import dataclass, field
from typing import List, Tuple, Set, Dict
from itertools import islice

from tqdm import tqdm
from elasticsearch import Elasticsearch, ConnectionError
from langchain_openai import ChatOpenAI
from langchain_core.messages.human import HumanMessage
from sentence_transformers import CrossEncoder

from retriever import (
    retrieve_best_passage,
    fetch_wikipedia_page,
    get_doc_score_from_passages,
    llm_select_next_passage_with_score
)

from question_generation import (
    load_openai_key,
    generate_seed_questions,
    generate_multihop_questions,
    revise_question,
    generate_simplified_question
)

from verification3 import (
    evaluate_question_naturalness_dynamic,
    get_required_passages,
    verify_question_N_docs,
    verify_question_phase2
)

from visualization import HistoryNode, QUESTION_ID_COUNTER, generate_output, PrettyPrinter

pp = PrettyPrinter()

# Configuration
load_openai_key()
FILE1 = "filtered_fever_with_wiki_updated.jsonl"
FILE2 = "reranked_output_5.jsonl"
OUTPUT_FILE = "results_iterative_phase1.jsonl"
OUTPUT_FILE_PHASE2 = "results_iterative_phase2.jsonl"
K_ITERATIONS = 10
MAX_CANDIDATE_DOCS = 6
ES_INDEX_NAME = "fever"
MAX_REVISIONS = 5

cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')
chat_for_eval = ChatOpenAI(temperature=0, model_name="gpt-4o", openai_api_key=os.getenv("OPENAI_API_KEY"))
tie_breaker = itertools.count()

# --- Elasticsearch Setup ---
try:
    es = Elasticsearch("http://localhost:9200", request_timeout=30)
    if not es.ping():
        raise ConnectionError("Could not connect to Elasticsearch")
    print("Successfully connected to Elasticsearch.")
except Exception as e:
    print(f"Elasticsearch connection failed: {e}")
    es = None


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
# --- PHASE 1 HELPERS ---
# ==============================================================================

def extract_common_attributes_with_llm(document_text: str, chat_model, num_attributes=5) -> List[str]:
    prompt = f"""You are a creative analyst. Read the document and identify {num_attributes} high-level attributes/themes.
DOCUMENT: {document_text[:3000]}
Return comma-separated string."""
    try:
        response = chat_model.invoke(prompt)
        attributes = [attr.strip() for attr in response.content.split(',') if attr.strip()]
        return attributes
    except Exception as e:
        print(f"Error during attribute extraction: {e}")
        return []

def retrieve_passages_with_bm25(query: str, es_client: Elasticsearch, index_name: str, size: int = 100) -> List[Tuple[str, str]]:
    if not es_client: return []
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

def find_anchor_and_bridge_documents(claim: str, wiki_titles: List[str], cache: dict, chat_model, cross_encoder):
    if len(wiki_titles) < 2: return []
    pp.step("Finding Anchor Document...", indent=2)
    doc_scores = [(title, get_doc_score_from_passages(claim, title, cache)) for title in wiki_titles]
    doc_scores = [ds for ds in doc_scores if ds[1] is not None]
    if not doc_scores: return []
    ranked_docs = sorted(doc_scores, key=lambda x: x[1], reverse=True)
    anchor_title, _ = ranked_docs[0]
    pp.success(f"Anchor found: '{anchor_title}'", indent=4)
    
    anchor_text = cache.get(anchor_title, "")
    common_attributes = extract_common_attributes_with_llm(anchor_text, chat_model)
    
    new_query = claim + " " + " ".join(common_attributes) if common_attributes else claim
    bridge_candidates = [ds for ds in ranked_docs if ds[0] != anchor_title]
    if not bridge_candidates: return [ranked_docs[0]]
    bridge_scores = [(title, get_doc_score_from_passages(new_query, title, cache)) for title, _ in bridge_candidates]
    bridge_scores = [ds for ds in bridge_scores if ds[1] is not None]
    if not bridge_scores: return [ranked_docs[0]]
    best_bridge_title, _ = sorted(bridge_scores, key=lambda x: x[1], reverse=True)[0]
    pp.success(f"Bridge found: '{best_bridge_title}'", indent=4)
    final_docs = [(t, s) for t, s in ranked_docs if t == anchor_title or t == best_bridge_title]
    return final_docs

def rerank_passages(question: str, passages_to_rerank: List[Tuple[str, str]], top_k: int) -> List[Tuple[str, str]]:
    if not passages_to_rerank: return []
    # print(f"[INFO] Reranking {len(passages_to_rerank)} passages...") # Silenced
    pairs = [[question, p_text] for _, p_text in passages_to_rerank]
    scores = cross_encoder.predict(pairs)
    scored_passages = list(zip(passages_to_rerank, scores))
    sorted_passages = sorted(scored_passages, key=lambda x: x[1], reverse=True)
    return [passage for passage, score in sorted_passages[:top_k]]

def find_next_passage_multistage(current_state: QuestionState, current_titles_used: Set[str]) -> Tuple[float, Tuple[str, str]]:
    question = current_state.question
    retrieved_docs = retrieve_passages_with_bm25(question, es, ES_INDEX_NAME, size=100)
    unique_titles = list(dict.fromkeys([title for title, _ in retrieved_docs]))
    candidate_titles = [title for title in unique_titles if title not in current_titles_used]
    if not candidate_titles: return -1.0, None

    best_passages_from_docs = []
    for title in tqdm(candidate_titles[:20], desc="Extracting Passages", leave=False, disable=True):
        passage_text = retrieve_best_passage(title, question, method='cross-encoder')
        if passage_text: best_passages_from_docs.append((title, passage_text))
    
    if not best_passages_from_docs: return -1.0, None
    top_10_passages = rerank_passages(question, best_passages_from_docs, top_k=10)
    return llm_select_next_passage_with_score(current_state, top_10_passages, chat_for_eval)

def parse_generated_text(text: str) -> List[dict]:
    pp.header("PARSING LLM OUTPUT")
    pattern = re.compile(r"Question:\s*(.*?)\s*Explanation:\s*(.*?)\s*Answer:\s*(.*)", re.DOTALL)
    parsed_data = []
    blocks = text.strip().split('---')
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
# --- PHASE 2 & DIVERSITY HELPERS ---
# ==============================================================================

def evaluate_lexical_diversity_llm(questions_list):
    if not questions_list:
        print("\n🧠 Diversity Analysis: No questions provided.")
        return
    print("\n🧠 Performing LLM-based Diversity Analysis...")
    q_text_list = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions_list)])
    
    prompt = f"""You are an expert linguist. Determine the "Question Template" for each question by replacing specific entities with placeholders like [PERSON], [MOVIE], [DATE].
    List:
    {q_text_list}
    Return a summary of Unique Types and their counts."""
    
    try:
        chat = ChatOpenAI(temperature=0.0, model_name="gpt-4o")
        response = chat.invoke([HumanMessage(content=prompt)])
        print("\n" + response.content)
    except Exception as e:
        print(f"Error during diversity evaluation: {e}")

def run_phase_2_top_down(successful_questions: List[QuestionState]):
    """
    Phase 2: Takes max-hop questions and simplifies them recursively.
    """
    pp.header("PHASE 2: TOP-DOWN SIMPLIFICATION")
    
    # Sort by Hops (Descending) and take top 5
    successful_questions.sort(key=lambda x: len(x.passages_used), reverse=True)
    candidates = successful_questions[:5]
    
    pp.info(f"Selected {len(candidates)} questions to simplify down to 1 hop.")
    phase2_questions_for_diversity = []

    for idx, start_q in enumerate(candidates):
        print(f"\n{pp.BLUE}>>> Simplifying Candidate {idx+1}: {start_q.question} ({len(start_q.passages_used)} hops){pp.ENDC}")
        current_state = start_q
        
        # While strictly > 1 hop, try to reduce to current_hops - 1
        while len(current_state.passages_used) > 1:
            current_hops = len(current_state.passages_used)
            target_hops = current_hops - 1
            
            # Remove one doc (Strategy: Remove the last one added)
            passages_to_keep = current_state.passages_used[:-1]
            removed_passage = current_state.passages_used[-1]
            pp.info(f"Reducing {current_hops} -> {target_hops}. Removing: '{removed_passage[0]}'", indent=2)
            
            kept_texts = [p[1] for p in passages_to_keep]
            new_q_text = generate_simplified_question(
                current_state.question, 
                current_state.answer, 
                kept_texts, 
                target_hops
            )
            
            parsed = parse_generated_text(new_q_text)
            if not parsed:
                pp.warning("Failed to parse simplified question. Stopping this branch.", indent=4)
                break
                
            cand = parsed[0]
            
            pp.step("Verifying simplified question...", indent=4)
            verification = verify_question_phase2(kept_texts, cand["question"], cand["answer"])
            
            status = "passed" if verification["requires_all_passages"] else "failed"

            if status == "passed":
                pp.success("Simplification Validated!", indent=6)
                
                new_state = QuestionState(
                    question=cand["question"],
                    explanation=cand["explanation"],
                    answer=cand["answer"],
                    passages_used=passages_to_keep
                )
                
                # Log Phase 2 with FULL PASSAGE DATA
                log_entry = {
                    "phase": "2_simplification",
                    "status": "success",
                    "parent_q": current_state.question,
                    "simplified_q": new_state.question,
                    "answer": new_state.answer,
                    "hops": target_hops,
                    "verification": verification,
                    "passages": [{"title": t, "text": txt} for t, txt in new_state.passages_used]
                }
                with open(OUTPUT_FILE_PHASE2, 'a') as f:
                    json.dump(log_entry, f); f.write('\n')
                
                phase2_questions_for_diversity.append(new_state.question)
                current_state = new_state 
            else:
                pp.warning(f"Verification Failed (Not strict {target_hops} hops). Reason: {verification}", indent=6)
                # Dump failed attempts too for debugging
                log_entry = {
                    "phase": "2_simplification",
                    "status": "failed",
                    "parent_q": current_state.question,
                    "attempted_q": cand["question"],
                    "hops": target_hops,
                    "verification": verification
                }
                with open(OUTPUT_FILE_PHASE2, 'a') as f:
                    json.dump(log_entry, f); f.write('\n')
                break

    evaluate_lexical_diversity_llm(phase2_questions_for_diversity)
    return phase2_questions_for_diversity


# ==============================================================================
# --- MAIN LOGIC ---
# ==============================================================================
def main():
    if not es:
        print("Cannot proceed without an Elasticsearch connection. Exiting.")
        return

    # --- Load Data ---
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

    fever_data = dict(islice(fever_data.items(), 2)) # Run for 2 claims
    print(f"Processing {len(fever_data)} FEVER entries...\n")

    for record in tqdm(list(fever_data.values())):
        claim = record['claim']
        wiki_titles = record.get('wiki_urls', [])
        pp.header(f"PROCESSING CLAIM: {claim}")

        # --- PHASE 1: BOTTOM-UP EXPANSION ---
        pp.header("PHASE 1: BOTTOM-UP EXPANSION")

        cache = {title: fetch_wikipedia_page(title) or "" for title in wiki_titles}
        ranked_docs = find_anchor_and_bridge_documents(claim, wiki_titles, cache, chat_for_eval, cross_encoder)

        if len(ranked_docs) < 2: continue

        expansion_history = []
        question_state_to_id = {}
        all_generated_questions_phase1 = [] 

        initial_passage_tuple = (ranked_docs[0][0], retrieve_best_passage(ranked_docs[0][0], claim, method='cross-encoder'))
        if not initial_passage_tuple[1]: continue

        generated_initial = generate_seed_questions(initial_passage_tuple[1], num_questions=3)
        parsed_initial = parse_generated_text(generated_initial)

        pqs = [[] for _ in range(MAX_CANDIDATE_DOCS)]

        for item in parsed_initial:
            initial_state = QuestionState(question=item["question"], explanation=item["explanation"], answer=item["answer"], passages_used=[initial_passage_tuple])
            all_generated_questions_phase1.append(initial_state)

            # Seeds are 1-hop, marked as 'passed' (trivial)
            current_q_id = next(QUESTION_ID_COUNTER)
            question_state_to_id[id(initial_state)] = f"{current_q_id}"
            expansion_history.append(HistoryNode(
                id=f"{current_q_id}", 
                parent_ids=[], 
                question_text=initial_state.question, 
                answer=initial_state.answer, 
                explanation=initial_state.explanation, 
                passages_used=initial_state.passages_used, 
                is_seed=True,
                verification_status="passed"
            ))

            current_titles_used = {initial_passage_tuple[0]}
            score, next_best_passage = find_next_passage_multistage(initial_state, current_titles_used)
            if next_best_passage:
                heapq.heappush(pqs[0], (-score, next(tie_breaker), initial_state, next_best_passage))

        for iteration_num in range(K_ITERATIONS):
            best_candidate_peek, best_pq_index = None, -1
            for i, pq in enumerate(pqs):
                if not pq: continue
                if best_candidate_peek is None or pq[0][0] < best_candidate_peek[0]:
                    best_candidate_peek, best_pq_index = pq[0], i
            if best_pq_index == -1: break

            neg_score, _, prev_state, passage_to_add = heapq.heappop(pqs[best_pq_index])
            pp.info(f"Expanding Q (Hops: {len(prev_state.passages_used)}) -> Adding '{passage_to_add[0]}'", indent=2)

            all_passage_tuples = prev_state.passages_used + [passage_to_add]
            passage_texts = [text for _, text in all_passage_tuples]
            multihop_text = generate_multihop_questions(passage_texts)
            parsed_multihop = parse_generated_text(multihop_text)
            
            if not parsed_multihop: continue
            
            for item in parsed_multihop:
                revision_attempts = 0
                is_successful = False
                item_to_process = item
                
                while revision_attempts < MAX_REVISIONS and not is_successful:
                    # 1. Structural Check
                    minimal_passages = get_required_passages(item_to_process["question"], item_to_process["answer"], all_passage_tuples)
                    new_hop_count = len({t for t, _ in minimal_passages})
                    
                    if new_hop_count <= len(prev_state.passages_used):
                        naturalness = evaluate_question_naturalness_dynamic(item_to_process["question"], passage_texts, chat_for_eval)
                        revised_text = revise_question(passage_texts, item_to_process, minimal_passages, naturalness)
                        parsed_rev = parse_generated_text(revised_text)
                        if parsed_rev:
                            item_to_process = parsed_rev[0]
                            revision_attempts += 1
                        else: break
                    else:
                        is_successful = True
                
                # 2. Strict Verification (Subsets & General Knowledge)
                status = "failed"
                verify_res = {}
                if is_successful:
                    passages_text = [p[1] for p in minimal_passages]
                    verify_res = verify_question_N_docs(passages_text, item_to_process["question"], item_to_process["answer"])
                    status = "passed" if verify_res["requires_all_passages"] else "failed"

                    # Console Print
                    if status == "passed":
                        pp.success(f"Verification PASSED: {item_to_process['question'][:50]}...", indent=4)
                    else:
                        pp.warning(f"Verification FAILED: {item_to_process['question'][:50]}...", indent=4)

                # 3. Handle Result
                new_state = QuestionState(question=item_to_process["question"], explanation=item_to_process["explanation"], answer=item_to_process["answer"], passages_used=minimal_passages)
                
                if status == "passed":
                    all_generated_questions_phase1.append(new_state)
                    # Add to queue only if passed
                    if len(new_state.passages_used) - 1 < MAX_CANDIDATE_DOCS:
                         used_titles = {t for t, _ in new_state.passages_used}
                         score, next_p = find_next_passage_multistage(new_state, used_titles)
                         if next_p:
                              heapq.heappush(pqs[len(new_state.passages_used)-1], (-score, next(tie_breaker), new_state, next_p))

                # Log History Node (Visualize ALL attempts)
                new_q_id = next(QUESTION_ID_COUNTER)
                expansion_history.append(HistoryNode(
                    id=f"{new_q_id}", 
                    parent_ids=[question_state_to_id.get(id(prev_state), "0")], 
                    question_text=new_state.question, 
                    answer=new_state.answer, 
                    explanation=new_state.explanation, 
                    passages_used=new_state.passages_used, 
                    is_seed=False,
                    verification_status=status
                ))
                question_state_to_id[id(new_state)] = f"{new_q_id}"
                
                # Dump JSONL Cleanly
                log_entry = {
                    "status": status,
                    "phase": "1_expansion",
                    "claim": claim,
                    "question": new_state.question,
                    "answer": new_state.answer,
                    "explanation": new_state.explanation,
                    "hops": len(new_state.passages_used),
                    "verification_details": verify_res,
                    "passages": [{"title": t, "text": txt} for t, txt in new_state.passages_used]
                }
                with open(OUTPUT_FILE, 'a') as f:
                    json.dump(log_entry, f); f.write('\n')

        # Generate Visualization
        generate_output(claim, expansion_history)
        
        # Diversity Phase 1
        phase1_q_list = [q.question for q in all_generated_questions_phase1 if len(q.passages_used) >= 2]
        evaluate_lexical_diversity_llm(phase1_q_list[:10])

        # --- PHASE 2: TOP-DOWN SIMPLIFICATION ---
        run_phase_2_top_down(all_generated_questions_phase1)

def calculate_metrics_phase1():
    print("\n--- PHASE 1 METRICS ---")
    if not os.path.exists(OUTPUT_FILE): return
    total, passed, hops = 0, 0, {}
    with open(OUTPUT_FILE, 'r') as f:
        for line in f:
            d = json.loads(line)
            total += 1
            if d.get("status") == "passed":
                passed += 1
                h = d.get("hops", 0)
                hops[h] = hops.get(h, 0) + 1
    print(f"Total Generated: {total}, Passed: {passed}, Passed Hops Dist: {hops}")

def calculate_metrics_phase2():
    print("\n--- PHASE 2 METRICS ---")
    if not os.path.exists(OUTPUT_FILE_PHASE2): return
    total, success = 0, 0
    with open(OUTPUT_FILE_PHASE2, 'r') as f:
        for line in f:
            d = json.loads(line)
            total += 1
            if d.get("status") == "success": success += 1
    print(f"Total Phase 2 Attempts: {total}, Success: {success}")

if __name__ == "__main__":
    main()
    calculate_metrics_phase1()
    calculate_metrics_phase2()
