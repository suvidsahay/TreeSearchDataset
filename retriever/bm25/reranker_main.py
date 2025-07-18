import json
import re
from retriever import retrieve_best_passage, fetch_wikipedia_page
from question_generation import load_openai_key, generate_questions
from verification3 import verify_question_v3  # <-- Use the new verification file
from verification2 import evaluate_question_naturalness  # Keep for naturalness scoring
from tqdm import tqdm
from langchain_openai import ChatOpenAI
import os
from sentence_transformers import CrossEncoder
from itertools import islice

# Load OpenAI API Key
load_openai_key()

OUTPUT_FILE = "results.jsonl"

# --- Clear the output file before starting a new run ---
print(f"Clearing old results from {OUTPUT_FILE}...")
with open(OUTPUT_FILE, 'w') as f:
    pass  # This empties the file

cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')

# --- Load Data ---

FILE1 = "filtered_fever_with_wiki_updated.jsonl"
FILE2 = "reranked_output_5.jsonl"
OUTPUT = "merged_claims_wiki.jsonl"

# Step 1: load claims+URLs from file1
fever_data = {}
with open(FILE1, 'r') as f1:
    for line in f1:
        if not line.strip():
            continue
        rec = json.loads(line)
        claim = rec.get("claim")
        urls = rec.get("wiki_urls", [])
        urls = [title.replace("_", " ") for title in urls]

        if not claim:
            continue
        fever_data[claim] = {
            "claim": claim,
            "wiki_urls": list(dict.fromkeys(urls))
        }

# Step 2: read file2, append titles to wiki_urls under the same claim/query
with open(FILE2, 'r') as f2:
    for line in f2:
        if not line.strip():
            continue
        rec = json.loads(line)
        query = rec.get("query")
        docs = rec.get("docs", [])
        if not query or query not in fever_data:
            continue
        existing = set(fever_data[query]["wiki_urls"])
        for d in docs:
            title = d.get("title").replace("_", " ")
            if title and title not in existing:
                fever_data[query]["wiki_urls"].append(title)
                existing.add(title)

fever_data = dict(islice(fever_data.items(), 20))

print(f"Processing {len(fever_data)} FEVER entries...\n")


# Initialize the chat model for evaluation once
chat_for_eval = ChatOpenAI(temperature=0, model_name="gpt-4o", openai_api_key=os.getenv("OPENAI_API_KEY"))

for record in tqdm(fever_data.values()):
    claim = record['claim']
    wiki_titles = record.get('wiki_urls', record.get('wiki_pages', []))

    if len(wiki_titles) < 2:
        continue

        # --- Step 1: Fetch wiki texts ---
    cache = {}
    cand_pairs = []
    for title in wiki_titles:
        if title in cache:
            text = cache[title]
        else:
            text = fetch_wikipedia_page(title) or ""
            cache[title] = text
        if text:
            cand_pairs.append((title, text))

    if len(cand_pairs) < 2:
        continue

    # --- Step 2: Rerank with cross-encoder ---
    cross_inp = [[claim, text] for _, text in cand_pairs]
    scores = cross_encoder.predict(cross_inp)

    # Attach scores and sort
    ranked = sorted(
        zip(cand_pairs, scores),
        key=lambda x: x[1],
        reverse=True
    )
    print("🏆 Top-ranked documents and scores:")
    for (title, _), score in ranked:
        print(f" - {title!r}: {score:.4f}")

    # --- Step 3: Pick top-2 unique titles ---
    selected = []
    seen_titles = set()
    for (title, text), score in ranked:
        if title not in seen_titles:
            seen_titles.add(title)
            selected.append((title, text))
        if len(selected) == 2:
            break

    if len(selected) < 2:
        continue

    # --- Step 4: Retrieve best passages via BM25 ---
    doc1 = retrieve_best_passage(selected[0][0], claim, method='cross-encoder')
    doc2 = retrieve_best_passage(selected[1][0], claim, method='cross-encoder')

    if not doc1 or not doc2:
        continue

    # --- Generate Questions, Explanations, and Ground Truth Answers ---
    generated_text = generate_questions(doc1, doc2)

    # New, robust parsing logic using the "---" separator
    blocks = generated_text.strip().split('---')

    questions_data = []
    for block in blocks:
        if not block.strip():
            continue

        # Extract Q, E, and A from each block
        try:
            q = re.search(r"Question:\s*(.*)", block).group(1).strip()
            e = re.search(r"Explanation:\s*(.*)", block).group(1).strip()
            a = re.search(r"Answer:\s*(.*)", block, re.DOTALL).group(1).strip()

            if not q: continue  # Skip if the question is empty after parsing

            print(f"\nProcessing Generated Question: {q}")

            # --- Verification and Evaluation ---
            # Step 1: Use the new verification method
            verification_details = verify_question_v3([doc1, doc2], q, a)

            # Step 2: Evaluate naturalness
            naturalness_details = evaluate_question_naturalness(q, chat_for_eval)

            # Combine all data for this question into one comprehensive record
            combined_details = {
                "question": q,
                "explanation": e,
                "ground_truth_answer": a,
                **verification_details,
                **naturalness_details
            }
            questions_data.append(combined_details)
            print("--- Verification & Evaluation Complete for this question ---")

        except AttributeError:
            # This handles cases where a block doesn't contain Q, E, and A
            print(f"Skipping malformed block:\n{block}")
            continue

    # --- Log Everything for this Claim ---
    log_entry = {
        "claim": claim,
        "passage_1": doc1,
        "passage_2": doc2,
        "questions": questions_data  # This now contains the full record
    }
    with open(OUTPUT_FILE, 'a') as f_out:
        json.dump(log_entry, f_out)
        f_out.write('\n')

# --- Metrics Calculation Section (will now be accurate) ---
count_2_passage = 0
count_A_passage = 0
count_B_passage = 0
count_no_passage = 0
count_error = 0
total_questions = 0
total_naturalness_score = 0
scored_questions = 0

with open(OUTPUT_FILE, 'r') as f:
    for line in f:
        entry = json.loads(line)
        for q in entry["questions"]:
            total_questions += 1
            if q.get("Correct_2_passage") is True:
                count_2_passage += 1
            elif q.get("Correct_A_passage") is True:
                count_A_passage += 1
            elif q.get("Correct_B_passage") is True:
                count_B_passage += 1
            elif q.get("Correct_no_passage") is True:
                count_no_passage += 1
            else:
                count_error += 1

            # Calculate average naturalness score
            if q.get("naturalness_score") is not None:
                total_naturalness_score += q["naturalness_score"]
                scored_questions += 1

if total_questions == 0:
    print("\nNo questions were processed!")
    exit()

average_naturalness = total_naturalness_score / scored_questions if scored_questions > 0 else 0

print("\n\n--- FINAL METRICS ---")
print(f"Total Questions Processed: {total_questions}")
print(f"✅ Needs Both Passages: {count_2_passage} ({(count_2_passage / total_questions) * 100:.2f}%)")
print(f"➡️  Only Passage A: {count_A_passage} ({(count_A_passage / total_questions) * 100:.2f}%)")
print(f"➡️  Only Passage B: {count_B_passage} ({(count_B_passage / total_questions) * 100:.2f}%)")
print(f"❌ Not Answerable / General Knowledge: {count_no_passage} ({(count_no_passage / total_questions) * 100:.2f}%)")
print(f"⚠️ Errors: {count_error} ({(count_error / total_questions) * 100:.2f}%)")
print(f"🌿 Average Naturalness Score: {average_naturalness:.2f} / 5.0")
