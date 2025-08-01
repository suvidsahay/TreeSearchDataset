import json
import re
from retriever import retrieve_best_passage
from question_generation import load_openai_key, generate_questions
from verification3 import verify_question_v3, evaluate_question_naturalness # <-- Use the new verification file
from tqdm import tqdm
from langchain_openai import ChatOpenAI
import os

# Load OpenAI API Key
load_openai_key()

OUTPUT_FILE = "results.jsonl"

# --- Clear the output file before starting a new run ---
print(f"Clearing old results from {OUTPUT_FILE}...")
with open(OUTPUT_FILE, 'w') as f:
    pass # This empties the file

# --- Load Data ---
with open('filtered_fever_with_wiki_updated.jsonl', 'r') as f:
    fever_data = [json.loads(line) for line in f.readlines()[:5]] # Using 5 for testing

print(f"Processing {len(fever_data)} FEVER entries...\n")

# Initialize the chat model for evaluation once
chat_for_eval = ChatOpenAI(temperature=0, model_name="gpt-4o", openai_api_key=os.getenv("OPENAI_API_KEY"))

for record in tqdm(fever_data):
    claim = record['claim']
    wiki_titles = record.get('wiki_urls', record.get('wiki_pages', []))

    if len(wiki_titles) < 2:
        continue

    # --- Retrieve Passages ---
    doc1 = retrieve_best_passage(wiki_titles[0], claim, method='bm25')
    doc2 = retrieve_best_passage(wiki_titles[1], claim, method='bm25')

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
            
            if not q: continue # Skip if the question is empty after parsing

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
        "questions": questions_data # This now contains the full record
    }
    with open(OUTPUT_FILE, 'a') as f_out:
        json.dump(log_entry, f_out)
        f_out.write('\n')


# --- Metrics Calculation Section ---
count_2_passage = 0
count_A_passage = 0
count_B_passage = 0
count_no_passage = 0
count_error = 0
total_questions = 0
naturalness_keys = [
    "clear_single_question_score",
    "combines_passages_score",
    "requires_both_score",
    "logical_dependency_score",
    "hotpot_style_score",
    "objectivity_score"
]

naturalness_totals = {k: 0 for k in naturalness_keys}
naturalness_counts = {k: 0 for k in naturalness_keys}
total_objectivity_score = 0
scored_objectivity_questions = 0

with open(OUTPUT_FILE, 'r') as f:
    for line in f:
        entry = json.loads(line)
        for q in entry["questions"]:
            total_questions += 1
            for key in naturalness_keys:
                if key in q and q[key] is not None:
            	    naturalness_totals[key] += q[key]
            	    naturalness_counts[key] += 1
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
           # if q.get("naturalness_score") is not None:
            #    total_naturalness_score += q["naturalness_score"]
             #   scored_naturalness_questions += 1

            # Calculate average objectivity score
            if q.get("objectivity_score") is not None:
                total_objectivity_score += q["objectivity_score"]
                scored_objectivity_questions += 1

if total_questions == 0:
    print("\nNo questions were processed!")
    exit()

#average_naturalness = total_naturalness_score / scored_naturalness_questions if scored_naturalness_questions > 0 else 0
average_objectivity = total_objectivity_score / scored_objectivity_questions if scored_objectivity_questions > 0 else 0

print("\n\n--- FINAL METRICS ---")
print(f"Total Questions Processed: {total_questions}")
print(f"✅ Needs Both Passages: {count_2_passage} ({(count_2_passage/total_questions)*100:.2f}%)")
print(f"➡️  Only Passage A: {count_A_passage} ({(count_A_passage/total_questions)*100:.2f}%)")
print(f"➡️  Only Passage B: {count_B_passage} ({(count_B_passage/total_questions)*100:.2f}%)")
print(f"❌ Not Answerable / General Knowledge: {count_no_passage} ({(count_no_passage/total_questions)*100:.2f}%)")
print(f"⚠️ Errors: {count_error} ({(count_error/total_questions)*100:.2f}%)")
print("-" * 25)
#print(f"🌿 Average Naturalness Score: {average_naturalness:.2f} / 5.0")
print(f"🎯 Average Objectivity Score: {average_objectivity:.2f} / 5.0")
print("\n🌿 Average Naturalness Scores by Dimension:")
for key in naturalness_keys:
    count = naturalness_counts[key]
    if count > 0:
        avg = naturalness_totals[key] / count
        print(f"- {key}: {avg:.2f} / 5.0")
    else:
        print(f"- {key}: No valid scores")