import json
from retriever import retrieve_best_passage
from question_generation import load_openai_key, generate_questions
from verification import verify_question
from tqdm import tqdm

# Load OpenAI API Key
load_openai_key()

OUTPUT_FILE = "results.jsonl"

# Load 100 FEVER claims
with open('fever_sample.jsonl', 'r') as f:
    fever_data = [json.loads(line) for line in f.readlines()[:100]]

print(f"Processing {len(fever_data)} FEVER entries...\n")

for record in tqdm(fever_data):
    claim = record['claim']
    wiki_titles = record.get('wiki_pages', [])

    if len(wiki_titles) < 2:
        continue

    doc1 = retrieve_best_passage(wiki_titles[0], claim)
    doc2 = retrieve_best_passage(wiki_titles[1], claim)

    if not doc1 or not doc2:
        continue

    # Generate 3 questions
    questions_text = generate_questions(doc1, doc2)
    questions_list = [q.strip() for q in questions_text.split('\n') if q.strip()]

    questions_data = []
    for q in questions_list:
        verification_details = verify_question([doc1, doc2], q)
        print("\nVerification Output:")
        print(verification_details)

        questions_data.append(verification_details)

    # Log the results for this claim
    log_entry = {
        "claim": claim,
        "wiki_pages": wiki_titles,
        "passage_1": doc1,
        "passage_2": doc2,
        "questions": questions_data
    }

    with open(OUTPUT_FILE, 'a') as f_out:
        json.dump(log_entry, f_out)
        f_out.write('\n')

# -----------------------------
# Metrics Calculation Section
# -----------------------------

count_2_passage = 0
count_A_passage = 0
count_B_passage = 0
count_no_passage = 0

with open(OUTPUT_FILE, 'r') as f:
    for line in f:
        entry = json.loads(line)
        for q in entry["questions"]:
            # Priority: If Correct_2_passage is True, ignore others
            if q.get("Correct_2_passage") == True:
                count_2_passage += 1
            elif q.get("Correct_A_passage") == True:
                count_A_passage += 1
            elif q.get("Correct_B_passage") == True:
                count_B_passage += 1
            elif q.get("Correct_no_passage") == True:
                count_no_passage += 1

total_questions = count_2_passage + count_A_passage + count_B_passage + count_no_passage
if total_questions == 0:
    print("\nNo questions were processed! Please check if generation or verification failed.")
    exit()
print("\n--- Verification Metrics ---")
print(f"Total Questions: {total_questions}")
print(f"✅ Needs Both Passages: {count_2_passage} ({(count_2_passage/total_questions)*100:.2f}%)")
print(f"➡️  Only Passage A: {count_A_passage} ({(count_A_passage/total_questions)*100:.2f}%)")
print(f"➡️  Only Passage B: {count_B_passage} ({(count_B_passage/total_questions)*100:.2f}%)")
print(f"❌ General Knowledge (No Passage Needed): {count_no_passage} ({(count_no_passage/total_questions)*100:.2f}%)")

# Multi-hop Effectiveness Score
score = (count_2_passage / total_questions) * 100
print(f"\nMulti-hop Effectiveness Score: {score:.2f}%")
print(f"\nDetailed results saved to {OUTPUT_FILE}")

