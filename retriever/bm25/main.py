import json
from retriever import retrieve_best_passage
from question_generation import load_openai_key, generate_questions
from verification import verify_question
from tqdm import tqdm

# Load OpenAI API Key
load_openai_key()

# Output file to store results
OUTPUT_FILE = "results.jsonl"

# Initialize counters
valid_count = 0
invalid_count = 0
total_questions = 0

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
        verification_result = verify_question([doc1, doc2], q)
        status = "VALID" if "VALID" in verification_result.upper() else "INVALID"
        questions_data.append({"question": q, "verification": status})

        if status == "VALID":
            valid_count += 1
        else:
            invalid_count += 1

        total_questions += 1

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

# Final Summary
print("\n--- Summary Report ---")
print(f"Total Questions Generated: {total_questions}")
print(f"✅ Valid (Needs 2 Passages): {valid_count}")
print(f"❌ Invalid (Needs 1 or 0 Passages): {invalid_count}")
print(f"Results saved to {OUTPUT_FILE}")

