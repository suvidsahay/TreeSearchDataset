from sentence_transformers import CrossEncoder
import json
from tqdm import tqdm
import wikipediaapi


# Load CrossEncoder model
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')

input_path = 'output.txt'           # Your original JSONL
output_path = 'reranked_output.jsonl'  # Output JSONL

wiki_wiki = wikipediaapi.Wikipedia(user_agent="multi_hop_agent", language="en")

def fetch_wikipedia_page(title: str) -> str | None:
    page = wiki_wiki.page(title)
    return page.text if page.exists() else None

# Count total lines first for tqdm
with open(input_path, 'r') as f:
    total_lines = sum(1 for _ in f)

# 🚀 Step 0: Load processed QIDs
processed_qids = set()
try:
    with open(output_path, 'r') as outfile:
        for line in outfile:
            if not line.strip(): continue
            item = json.loads(line)
            processed_qids.add(item.get('qid'))
    print(f"Skipping {len(processed_qids)} already-processed queries.")
except FileNotFoundError:
    pass  # No output file yet; no QIDs to skip

# Reranking loop
with open(input_path, 'r') as infile, open(output_path, 'a') as outfile:
    cache = {}
    for line in tqdm(infile, total=total_lines, desc="Reranking queries"):
        if not line.strip(): continue

        data = json.loads(line)
        qid = data.get('qid')
        if qid in processed_qids:
            print(f"Skipping {qid} already-processed query")
            continue  # skip already done

        query = data['query']
        docs = data.get('docs', [])
        if not docs:
            continue

        # Cross-encoder preparation
        cross_inp = []
        for doc in docs:
            title = doc.get("title", "")
            if title not in cache:
                cache[title] = fetch_wikipedia_page(title) or ""
            wiki_text = cache[title]
            cross_inp.append([query, wiki_text])
            doc["full_wiki_text"] = wiki_text

        # Scoring
        cross_scores = cross_encoder.predict(cross_inp)
        for idx, score in enumerate(cross_scores):
            docs[idx]['cross_score'] = float(score)

        data['docs'] = sorted(docs, key=lambda x: x['cross_score'], reverse=True)
        outfile.write(json.dumps(data) + "\n")
        processed_qids.add(qid)


print("✅ All queries reranked and saved to:", output_path)
