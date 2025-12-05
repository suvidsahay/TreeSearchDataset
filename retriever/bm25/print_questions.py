import json
from collections import defaultdict

file_path = "results_iterative.jsonl"

# hop_count -> list of entries
by_hops = defaultdict(list)

with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)

        # if data.get("status") != "success":
        #     continue  # skip failures

        hops = data.get("num_hops", "unknown")
        by_hops[hops].append({
            "question": data.get("final_question"),
            "answer": data.get("answer"),
            "claim": data.get("claim"),
        })

# Pretty print
for hops in sorted(by_hops, key=lambda x: (isinstance(x, str), x)):
    print(f"{hops} Hop Questions:")
    for i, item in enumerate(by_hops[hops], start=1):
        print(f"{item['question']}")
        # print(f"     A: {item['answer']}")
        # print(f"     Claim: {item['claim']}")
    print()
