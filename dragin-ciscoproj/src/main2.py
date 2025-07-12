from adaptive_rag import AdaptiveRAG
from verification import Verifier

model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"   # or your bigger model
bm25_index = "wiki_index"

rag = AdaptiveRAG(model_name_or_path=model_path, bm25_index=bm25_index, threshold=-1.5, k=3, max_length=100)
verifier = Verifier(model_name_or_path=model_path, bm25_index=bm25_index)

question = "Who was the president of the USA in 2020?"

# STEP 1: Generate candidate answers
candidate_answers = rag.run(question)

# STEP 2: Verify candidate answers
results = verifier.verify(question, candidate_answers)

# STEP 3: Print results
for idx, (answer, score, reasoning) in enumerate(results):
    print(f"Candidate {idx+1}: {answer}")
    print(f"Verification Score: {score}")
    print(f"Reasoning:\\n{reasoning}\\n")

