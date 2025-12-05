import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from retriever import BM25  # Assuming you have BM25 implemented like in DRAGIN

class AdaptiveGenerator:
    def __init__(self, model_name_or_path, device=None):
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        print(f"Loading model on device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map="auto" if self.device == 'cuda' else {"": self.device}
        )

    def generate_k_no_retrieval(self, input_text, k, max_length):
        candidates = []
        formatted_input = f"Answer the following question:\n\n{input_text}\n\nAnswer:"
        input_ids = self.tokenizer.encode(formatted_input, return_tensors="pt").to(self.device)

        for _ in range(k):
            outputs = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_length,
                return_dict_in_generate=True,
                output_scores=True
            )
            generated_tokens = outputs.sequences[:, input_ids.shape[1]:]
            text = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

            transition_scores = self.model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)
            logprobs = transition_scores[0].cpu().numpy()
            confidence = np.mean(logprobs)

            candidates.append((text, confidence))
        return candidates

    def generate_with_context(self, prompt, k, max_length):
        candidates = []
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        for _ in range(k):
            outputs = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_length
            )
            generated_tokens = outputs[:, input_ids.shape[1]:]
            text = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
            candidates.append(text)
        return candidates

class AdaptiveRAG:
    def __init__(self, model_name_or_path, bm25_index, threshold= -1.5, k=3, max_length=100, device=None):
        self.generator = AdaptiveGenerator(model_name_or_path, device)
        self.retriever = BM25(index_name=bm25_index)
        self.threshold = threshold
        self.k = k
        self.max_length = max_length

    def build_prompt(self, question, docs):
        prompt = f"Question: {question}\nContext:\n"
        for i, doc in enumerate(docs):
            prompt += f"[{i+1}] {doc}\n"
        prompt += "Answer:"
        return prompt

    def run(self, question):
        no_retrieval_candidates = self.generator.generate_k_no_retrieval(question, self.k, self.max_length)
        best_candidate = max(no_retrieval_candidates, key=lambda x: x[1])

        if best_candidate[1] >= self.threshold:
            print("High confidence from pretrained knowledge.")
            return [c[0] for c in no_retrieval_candidates]
        else:
            print("Low confidence. Triggering retrieval...")
            docs = self.retriever.retrieve(question, topk=5)
            prompt = self.build_prompt(question, docs)
            retrieved_candidates = self.generator.generate_with_context(prompt, self.k, self.max_length)
            return retrieved_candidates

if __name__ == "__main__":
    model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Switched to smaller model for testing
    bm25_index = "wiki_index"

    rag = AdaptiveRAG(model_name_or_path=model_path, bm25_index=bm25_index, threshold=-1.5, k=3, max_length=100)

    question = "Who was the president of the USA in 2020?"
    answers = rag.run(question)

    print("\nGenerated Answers:")
    for idx, ans in enumerate(answers):
        print(f"Answer {idx+1}: {ans}")

    # Next step: Pass 'answers' to your verification pipeline
    from verification import Verifier

    verifier = Verifier(model_name_or_path="distilgpt2", bm25_index="wiki_index")

# Assume you have a question and k candidate answers
    results = verifier.verify(question, candidate_answers)

# Select best answer
    best_answer = max(results, key=lambda x: x[1])[0]  # highest verification score

