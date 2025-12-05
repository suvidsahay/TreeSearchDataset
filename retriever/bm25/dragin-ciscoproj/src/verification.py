import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from retriever import BM25

class Verifier:
    def __init__(self, model_name_or_path, bm25_index, device=None):
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        print(f"[Verifier] Loading model on device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map="auto" if self.device == 'cuda' else {"": self.device}
        )
        self.retriever = BM25(index_name=bm25_index)

    def build_verification_prompt(self, question, answer, passages):
        prompt = f"Verify if the following answer to the question is correct based on the provided context.\n"
        prompt += f"Question: {question}\n"
        prompt += f"Proposed Answer: {answer}\n"
        prompt += "Context:\n"
        for i, passage in enumerate(passages):
            prompt += f"[{i+1}] {passage}\n"
        prompt += "Reasoning and Conclusion:"  # Let model reason
        return prompt

    def generate_reasoning(self, prompt, max_length=200):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=max_length,
            pad_token_id=self.tokenizer.eos_token_id
        )
        generated_tokens = outputs[:, input_ids.shape[1]:]
        reasoning = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        return reasoning

    def score_reasoning(self, reasoning, answer):
        # Simple scoring: check if answer appears in reasoning
        if answer.lower() in reasoning.lower():
            return 1  # Good
        else:
            return 0  # Bad

    def verify(self, question, candidate_answers, topk_passages=5):
        scored_candidates = []

        for answer in candidate_answers:
            verification_query = f"{question} {answer}"
            retrieved_passages = [
            "Joe Biden won the 2020 U.S. Presidential election and took office in January 2021.",
            "Donald Trump was the president before Joe Biden, serving from 2017 to 2021.",
            "The President of the United States is elected every 4 years.",
            "The Vice President during Joe Biden's term is Kamala Harris.",
            "Barack Obama was president from 2009 to 2017."
            ]

            verification_prompt = self.build_verification_prompt(question, answer, retrieved_passages)
            reasoning = self.generate_reasoning(verification_prompt)
            score = self.score_reasoning(reasoning, answer)

            scored_candidates.append((answer, score, reasoning))

        return scored_candidates

if __name__ == "__main__":
    model_path = "distilgpt2"  # For testing
    bm25_index = "wiki_index"

    verifier = Verifier(model_name_or_path=model_path, bm25_index=bm25_index)

    question = "Who was the president of the USA in 2020?"
    candidate_answers = [
        "Joe Biden",
        "Donald Trump",
        "Barack Obama"
    ]

    results = verifier.verify(question, candidate_answers)

    for idx, (answer, score, reasoning) in enumerate(results):
        print(f"Candidate {idx+1}: {answer}")
        print(f"Verification Score: {score}")
        print(f"Reasoning:\n{reasoning}\n")

