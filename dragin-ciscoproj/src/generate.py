import numpy as np
import logging
import spacy
import torch
from math import exp
from scipy.special import softmax
from retriever import BM25, SGPT
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import re # Import re for parsing the decomposition

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

nlp = spacy.load("en_core_web_sm")

class BasicGenerator:
    def __init__(self, model_name_or_path):
        logger.info(f"Loading model from {model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model_config = AutoConfig.from_pretrained(model_name_or_path,
                    trust_remote_code = "falcon" in model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto",
                    trust_remote_code = "falcon" in model_name_or_path)
        if self.model_config.model_type == "llama": #TODO Create the custom configuration
            self.space_token = " "
        else:
            self.space_token = self.tokenizer.tokenize(' ')[0]

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, input_text, max_length, return_logprobs=False):
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        input_ids = input_ids.to(self.model.device)
        input_length = input_ids.shape[1]
        attention_mask = torch.ones_like(input_ids)

        if return_logprobs:
            outputs = self.model.generate(
                input_ids = input_ids,
                attention_mask = attention_mask,
                max_new_tokens = max_length,
                return_dict_in_generate = True,
                output_scores = True,
            )
            transition_scores = self.model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True
            )

            generated_tokens = outputs.sequences[:, input_length:]
            text = self.tokenizer.decode(generated_tokens[0]) # text = "".join(tokens)
            tokens = [self.tokenizer.decode(t) for t in generated_tokens[0]]
            logprobs = transition_scores[0]
            logprobs = [p.cpu().numpy() for p in logprobs]
            assert len(tokens) == len(logprobs)
            return text, tokens, logprobs

        else:
            outputs = self.model.generate(
                input_ids = input_ids,
                max_new_tokens = max_length,
                attention_mask = attention_mask,
            )
            generated_tokens = outputs[:, input_length:]
            text = self.tokenizer.decode(generated_tokens[0])
            return text, None, None

    def generate_attn(self, input_text, max_length, solver="max", use_entropy = False, use_logprob = False):
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        input_ids = input_ids.to(self.model.device)
        input_length = input_ids.shape[1]
        attention_mask = torch.ones_like(input_ids)

        outputs = self.model.generate(
            input_ids = input_ids,
            attention_mask = attention_mask,
            max_new_tokens = max_length,
            return_dict_in_generate = True,
            output_scores = True,
        )
        generated_tokens = outputs.sequences[:, input_length:]
        tokens = self.tokenizer.convert_ids_to_tokens(generated_tokens[0])
        text = self.tokenizer.decode(generated_tokens[0])

        # merge tokens
        range_ = []
        for i, t in enumerate(tokens):
            if i == 0 or t.startswith(self.space_token) or generated_tokens[0][i] == 13 or tokens[i-1] == '</s>':
                range_.append([i, i])
            else:
                range_[-1][-1] += 1

        # attention
        atten = self.model(generated_tokens, output_attentions=True).attentions[-1][0]
        if solver == "max":
            mean_atten, _ = torch.max(atten, dim=1)
            mean_atten = torch.mean(mean_atten, dim=0)
        elif solver == "avg":
            mean_atten = torch.sum(atten, dim=1)
            mean_atten = torch.mean(mean_atten, dim=0)
            for i in range(mean_atten.shape[0]):
                mean_atten[i] /= (mean_atten.shape[0] - i)
        elif solver == "last_token":
            mean_atten = torch.mean(atten[:, -1], dim=0)
        else:
            raise NotImplementedError
        if mean_atten.shape[0] > 1 and tokens[0] == '</s>':
            mean_atten = mean_atten / sum(mean_atten[1:]).item()
        # mean_atten = mean_atten[tl:tr]

        # regular tokens
        seqlist = []
        attns = []
        for r in range_:
            tokenseq = "".join(tokens[r[0]: r[1]+1]).replace(self.space_token, "")
            value = sum(mean_atten[r[0]: r[1]+1]).item()
            seqlist.append(tokenseq)
            attns.append(value)

        # -log prob
        if use_logprob:
            transition_scores = self.model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True
            )
            logprobs = transition_scores[0]
            logprobs = [p.cpu().numpy() for p in logprobs]
            assert len(tokens) == len(logprobs)
            seqlogprobs = []
            for r in range_:
                logprobseq = sum(logprobs[r[0]:r[1]+1]) / (r[1] - r[0] + 1)
                seqlogprobs.append(logprobseq)
        else:
            seqlogprobs = None

        # entropy
        if use_entropy:
            tmp = []
            for v in outputs.scores:
                tmp.append(v.cpu())
            softmax_probs = softmax(tmp, axis=-1)
            entropies = -np.sum(softmax_probs * np.log(softmax_probs + 1e-10), axis=-1)
            entropies = [v[0] for v in entropies]
            seqentropies = []
            for r in range_:
                entropyseq = sum(entropies[r[0]:r[1]+1]) / (r[1] - r[0] + 1)
                seqentropies.append(entropyseq)
        else:
            seqentropies = None

        return text, seqlist, attns, seqlogprobs, seqentropies


class Counter:
    def __init__(self):
        self.retrieve = 0
        self.generate = 0
        self.hallucinated = 0
        self.token = 0
        self.sentence = 0

    def add_generate(self, text, tokenizer):
        self.generate += 1
        ids = tokenizer(text, return_tensors="pt")['input_ids'][0].tolist()
        self.token += len(ids)
        sentences = [sent.text for sent in nlp(text).sents]
        self.sentence += len(sentences)

    def calc(self, other_counter):
        return {
            "retrieve_count": self.retrieve - other_counter.retrieve,
            "generate_count": self.generate - other_counter.generate,
            "hallucinated_count": self.hallucinated - other_counter.hallucinated,
            "token_count": self.token - other_counter.token,
            "sentence_count": self.sentence - other_counter.sentence
        }


class BasicRAG:
    def __init__(self, args):
        args = args.__dict__
        for k, v in args.items():
            setattr(self, k, v)
        self.generator = BasicGenerator(self.model_name_or_path)
        if "retriever" in self.__dict__:
            self.retriever_type = self.retriever
            if self.retriever_type == "BM25":
                # gpt2_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
                self.retriever = BM25(
                    tokenizer = self.generator.tokenizer,
                    index_name = "wiki" if "es_index_name" not in args else self.es_index_name,
                    engine = "elasticsearch",
                )
            elif self.retriever_type == "SGPT":
                self.retriever = SGPT(
                    model_name_or_path = self.sgpt_model_name_or_path,
                    sgpt_encode_file_path = self.sgpt_encode_file_path,
                    passage_file = self.passage_file
                )
            else:
                raise NotImplementedError

        self.counter = Counter()

    def retrieve(self, query, topk=1, max_query_length=64):
        self.counter.retrieve += 1
        if self.retriever_type == "BM25":
            _docs_ids, docs = self.retriever.retrieve(
                queries = [query],
                topk = topk,
                max_query_length = max_query_length,
            )
            return docs[0]
        elif self.retriever_type == "SGPT":
            docs = self.retriever.retrieve(
                queries = [query],
                topk = topk,
            )
            return docs[0]
        else:
            raise NotImplementedError

    def get_top_sentence(self, text):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]
        return sentences[0] if len(sentences) > 0 else ""

    def get_last_sentence(self, text):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]
        return sentences[-1] if len(sentences) > 0 else ""

    def inference(self, question, demo, case):
        # non-retrieval
        assert self.query_formulation == "direct"
        prompt = "".join([d["case"]+"\n" for d in demo])
        prompt += case
        text, _, _ = self.generator.generate(prompt, self.generate_max_length)
        if self.use_counter == True:
            self.counter.add_generate(text, self.generator.tokenizer)
        return text


class SingleRAG(BasicRAG):
    def __init__(self, args):
        super().__init__(args)

    def inference(self, question, demo, case):
        assert self.query_formulation == "direct"
        docs = self.retrieve(question, topk=self.retrieve_topk)
        # 对 topk 个 passage 生成 prompt
        prompt = "".join([d["case"]+"\n" for d in demo])
        prompt += "Context:\n"
        for i, doc in enumerate(docs):
            prompt += f"[{i+1}] {doc}\n"
        prompt += "Answer in the same format as before.\n"
        prompt += case
        text, _, _ = self.generator.generate(prompt, self.generate_max_length)
        if self.use_counter == True:
            self.counter.add_generate(text, self.generator.tokenizer)
        return text


class FixLengthRAG(BasicRAG):
    def __init__(self, args):
        super().__init__(args)

    def inference(self, question, demo, case):
        assert self.query_formulation == "direct"
        text = ""
        retrieve_question = question
        while True:
            old_len = len(text)
            docs = self.retrieve(retrieve_question, topk=self.retrieve_topk)
            prompt = "".join([d["case"]+"\n" for d in demo])
            prompt += "Context:\n"
            for i, doc in enumerate(docs):
                prompt += f"[{i+1}] {doc}\n"
            prompt += "Answer in t he same format as before.\n"
            prompt += case + " " + text
            if self.method == "fix-length-retrieval":
                new_text, _, _ = self.generator.generate(prompt, self.fix_length)
                if self.use_counter == True:
                    self.counter.add_generate(new_text, self.generator.tokenizer)
                text = text.strip() + " " + new_text.strip()
                retrieve_question = new_text.strip()
            else:
                # fix sentence
                new_text, _, _ = self.generator.generate(prompt, self.generate_max_length)
                if self.use_counter == True:
                    self.counter.add_generate(new_text, self.generator.tokenizer)
                new_text = new_text.strip()
                sentences = list(nlp(new_text).sents)
                sentences = [str(sent).strip() for sent in sentences]
                if len(sentences) == 0:
                    break
                text = text.strip() + " " + str(sentences[0])
                retrieve_question = sentences[0]

            # 判断 token 的个数要少于 generate_max_length
            tokens_count = len(self.generator.tokenizer.encode(text))
            if tokens_count > self.generate_max_length or len(text) <= old_len or "the answer is" in text:
                break
        return text


class TokenRAG(BasicRAG):
    def __init__(self, args):
        super().__init__(args)

    def modifier(self, text, tokens, logprobs):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]

        tid = 0
        for sid, sent in enumerate(sentences):
            pos = 0
            tr = tid
            while tr < len(tokens):
                apr = sent[pos:].find(tokens[tr])
                if apr == -1:
                    break
                pos = apr + len(tokens[tr])
                tr += 1
            probs = [1 - exp(v) for v in logprobs[tid:tr+1]]
            probs = np.array(probs)
            p = {
                "avg": np.mean,
                "max": np.max,
                "min": np.min,
            }.get(self.sentence_solver, lambda x: 0)(probs)
            if p > self.hallucination_threshold: # hallucination
                # keep sentences before hallucination
                prev = "" if sid == 0 else " ".join(sentences[:sid])
                # replace all hallucinated tokens in current sentence with [xxx]
                curr = sentences[sid]
                pos = 0
                # # 这里改成了替换掉最大的那个，而不是所有的
                # max_prob = 0
                # for prob, tok in zip(probs, tokens[tid:tr+1]):
                #     max_prob = max(prob, max_prob)
                for prob, tok in zip(probs, tokens[tid:tr+1]):
                    apr = curr[pos:].find(tok) + pos
                    if prob > self.hallucination_threshold:
                    # if prob == max_prob:
                        curr = curr[:apr] + "[xxx]" + curr[apr+len(tok):]
                        pos = apr + len("[xxx]")
                    else:
                        pos = apr + len(tok)
                return prev, curr, True
            tid = tr + 1

        # No hallucination
        return text, None, False

    def inference(self, question, demo, case):
        # assert self.query_formulation == "direct"
        text = ""
        while True:
            old_len = len(text)
            prompt = "".join([d["case"]+"\n" for d in demo])
            prompt += case + " " + text
            new_text, tokens, logprobs = self.generator.generate(
                prompt,
                self.generate_max_length,
                return_logprobs=True
            )
            if self.use_counter == True:
                self.counter.add_generate(new_text, self.generator.tokenizer)
            ptext, curr, hallucination = self.modifier(new_text, tokens, logprobs)
            if not hallucination:
                text = text.strip() + " " + new_text.strip()
            else:
                if self.query_formulation == "direct":
                    retrieve_question = curr.replace("[xxx]", "")
                elif self.query_formulation == "forward_all":
                    tmp_all = [question, text, ptext]
                    retrieve_question = " ".join(s for s in tmp_all if len(s) > 0)
                else:
                    raise NotImplemented

                docs = self.retrieve(retrieve_question, topk=self.retrieve_topk)
                prompt = "".join([d["case"]+"\n" for d in demo])
                prompt += "Context:\n"
                for i, doc in enumerate(docs):
                    prompt += f"[{i+1}] {doc}\n"
                prompt += "Answer in the same format as before.\n"
                prompt += case + " " + text + " " + ptext.strip()
                new_text, _, _ = self.generator.generate(prompt, self.generate_max_length)
                if self.use_counter == True:
                    self.counter.add_generate(new_text, self.generator.tokenizer)
                    self.counter.hallucinated += 1
                text = text.strip() + " " + ptext.strip() + " " + new_text.strip()

            # 判断 token 的个数要少于 generate_max_length
            tokens_count = len(self.generator.tokenizer.encode(text))
            if tokens_count > self.generate_max_length or len(text) <= old_len or "the answer is" in text:
                break
        return text


class EntityRAG(TokenRAG):
    def __init__(self, args):
        super().__init__(args)

    def modifier(self, text, tokens, logprobs):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]

        entity = []
        for sent in sentences:
            doc = nlp(sent)
            li = [ent.text for ent in doc.ents]
            entity.append(li)

        belonging = [-1] * len(text)
        pos = 0
        for tid, tok in enumerate(tokens):
            apr = text[pos:].find(tok) + pos
            assert apr != -1
            for j in range(pos, apr+len(tok)):
                belonging[j] = tid
            pos = apr + len(tok)

        entity_intv = []
        for sid, sent in enumerate(sentences):
            tmp = []
            pos = text.find(sent)
            for ent in entity[sid]:
                apr = text[pos:].find(ent) + pos
                el = belonging[apr]
                er = belonging[apr + len(ent) - 1]
                tmp.append((el, er))
                pos = apr + len(ent)
            entity_intv.append(tmp)

        entity_prob = []
        for ent_itv_per_sent in entity_intv:
            tmp = []
            for itv in ent_itv_per_sent:
                probs = np.array(logprobs[itv[0]:itv[1]+1])
                p = {
                    "avg": np.mean,
                    "max": np.max,
                    "min": np.min,
                    "first": lambda x: x[0] if len(x) > 0 else 0
                }.get(self.entity_solver, lambda x: 0)(probs)
                tmp.append(p)
            entity_prob.append(tmp)

        for sid in range(len(sentences)):
            if len(entity_prob[sid]) == 0:
                continue
            probs = [1 - exp(v) for v in entity_prob[sid]]
            probs = np.array(probs)
            p = {
                "avg": np.mean,
                "max": np.max,
                "min": np.min,
            }.get(self.sentence_solver, lambda x: 0)(probs)
            if p > self.hallucination_threshold: # hallucination
                # keep sentences before hallucination
                prev = "" if sid == 0 else " ".join(sentences[:sid])
                # replace all hallucinated entities in current sentence with [xxx]
                curr = sentences[sid]
                pos = 0
                for prob, ent in zip(probs, entity[sid]):
                    apr = curr[pos:].find(ent) + pos
                    if prob > self.hallucination_threshold:
                        curr = curr[:apr] + "[xxx]" + curr[apr+len(ent):]
                        pos = apr + len("[xxx]")
                    else:
                        pos = apr + len(ent)
                return prev, curr, True
        # No hallucination
        return text, None, False

    def inference(self, question, demo, case):
        return super().inference(question, demo, case)


class AttnWeightRAG(BasicRAG):
    def __init__(self, args):
        super().__init__(args)

    def modifier(self, text, tokens, attentions, weight):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]
        tid = 0
        for sid, sent in enumerate(sentences):
            tl, tr = tid, tid
            if sid == len(sentences) - 1:
                tl, tr = tid, len(tokens)
            else:
                for i in range(tid + 1, len(tokens)):
                    seq = " ".join(tokens[tl:i])
                    if sent in seq:
                        tr = i
                        break
                tid = tr
            # value = attenion * (-log prob)
            attns = attentions[tl:tr]
            attns = np.array(attns) / sum(attns)
            value = [attns[i-tl] * weight[i] * (tr-tl) for i in range(tl, tr)]
            thres = [1 if v > self.hallucination_threshold else 0 for v in value]
            if 1 in thres:
                # hallucinated
                if "check_real_words" in self.__dict__ and self.check_real_words:
                    doc = nlp(sent)
                    real_words = set(token.text for token in doc if token.pos_ in
                        ['NOUN', 'ADJ', 'VERB', 'PROPN', 'NUM'])
                    def match(tok):
                        for word in real_words:
                            if word in tok:
                                return True
                        return False
                    for i in range(len(thres)):
                        if not match(tokens[tl+i]):
                            thres[i] = 0

                prev = "" if sid == 0 else " ".join(sentences[:sid])
                # curr = " ".join(
                #     [tokens[i] if thres[i] == 0 else "[xxx]" for i in range(len(thres))]
                # )
                return True, prev, tokens[tl:tr], thres
        return False, text, None, None

    def keep_real_words(self, prev_text, curr_tokens, curr_hit):
        curr_text = " ".join(curr_tokens)
        all_text = prev_text + " " + curr_text
        input_ids = self.generator.tokenizer.encode(all_text, return_tensors="pt")
        input_length = input_ids.shape[1]
        tokens_tmp = self.generator.tokenizer.convert_ids_to_tokens(input_ids[0])

        atten_tmp = self.generator.model(input_ids, output_attentions=True).attentions[-1][0]

        # merge tokens
        range_ = []
        for i, t in enumerate(tokens_tmp):
            if i == 0 or t.startswith(self.generator.space_token) or input_ids[0][i] == 13:
                range_.append([i, i])
            else:
                range_[-1][-1] += 1
        tokens = []
        for r in range_:
            tokenseq = "".join(tokens_tmp[r[0]: r[1]+1]).replace(self.generator.space_token, "")
            tokens.append(tokenseq)

        # 获取幻觉词对应的 attention
        curr_st = len(tokens) - len(curr_tokens)
        atten_tmp = torch.mean(atten_tmp, dim=0)
        attns = []
        for r in range_:
            # att = torch.zeros(atten_tmp.shape[0], input_length)
            att = torch.zeros(input_length)
            for i in range(r[0], r[1] + 1):
                if i == 0:
                    continue
                v = atten_tmp[i-1][:r[0]] # 上一位的
                v = v / v.sum()
                t = torch.zeros(input_length)
                t[:r[0]] = v
                att += t
            att /= (r[1] - r[0] + 1)
            # merge token for att
            att = torch.tensor([att[rr[0]:rr[1]+1].sum() for rr in range_])
            attns.append(att)

        # 计算每个超过阈值的 token 在前文的 attentions
        forward_attns = torch.zeros(len(tokens))
        hit_cnt = 0
        for i in range(len(curr_hit)):
            if curr_hit[i] == 1:
                forward_attns += attns[curr_st + i]
                hit_cnt += 1
        forward_attns /= hit_cnt
        forward_attns = forward_attns.tolist()

        # 分析词性，保留实词对应的 attns
        doc = nlp(all_text)
        real_words = set(token.text for token in doc if token.pos_ in
                      ['NOUN', 'ADJ', 'VERB', 'PROPN', 'NUM'])

        def match(token):
            for word in real_words:
                if word in token:
                    return True
            return False

        real_pairs = []
        for i in range(len(tokens)):
            tok, att = tokens[i], forward_attns[i]
            if i >= curr_st and curr_hit[i - curr_st]:
                continue
            if match(tok):
                real_pairs.append((att, tok, i))

        if "retrieve_keep_top_k" in self.__dict__:
            top_k = min(self.retrieve_keep_top_k, len(real_pairs))
        elif "retrieve_keep_ratio" in self.__dict__:
            top_k = int(len(real_pairs) * self.retrieve_keep_ratio)

        real_pairs = sorted(real_pairs, key = lambda x:x[0], reverse=True)
        real_pairs = real_pairs[:top_k]
        real_pairs = sorted(real_pairs, key = lambda x:x[2])
        return " ".join([x[1] for x in real_pairs])

    # New method to decompose the question into sub-queries used by inference
    def decompose_question(self, question):
        """
        Uses an LLM to break down a complex question into simpler sub-questions or key concepts.
        """
        # Define the prompt for question decomposition
        DECOMPOSE_PROMPT = f"""You are an expert at analyzing complex questions and breaking them down into a concise set of independent, non-redundant sub-questions or essential information needs. Your goal is to identify the key components required to fully answer the original question.

        Your task is to decompose the following complex question into a list of sub-questions or key information needs.

        Instructions:
        - Generate a list containing a maximum of 5 items. For simpler questions, fewer items are acceptable if they cover the necessary aspects.
        - Each item in the list must be a distinct sub-question or a crucial piece of information required to answer the original question.
        - Avoid any overlap, redundancy, or duplication between the items in the list.
        - Present the decomposition as a numbered list.

        QUESTION: {question}

        DECOMPOSITION:
        """

        logger.info(f"### Decomposing question ###: {question} \n")
        # Use the generator to get the LLM's decomposition
        # Adjust max_length as needed for the decomposition output
        decomposition_text, _, _ = self.generator.generate(DECOMPOSE_PROMPT, max_length=200)
        logger.info(f"### Decomposition output ###: {decomposition_text} \n")

        # Parse the decomposition text into a list of queries
        decomposed_queries = []
        # Split by lines and extract points starting with hyphen or number
        for line in decomposition_text.split('\\n'):
            line = line.strip()
            if line and not line.startswith("QUESTION:") and not line.startswith("DECOMPOSITION:"):
                 # Use regex to find lines starting with hyphen or number followed by a space
                 match = re.match(r'^[\-\d]+\.?\s*(.*)', line)
                 if match:
                     query = match.group(1).strip()
                     if query: # Ensure the extracted query is not empty
                         decomposed_queries.append(query)
                 elif line: # Add lines that don't match the expected format but are not empty or prompt remnants
                      # You might want to refine this to only include valid looking sub-queries
                      decomposed_queries.append(line)

        # If decomposition failed or returned nothing, use the original question as a fallback
        if not decomposed_queries:
            logger.warning("-------^^^ Question decomposition failed or returned empty. Using original question for retrieval. ^^^-------")
            decomposed_queries.append(question)

        return decomposed_queries


    def inference(self, question, demo, case):

        if hasattr(self, "use_decomposition") and not self.use_decomposition:
        # ===== ABLATION MODE: No decomposition =====
            decomposed_queries = [question]
        else:
        # ===== Original DRAGIN Mode =====
            decomposed_queries = self.decompose_question(question)

        # 2. Retrieve documents for each decomposed query
        all_retrieved_docs = []
        # You might want to adjust topk for retrieval based on decomposed queries
        retrieval_topk_per_query = max(1, self.retrieve_topk // len(decomposed_queries)) # Simple distribution
        if retrieval_topk_per_query == 0: retrieval_topk_per_query = 1 # Ensure at least 1 doc per query if topk is very small

        for query in decomposed_queries:
            # Retrieve documents for each decomposed query part
            docs = self.retrieve(query, topk=retrieval_topk_per_query)
            all_retrieved_docs.extend(docs)

        # Remove duplicate documents if necessary (based on content) - not strictly needed for prompt but can clean up
        all_retrieved_docs = list(dict.fromkeys(all_retrieved_docs))

        text = "" # This will build the generated answer iteratively
        # Keep track of context, starting with the documents retrieved from decomposition
        current_context_docs = all_retrieved_docs

        # 3. Start the iterative generation loop, using the initially retrieved context
        while True:
            old_len = len(text)

            # Formulate prompt with current generated text and accumulated context
            prompt = "".join([d["case"]+"\n" for d in demo]) # Few-shot examples
            prompt += "Context:\n"
            # Add all accumulated context documents to the prompt
            for i, doc in enumerate(current_context_docs):
                 prompt += f"[{i+1}] {doc}\n"

            prompt += "Answer in the same format as before.\n"
            # Append the case (the current question formatting) and the text generated so far
            tmp_li = [case, text]
            prompt += " ".join(s for s in tmp_li if len(s) > 0)

            print(prompt)
            # Generate the next part of the answer
            new_text, tokens, attns, logprobs, entropies = self.generator.generate_attn(
                prompt,
                self.generate_max_length, # Maximum tokens for this generation step
                use_entropy = self.method == "dragin",
                use_logprob = self.method == "attn_prob"
            )

            if self.use_counter == True:
                self.counter.add_generate(new_text, self.generator.tokenizer)

            # Determine weight for hallucination detection
            weight = entropies if self.method == "dragin" else [-v for v in logprobs]

            # Check for hallucination in the newly generated text
            hallucination, ptext, curr_tokens, curr_hit =  self.modifier(new_text, tokens, attns, weight)

            if not hallucination:
                # If no hallucination, append the new text to the current answer
                text = text.strip() + " " + new_text.strip()
                logger.info(f"Generated (no hallucination): {new_text.strip()}")

                # Check stopping conditions
                tokens_count = len(self.generator.tokenizer.encode(text))
                # Stop if max length reached, text didn't change much, or answer phrase is included
                if tokens_count > self.generate_max_length or len(text) <= old_len or "the answer is" in text.lower():
                     logger.info("Stopping condition met (no hallucination).")
                     break

            else:
                # If hallucination detected
                if self.use_counter == True:
                     self.counter.hallucinated += 1
                logger.warning(f"Hallucination detected. Hallucinated tokens: {' '.join(curr_tokens)}")
                logger.warning(f"Non-hallucinated prefix: {ptext.strip()}")

                # Formulate retrieval query based on the hallucinated part
                # This helps retrieve relevant context to correct the hallucination
                retrieve_question = " ".join(curr_tokens) # Query is the hallucinated part

                # Retrieve additional documents based on the hallucinated part
                # You might want a smaller topk for this targeted retrieval
                newly_retrieved_docs = self.retrieve(retrieve_question, topk=max(1, self.retrieve_topk // 2)) # Example: half of initial topk

                # Add newly retrieved docs to the overall context for the next iteration
                current_context_docs.extend(newly_retrieved_docs)
                # Optional: Remove duplicates from current_context_docs again

                # Append the non-hallucinated part (ptext) to the current text
                text = text.strip() + " " + ptext.strip()
                logger.info(f"Appended non-hallucinated prefix: {ptext.strip()}")

                # Check stopping conditions after updating text (with the non-hallucinated part)
                tokens_count = len(self.generator.tokenizer.encode(text))
                if tokens_count > self.generate_max_length or len(text) <= old_len or "the answer is" in text.lower():
                     logger.info("Stopping condition met (after hallucination handling).")
                     break

        return text.strip() # Return the final generated text
