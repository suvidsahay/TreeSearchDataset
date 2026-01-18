import os
import json
import argparse
from tqdm import tqdm
import logging
from data import StrategyQA, WikiMultiHopQA, HotpotQA, IIRC, Fever
from elasticsearch import Elasticsearch
import wikipediaapi

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

wiki_wiki = wikipediaapi.Wikipedia(user_agent="multi_hop_agent", language="en")

def fetch_wikipedia_page(title: str) -> str | None:
    page = wiki_wiki.page(title)
    return page.text if page.exists() else None


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", type=str, required=True)
    args = parser.parse_args()
    config_path = args.config_path
    with open(config_path, "r") as f:
        args = json.load(f)
    args = argparse.Namespace(**args)
    args.config_path = config_path
    if "shuffle" not in args:
        args.shuffle = False
    if "use_counter" not in args:
        args.use_counter = True
    return args


def main():
    args = get_args()
    logger.info(f"{args}")

    # output dir
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)
    dir_name = os.listdir(args.output_dir)
    for i in range(10000):
        if str(i) not in dir_name:
            args.output_dir = os.path.join(args.output_dir, str(i))
            os.makedirs(args.output_dir)
            break
    logger.info(f"output dir: {args.output_dir}")
    # save config
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(args.__dict__, f, indent=4)
    # create output file
    output_file = open(os.path.join(args.output_dir, "output.txt"), "w")

    # load data
    if args.dataset == "strategyqa":
        data = StrategyQA(args.data_path)
    elif args.dataset == "2wikimultihopqa":
        data = WikiMultiHopQA(args.data_path)
    elif args.dataset == "hotpotqa":
        data = HotpotQA(args.data_path)
    elif args.dataset == "iirc":
        data = IIRC(args.data_path)
    elif args.dataset == "fever":
        data = Fever(args.data_path)
    else:
        raise NotImplementedError
    data.format(fewshot=args.fewshot)
    data = data.dataset
    if args.shuffle:
        data = data.shuffle()
    if args.sample != -1:
        samples = min(len(data), args.sample)
        data = data.select(range(samples))

    es = Elasticsearch("http://localhost:9200")

    for example in tqdm(data):
        query = example["question"]
        request = ([
            {"index": args.es_index_name, "search_type": "dfs_query_then_fetch"},
            {
                "_source": True,
                "query": {
                    "multi_match": {
                        "query": query,
                        "type": "best_fields",
                        "fields": ["txt"],
                        "tie_breaker": 0.5
                    }
                },
                "size": args.retrieve_topk
            }
        ])
        resp = es.msearch(body=request)  # as above
        # Parse the response
        docs = []
        for r in resp["responses"]:
            for hit in r["hits"]["hits"]:
                title = hit["_source"].get("title", "")
                # full_text = fetch_wikipedia_page(title) or ""
                docs.append({
                    "doc_id": hit["_id"],
                    "title": title,
                    "score": hit["_score"],
                    "txt": hit["_source"].get("txt", "")
                    # "full_wikipedia_txt": full_text
                })

        ret = {
            "qid": example["qid"],
            "query": query,
            "docs": docs
        }
        output_file.write(json.dumps(ret) + "\n")

    # # Initialize BM25 retriever once (no model)
    # bm25 = BM25(
    #     tokenizer=None,  # optional if BM25Search doesn't need it
    #     index_name=args.es_index_name,
    #     engine='elasticsearch'
    # )
    #
    # # Iterate over data
    # output_file = open(os.path.join(args.output_dir, "output.txt"), "w")
    # for example in tqdm(data):
    #     qid = example["qid"]
    #     query = example["question"]
    #
    #     # Perform BM25 retrieval
    #     docids, docs = bm25.retrieve([query], topk=args.retrieve_topk)
    #
    #     # Prepare JSON response
    #     ret = {
    #         "qid": qid,
    #         "query": query,
    #         "docids": docids.tolist()[0],
    #         "docs": docs.tolist(),
    #     }
    #
    #     output_file.write(json.dumps(ret) + "\n")
    #
    # output_file.close()
    # logger.info("✅ Retrieval done.")


if __name__ == "__main__":
    main()
