import os
import json
import argparse
from tqdm import tqdm
from copy import copy
import logging
from data import StrategyQA, WikiMultiHopQA, HotpotQA, IIRC, Fever, DPR
from generate import *

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)


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
    elif args.dataset == "dpr":
        data = DPR(args.data_path)
    else:
        raise NotImplementedError
    data.format(fewshot=args.fewshot)
    data = data.dataset
    if args.shuffle:
        data = data.shuffle()
    if args.sample != -1:
        samples = min(len(data), args.sample)
        data = data.select(range(samples))
   
    # 根据 method 选择不同的生成策略
    if args.method == "non-retrieval":
        print("[DEBUG] Instantiating BasicRAG...")
        model = BasicRAG(args)
        print("[DEBUG] BasicRAG instantiated.")
    elif args.method == "single-retrieval":
        model = SingleRAG(args)
    elif args.method == "fix-length-retrieval" or args.method == "fix-sentence-retrieval":
        model = FixLengthRAG(args)
    elif args.method == "token":
        model = TokenRAG(args)
    elif args.method == "entity":
        model = EntityRAG(args)
    elif args.method == "attn_prob" or args.method == "dragin":
        model = AttnWeightRAG(args)
    else:
        raise NotImplementedError
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("start inference")
    
    for i in tqdm(range(len(data))):
        last_counter = copy(model.counter)
        batch = data[i]
        pred = model.inference(batch["question"], batch["demo"], batch["case"])
        #ret["context"] = contexts  # If you want to save this to output.txt as well
        #pred = model.inference(batch["question"], batch["demo"], batch["case"])
        pred = pred.strip()
        ret = {
            "qid": batch["qid"], 
            #"question": batch["question"],
            "prediction": pred,
        }
        #ret["contexts"] = contexts
        if args.use_counter:
            ret.update(model.counter.calc(last_counter))
        output_file.write(json.dumps(ret)+"\n")
        #ret["context"] = contexts
        #output_file.write(f"Question: {ret['question']}\n")
        #output_file.write(f"Answer: {ret['prediction']}\n")
        #if "context" in ret:
        #    output_file.write(f"{ret['context']}\n")
        #output_file.write("Answer in the same format as before.\n\n")
    

if __name__ == "__main__":
    main()
