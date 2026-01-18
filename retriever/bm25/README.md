## Project Setup and Execution Guide

This guide provides step-by-step instructions to configure the environment, launch necessary services, index data, and run the main project script.

---

## Repository Structure

**Root directory (important):** All commands below assume you are inside this directory unless stated otherwise.

```
TreeSearchDataset/retriever/bm25/
```

---

### 1. Login to Unity

```bash
ssh -v -i ~/.ssh/<unity-privkey>.key <username>@unity.rc.umass.edu
```

---

### 2. Allocate a GPU Node

```bash
srun \
  --partition=gpu-preempt \
  --gres=gpu:1 \
  --mem=40G \
  --cpus-per-task=4 \
  --time=04:00:00 \
  --pty bash
```

---

### 3. Start Elasticsearch (Required)

```bash
cd data/elasticsearch-7.17.9/

export ES_JAVA_OPTS="-Xms1g -Xmx1g"

nohup bin/elasticsearch > ~/es.log 2>&1 &
```

Verify Elasticsearch is running:

```bash
tail ~/es.log

curl localhost:9200/_cluster/health?pretty
```

You should see `"status" : "green"` or `"yellow"`.

---

### 4. Activate Python Environment

```bash
cd ./TreeSearchDataset/retriever/bm25

source <path to your virtual env>
```

### 5. (Optional) Index the FEVER Dataset
Don't do this step if you are just running the pipeline. Skip to **Project Execution**

Run the preparation script only to load and index the FEVER dataset into Elasticsearch. This process only needs to be performed once.

```bash
python prep_elastic.py --data_path data/dpr/psgs_w100.tsv --index_name fever
```

---

## Project Execution

If you are using open-source models with vllm-backend for `Llama-3.1-8B-Instruct` or `google/gemma-3-4b-it`: follow Steps 5, 6, and then 7

If you are using direct API calls to GPT/Claude models: run Step 7

### 5. Start vLLM Server (LLM Backend)

Choose the model server in **one terminal**:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model google/gemma-3-4b-it \
  --port 8001 \
  --dtype auto \
  --max-model-len 2048
```

```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --port 8002 \
  --dtype auto \
  --max-model-len 2048
```

This exposes an OpenAI-compatible API locally.

---

### 6. Run the Pipeline (New Terminal, Same Job)

Open a **second terminal in the same `srun` session**.

```bash
export VLLM_BASE_URL="http://localhost:8001/v1"

export VLLM_BASE_URL="http://localhost:8002/v1"
```

### 7. Run the pipeline 

```bash
python tree_construct.py
```

---

### Free Disk Space (Optional but Recommended)

Unity home directories can fill up quickly.

```bash
ssh unity.rc.umass.edu 'rm -rf ~/.vscode-server ~/.vscode-server-insiders'
```

---