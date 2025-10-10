# Project Setup and Execution Guide

This guide provides step-by-step instructions to configure the environment, launch necessary services, index data, and run the main project script.

---

## Step 0: Initial Login and Directory Setup
Begin by logging into the designated account and navigating to your primary working directory.

```bash
# Login to the nithyarajkum account...

---

## Step 1: Launch a GPU-Enabled Interactive Job

Request computational resources on a GPU node using the SLURM scheduler.

```bash
srun --partition=superpod-a100 --gres=gpu:1 --mem=40G --cpus-per-task=4 --time=04:00:00 --pty bash
```

---

## Step 2: Start Elasticsearch

This step initializes and runs the Elasticsearch service, which is required for data indexing and retrieval.

1. **Navigate to the Elasticsearch Directory:**

   ```bash
   cd data/elasticsearch-7.17.9/
   ```

2. **Set Java Heap Size:**

   ```bash
   export ES_JAVA_OPTS="-Xms1g -Xmx1g"
   ```

3. **Clear Previous Node Data (For Fresh Setups Only):**
   ⚠️ Only run this command if this is a first-time setup. Running it will delete all indexed data.

   ```bash
   rm -rf data/nodes
   ```

4. **Run Elasticsearch in the Background:**

   ```bash
   nohup bin/elasticsearch > ~/es.log 2>&1 &
   ```

---

## Step 3: Activate Conda Environment

Activate the project's Python environment and return to the root directory.

```bash
# Activate the 'dragin' Conda environment
conda activate dragin

# Navigate back to the project's root directory
cd ../..
```

---

## Step 4: Check Elasticsearch Cluster Health

Verify that the Elasticsearch cluster started successfully. It may take 30–60 seconds to initialize.

```bash
curl -s http://localhost:9200/_cluster/health
```

* A healthy cluster will show a status of `green` or `yellow`.

---

## Step 5: Index the FEVER Dataset

Run the preparation script to load and index the FEVER dataset into Elasticsearch.
This process only needs to be performed once.

```bash
python prep_elastic.py --data_path data/dpr/psgs_w100.tsv --index_name fever
```

---

## Step 6: Final Preparation and Execution

Complete the final setup steps before running the main analysis script.

1. **Navigate to the Source Directory:**

   ```bash
   cd src
   ```

2. **Verify Indices:**

   ```bash
   curl -X GET "localhost:9200/_cat/indices?v"
   ```

3. **Go to the Script Directory:**

   ```bash
   cd retriever/bm25
   ```

4. **Copy Required Data Files:**

   ```bash
   cp /work/pi_hzamani_umass_edu/nithyarajkum_umass_edu/output/filtered_fever_with_wiki_updated.jsonl .
   cp /work/pi_hzamani_umass_edu/nithyarajkum_umass_edu/output/reranked_output_5.jsonl .
   ```

5. **Export OpenAI API Key:**

   ```bash
   export OPENAI_API_KEY="your_api_key_here"
   ```

6. **Run the Main Script with either bm25 retrieval or llm retrieval:**

   ```bash
   python tree_construct.py --retrieval_method bm25
   python tree_construct.py --retrieval_method llm 
   ```

---

