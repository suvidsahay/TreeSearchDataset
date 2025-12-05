Here’s a clean, structured `README.md` draft for your project based on the directory and instructions you provided:

---

# DRAGIN: Adaptive Retrieval-Augmented Generation with Inference

This repository contains the implementation of **DRAGIN**, an adaptive retrieval-augmented generation framework for fact-checking and verification tasks. The project integrates Elasticsearch, BM25, cross-encoder reranking, and adaptive retrieval strategies.

---

## 🚀 Setup Instructions

### 0. Login into nithyarajkum account and change the directory to

```bash
cd /work/pi_hzamani_umass_edu/nithyarajkum_umass_edu/
```

### 1. Launch a GPU-enabled interactive job (for SLURM users)

```bash
srun --partition=superpod-a100 --gres=gpu:1 --mem=40G --cpus-per-task=4 --time=04:00:00 --pty bash
```

### 2. Start Elasticsearch

Navigate to the Elasticsearch directory:

```bash
cd data/elasticsearch-7.17.9/
export ES_JAVA_OPTS="-Xms1g -Xmx1g"
rm -rf data/nodes
nohup bin/elasticsearch > ~/es.log 2>&1 &
```

⚠️ **Note:** Do not rerun this step if the FEVER dataset has already been indexed (see step 4).

### 3. Activate the Conda Environment

```bash
conda activate dragin
cd ../..
```

### 4. Check Elasticsearch cluster health

```bash
curl -s http://localhost:9200/_cluster/health
```

### 5. Index the FEVER dataset

```bash
python prep_elastic.py --data_path data/dpr/psgs_w100.tsv --index_name fever
```

### 6. Verify indices

```bash
cd src
curl -X GET "localhost:9200/_cat/indices?v"
```

### 7. Run DRAGIN training/inference

```bash
python -u main.py -c ablation_config.json > ~/train.log 2>&1 &
```

---

## 📊 Logs & Outputs

* **Logs:** Written to `~/train.log`.
* **Elasticsearch logs:** Stored in `~/es.log`.

---

## ⚙️ Configuration

Modify `ablation_config.json` to adjust:

* Retriever settings
* Cross-encoder parameters
* Adaptive generation thresholds
* Training/inference modes

---

## 📌 Notes

* `main2.py`, `og_generate.py`, and `old_generate.py` are experimental scripts; prefer using `main.py` for standard runs.
* Ensure **Elasticsearch (7.17.9)** is running before indexing or retrieving.
* FEVER dataset must be available at `data/dpr/psgs_w100.tsv`.

---

Do you want me to also include a **Usage Examples** section (e.g., running generation only, evaluation only, etc.) so that new users don’t need to dig into `main.py` flags?
