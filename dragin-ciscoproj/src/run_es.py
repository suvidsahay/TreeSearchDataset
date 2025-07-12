from elasticsearch import Elasticsearch, helpers
import json

es = Elasticsearch("http://localhost:9200")

INDEX_NAME = "34051_wiki"

# Define your index mapping (optional but good practice)
if not es.indices.exists(index=INDEX_NAME):
    es.indices.create(index=INDEX_NAME, body={
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0
        },
        "mappings": {
            "properties": {
                "title": {"type": "text"},
                "txt": {"type": "text"}
            }
        }
    })

# Load and index your data
with open("path/to/filtered_fever_wiki.json") as f:
    data = json.load(f)

def gen_docs():
    for item in data:
        for title, text in zip(item["wiki_urls"], item["wikipedia_texts"]):
            yield {
                "_index": INDEX_NAME,
                "_source": {
                    "title": title,
                    "txt": text
                }
            }

helpers.bulk(es, gen_docs())
print("Indexing complete.")

