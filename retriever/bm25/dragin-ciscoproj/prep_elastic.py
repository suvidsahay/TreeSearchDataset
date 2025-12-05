from typing import List, Tuple, Union, Dict
import argparse
import glob
import time
from elasticsearch import Elasticsearch
import csv
import json
import logging
from tqdm import tqdm
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.search.lexical.elastic_search import ElasticSearch

def build_elasticsearch(beir_corpus_file_pattern: str, index_name: str):
    try:
        # 1. Initialize Elasticsearch client (ignore security warning)
        es_client = Elasticsearch(
            hosts=['http://localhost:9200'],
            request_timeout=100,
            max_retries=3,
            retry_on_timeout=True
        )

        if not es_client.ping():
            raise ConnectionError("Failed to connect to Elasticsearch")

        # 2. Complete BEIR configuration with ALL possible parameters
        config = {
       'hostname': 'http://localhost:9200',  # Use full URL
        'timeout': 100,
        'retry_on_timeout': True,
        'maxsize': 24,
        'index_name': index_name,
        'keys': {
        'title': 'title',
        'body': 'txt'
        },
        'language': 'english',
        'number_of_shards': 1,
        'number_of_replicas': 0,
        'es_client': es_client,
        'analyzer': 'standard',
        'similarity': 'BM25'
    }


        # 3. Initialize BEIR ElasticSearch
        es = ElasticSearch(config)
        print(f"✅ Successfully connected to Elasticsearch")

        # 4. Index management
        print(f'Creating index {index_name}')
        es.delete_index()
        time.sleep(5)
        es.create_index()

        # 5. Index documents
        beir_corpus_files = glob.glob(beir_corpus_file_pattern)
        print(f'#files {len(beir_corpus_files)}')

        def generate_actions():
            for beir_corpus_file in beir_corpus_files:
                with open(beir_corpus_file, 'r') as fin:
                    reader = csv.reader(fin, delimiter='\t')
                    header = next(reader)  # skip header
                    for row in reader:
                        yield {
                            '_id': row[0],
                            '_op_type': 'index',
                            'title': row[2],
                            'txt': row[1]
                        }

        progress = tqdm(unit='docs')
        es.bulk_add_to_index(
            generate_actions=generate_actions(),
            progress=progress
        )
        print("✅ Indexing completed successfully")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--index_name', required=True)
    args = parser.parse_args()

    build_elasticsearch(args.data_path, args.index_name)
