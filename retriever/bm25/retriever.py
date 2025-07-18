import wikipediaapi
import nltk
from nltk.tokenize import sent_tokenize
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sentence_transformers.cross_encoder import CrossEncoder

cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')

print("Loading SentenceTransformer model...")
sbert_model = SentenceTransformer('all-mpnet-base-v2')
print("Model loaded.")

# This line is only needed once. If it's already downloaded, this does nothing.
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

wiki_wiki = wikipediaapi.Wikipedia(user_agent="multi_hop_agent", language="en")

def fetch_wikipedia_page(title):
    """Fetches the full text of a Wikipedia page."""
    page = wiki_wiki.page(title)
    return page.text if page.exists() else None

def split_into_passages(text, chunk_size=300, overlap=20):
    """
    A more robust passage splitter that splits by words.
    """
    if not text:
        return []
    words = text.split()
    passages = []
    for i in range(0, len(words), chunk_size - overlap):
        passages.append(" ".join(words[i:i + chunk_size]))
    return passages

def retrieve_sbert_passage(title, query):
    """Retrieves the best passage from a Wikipedia article using SBERT."""
    text = fetch_wikipedia_page(title)
    if not text:
        print(f"SBERT: Page '{title}' does not exist.")
        return None

    passages = split_into_passages(text)
    if not passages:
        print(f"SBERT: Could not split page '{title}' into passages.")
        return None

    # Encode the query and all passages
    passage_embeddings = sbert_model.encode(passages, convert_to_numpy=True)
    claim_embedding = sbert_model.encode([query], convert_to_numpy=True)

    # Compute cosine similarity and find the best passage
    similarities = cosine_similarity(claim_embedding, passage_embeddings)
    best_idx = np.argmax(similarities[0])

    return passages[best_idx]


def retrieve_bm25_passage(title, query):
    """Retrieves the best passage from a Wikipedia article using BM25."""
    text = fetch_wikipedia_page(title)
    if not text:
        print(f"BM25: Page '{title}' does not exist.")
        return None

    passages = split_into_passages(text)
    if not passages:
        print(f"BM25: Could not split page '{title}' into passages.")
        return None

    # Tokenize and use BM25 to find the best passage
    tokenized_passages = [p.lower().split() for p in passages]
    bm25 = BM25Okapi(tokenized_passages)
    scores = bm25.get_scores(query.lower().split())
    best_idx = scores.argmax()

    return passages[best_idx]

def retrieve_cross_encoder(title, query):
    text = fetch_wikipedia_page(title)
    if not text:
        print(f"Cross Encoder: Page '{title}' does not exist.")
        return None

    passages = split_into_passages(text)
    if not passages:
        print(f"Cross Encoder: Could not split page '{title}' into passages.")
        return None

    pairs = [[query, passage] for passage in passages]
    rerank_scores = cross_encoder.predict(pairs)
    best_idx = np.argmax(rerank_scores)

    return passages[best_idx]


def retrieve_best_passage(title, query, method='sbert'):
    print(f"Retrieving from '{title}' using {method.upper()}...")
    if method == 'sbert':
        return retrieve_sbert_passage(title, query)
    elif method == 'bm25':
        return retrieve_bm25_passage(title, query)
    elif method == "cross-encoder":
        return retrieve_cross_encoder(title, query)
    else:
        raise ValueError("Method must be either 'sbert','bm25' or 'cross-encoder'")
