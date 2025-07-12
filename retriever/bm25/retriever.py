import wikipediaapi
import nltk
from nltk.tokenize import sent_tokenize
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


nltk.download('punkt')

wiki_wiki = wikipediaapi.Wikipedia(user_agent="multi_hop_agent", language="en")
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')



def fetch_wikipedia_page(title):
    page = wiki_wiki.page(title)
    return page.text if page.exists() else None

def split_into_passages(text, chunk_size=100):
    sentences = sent_tokenize(text)
    passages = []
    chunk = ""
    for sentence in sentences:
        if len(chunk.split()) + len(sentence.split()) <= chunk_size:
            chunk += " " + sentence
        else:
            passages.append(chunk.strip())
            chunk = sentence
    if chunk:
        passages.append(chunk.strip())
    return passages

def retrieve_best_passage(title, query):
    text = fetch_wikipedia_page(title)
    if not text:
        return None
    passages = split_into_passages(text)
    tokenized_passages = [p.lower().split() for p in passages]
    bm25 = BM25Okapi(tokenized_passages)
    scores = bm25.get_scores(query.lower().split())
    best_idx = scores.argmax()
    return passages[best_idx]


def retrieve_best_passage_cross_encoder(title, query, chunk_size=100):
    text = fetch_wikipedia_page(title)
    if not text:
        return None

    passages = split_into_passages(text, chunk_size)

    if not passages:
        return None

    # Prepare query-passage pairs
    cross_inp = [[query, passage] for passage in passages]

    # Score passages
    cross_scores = cross_encoder.predict(cross_inp)

    # Get best passage
    best_idx = cross_scores.argmax()
    return passages[best_idx]

