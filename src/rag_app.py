print("🚀 Starting RAG app...")

import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from difflib import SequenceMatcher

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load data
with open("data/sample.txt", "r", encoding="utf-8") as f:
    documents = [line.strip() for line in f if line.strip()]

# Create embeddings
doc_embeddings = embedder.encode(documents)

# Build FAISS index
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(doc_embeddings))

print("✅ RAG system ready\n")

stop_words = {"what", "is", "the", "a", "an", "of", "in", "to"}
question_words = {"what", "who", "which", "define"}

def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def normalize(token: str) -> str:
    token = token.lower()
    token = re.sub(r'[^a-z0-9]', '', token)
    token = re.sub(r'(.)\1{2,}', r'\1', token)  # allow max 2 repeats
    return token

def clean_query(query):
    query = query.lower()
    query = re.sub(r"[^a-z0-9\s]", "", query)

    query = re.sub(r"\bwhats\b", "what is", query)
    query = re.sub(r"\bwhos\b", "who is", query)

    if query.startswith("what ") and " is " not in query:
        query = query.replace("what ", "what is ", 1)

    query = re.sub(r"\s+", " ", query).strip()
    return query

def extract_keywords(query):
    words = re.findall(r'\b\w+\b', query.lower())
    return [normalize(w) for w in words if w not in stop_words and w not in question_words]

def is_person_question(query):
    return query.startswith("who")

def is_metadata(line):
    return line.lower().startswith("author")

# 🔥 FIX: remove strict distance filter
def retrieve(query, k=3):
    query_embedding = embedder.encode([query])
    _, indices = index.search(np.array(query_embedding), k)
    return [documents[i] for i in indices[0]]

def best_context(query, contexts):
    keywords = extract_keywords(query)

    best = None
    best_score = 0

    for ctx in contexts:
        ctx_words = [normalize(w) for w in re.findall(r'\b\w+\b', ctx)]

        score = 0
        for kw in keywords:
            if len(kw) < 3:   # 🚫 ignore very short noisy tokens
                continue

            for w in ctx_words:
                if kw == w or similarity(kw, w) >= 0.82:
                    score += 1

        if score > best_score:
            best_score = score
            best = ctx

    return best, best_score


while True:
    raw_query = input("Ask: ")
    if raw_query.lower() in ["exit", "quit"]:
        break

    query = clean_query(raw_query)

    contexts = [c for c in retrieve(query, k=3) if not is_metadata(c)]

    if is_person_question(query):
        print("Contexts:", contexts)
        print("Answer: I don't know")
        continue

    best, score = best_context(query, contexts)

    # require at least 1 strong match
    if score == 0:
        print("Contexts:", contexts)
        print("Answer: I don't know")
        continue

    print("Contexts:", contexts)
    print("Answer:", best)