print("🚀 Starting RAG app...")

import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from difflib import SequenceMatcher
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

SHOW_CONTEXT = True

# Load models
embedder = SentenceTransformer("all-MiniLM-L6-v2")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

# Load data
with open("data/sample.txt", "r", encoding="utf-8") as f:
    documents = [line.strip() for line in f if line.strip()]

# Build FAISS index
doc_embeddings = embedder.encode(documents)
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(doc_embeddings))

print("✅ RAG system ready\n")

# Config
stop_words = {"what", "is", "the", "a", "an", "of", "in", "to"}
question_words = {"what", "who", "which", "define"}

ALIASES = {
    "iac": "infrastructure as code",
    "infra": "infrastructure",
    "terra": "terraform",
    "k8s": "kubernetes",
    "aws": "amazon web services",
    "s3": "amazon s3",
    "ec2": "amazon ec2",
    "iam": "identity and access management"
}

# -------- UTILS --------
def apply_aliases(query):
    return " ".join([ALIASES.get(w, w) for w in query.split()])

def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def normalize(token):
    token = token.lower()
    token = re.sub(r'[^a-z0-9]', '', token)
    token = re.sub(r'(.)\1{2,}', r'\1', token)
    return token

def correct_typos(query):
    vocab = set()
    for doc in documents:
        for w in re.findall(r'\b\w+\b', doc.lower()):
            vocab.add(w)

    corrected = []
    for word in query.split():
        best_match = word
        best_score = 0

        for v in vocab:
            score = similarity(word, v)
            if score > best_score:
                best_score = score
                best_match = v

        if best_score > 0.8:
            corrected.append(best_match)
        else:
            corrected.append(word)

    return " ".join(corrected)

def clean_query(query):
    query = query.lower()
    query = re.sub(r"[^a-z0-9\s]", "", query)

    query = re.sub(r"\bwhats\b", "what is", query)
    query = re.sub(r"\bwhos\b", "who is", query)

    if query.startswith("what ") and " is " not in query:
        query = query.replace("what ", "what is ", 1)

    query = re.sub(r"\s+", " ", query).strip()

    query = apply_aliases(query)
    query = correct_typos(query)

    return query

def extract_keywords(query):
    words = re.findall(r'\b\w+\b', query.lower())
    return [normalize(w) for w in words if w not in stop_words and w not in question_words]

def is_person_question(query):
    return query.startswith("who")

def is_unsupported_intent(query):
    return query.startswith("why") or query.startswith("how")

def is_generic_question(query):
    generic_words = {"thing", "things", "tools", "stuff", "it", "this"}
    return any(word in query.split() for word in generic_words)

def retrieve(query, k=3):
    query_embedding = embedder.encode([query])
    _, indices = index.search(np.array(query_embedding), k)
    return [documents[i] for i in indices[0]]

# -------- CONTEXT SELECTION --------
def best_context(query, contexts):
    keywords = extract_keywords(query)

    best = None
    best_score = 0

    for ctx in contexts:
        ctx_lower = ctx.lower()
        ctx_words = [normalize(w) for w in re.findall(r'\b\w+\b', ctx)]

        score = 0

        # Prefer definition
        if " is " in ctx_lower:
            score += 2

        for kw in keywords:
            if len(kw) < 3:
                continue

            if kw in ctx_lower:
                score += 3  # strong signal

            for w in ctx_words:
                sim = similarity(kw, w)
                if sim > 0.9:
                    score += 2
                elif sim > 0.75:
                    score += 1

        if score > best_score:
            best_score = score
            best = ctx

    return best, best_score

# -------- LLM --------
def generate_answer(context, query):
    prompt = f"""
You are an AI assistant.

Answer the question using EXACT information from the context.
Return a FULL sentence from the context.
Do NOT summarize or shorten.

Context:
{context}

Question:
{query}

Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    if answer:
        answer = answer[0].upper() + answer[1:]
    return answer

# -------- MAIN LOOP --------
while True:
    raw_query = input("Ask: ")
    if raw_query.lower() in ["exit", "quit"]:
        break

    query = clean_query(raw_query)
    contexts = retrieve(query, k=3)

    # Guardrails
    if (
        is_person_question(query)
        or is_unsupported_intent(query)
        or is_generic_question(query)
    ):
        if SHOW_CONTEXT:
            print("Contexts:", contexts)
        print("Answer: I don't know")
        continue

    best, score = best_context(query, contexts)

    if score == 0 or best is None:
        if SHOW_CONTEXT:
            print("Contexts:", contexts)
        print("Answer: I don't know")
        continue

    if SHOW_CONTEXT:
        print("Contexts:", contexts)

    answer = generate_answer(best, query)

    print("Answer:", answer)