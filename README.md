# RAG Chatbot (FAISS + Semantic Retrieval)

A retrieval-based AI system that answers user queries using semantic search, fuzzy matching, and strict grounding to avoid hallucinations.

---

## 🚀 Features

- Semantic search using SentenceTransformers
- FAISS vector database for fast retrieval
- Fuzzy matching for typo tolerance
- Query normalization (handles noisy input)
- Intent filtering (rejects invalid queries like "who is docker")
- Hallucination-free responses (only grounded answers)

---

## 🧱 Architecture

Query → Embedding → FAISS → Context Retrieval → Filtering → Answer

---

## 📂 Project Structure

```
src/
  rag_app.py
data/
  sample.txt
requirements.txt
README.md
```

---

## ⚙️ Setup

```bash
pip install -r requirements.txt
```

---

## ▶️ Run

```bash
python src/rag_app.py
```

---

## 🧪 Example

```
Ask: what is docker
Answer: Docker is used for containerization.

Ask: what is llm
Answer: I don't know
```

---

## 🧠 Key Learnings

- Semantic vs keyword retrieval
- Handling noisy inputs in AI systems
- Importance of grounding to avoid hallucination
- Basics of RAG architecture

---

## 👤 Author

Vaibhav Pohankar