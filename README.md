# 🧠 RAG Chatbot (Retrieval-Augmented Generation)

This project demonstrates the evolution from a rule-based retrieval system to a real RAG pipeline using embeddings and vector search.

---

## 🚀 Features

- 🔍 Keyword + fuzzy matching (baseline system)
- 🧠 Semantic retrieval using embeddings
- ⚡ FAISS vector database for similarity search
- 🤖 LLM-based answer generation
- ❌ Hallucination reduction using context grounding

---

## 🧱 Architecture

User Query → Embedding → FAISS → Retrieve Context → LLM → Answer

---

## 📂 Project Structure

src/
- baseline_app.py → rule-based retrieval
- rag_app.py → RAG pipeline

data/
- sample.txt → knowledge base

---

## ⚙️ Setup

```bash
pip install -r requirements.txt