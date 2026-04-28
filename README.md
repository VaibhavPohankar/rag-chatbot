# 🧠 RAG Chatbot (FAISS + Semantic Retrieval + Grounded LLM)

A Retrieval-Augmented Generation (RAG) system that answers user queries using semantic search, fuzzy matching, and a grounded LLM to avoid hallucinations.

---

## 🚀 Features

* 🔍 **Semantic Search** using SentenceTransformers
* ⚡ **FAISS Vector Database** for fast similarity retrieval
* 🧩 **Fuzzy Matching & Typo Handling** (e.g., *dockr*, *kubernets*)
* 🔄 **Alias Normalization** (e.g., *iac → infrastructure as code*)
* 🧠 **Definition-Priority Ranking** for accurate answers
* 🚫 **Hallucination-Free Responses** (answers only from context)
* 🛡️ **Guardrails** (rejects vague or unsupported queries)
* 🤖 **LLM Integration (FLAN-T5)** for clean full-sentence responses

---

## 🧱 Architecture

User Query
→ Query Cleaning (normalize + alias + typo handling)
→ Embedding (SentenceTransformer)
→ FAISS Retrieval
→ Context Ranking (keyword + definition scoring)
→ LLM (FLAN-T5)
→ Final Answer

---

## 📂 Project Structure

```
rag-chatbot/
├── src/
│   └── rag_app.py
├── data/
│   └── sample.txt
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### 1. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Install dependencies

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
Answer: Docker is a containerization platform used to package applications and their dependencies.

Ask: what is kubernetse  
Answer: Kubernetes is a container orchestration system used to manage containerized applications.

Ask: what is iac  
Answer: Infrastructure as code is a practice used to manage and provision infrastructure through code instead of manual processes.

Ask: what is randomstuff  
Answer: I don't know
```

---

## 🧠 Key Concepts Demonstrated

* Retrieval-Augmented Generation (RAG)
* Semantic vs Keyword Search
* Handling Noisy User Input
* Grounded AI (no hallucination design)
* Context Ranking & Heuristic Scoring
* LLM Prompt Engineering

---

## 🔧 Tech Stack

* Python
* FAISS
* SentenceTransformers
* HuggingFace Transformers (FLAN-T5)
* NumPy

---

## 🚀 Future Improvements

* Streamlit UI for interactive usage
* PDF / document ingestion
* Confidence scoring for answers
* API deployment (FastAPI / Flask)

---

## 👤 Author

**Vaibhav Pohankar**
DevOps | Cloud | AI Systems

GitHub: https://github.com/VaibhavPohankar

---

## 🎯 Summary

This project demonstrates a **production-style RAG system** that:

* retrieves relevant context
* handles noisy inputs
* generates grounded, reliable answers

Designed with a focus on **accuracy, robustness, and explainability**.
