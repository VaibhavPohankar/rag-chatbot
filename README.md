# 🧠 RAG Chatbot (FAISS + Semantic Retrieval)

A retrieval-based AI system that answers user queries using semantic search, fuzzy matching, and strict grounding to avoid hallucinations.

---

## 🚀 Features

- 🔍 Semantic search using SentenceTransformers  
- ⚡ FAISS vector database for fast similarity search  
- 🧩 Fuzzy matching for typo tolerance (e.g., "dockr", "kubernets")  
- 🧹 Query normalization for noisy inputs  
- 🚫 Intent filtering (rejects invalid queries like "who is docker")  
- ✅ Hallucination-free responses (answers only from retrieved context)  

---

## 🧱 Architecture

```
User Query → Cleaning → Embedding → FAISS → Context Retrieval → Filtering → Answer
```

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

Ask: what is kubernetse
Answer: Kubernetes is used for container orchestration.

Ask: what is llm
Answer: I don't know
```

---

## 🧠 Key Learnings

- Difference between keyword-based and semantic retrieval  
- Handling noisy user inputs and typos  
- Importance of grounding to prevent hallucination  
- Core concepts of Retrieval-Augmented Generation (RAG)  
- Trade-offs between deterministic vs generative AI systems  

---

## 🚀 Future Improvements

- Add LLM-based answer generation  
- Support document ingestion (PDF, TXT, etc.)  
- Deploy as a web app (Streamlit / FastAPI)  
- Improve multi-context reasoning  

---

## 👤 Author

**Vaibhav Pohankar**  
DevOps | Cloud | AI Systems  