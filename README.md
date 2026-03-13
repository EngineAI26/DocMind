# 🧠 DocMind — Multi-modal Document Q&A with RAG

A Streamlit app that lets you upload any PDF or image and chat with it using
Retrieval-Augmented Generation (RAG) powered by the Claude API.

## Tech Stack
- **Frontend/Server** — Streamlit
- **LLM + Vision** — Anthropic Claude (claude-sonnet-4)
- **PDF Parsing** — PyMuPDF (fitz)
- **Embeddings** — TF-IDF vectors (built from scratch, no external DB)
- **Retrieval** — Cosine similarity search in-memory
- **Image OCR** — Claude Vision API

## RAG Pipeline
1. Document → extract raw text (PDF.js / Claude Vision)
2. Text → overlapping chunks (~400 words, 80 overlap)
3. Chunks → TF-IDF vector embeddings
4. Query → embed → cosine similarity → top-4 chunks
5. Top chunks + query → Claude → grounded answer

---

## 🚀 Deploy to Streamlit Cloud (Free, Shareable URL)

### Step 1 — Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/docmind.git
git push -u origin main
```

### Step 2 — Deploy on Streamlit Cloud
1. Go to https://share.streamlit.io
2. Sign in with GitHub
3. Click **New app**
4. Select your repo → branch: `main` → file: `app.py`
5. Click **Deploy**

> Your live URL will be: `https://YOUR_USERNAME-docmind-app-XXXX.streamlit.app`

### Step 3 — Add API Key as a Secret (optional but recommended)
In Streamlit Cloud → your app → **Settings → Secrets**:
```toml
ANTHROPIC_API_KEY = "sk-ant-..."
```
Then in `app.py` replace the sidebar input with:
```python
import os
api_key = os.environ.get("ANTHROPIC_API_KEY", "")
```

---

## 🖥 Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501

---

## Interview Talking Points

### What is RAG?
"RAG solves the problem of LLMs not knowing your private documents.
Instead of fine-tuning, I split the doc into chunks, find the most relevant
chunks to the question using vector similarity, and inject only those chunks
into the LLM prompt as context. The model answers from the document, not from hallucination."

### Why TF-IDF instead of OpenAI embeddings?
"To keep it self-contained with zero external dependencies beyond the LLM.
In production I'd swap this for a proper embedding model (e.g. text-embedding-3-small)
and a vector DB like Pinecone or Qdrant for scale."

### How does chunking work?
"I split text into overlapping windows — 400-word chunks with 80-word overlap.
The overlap prevents losing context when an answer spans a chunk boundary."

### What would you improve for production?
- Replace TF-IDF with dense embeddings (OpenAI / Cohere / BGE)
- Use a real vector DB (Pinecone, Qdrant, ChromaDB)
- Add re-ranking with a cross-encoder model
- Stream responses token-by-token
- Add document metadata filtering
