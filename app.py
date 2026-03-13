import streamlit as st
import google.generativeai as genai
import numpy as np
import fitz  # PyMuPDF
import math
import re
from collections import Counter
from PIL import Image
import io
import time

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocMind – RAG Document Q&A",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── API KEY FROM STREAMLIT SECRETS ───────────────────────────────────────────
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=API_KEY)
except Exception:
    st.error("⚠️ API key not configured. Please add GEMINI_API_KEY to Streamlit secrets.")
    st.stop()

# ─── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.hero { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem; border-radius: 16px; margin-bottom: 1.5rem;
        border: 1px solid #2a2d4a; }
.hero h1 { color: #e8e9ed; font-size: 2rem; font-weight: 600;
           letter-spacing: -0.03em; margin: 0; }
.hero p  { color: #8b92b3; font-size: 0.95rem; margin: 0.4rem 0 0; }
.hero .accent { color: #a89cff; }

.how-card { background: #1e2128; border: 1px solid #2a2d36; border-radius: 14px;
            padding: 1.4rem; height: 100%; }
.how-icon { font-size: 2rem; margin-bottom: 0.6rem; }
.how-title { color: #e8e9ed; font-size: 1rem; font-weight: 600; margin-bottom: 0.4rem; }
.how-desc  { color: #6b7280; font-size: 0.85rem; line-height: 1.6; }

.step-row  { display: flex; align-items: center; gap: 14px; padding: 0.9rem 0;
             border-bottom: 1px solid #2a2d36; }
.step-row:last-child { border-bottom: none; }
.step-circle { background: #6c63ff; color: white; width: 30px; height: 30px;
               border-radius: 50%; display: flex; align-items: center;
               justify-content: center; font-size: 0.8rem; font-weight: 600;
               flex-shrink: 0; }
.step-body strong { color: #e8e9ed; font-size: 0.9rem; display: block; margin-bottom: 2px; }
.step-body span   { color: #6b7280; font-size: 0.82rem; }

.chunk-card { background: #1e2128; border: 1px solid #2a2d36; border-radius: 10px;
              padding: 0.8rem 1rem; margin-bottom: 0.5rem;
              font-size: 0.8rem; color: #6b7280; font-family: 'DM Mono', monospace; }
.chunk-num  { color: #a89cff; font-size: 0.7rem; margin-bottom: 4px; }

.source-box   { background: #12141a; border-left: 3px solid #6c63ff;
                border-radius: 0 8px 8px 0; padding: 0.75rem 1rem;
                margin-top: 0.5rem; font-size: 0.82rem; color: #6b7280; }
.source-label { color: #a89cff; font-family: 'DM Mono', monospace;
                font-size: 0.7rem; margin-bottom: 4px; }

.badge { display: inline-block; background: #1a1f2e; border: 1px solid #2d3a5e;
         color: #a89cff; padding: 3px 10px; border-radius: 20px;
         font-size: 0.72rem; font-family: 'DM Mono', monospace; margin: 2px; }
</style>
""", unsafe_allow_html=True)

# ─── SESSION STATE ────────────────────────────────────────────────────────────
for key, default in {
    "chunks": [], "embeddings": [], "vocab": {}, "idf": {},
    "chat_history": [], "doc_ready": False, "doc_name": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 400
CHUNK_OVERLAP = 80
TOP_K         = 4
MODEL         = "gemini-2.0-flash"

# ─── CHUNKING ─────────────────────────────────────────────────────────────────
def chunk_text(text: str) -> list[str]:
    words  = text.split()
    stride = CHUNK_SIZE - CHUNK_OVERLAP
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i : i + CHUNK_SIZE])
        if len(chunk.strip()) > 40:
            chunks.append(chunk)
        i += stride
    return chunks

# ─── TF-IDF EMBEDDINGS ────────────────────────────────────────────────────────
def tokenize(text: str) -> list[str]:
    return [w for w in re.sub(r"[^a-z0-9\s]", " ", text.lower()).split() if len(w) > 1]

def build_vocab(texts: list[str]) -> dict:
    vocab, idx = {}, 0
    for t in texts:
        for w in tokenize(t):
            if w not in vocab:
                vocab[w] = idx; idx += 1
    return vocab

def compute_idf(chunks: list[str]) -> dict:
    N, df = len(chunks), {}
    for c in chunks:
        for w in set(tokenize(c)):
            df[w] = df.get(w, 0) + 1
    return {w: math.log((N + 1) / (d + 1)) + 1 for w, d in df.items()}

def embed(text: str, vocab: dict, idf: dict) -> np.ndarray:
    tokens = tokenize(text)
    tf  = Counter(tokens)
    n   = len(tokens) or 1
    vec = np.zeros(len(vocab), dtype=np.float32)
    for w, cnt in tf.items():
        if w in vocab:
            vec[vocab[w]] = (cnt / n) * idf.get(w, 1.0)
    return vec

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0

# ─── RETRIEVAL ────────────────────────────────────────────────────────────────
def retrieve(query: str, k: int = TOP_K) -> list[dict]:
    qv     = embed(query, st.session_state.vocab, st.session_state.idf)
    scored = [
        {"idx": i, "chunk": c, "score": cosine_sim(qv, ev)}
        for i, (c, ev) in enumerate(zip(st.session_state.chunks, st.session_state.embeddings))
    ]
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:k]

# ─── DOCUMENT PARSING ─────────────────────────────────────────────────────────
def extract_pdf_text(file_bytes: bytes) -> str:
    doc  = fitz.open(stream=file_bytes, filetype="pdf")
    text = "".join(page.get_text() + "\n" for page in doc)
    return text.strip()

def extract_image_text(file_bytes: bytes) -> str:
    model  = genai.GenerativeModel(MODEL)
    image  = Image.open(io.BytesIO(file_bytes))
    result = model.generate_content([
        "Extract ALL visible text from this image. Return only the raw text, no commentary.",
        image
    ])
    return result.text.strip()

# ─── BUILD INDEX ──────────────────────────────────────────────────────────────
def build_index(text: str):
    chunks = chunk_text(text)
    vocab  = build_vocab(chunks)
    idf    = compute_idf(chunks)
    st.session_state.chunks     = chunks
    st.session_state.vocab      = vocab
    st.session_state.idf        = idf
    st.session_state.embeddings = [embed(c, vocab, idf) for c in chunks]
    st.session_state.doc_ready  = True

# ─── GEMINI ANSWER ────────────────────────────────────────────────────────────
def ask_gemini(question: str, retrieved: list[dict]) -> str:
    context = "\n\n---\n\n".join(
        f"[Chunk {r['idx']+1}]\n{r['chunk']}" for r in retrieved
    )
    system_instruction = (
        "You are a helpful document assistant. Answer the user's question based ONLY "
        "on the provided document context. Be clear, concise, and accurate. "
        "If the answer isn't in the context, say so honestly.\n\n"
        f"DOCUMENT CONTEXT:\n{context}"
    )
    model   = genai.GenerativeModel(MODEL, system_instruction=system_instruction)
    history = [
        {"role": "user" if m["role"] == "user" else "model", "parts": [m["content"]]}
        for m in st.session_state.chat_history
    ]
    try:
        time.sleep(3)  # avoid hitting free tier rate limit
        chat   = model.start_chat(history=history)
        result = chat.send_message(question)
        return result.text.strip()
    except Exception as e:
        err = str(e).lower()
        if "resource exhausted" in err or "quota" in err or "429" in err:
            return "⚠️ The AI is a bit busy right now (free tier rate limit). Please wait 30–60 seconds and try again."
        raise e

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='margin-bottom:1rem'>
        <span style='font-size:1.4rem;font-weight:600;letter-spacing:-0.02em;color:#e8e9ed'>
        🧠 Doc<span style='color:#a89cff'>Mind</span></span><br>
        <span style='font-size:0.78rem;color:#6b7280'>Multimodal RAG · AI Document Assistant</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**📄 Upload your document**")
    uploaded = st.file_uploader(
        "", type=["pdf", "png", "jpg", "jpeg", "webp"],
        label_visibility="collapsed"
    )

    if uploaded:
        if st.button("⚡ Process Document", use_container_width=True, type="primary"):
            with st.spinner("Extracting text & building index…"):
                try:
                    raw = uploaded.read()
                    text = extract_pdf_text(raw) if uploaded.type == "application/pdf" \
                           else extract_image_text(raw)
                    if len(text.strip()) < 30:
                        st.error("Could not extract text. Try another file.")
                    else:
                        build_index(text)
                        st.session_state.doc_name     = uploaded.name
                        st.session_state.chat_history = []
                        st.success(f"✅ {len(st.session_state.chunks)} chunks indexed!")
                except Exception as e:
                    st.error(f"Error: {e}")

    if st.session_state.doc_ready:
        st.divider()
        st.markdown(f"**📑 {st.session_state.doc_name}**")
        st.markdown(
            f'<span class="badge">📦 {len(st.session_state.chunks)} chunks</span> '
            f'<span class="badge">🔠 {len(st.session_state.vocab)} vocab</span>',
            unsafe_allow_html=True
        )
        with st.expander(f"View chunks ({len(st.session_state.chunks)})"):
            for i, c in enumerate(st.session_state.chunks):
                st.markdown(f"""
                <div class='chunk-card'>
                    <div class='chunk-num'>CHUNK {i+1}</div>
                    {c[:200]}{'…' if len(c) > 200 else ''}
                </div>""", unsafe_allow_html=True)
        st.divider()
        if st.button("🗑 Reset / New Document", use_container_width=True):
            for k in ["chunks","embeddings","vocab","idf","chat_history","doc_ready","doc_name"]:
                st.session_state[k] = (
                    [] if k in ["chunks","embeddings","chat_history"] else
                    {} if k in ["vocab","idf"] else
                    False if k == "doc_ready" else ""
                )
            st.rerun()

# ─── MAIN AREA ────────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero'>
    <h1>🧠 Doc<span class='accent'>Mind</span></h1>
    <p>Upload any document · Ask questions · Get instant AI-powered answers</p>
</div>
""", unsafe_allow_html=True)

# ── LANDING PAGE (no doc uploaded yet) ────────────────────────────────────────
if not st.session_state.doc_ready:
    st.markdown("## How to use DocMind")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class='how-card'>
            <div class='how-icon'>📄</div>
            <div class='how-title'>Step 1 — Upload</div>
            <div class='how-desc'>Click <b>Browse files</b> in the sidebar and upload any <b>PDF</b> or <b>image</b> (JPG, PNG, WebP). Works with scanned documents too.</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='how-card'>
            <div class='how-icon'>⚡</div>
            <div class='how-title'>Step 2 — Process</div>
            <div class='how-desc'>Hit <b>Process Document</b>. The app extracts the text, splits it into chunks, and builds a vector index — all in seconds.</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class='how-card'>
            <div class='how-icon'>💬</div>
            <div class='how-title'>Step 3 — Ask</div>
            <div class='how-desc'>Type any question about your document in the chat box. Get instant, grounded answers with the source chunks highlighted.</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### What happens under the hood")

    st.markdown("""
    <div style='background:#1e2128;border:1px solid #2a2d36;border-radius:14px;padding:1.2rem 1.6rem;'>
        <div class='step-row'>
            <div class='step-circle'>1</div>
            <div class='step-body'>
                <strong>Text Extraction</strong>
                <span>PDFs parsed with PyMuPDF · Images read via Gemini Vision OCR</span>
            </div>
        </div>
        <div class='step-row'>
            <div class='step-circle'>2</div>
            <div class='step-body'>
                <strong>Chunking</strong>
                <span>Text split into overlapping ~400-word windows (80-word overlap) to preserve context</span>
            </div>
        </div>
        <div class='step-row'>
            <div class='step-circle'>3</div>
            <div class='step-body'>
                <strong>TF-IDF Embedding</strong>
                <span>Each chunk converted into a vector — term frequency × inverse document frequency</span>
            </div>
        </div>
        <div class='step-row'>
            <div class='step-circle'>4</div>
            <div class='step-body'>
                <strong>Semantic Retrieval</strong>
                <span>Your question is embedded and compared against all chunks via cosine similarity — top 4 retrieved</span>
            </div>
        </div>
        <div class='step-row'>
            <div class='step-circle'>5</div>
            <div class='step-body'>
                <strong>Grounded Generation</strong>
                <span>Retrieved chunks injected into Gemini's prompt — answers come only from your document, no hallucination</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Supported file types")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        **📄 PDF**
        - Research papers
        - Reports & articles
        - Contracts & invoices
        - Textbook chapters
        """)
    with c2:
        st.markdown("""
        **🖼️ Images (JPG, PNG, WebP)**
        - Scanned documents
        - Screenshots of text
        - Handwritten notes (printed)
        - Whiteboards & slides
        """)

# ── CHAT INTERFACE (doc ready) ────────────────────────────────────────────────
else:
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🧠"):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(f"📎 {len(msg['sources'])} source chunks used"):
                    for s in msg["sources"]:
                        st.markdown(f"""
                        <div class='source-box'>
                            <div class='source-label'>CHUNK {s['idx']+1} · similarity {s['score']*100:.0f}%</div>
                            {s['chunk'][:300]}{'…' if len(s['chunk'])>300 else ''}
                        </div>""", unsafe_allow_html=True)

    if not st.session_state.chat_history:
        st.markdown("**💡 Try asking:**")
        cols = st.columns(3)
        suggestions = [
            "Summarize this document in 3 sentences",
            "What are the key topics or themes?",
            "List the main conclusions or findings",
        ]
        for i, s in enumerate(suggestions):
            if cols[i].button(s, use_container_width=True):
                st.session_state._quick_ask = s
                st.rerun()

    if hasattr(st.session_state, "_quick_ask"):
        prompt = st.session_state._quick_ask
        del st.session_state._quick_ask
        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)
        with st.chat_message("assistant", avatar="🧠"):
            with st.spinner("Retrieving chunks & generating answer…"):
                retrieved = retrieve(prompt)
                answer    = ask_gemini(prompt, retrieved)
            st.markdown(answer)
            with st.expander(f"📎 {len(retrieved)} source chunks used"):
                for s in retrieved:
                    st.markdown(f"""
                    <div class='source-box'>
                        <div class='source-label'>CHUNK {s['idx']+1} · similarity {s['score']*100:.0f}%</div>
                        {s['chunk'][:300]}{'…' if len(s['chunk'])>300 else ''}
                    </div>""", unsafe_allow_html=True)
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        st.session_state.chat_history.append({"role": "assistant", "content": answer, "sources": retrieved})
        st.rerun()

    if prompt := st.chat_input("Ask anything about your document…"):
        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)
        with st.chat_message("assistant", avatar="🧠"):
            with st.spinner("Retrieving chunks & generating answer…"):
                retrieved = retrieve(prompt)
                answer    = ask_gemini(prompt, retrieved)
            st.markdown(answer)
            with st.expander(f"📎 {len(retrieved)} source chunks used"):
                for s in retrieved:
                    st.markdown(f"""
                    <div class='source-box'>
                        <div class='source-label'>CHUNK {s['idx']+1} · similarity {s['score']*100:.0f}%</div>
                        {s['chunk'][:300]}{'…' if len(s['chunk'])>300 else ''}
                    </div>""", unsafe_allow_html=True)
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        st.session_state.chat_history.append({"role": "assistant", "content": answer, "sources": retrieved})
        st.rerun()
