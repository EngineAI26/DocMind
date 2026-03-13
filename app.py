import streamlit as st
import google.generativeai as genai
import numpy as np
import fitz  # PyMuPDF
import math
import re
from collections import Counter
from PIL import Image
import io

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocMind – RAG Document Q&A",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

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

.chunk-card { background: #1e2128; border: 1px solid #2a2d36;
              border-radius: 10px; padding: 0.8rem 1rem; margin-bottom: 0.5rem;
              font-size: 0.8rem; color: #6b7280; font-family: 'DM Mono', monospace; }
.chunk-num  { color: #a89cff; font-size: 0.7rem; margin-bottom: 4px; }

.source-box { background: #12141a; border-left: 3px solid #6c63ff;
              border-radius: 0 8px 8px 0; padding: 0.75rem 1rem;
              margin-top: 0.5rem; font-size: 0.82rem; color: #6b7280; }
.source-label { color: #a89cff; font-family: 'DM Mono', monospace;
                font-size: 0.7rem; margin-bottom: 4px; }

.badge { display: inline-block; background: #1a1f2e; border: 1px solid #2d3a5e;
         color: #a89cff; padding: 3px 10px; border-radius: 20px;
         font-size: 0.72rem; font-family: 'DM Mono', monospace; margin: 2px; }

.rag-step  { display: flex; align-items: flex-start; gap: 10px; margin-bottom: 0.8rem; }
.step-num  { background: #6c63ff; color: white; width: 24px; height: 24px;
             border-radius: 50%; display: flex; align-items: center;
             justify-content: center; font-size: 0.75rem; font-weight: 600;
             flex-shrink: 0; margin-top: 2px; }
.step-text { color: #8b92b3; font-size: 0.85rem; line-height: 1.5; }
.step-text b { color: #e8e9ed; }
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
MODEL         = "gemini-2.0-flash"   # free tier, fast, multimodal

# ─── CHUNKING ─────────────────────────────────────────────────────────────────
def chunk_text(text: str) -> list[str]:
    words  = text.split()
    stride = CHUNK_SIZE - CHUNK_OVERLAP
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + CHUNK_SIZE])
        if len(chunk.strip()) > 40:
            chunks.append(chunk)
        i += stride
    return chunks

# ─── TF-IDF EMBEDDINGS ────────────────────────────────────────────────────────
def tokenize(text: str) -> list[str]:
    return [w for w in re.sub(r"[^a-z0-9\s]", " ", text.lower()).split() if len(w) > 1]

def build_vocab(texts: list[str]) -> dict[str, int]:
    vocab, idx = {}, 0
    for t in texts:
        for w in tokenize(t):
            if w not in vocab:
                vocab[w] = idx
                idx += 1
    return vocab

def compute_idf(chunks: list[str]) -> dict[str, float]:
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
    text = ""
    for page in doc:
        text += page.get_text() + "\n"
    return text.strip()

def extract_image_text(file_bytes: bytes) -> str:
    """Gemini Vision OCR."""
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
    embs   = [embed(c, vocab, idf) for c in chunks]
    st.session_state.chunks     = chunks
    st.session_state.vocab      = vocab
    st.session_state.idf        = idf
    st.session_state.embeddings = embs
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
    model = genai.GenerativeModel(MODEL, system_instruction=system_instruction)

    # Rebuild Gemini-compatible history (roles: "user" / "model")
    history = []
    for msg in st.session_state.chat_history:
        role = "user" if msg["role"] == "user" else "model"
        history.append({"role": role, "parts": [msg["content"]]})

    chat   = model.start_chat(history=history)
    result = chat.send_message(question)
    return result.text.strip()

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='margin-bottom:1rem'>
        <span style='font-size:1.4rem;font-weight:600;letter-spacing:-0.02em;color:#e8e9ed'>
        🧠 Doc<span style='color:#a89cff'>Mind</span></span><br>
        <span style='font-size:0.78rem;color:#6b7280'>Multimodal RAG · Gemini API (Free)</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**🔑 Google Gemini API Key**")
    st.caption("Free at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)")
    api_key = st.text_input("", type="password", placeholder="AIza...", label_visibility="collapsed")

    if api_key:
        genai.configure(api_key=api_key)

    st.divider()

    st.markdown("**📄 Upload Document**")
    uploaded = st.file_uploader(
        "", type=["pdf", "png", "jpg", "jpeg", "webp"],
        label_visibility="collapsed"
    )

    if uploaded and api_key:
        if st.button("⚡ Process Document", use_container_width=True, type="primary"):
            with st.spinner("Processing…"):
                try:
                    raw = uploaded.read()
                    if uploaded.type == "application/pdf":
                        text = extract_pdf_text(raw)
                    else:
                        text = extract_image_text(raw)

                    if len(text.strip()) < 30:
                        st.error("Could not extract text. Try another file.")
                    else:
                        build_index(text)
                        st.session_state.doc_name     = uploaded.name
                        st.session_state.chat_history = []
                        st.success(f"✅ {len(st.session_state.chunks)} chunks indexed!")
                except Exception as e:
                    st.error(f"Error: {e}")
    elif uploaded and not api_key:
        st.warning("Enter your Gemini API key first.")

    st.divider()

    with st.expander("⚡ How RAG works"):
        st.markdown("""
<div class='rag-step'><div class='step-num'>1</div><div class='step-text'><b>Chunk</b> — Document split into overlapping ~400-word pieces</div></div>
<div class='rag-step'><div class='step-num'>2</div><div class='step-text'><b>Embed</b> — Each chunk → TF-IDF vector (built from scratch)</div></div>
<div class='rag-step'><div class='step-num'>3</div><div class='step-text'><b>Retrieve</b> — Query embedded; cosine similarity finds top-4 chunks</div></div>
<div class='rag-step'><div class='step-num'>4</div><div class='step-text'><b>Generate</b> — Chunks injected into Gemini prompt as context</div></div>
<div class='rag-step'><div class='step-num'>5</div><div class='step-text'><b>Answer</b> — Grounded response, no hallucination outside the doc</div></div>
""", unsafe_allow_html=True)

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

        if st.button("🗑 Reset", use_container_width=True):
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
    <p>Multi-modal Document Q&amp;A · Retrieval-Augmented Generation · Gemini 2.0 Flash (Free)</p>
</div>
""", unsafe_allow_html=True)

if not api_key:
    st.info("👈 Paste your **free** Google Gemini API key in the sidebar to get started.")
    st.markdown("""
**Get your free key in 30 seconds:**
1. Go to [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
2. Sign in with Google → click **Create API Key**
3. Copy and paste it in the sidebar

**What this app demonstrates:**
- 📄 PDF parsing with PyMuPDF
- 🖼️ Image OCR via Gemini Vision API
- 🔢 TF-IDF vector embeddings built from scratch in Python
- 🔍 Cosine similarity retrieval — no vector DB needed
- 🤖 Grounded generation — answers only from your document
- 💬 Multi-turn chat with full conversation history
    """)

elif not st.session_state.doc_ready:
    st.markdown("### Upload a document in the sidebar to begin")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Supported formats:**\n- 📄 PDF\n- 🖼️ Images: JPG, PNG, WebP")
    with col2:
        st.markdown("**Good test docs:**\n- Research papers\n- Reports / articles\n- Any text-heavy image")

else:
    # ── Render chat history ──
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

    # ── Suggestion chips (only on fresh doc) ──
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

    # ── Handle suggestion button click ──
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

    # ── Chat input ──
    if prompt := st.chat_input("Ask a question about your document…"):
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
