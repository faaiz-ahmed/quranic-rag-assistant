import hashlib
import io
import os
import re
from pathlib import Path
import pdfplumber
import pypdfium2 as pdfium
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
try:
    import pytesseract
except ImportError:
    pytesseract = None
APP_DIR = Path(__file__).parent
DEFAULT_PDF = APP_DIR / "Surah-Fatiha.pdf"
CACHE_DIR = APP_DIR / "ocr_cache"
MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "qwen/qwen3-32b"]
EXAMPLES = [
    "What does the word Rabb mean?",
    "What is said about reciting Surat Al-Fatihah in prayer?",
    "What are the two kinds of guidance?",
    "Who are those on whom Allah bestowed His Grace?",
    "What is the reward for saying Amin?",
    "What does Al-Fatihah mean by the Straight Way?",
]
st.set_page_config(page_title="Quranic AI Assistant", page_icon="📖", layout="centered")
st.markdown(
    """
<style>
.hero{background:linear-gradient(135deg,#0f766e,#134e4a);color:#fff;
      padding:1.5rem 1.8rem;border-radius:16px;margin-bottom:.8rem}
.hero h1{margin:0;font-size:1.8rem;color:#fff;padding:0}
.hero p{margin:.4rem 0 0;opacity:.92;font-size:.98rem}
.note{font-size:.82rem;color:#6b7280;margin-top:.5rem}
</style>
<div class="hero">
  <h1>📖 Quranic AI Assistant</h1>
  <p>Ask questions about Surah Al-Fatihah. Answers come only from the PDF
     translation, using Retrieval-Augmented Generation (RAG).</p>
</div>
""",
    unsafe_allow_html=True,
)


def clean_page(text: str) -> str:
    text = re.sub(r"-\n(?=[a-z])", "", text)
    kept = []
    for para in re.split(r"\n\s*\n", text):
        para = re.sub(r"\s+", " ", para).strip()
        if len(re.findall(r"[A-Za-z]{3,}", para)) >= 3:
            kept.append(para)
    return "\n\n".join(kept)

def read_text_layer(file_bytes: bytes) -> list[str]:
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            return [p.extract_text() or "" for p in pdf.pages]
    except Exception:
        return []

def ocr_pdf(file_bytes: bytes, dpi: int = 200) -> list[str]:
    if pytesseract is None:
        raise RuntimeError("pytesseract is not installed (see requirements.txt).")
    pdf = pdfium.PdfDocument(file_bytes)
    return [
        pytesseract.image_to_string(page.render(scale=dpi / 72).to_pil(), lang="eng")
        for page in pdf
    ]

@st.cache_data(show_spinner=False)
def load_pdf(file_bytes: bytes):
    cache_file = CACHE_DIR / f"{hashlib.sha256(file_bytes).hexdigest()[:16]}.txt"
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8").split("\f"), "OCR (cached)"
    raw = read_text_layer(file_bytes)
    if sum(len(p.strip()) for p in raw) > 100:
        method = "Text layer"
    else:
        raw, method = ocr_pdf(file_bytes), "OCR (scanned PDF)"
    pages = [clean_page(p) for p in raw]
    if not any(pages):
        raise ValueError("No readable text found in this PDF.")
    try:
        CACHE_DIR.mkdir(exist_ok=True)
        cache_file.write_text("\f".join(pages), encoding="utf-8")
    except OSError:
        pass
    return pages, method

@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=False)
def build_index(pages: tuple, chunk_size: int = 500, overlap: int = 80):
    docs = [
        Document(page_content=t, metadata={"page": i + 1})
        for i, t in enumerate(pages)
        if t.strip()
    ]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    return FAISS.from_documents(chunks, get_embeddings()), len(chunks)

PROMPT = ChatPromptTemplate.from_template(
    """You are a careful assistant for studying Surah Al-Fatihah.
Answer ONLY from the passages below, which come from the uploaded PDF.

Rules:
- If the passages do not contain the answer, say: "The provided text does not mention this."
- Do not add outside knowledge, personal interpretation, or religious rulings (fatwa).
- Be concise and well organised (short paragraphs or bullets).
- Mention the page, e.g. (Page 2), when you use a passage.

Passages:
{context}

Question: {question}

Answer:"""
)

def format_context(docs) -> str:
    return "\n\n".join(
        f"[Passage {i} | Page {d.metadata.get('page', '?')}]\n{d.page_content}"
        for i, d in enumerate(docs, 1)
    )

def get_api_key():
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return os.getenv("GROQ_API_KEY") or st.text_input(
        "Groq API key",
        type="password",
        help="Free key from console.groq.com",
    )

def render_message(msg: dict):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📚 Sources used ({len(msg['sources'])})"):
                for i, (page, text) in enumerate(msg["sources"], 1):
                    st.markdown(f"**Passage {i} · Page {page}**")
                    st.markdown("> " + text.replace("\n", "\n> "))

with st.sidebar:
    st.header("⚙️ Settings")
    api_key = get_api_key()
    model = st.selectbox("Groq model", MODELS)
    top_k = st.slider("Passages to retrieve (k)", 2, 8, 4)
    st.subheader("📄 Document")
    uploaded = st.file_uploader("Use a different PDF (optional)", type=["pdf"])
if not api_key:
    st.info("👈 Enter your Groq API key in the sidebar to start.")
    st.stop()

if uploaded:
    pdf_bytes, source_name = uploaded.getvalue(), uploaded.name
elif DEFAULT_PDF.exists():
    pdf_bytes, source_name = DEFAULT_PDF.read_bytes(), DEFAULT_PDF.name
else:
    st.error("Default PDF not found. Upload a surah PDF in the sidebar.")
    st.stop()
try:
    with st.spinner("Reading the PDF (OCR on first run, may take a few seconds)..."):
        pages, method = load_pdf(pdf_bytes)
    with st.spinner("Building the search index..."):
        vector_db, n_chunks = build_index(tuple(pages))
except Exception as e:
    st.error(f"Could not process the PDF: {e}")
    st.stop()
st.caption(f"📄 **{source_name}** · {len(pages)} pages · {n_chunks} passages · {method}")
with st.sidebar:
    c1, c2 = st.columns(2)
    c1.metric("Pages", len(pages))
    c2.metric("Passages", n_chunks)
    with st.expander("Preview extracted text"):
        st.text(("\n\n".join(pages))[:1500] + " ...")
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.caption(
        "Study project. Not a source of religious rulings. "
        "Please consult qualified scholars."
    )

st.session_state.setdefault("messages", [])
if not st.session_state.messages:
    st.markdown("**Try asking:**")
    cols = st.columns(2)
    for i, q in enumerate(EXAMPLES):
        if cols[i % 2].button(q, key=f"ex{i}", use_container_width=True):
            st.session_state.pending = q
            st.rerun()
for m in st.session_state.messages:
    render_message(m)
question = st.chat_input("Ask a question about the Surah...") or st.session_state.pop(
    "pending", None
)
if question:
    user_msg = {"role": "user", "content": question}
    st.session_state.messages.append(user_msg)
    render_message(user_msg)
    with st.chat_message("assistant"):
        with st.spinner("Searching the PDF and writing an answer..."):
            try:
                docs = vector_db.similarity_search(question, k=top_k)
                llm = ChatGroq(groq_api_key=api_key, model_name=model, temperature=0)
                chain = PROMPT | llm | StrOutputParser()
                answer = chain.invoke(
                    {"context": format_context(docs), "question": question}
                )
                answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
                sources = [(d.metadata.get("page", "?"), d.page_content) for d in docs]
            except Exception as e:
                answer, sources = f"⚠️ Could not get an answer: {e}", []
        bot_msg = {"role": "assistant", "content": answer, "sources": sources}
        st.session_state.messages.append(bot_msg)
        st.markdown(answer)
        if sources:
            with st.expander(f"📚 Sources used ({len(sources)})"):
                for i, (page, text) in enumerate(sources, 1):
                    st.markdown(f"**Passage {i} · Page {page}**")
                    st.markdown("> " + text.replace("\n", "\n> "))
