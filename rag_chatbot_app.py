"""SEC filing Q&A chatbot (RAG) in ONE file.

Put your PDF(s) in a folder named  docs/  next to this file. They are indexed
automatically the first time the app starts.

Run with:  streamlit run app.py
Needs:     pip install anthropic chromadb sentence-transformers streamlit pypdf
           and the ANTHROPIC_API_KEY environment variable (or Streamlit secret).
"""
import hashlib
import io
import os
from pathlib import Path

import anthropic
import chromadb
import streamlit as st
from pypdf import PdfReader

# =============================================================================
# SETTINGS (tune these)
# =============================================================================
MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1500      # characters per chunk
CHUNK_OVERLAP = 300    # characters shared between neighbouring chunks
TOP_K = 6              # chunks retrieved per question
MAX_HISTORY = 6        # past messages sent back for follow-up questions
MAX_QUESTIONS_PER_SESSION = 5   # protects your API bill on a public demo

DOCS_DIR = Path("docs")   # folder (next to app.py) that holds your PDFs
DOC_TYPES = {".pdf", ".txt", ".md"}

SYSTEM_PROMPT = (
    "You are an assistant that answers questions about company financial filings "
    "(such as SEC Form 10-K) using ONLY the provided context passages. "
    "Quote figures exactly as written and always state the unit (for example, "
    "millions of dollars) and the fiscal year or period. If the answer is not in the "
    "context, say you don't know based on the uploaded filings. Do not give investment "
    "advice, recommendations or price predictions. Keep answers concise and mention "
    "the source file and page, e.g. (Microsoft_10-K.pdf, p. 54)."
)

DISCLAIMER = "For information only. Not financial advice. Always check figures against the original filing."


# =============================================================================
# CORE RAG LOGIC
# =============================================================================
@st.cache_resource
def get_embedder():
    """Load the embedding model once and reuse it."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL)


@st.cache_resource
def get_collection():
    """Open (or create) the vector database stored in the ./db folder."""
    return chromadb.PersistentClient("./db").get_or_create_collection("docs")


def embed(texts):
    return get_embedder().encode(texts).tolist()


def load_pages(filename, data: bytes):
    """Return [(page_number, text), ...] for a PDF or plain-text file."""
    if filename.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        return [(i + 1, p.extract_text() or "") for i, p in enumerate(reader.pages)]
    return [(1, data.decode("utf-8", errors="ignore"))]


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = " ".join(text.split())  # collapse whitespace
    if not text:
        return []
    step = size - overlap
    return [text[i:i + size] for i in range(0, len(text), step)]


def ingest(collection, filename, data: bytes):
    """Chunk, embed and store a file. Returns the number of chunks added."""
    ids, docs, metas = [], [], []
    for page, text in load_pages(filename, data):
        for n, chunk in enumerate(chunk_text(text)):
            ids.append(hashlib.md5(f"{filename}-{page}-{n}".encode()).hexdigest())
            docs.append(chunk)
            metas.append({"source": filename, "page": page})
    if docs:
        # upsert: re-uploading the same file will not create duplicates
        collection.upsert(ids=ids, documents=docs, embeddings=embed(docs), metadatas=metas)
    return len(docs)


def pending_docs(collection):
    """PDF/text files in docs/ that have not been indexed yet."""
    if not DOCS_DIR.is_dir():
        return []
    todo = []
    for path in sorted(DOCS_DIR.iterdir()):
        if path.suffix.lower() in DOC_TYPES:
            already = collection.get(where={"source": path.name}, limit=1)
            if not already["ids"]:
                todo.append(path)
    return todo


def retrieve(collection, question, k=TOP_K):
    if collection.count() == 0:
        return []
    res = collection.query(query_embeddings=embed([question]), n_results=k)
    return [
        {"text": d, "source": m["source"], "page": m["page"]}
        for d, m in zip(res["documents"][0], res["metadatas"][0])
    ]


def build_messages(question, passages, past):
    context = "\n\n".join(f"[{p['source']}, p. {p['page']}]\n{p['text']}" for p in passages)
    msgs = list(past[-MAX_HISTORY:])
    msgs.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"})
    return msgs


def stream_answer(question, passages, past=()):
    """Generator that yields Claude's answer piece by piece."""
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    with client.messages.stream(
        model=MODEL,
        max_tokens=800,
        system=SYSTEM_PROMPT,
        messages=build_messages(question, passages, list(past)),
    ) as stream:
        for piece in stream.text_stream:
            yield piece


# =============================================================================
# STREAMLIT USER INTERFACE
# =============================================================================
def show_sources(passages):
    with st.expander("Sources"):
        for s in passages:
            st.markdown(f"**{s['source']}, p. {s['page']}**")
            st.caption(s["text"][:300] + "...")


def main():
    st.set_page_config(page_title="SEC Filing Q&A Chatbot", page_icon="📈")
    st.title("📈 Microsoft 2026 Annual Report Q&A Chatbot")
    st.caption("Ask questions about Microsoft's 2026 annual financial report in the knowledge base. Answers come only from those documents.")
    st.caption("Each session has a limit of 5 questions to avoid over-incurring API costs")
    st.caption(DISCLAIMER)

    # API key: Streamlit secrets (when deployed) or environment variable (locally)
    try:
        if "ANTHROPIC_API_KEY" in st.secrets:
            os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        pass  # no secrets file locally; fall back to the environment variable

    if not os.getenv("ANTHROPIC_API_KEY"):
        st.error("No API key found. Set the ANTHROPIC_API_KEY environment variable and restart.")
        st.stop()

    collection = get_collection()

    # ---- Index any new files from the docs/ folder (first run only) ----
    todo = pending_docs(collection)
    if todo:
        with st.spinner("Getting the knowledge base ready. This only happens on first start and can take a few minutes..."):
            for path in todo:
                ingest(collection, path.name, path.read_bytes())
    if collection.count() == 0:
        st.warning("The knowledge base is empty. Put a PDF in the docs/ folder (or upload one in the sidebar).")

    # ---- Sidebar: add documents ----
    with st.sidebar:
        st.header("Documents")
        st.caption("Microsoft 2026 Annual Financial Report from U.S. Securities and Exchange Commission")
        st.divider()
        st.subheader("Try asking")
        st.markdown(
            "- What are the main risk factors?\n"
            "- What were total revenues, and how did they change from the prior year?\n"
            "- What business segments does the company report?\n"
            "- What does the annual report say about competition?"
        )

    # ---- Chat state ----
    if "messages" not in st.session_state:
        st.session_state.messages = []   # each: {"role", "content", "sources"?}
    if "asked" not in st.session_state:
        st.session_state.asked = 0

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("sources"):
                show_sources(m["sources"])

    # ---- Handle a new question ----
    question = st.chat_input("Ask something about your documents")
    if not question:
        return
    if st.session_state.asked >= MAX_QUESTIONS_PER_SESSION:
        st.warning("Question limit reached for this session. Refresh the page to start over.")
        return
    st.session_state.asked += 1

    with st.chat_message("user"):
        st.markdown(question)

    past = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    with st.spinner("Searching the filing for the most relevant information..."):
        passages = retrieve(collection, question)

    with st.chat_message("assistant"):
        if not passages:
            reply = "Please upload and index a document first."
            st.markdown(reply)
        else:
            reply = st.write_stream(stream_answer(question, passages, past))
            show_sources(passages)

    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append({"role": "assistant", "content": reply, "sources": passages})


main()
