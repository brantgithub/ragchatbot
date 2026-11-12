"""Document Q&A chatbot (RAG) in ONE file.

Run with:  streamlit run app.py
Needs:     pip install anthropic chromadb sentence-transformers streamlit pypdf
           and the ANTHROPIC_API_KEY environment variable (or Streamlit secret).
"""
import hashlib
import io
import os

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
TOP_K = 4              # chunks retrieved per question
MAX_HISTORY = 6        # past messages sent back for follow-up questions
MAX_QUESTIONS_PER_SESSION = 20   # protects your API bill on a public demo

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the provided "
    "context passages. If the answer is not in the context, say you don't know "
    "based on the uploaded documents. Keep answers concise. When useful, mention "
    "the source file name and page, e.g. (handbook.pdf, p. 3)."
)

SAMPLE_DOC = """Maple Tech Remote Work Policy (sample document)

Eligibility: Employees who have completed their 90-day probation may work remotely up to three days per week. Managers may approve additional remote days case by case.

Equipment: The company provides a laptop and a monitor. Employees receive a one-time home office stipend of $500, claimable within the first six months of remote work.

Core hours: Everyone must be reachable between 10:00 and 15:00 Pacific Time, regardless of where they work.

Expenses: Internet costs are reimbursed up to $60 per month with a valid receipt submitted through the finance portal by the 5th of the following month.

Security: Remote employees must use the company VPN and enable full-disk encryption on all work devices. Lost devices must be reported to IT within 24 hours.
"""


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
    st.set_page_config(page_title="Document Q&A Chatbot", page_icon="💬")
    st.title("💬 Document Q&A Chatbot")
    st.caption("Upload documents, then ask questions. Answers come only from your files.")

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

    # ---- Sidebar: add documents ----
    with st.sidebar:
        st.header("Documents")
        files = st.file_uploader(
            "Upload PDF or text files", type=["pdf", "txt", "md"], accept_multiple_files=True
        )
        if st.button("Add to knowledge base", disabled=not files):
            with st.spinner("Reading and indexing..."):
                for f in files:
                    st.success(f"{f.name}: {ingest(collection, f.name, f.getvalue())} chunks")
        if st.button("Load sample document"):
            with st.spinner("Indexing..."):
                n = ingest(collection, "sample_policy.txt", SAMPLE_DOC.encode())
            st.success(f"sample_policy.txt: {n} chunks")
        st.caption(f"Chunks stored: {collection.count()}")

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
