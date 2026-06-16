"""SpecterSummarizer — Streamlit app for legal judgment summarization + chat.

Run with:  streamlit run app.py
"""
import streamlit as st

from src.inference import load_summarizer, summarize
from src.rag_chat import JudgmentChat
from src.utils import clean_text, extract_pdf_text

st.set_page_config(page_title="SpecterSummarizer", page_icon="⚖️", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 2.5rem; max-width: 1100px;}
      h1 {font-weight: 700;}
      .stChatMessage {border-radius: 12px;}
      .tagline {color: #6b7280; font-size: 1.05rem; margin-top: -0.6rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("⚖️ SpecterSummarizer")
st.markdown(
    '<p class="tagline">Local AI for summarizing Indian court judgments — '
    "upload a PDF, generate a concise summary, and chat with the document. "
    "Runs fully offline, no external APIs.</p>",
    unsafe_allow_html=True,
)


def reset_doc(text: str):
    """Store a freshly loaded document and clear stale summary/chat state."""
    st.session_state.doc_text = text
    st.session_state.summary = None
    st.session_state.chat = None
    st.session_state.messages = []


# ---- Sidebar: input ---------------------------------------------------------
with st.sidebar:
    st.header("📄 Input")
    pdf = st.file_uploader("Upload a judgment PDF", type="pdf")
    if pdf and st.session_state.get("loaded_name") != pdf.name:
        with st.spinner("Extracting text…"):
            reset_doc(extract_pdf_text(pdf))
            st.session_state.loaded_name = pdf.name

    if st.button("Use sample judgment", use_container_width=True):
        try:
            from src.utils import load_jsonl

            sample = next(load_jsonl("data/train_judg.jsonl"))
            reset_doc(clean_text(sample["Judgment"]))
            st.session_state.loaded_name = f"sample · {sample['ID']}"
        except (FileNotFoundError, StopIteration):
            st.warning("No sample data found in data/train_judg.jsonl.")

    st.caption(f"Model: `{load_summarizer()[2]}`")

# ---- Main ------------------------------------------------------------------
if not st.session_state.get("doc_text"):
    st.info("⬅️ Upload a judgment PDF or load the sample to get started.")
    st.stop()

st.success(f"Loaded: **{st.session_state.loaded_name}**")
tab_summary, tab_chat, tab_text = st.tabs(["📝 Summary", "💬 Chat", "📃 Full text"])

with tab_summary:
    col1, col2 = st.columns([1, 3])
    with col1:
        length = st.select_slider(
            "Summary length", options=["Short", "Medium", "Long"], value="Medium"
        )
        go = st.button("Generate summary", type="primary", use_container_width=True)
    bounds = {"Short": (40, 130), "Medium": (60, 220), "Long": (120, 350)}
    if go:
        lo, hi = bounds[length]
        with st.spinner("Summarizing…"):
            st.session_state.summary = summarize(
                st.session_state.doc_text, min_length=lo, max_length=hi
            )
    if st.session_state.get("summary"):
        st.subheader("Summary")
        st.write(st.session_state.summary)

with tab_chat:
    st.caption("Ask about the judgment. Answers are drawn from the document itself.")
    if st.session_state.get("chat") is None:
        with st.spinner("Indexing document for chat…"):
            st.session_state.chat = JudgmentChat(st.session_state.doc_text)

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if question := st.chat_input("e.g. What was the court's decision?"):
        st.session_state.messages.append({"role": "user", "content": question})
        st.chat_message("user").write(question)
        with st.spinner("Searching…"):
            answer = st.session_state.chat.answer(question)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)

with tab_text:
    st.text_area(
        "Extracted text", st.session_state.doc_text, height=500, label_visibility="collapsed"
    )
