from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src.rag import collection_name_for_channel, index_channel, query_channel

st.set_page_config(page_title="YouTube Channel RAG", layout="centered")

if "active_collection_name" not in st.session_state:
    st.session_state.active_collection_name = None
if "last_channel_url" not in st.session_state:
    st.session_state.last_channel_url = ""
if "last_index_stats" not in st.session_state:
    st.session_state.last_index_stats = None


st.title("YouTube Channel RAG")
st.caption("Ask questions across public transcripts from a channel or playlist.")

with st.sidebar:
    st.subheader("How it works")
    st.markdown(
        """
        1. Paste a **channel** (`@handle` or `/channel/UC…`) or **playlist** URL.
           We append `/videos` for channel home links so real video IDs are listed.
        2. Click **Index channel** (uses transcripts + OpenAI embeddings).
        3. Ask a question; answers use **retrieved transcript chunks** only.
        """
    )
    st.divider()
    st.caption("Requires `OPENAI_API_KEY` in environment or `.env`.")

channel_url = st.text_input(
    "Channel or playlist URL",
    placeholder="https://www.youtube.com/@MrBeast",
    value=st.session_state.last_channel_url or "",
)
max_videos = st.number_input("Max videos to scan", min_value=1, max_value=500, value=100)
replace_index = st.checkbox(
    "Replace existing index for this URL",
    value=True,
    help="If checked, deletes the stored vectors for this channel URL before re-indexing. "
    "Uncheck to upsert chunks (same video IDs update; orphans may remain).",
)

col_a, col_b = st.columns(2)
with col_a:
    index_clicked = st.button("Index channel", type="primary")
with col_b:
    if st.session_state.active_collection_name:
        st.caption(f"Active index: `{st.session_state.active_collection_name}`")

if index_clicked and channel_url.strip():
    st.session_state.last_channel_url = channel_url.strip()
    progress = st.progress(0.0, text="Starting…")
    status = st.empty()

    def on_progress(p: dict) -> None:
        msg = p.get("message", "")
        if p.get("stage") == "list":
            progress.progress(0.05, text=msg or "Listed videos")
        elif p.get("stage") == "video":
            total = max(int(p.get("total") or 1), 1)
            cur = int(p.get("current") or 0)
            frac = min(0.05 + (cur / total) * 0.95, 1.0)
            progress.progress(frac, text=f"{cur}/{total} — {msg}")
        status.write(msg)

    try:
        with st.spinner("Indexing…"):
            stats = index_channel(
                channel_url.strip(),
                max_videos=int(max_videos),
                replace=replace_index,
                on_progress=on_progress,
            )
        st.session_state.active_collection_name = stats["collection_name"]
        st.session_state.last_index_stats = stats
        progress.progress(1.0, text="Done")
        parts = [
            f"Indexed **{stats['videos_indexed']}** videos "
            f"({stats['chunks_written']} chunks)."
        ]
        if stats.get("videos_skipped_no_transcript"):
            parts.append(
                f"No transcript: **{stats['videos_skipped_no_transcript']}**."
            )
        if stats.get("videos_skipped_invalid_id"):
            parts.append(
                f"Invalid/listing ID (use /@channel or /channel/UC…/videos): "
                f"**{stats['videos_skipped_invalid_id']}**."
            )
        if stats.get("videos_skipped_ip_blocked"):
            parts.append(
                f"YouTube IP block: **{stats['videos_skipped_ip_blocked']}** "
                "(wait and re-index, or set `YOUTUBE_COOKIES_PATH` in `.env`)."
            )
        st.success(" ".join(parts))
        if stats.get("error"):
            st.warning(stats["error"])
    except Exception as e:
        progress.progress(0.0, text="Failed")
        st.error(str(e))

with st.expander("Index details (debug)"):
    if st.session_state.last_index_stats:
        st.json(st.session_state.last_index_stats)
    if channel_url.strip():
        st.code(collection_name_for_channel(channel_url.strip()), language="text")

st.divider()
question = st.text_input("Your question")

if st.button("Ask") and question.strip():
    coll = st.session_state.active_collection_name
    if not coll:
        st.warning("Index a channel first.")
    else:
        try:
            with st.spinner("Searching…"):
                result = query_channel(question.strip(), coll)
            st.markdown("**Answer**")
            st.markdown(result.get("answer") or "")
            st.markdown("**Sources**")
            for url in result.get("sources") or []:
                st.markdown(f"- [{url}]({url})")
        except Exception as e:
            st.error(str(e))
