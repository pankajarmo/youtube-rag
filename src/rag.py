from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from typing import Any

from openai import OpenAI

from src.chunking import chunk_text
from src.config import OPENAI_CHAT_MODEL, require_openai_key
from src.embeddings import embed_query, embed_texts
from src.store import (
    delete_collection_if_exists,
    get_collection,
    get_or_create_collection,
)
from src.transcripts import get_transcript_text
from src.youtube_channel import get_channel_video_entries


def normalize_channel_url(url: str) -> str:
    return url.strip()


def collection_name_for_channel(channel_url: str) -> str:
    normalized = normalize_channel_url(channel_url).lower()
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]
    return f"yt_rag_{digest}"


ProgressFn = Callable[[dict[str, Any]], None]


def index_channel(
    channel_url: str,
    max_videos: int = 100,
    *,
    replace: bool = False,
    on_progress: ProgressFn | None = None,
    throttle_seconds: float = 0.15,
) -> dict[str, Any]:
    """
    Fetch transcripts, chunk, embed, and store in a Chroma collection dedicated
    to this channel URL (hash-based name). If replace is True, drops any
    existing collection with that name first.
    """
    url = normalize_channel_url(channel_url)
    name = collection_name_for_channel(url)
    if replace:
        delete_collection_if_exists(name)
    collection = get_or_create_collection(name)
    client = OpenAI(api_key=require_openai_key())

    entries = get_channel_video_entries(url, playlist_max=max_videos)

    def prog(payload: dict[str, Any]) -> None:
        if on_progress:
            on_progress(payload)

    prog({"stage": "list", "total_videos": len(entries), "message": "Listed videos"})

    indexed_videos = 0
    skipped_no_transcript = 0
    total_chunks = 0

    for idx, entry in enumerate(entries):
        video_id = entry["id"]
        title = entry.get("title") or ""
        transcript = get_transcript_text(video_id)
        if not transcript:
            skipped_no_transcript += 1
            prog(
                {
                    "stage": "video",
                    "current": idx + 1,
                    "total": len(entries),
                    "video_id": video_id,
                    "message": "No transcript; skipped",
                }
            )
            continue

        chunks = chunk_text(transcript)
        if not chunks:
            skipped_no_transcript += 1
            continue

        embeddings = embed_texts(chunks, client=client)
        watch_url = f"https://www.youtube.com/watch?v={video_id}"
        ids = [f"{video_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "video_id": video_id,
                "url": watch_url,
                "title": title if isinstance(title, str) else "",
            }
            for _ in chunks
        ]

        collection.upsert(
            documents=chunks,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas,
        )
        indexed_videos += 1
        total_chunks += len(chunks)
        prog(
            {
                "stage": "video",
                "current": idx + 1,
                "total": len(entries),
                "video_id": video_id,
                "chunks": len(chunks),
                "message": f"Indexed {len(chunks)} chunks",
            }
        )
        if throttle_seconds > 0:
            time.sleep(throttle_seconds)

    return {
        "collection_name": name,
        "channel_url": url,
        "videos_listed": len(entries),
        "videos_indexed": indexed_videos,
        "videos_skipped_no_transcript": skipped_no_transcript,
        "chunks_written": total_chunks,
    }


def query_channel(
    question: str,
    collection_name: str,
    *,
    n_results: int = 5,
    client: OpenAI | None = None,
) -> dict[str, Any]:
    """Retrieve top chunks and generate a grounded answer with source URLs."""
    cli = client or OpenAI(api_key=require_openai_key())
    collection = get_collection(collection_name)
    if collection is None:
        return {
            "answer": "No index found for this channel. Index the channel first.",
            "sources": [],
            "source_titles": [],
        }
    q_embedding = embed_query(question, client=cli)
    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=n_results,
        include=["documents", "metadatas"],
    )
    docs = (results.get("documents") or [[]])[0]
    metas = (results.get("metadatas") or [[]])[0]
    if not docs:
        return {
            "answer": "No indexed content found for this channel. Index the channel first.",
            "sources": [],
            "source_titles": [],
        }

    context = "\n\n".join(docs)
    sources: list[str] = []
    titles: list[str] = []
    for m in metas:
        if not m:
            continue
        u = m.get("url")
        if u and u not in sources:
            sources.append(u)
        t = m.get("title")
        if t and t not in titles:
            titles.append(str(t))

    prompt = f"""Answer the question using ONLY the YouTube transcript context below.
Include which video(s) the information came from when possible (use titles or URLs from metadata implied by the context).
If the context does not contain enough information, say you do not know from the indexed transcripts.

Context:
{context}

Question: {question}

Answer:"""

    response = cli.chat.completions.create(
        model=OPENAI_CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    answer = response.choices[0].message.content or ""
    return {
        "answer": answer,
        "sources": list(dict.fromkeys(sources)),
        "source_titles": titles,
    }
