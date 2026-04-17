# YouTube Channel RAG — concepts and architecture

This document explains the ideas behind **retrieval-augmented generation (RAG)** as used in this project, why each part of the stack was chosen, and how the pieces fit together end to end.

---

## 1. Problem framing

A YouTube channel is not one document: it is **hundreds of long videos**, often overlapping in topic, published over time. Naively pasting every transcript into a single LLM prompt is impossible (context limits, cost, latency) and would drown the model in irrelevant text.

**Channel-scale Q&A** therefore needs:

- A way to **find the small slices** of text that are most relevant to a question.
- A way to **compose an answer** from those slices while **staying tied to what was actually said** in the videos.

That is exactly the RAG pattern: **retrieve** candidate evidence, then **generate** an answer conditioned on that evidence.

---

## 2. RAG in one pass

### Retrieval

1. **Ingest**: split each transcript into **chunks** (overlapping windows of words).
2. **Embed**: turn each chunk into a **vector** (a list of numbers) using an embedding model. Vectors are positioned in space so that **semantically similar** text ends up **geometrically close**.
3. **Store**: save vectors in a **vector database** with metadata (video id, URL, title).

At query time:

4. **Embed the question** with the **same** embedding model.
5. **Nearest-neighbor search**: fetch the top **K** chunks whose vectors are closest to the question vector.

### Generation

6. Concatenate retrieved chunks into a **context** string.
7. Prompt an LLM to answer **using only that context**, and to **cite** which videos the information came from.

### Why grounding matters

Large language models can **hallucinate** plausible-sounding claims. RAG reduces that risk by **binding** the model to retrieved transcript text. The prompt explicitly instructs the model to stay within the provided context and to admit ignorance when the context is insufficient.

---

## 3. Data source: YouTube transcripts

This project uses **public captions / transcripts** (for example auto-generated captions) via the `youtube-transcript-api` library, which talks to the same caption tracks the YouTube player can show. **Video listing** uses **yt-dlp** in **flat** mode (metadata only, no video download).

### Why not scrape the watch page?

Transcripts are structured, clean text aligned to speech. Scraping HTML is brittle, heavier, and often unnecessary when captions exist.

### Limitations (important)

- **Not every video** has a retrievable transcript (disabled captions, restrictions, or API errors). Those videos are skipped.
- **Auto captions** can be wrong or noisy; answers inherit that noise.
- **Language**: transcripts are in whatever languages YouTube exposes; mixing languages in one index can affect retrieval quality unless you filter.

Always respect **YouTube/Google terms of use**, rate limits, and copyright. This tool is for **personal learning and research** on content you are allowed to access.

---

## 4. Chunking: overlapping word windows

Transcripts are long. Embeddings and retrieval work best on **paragraph-scale** units, not an entire hour-long video at once.

This repo uses **fixed-size word windows** with **overlap** (defaults aligned with the weekend-project PDF: on the order of **~500 words** per chunk with **~50 words** of overlap).

### Why overlap?

Without overlap, a sentence that sits exactly on a **boundary** between two chunks might be split awkwardly, hurting embedding quality and recall. Overlap gives neighboring chunks **shared context**, so at least one chunk is likely to contain a coherent phrasing of the idea.

### Tradeoffs

- **Smaller chunks**: more precise retrieval, but less surrounding context per hit.
- **Larger chunks**: more context per hit, but more irrelevant text inside each chunk, which can confuse generation or dilute relevance scores.

---

## 5. Embeddings: `text-embedding-3-small`

An **embedding model** maps text to a high-dimensional vector. Similar meaning (not identical wording) tends to map to **similar vectors**, which enables semantic search.

**`text-embedding-3-small`** is a practical default because it balances **cost**, **latency**, and **quality** for English-heavy Q&A over web-scale text like transcripts.

### Same model for documents and queries

The question and every stored chunk must be embedded with the **same model** (and typically the same preprocessing rules). Mixing models breaks the geometry: distances become meaningless.

### Truncation

Very long chunks can exceed embedding input limits. This project **truncates** inputs to a conservative character budget before calling the API (similar to the PDF’s shared utility pattern).

---

## 6. Vector database: ChromaDB (persistent)

**Chroma** is embedded in the app process, needs minimal setup, and is **Python-friendly**—ideal for a weekend MVP and portfolio demos.

We use a **`PersistentClient`** so indexes **survive restarts** (unlike an in-memory client). Data lives under `CHROMA_PATH` (default `./chroma_db`).

### Collection naming

Each **normalized channel URL** maps to a **dedicated collection name** derived from a **hash** of the URL. That **isolates** channels so you do not accidentally retrieve chunks from another channel when you switch URLs.

### When you might outgrow Chroma

- **Multi-user production** with strict isolation, backups, and monitoring.
- **Hybrid search** (vector + keyword/BM25) at large scale.
- **Managed** vector SaaS with SLAs.

Those are not required to learn RAG or ship a credible v1.

---

## 7. Retrieval configuration

The query step asks for the top **`n_results`** chunks (default **5** in line with the PDF).

### Failure modes

- **Too few results**: the model may not see enough evidence; answers become vague or “unknown” more often.
- **Too many / noisy results**: unrelated transcript snippets add confusion and can trigger **hallucination** or contradictions.

Tuning `n_results`, chunk size, and overlap is the main “knob set” for retrieval quality before moving to advanced reranking.

---

## 8. LLM choice: `gpt-4o-mini`

Here the LLM’s job is mostly **synthesis and citation** over **short** retrieved excerpts, not open-ended world knowledge. A **smaller / cheaper** chat model is appropriate: faster iterations, lower cost, still strong at following instructions like “only use the context.”

Upgrade to a larger model if you need **nuanced reasoning** over many conflicting clips, or richer summarization across long contexts (with corresponding cost controls).

---

## 9. Architecture choices (alternatives considered)

| Choice | Why here |
|--------|----------|
| **Streamlit UI** | Fastest path to a usable demo; matches the PDF and avoids frontend build tooling. |
| **Direct OpenAI SDK** (no LangChain) | Fewer abstractions, easier to read line-by-line in a portfolio repo. |
| **Chroma persistent** | Survives restarts; good enough for local/small deployments. |
| **Per-channel hashed collection** | Prevents cross-channel contamination without a complex schema. |
| **yt-dlp flat + transcript API** | Reliable listing + captions without downloading video files. |

Reasonable alternatives:

- **LangChain / LlamaIndex**: faster integration of loaders, splitters, and retrievers; adds dependency weight and “magic” you may not want while learning.
- **FastAPI + React**: better for multi-client production; more engineering than needed for v1.
- **Remote vector DB** (Pinecone, Weaviate, pgvector): better for teams and scale; more ops.

---

## 10. Ethics and terms of service

- Use this on **public** content you have rights or permission to analyze for your use case.
- Do not attempt to bypass **private**, **unlisted**, or **paid** restrictions.
- Be mindful of **API and provider rate limits** (OpenAI billing, YouTube behavior).

---

## 11. Roadmap (from the source project ideas)

Ideas that map naturally onto this codebase:

1. **Timestamps in citations**: chunk transcripts using `(text, start, end)` segments from the transcript API instead of a single flattened string.
2. **Date filters**: store `upload_date` in metadata from yt-dlp and filter Chroma queries with `where`.
3. **Playlist-first workflows**: treat playlist URLs as first-class (already supported if yt-dlp lists entries).
4. **Multi-channel comparison**: index two collections and run two retrievals into one prompt, or merge with a `channel_id` metadata filter.

---

## Appendix A — How to run

See the repository [README](../README.md): create a virtualenv, `pip install -r requirements.txt`, set `OPENAI_API_KEY`, run `streamlit run app.py`.

---

## Appendix B — Manual smoke checklist

Use a **small public playlist** (few videos, known transcripts) to keep cost and time low.

1. Set `OPENAI_API_KEY` in `.env` or the shell.
2. Run `streamlit run app.py`.
3. Paste the playlist URL, set **Max videos** to **3–5**, enable **Replace index**, click **Index channel**.
4. Confirm success message shows **videos_indexed > 0** and **chunks_written > 0**.
5. Ask a question that should be answerable from a known video title or topic.
6. Confirm **Sources** lists plausible `watch?v=` URLs.
7. Run `pytest tests/ -q` for automated chunking checks.

This validates listing, transcript fetch, embedding, Chroma writes, retrieval, and chat in one pass.
