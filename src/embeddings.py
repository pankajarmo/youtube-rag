from __future__ import annotations

from openai import OpenAI

from src.config import (
    MAX_EMBEDDING_INPUT_CHARS,
    OPENAI_EMBEDDING_MODEL,
    require_openai_key,
)


def _truncate(s: str) -> str:
    if len(s) <= MAX_EMBEDDING_INPUT_CHARS:
        return s
    return s[:MAX_EMBEDDING_INPUT_CHARS]


def embed_texts(
    texts: list[str],
    client: OpenAI | None = None,
    model: str | None = None,
) -> list[list[float]]:
    """Return embeddings for each string (same order). Batches one API call per chunk batch."""
    if not texts:
        return []
    cli = client or OpenAI(api_key=require_openai_key())
    m = model or OPENAI_EMBEDDING_MODEL
    inputs = [_truncate(t) for t in texts]
    # OpenAI allows multiple inputs per request; keep batches modest for payload size
    batch_size = 64
    out: list[list[float]] = []
    for i in range(0, len(inputs), batch_size):
        batch = inputs[i : i + batch_size]
        resp = cli.embeddings.create(model=m, input=batch)
        ordered = sorted(resp.data, key=lambda d: d.index)
        out.extend(d.embedding for d in ordered)
    return out


def embed_query(text: str, client: OpenAI | None = None) -> list[float]:
    return embed_texts([text], client=client)[0]
