from __future__ import annotations

# Match PDF semantics: ~500 words per chunk, ~50 word stride (overlap)
DEFAULT_CHUNK_WORDS = 500
DEFAULT_OVERLAP_WORDS = 50


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_WORDS,
    overlap: int = DEFAULT_OVERLAP_WORDS,
) -> list[str]:
    """Split transcript into overlapping word windows."""
    words = text.split()
    if not words:
        return []
    stride = max(chunk_size - overlap, 1)
    chunks: list[str] = []
    for i in range(0, len(words), stride):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks
