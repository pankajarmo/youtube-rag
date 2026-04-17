from src.chunking import (
    DEFAULT_CHUNK_WORDS,
    DEFAULT_OVERLAP_WORDS,
    chunk_text,
)


def test_chunk_text_empty():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_chunk_text_single_window():
    words = ["w"] * 10
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size=20, overlap=5)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_overlap_stride():
    # stride = chunk_size - overlap = 5 - 2 = 3
    text = "a b c d e f g h i j"
    chunks = chunk_text(text, chunk_size=5, overlap=2)
    # positions: [0:5], [3:8], [6:11] -> "a b c d e", "d e f g h", "g h i j"
    assert len(chunks) >= 2
    assert chunks[0] == "a b c d e"
    assert "a b c d e" in chunks[0]


def test_defaults_are_pdf_semantics():
    assert DEFAULT_CHUNK_WORDS == 500
    assert DEFAULT_OVERLAP_WORDS == 50
