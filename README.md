# YouTube Channel RAG

Ask questions across a YouTube channel’s public transcripts using retrieval-augmented generation (RAG).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and set OPENAI_API_KEY
```

## Run

```bash
streamlit run app.py
```

Use a public channel URL (for example `https://www.youtube.com/@SomeChannel/videos`), index up to N videos, then ask questions.

Concepts and architecture: [docs/concepts-and-architecture.md](docs/concepts-and-architecture.md).

## Tests

```bash
pytest tests/ -q
```
