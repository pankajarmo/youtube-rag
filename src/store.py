from __future__ import annotations

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection

from src.config import CHROMA_PATH


def get_chroma_client() -> ClientAPI:
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_PATH))


def get_or_create_collection(name: str) -> Collection:
    client = get_chroma_client()
    return client.get_or_create_collection(name=name)


def delete_collection_if_exists(name: str) -> None:
    client = get_chroma_client()
    try:
        client.delete_collection(name)
    except Exception:
        pass


def get_collection(name: str) -> Collection | None:
    """Return collection if it exists; None otherwise."""
    client = get_chroma_client()
    try:
        return client.get_collection(name=name)
    except Exception:
        return None
