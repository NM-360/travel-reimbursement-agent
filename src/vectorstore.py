"""
Persistent ChromaDB vector store for policy retrieval.

Rules are embedded with `nomic-embed-text` (via our own Ollama client, so the
embeddings match everything else in the app) and stored in a persistent Chroma
collection on disk. We pass embeddings to Chroma explicitly rather than relying
on a Chroma embedding-function, which keeps this robust across Chroma versions.

The collection name includes an 8-char hash of the policy content, so editing
`travel_policy.md` automatically produces a fresh collection — no manual
re-index step needed.
"""
from __future__ import annotations

import hashlib
import os
from typing import Any

import chromadb

import llm

_CHROMA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".chroma")

_client: Any = None
_collections: dict[str, Any] = {}


def _get_client() -> Any:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=_CHROMA_DIR)
    return _client


def _content_hash(rules: list[dict[str, str]]) -> str:
    blob = "|".join(f"{r['policy_id']}:{r['text']}" for r in rules) + "|" + llm.OLLAMA_EMBED_MODEL
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:8]


def get_collection(rules: list[dict[str, str]]) -> Any:
    """
    Return a Chroma collection populated with the policy rules, building/persisting
    it on first use. Raises llm.LLMError if embeddings can't be produced.
    """
    name = f"policy_rules_{_content_hash(rules)}"
    if name in _collections:
        return _collections[name]

    client = _get_client()
    # Cosine space so query distances map cleanly to similarity = 1 - distance.
    coll = client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})

    if coll.count() < len(rules):
        embeddings = [llm.embed(f"{r['policy_id']}: {r['text']}") for r in rules]
        coll.upsert(
            ids=[r["policy_id"] for r in rules],
            embeddings=embeddings,
            documents=[r["text"] for r in rules],
            metadatas=[{"policy_id": r["policy_id"]} for r in rules],
        )

    _collections[name] = coll
    return coll


def query(rules: list[dict[str, str]], text: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Semantic search the policy collection. Returns [{policy_id, text, score}]."""
    coll = get_collection(rules)
    q_vec = llm.embed(text or "")
    res = coll.query(query_embeddings=[q_vec], n_results=min(top_k, len(rules)))

    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    dists = res.get("distances", [[]])[0]
    out = []
    for pid, doc, dist in zip(ids, docs, dists):
        out.append({"policy_id": pid, "text": doc, "score": round(1.0 - float(dist), 4)})
    return out
