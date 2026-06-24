"""Embedding subsystem — provider-agnostic interface and factory."""

from context_db.embeddings.embedder import DEFAULT_MODEL, Embedder, create_embedder

__all__ = ["DEFAULT_MODEL", "Embedder", "create_embedder"]
