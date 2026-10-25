"""Qdrant: collections, upload of the vectors written by the embedding step, search.

Local (Docker) and cloud differ only by settings: GROWTH_RAG_QDRANT_URL and
GROWTH_RAG_QDRANT_API_KEY. GROWTH_RAG_QDRANT_PATH runs Qdrant embedded in the
Python process (no server), used by the tests.
"""
