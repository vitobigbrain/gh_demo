"""
FAISS Index Management Module

Provides reusable functions for:
- Creating and managing FAISS indices
- Embedding texts and adding to index
- Searching with filtering and sorting
- Persistence (load/save index and metadata)

Supports both legacy string-based metadata and new structured metadata.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import faiss
import numpy as np

from embedding import local_embedding
from document_process import chunk_pdf_texts, DocumentChunk

# ========= Default Paths =========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_FILE_PATH = os.path.join(BASE_DIR, "Speech and Language Processing.pdf")
OUT_DIR = os.path.join(BASE_DIR, "faiss_out")
INDEX_FILE_PATH = os.path.join(OUT_DIR, "slp.index.faiss")
META_FILE_PATH = os.path.join(OUT_DIR, "slp.meta.json")

# ========= Config =========
EMBEDDING_DIM = 1024
DEFAULT_TOP_K = 10
BATCH_SIZE = 25


# ========= Metadata Types =========
# New structured metadata format
ChunkMeta = Dict[str, Any]  # {text, source_file, page, chunk_index}
MetaStore = Dict[int, ChunkMeta]

# Legacy metadata format (for backward compatibility)
LegacyMetaStore = Dict[int, str]


def _batched(items: List[str], batch_size: int) -> Iterable[List[str]]:
    """Yield batches of items."""
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def _embed_texts(texts: List[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
    """
    Embed texts using the local embedding service.
    
    Returns: float32 matrix of shape (n, EMBEDDING_DIM)
    """
    vectors: List[np.ndarray] = []
    for batch in _batched(texts, batch_size=batch_size):
        try:
            batch_vecs = local_embedding(batch)
        except Exception as e:
            raise RuntimeError(
                "Embedding request failed. Check `EMBEDDING_URL` in `embedding.py` "
                "and that the service is reachable."
            ) from e

        arr = np.asarray(batch_vecs, dtype="float32")
        if arr.ndim != 2 or arr.shape[1] != EMBEDDING_DIM:
            raise ValueError(f"Expected embeddings shape (n, {EMBEDDING_DIM}), got {arr.shape}")
        vectors.append(arr)

    if not vectors:
        return np.zeros((0, EMBEDDING_DIM), dtype="float32")

    out = np.vstack(vectors).astype("float32", copy=False)
    return out


def _normalize(vectors: np.ndarray) -> np.ndarray:
    """L2 normalize vectors in-place for cosine similarity."""
    faiss.normalize_L2(vectors)
    return vectors


# ========= Index Creation =========

def create_index(dim: int = EMBEDDING_DIM) -> faiss.Index:
    """
    Create a new FAISS index for cosine similarity.
    
    Uses Inner Product on L2-normalized vectors with ID mapping.
    """
    base = faiss.IndexFlatIP(dim)
    return faiss.IndexIDMap2(base)


# ========= Index Persistence =========

def load_index(index_path: str) -> faiss.Index:
    """Load a FAISS index from disk."""
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index file not found: {index_path}")
    return faiss.read_index(index_path)


def save_index(index: faiss.Index, index_path: str) -> None:
    """Save a FAISS index to disk."""
    os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
    faiss.write_index(index, index_path)


# ========= Metadata Persistence =========

def load_meta(meta_path: str) -> MetaStore:
    """
    Load metadata from disk.
    
    Handles both legacy format (id -> text string) and new format (id -> dict).
    """
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    
    meta: MetaStore = {}
    for k, v in raw.items():
        int_key = int(k)
        if isinstance(v, str):
            # Legacy format: convert to new format
            meta[int_key] = {
                "text": v,
                "source_file": "unknown",
                "page": None,
                "chunk_index": int_key,
            }
        else:
            # New format
            meta[int_key] = v
    return meta


def save_meta(meta_path: str, meta: MetaStore) -> None:
    """Save metadata to disk."""
    os.makedirs(os.path.dirname(meta_path) or ".", exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in meta.items()}, f, ensure_ascii=False, indent=2)


# ========= Index Building =========

def add_chunks(
    index: faiss.Index,
    meta: MetaStore,
    chunks: List[DocumentChunk],
) -> Tuple[faiss.Index, MetaStore]:
    """
    Add DocumentChunks to the index with structured metadata.
    
    Args:
        index: FAISS index to add to
        meta: Metadata store to update
        chunks: List of DocumentChunk objects
    
    Returns:
        Updated (index, meta) tuple
    """
    if not chunks:
        return index, meta

    texts = [c.text for c in chunks]
    vectors = _embed_texts(texts)
    _normalize(vectors)

    start_id = (max(meta.keys()) + 1) if meta else 0
    ids = np.arange(start_id, start_id + len(chunks)).astype("int64")

    index.add_with_ids(vectors, ids)
    
    for i, chunk in zip(ids.tolist(), chunks):
        meta[i] = {
            "text": chunk.text,
            "source_file": chunk.source_file,
            "page": chunk.page,
            "chunk_index": chunk.chunk_index,
        }

    return index, meta


def add_texts(
    index: faiss.Index,
    meta: Union[MetaStore, LegacyMetaStore],
    texts: List[str],
) -> Tuple[faiss.Index, MetaStore]:
    """
    Add plain text strings to the index (backward compatible).
    
    Args:
        index: FAISS index to add to
        meta: Metadata store to update
        texts: List of text strings
    
    Returns:
        Updated (index, meta) tuple
    """
    if not texts:
        return index, meta

    vectors = _embed_texts(texts)
    _normalize(vectors)

    # Convert legacy meta to new format if needed
    new_meta: MetaStore = {}
    for k, v in meta.items():
        if isinstance(v, str):
            new_meta[k] = {
                "text": v,
                "source_file": "unknown",
                "page": None,
                "chunk_index": k,
            }
        else:
            new_meta[k] = v

    start_id = (max(new_meta.keys()) + 1) if new_meta else 0
    ids = np.arange(start_id, start_id + len(texts)).astype("int64")

    index.add_with_ids(vectors, ids)
    
    for i, t in zip(ids.tolist(), texts):
        new_meta[i] = {
            "text": t,
            "source_file": "unknown",
            "page": None,
            "chunk_index": int(i),
        }

    return index, new_meta


def build_or_update_index(
    chunks: List[DocumentChunk],
    existing_index: Optional[faiss.Index] = None,
    existing_meta: Optional[MetaStore] = None,
    dim: int = EMBEDDING_DIM,
) -> Tuple[faiss.Index, MetaStore]:
    """
    Build a new index or update an existing one with new chunks.
    
    Args:
        chunks: List of DocumentChunk objects to add
        existing_index: Optional existing index to update
        existing_meta: Optional existing metadata to update
        dim: Embedding dimension (only used if creating new index)
    
    Returns:
        (index, meta) tuple
    """
    if existing_index is None:
        index = create_index(dim=dim)
        meta: MetaStore = {}
    else:
        index = existing_index
        meta = existing_meta or {}
    
    return add_chunks(index, meta, chunks)


# ========= Search =========

def search(
    index: faiss.Index,
    meta: MetaStore,
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Search the index for similar chunks.
    
    Args:
        index: FAISS index to search
        meta: Metadata store
        query: Query string
        top_k: Maximum number of results to return
        score_threshold: Minimum similarity score (0-1 for cosine)
    
    Returns:
        List of results sorted by score desc, each containing:
        - id: chunk ID
        - score: similarity score
        - text: chunk text
        - source_file: source filename
        - page: page number (if applicable)
        - chunk_index: index within source
    """
    q_vec = _embed_texts([query])
    _normalize(q_vec)

    scores, ids = index.search(q_vec, top_k)
    
    results: List[Dict[str, Any]] = []
    for score, _id in zip(scores[0].tolist(), ids[0].tolist()):
        if _id == -1:
            continue
        
        # Apply score threshold filter
        if score_threshold is not None and score < score_threshold:
            continue
        
        chunk_meta = meta.get(int(_id), {})
        
        # Handle both legacy and new metadata formats
        if isinstance(chunk_meta, str):
            text = chunk_meta
            source_file = "unknown"
            page = None
            chunk_index = int(_id)
        else:
            text = chunk_meta.get("text", "")
            source_file = chunk_meta.get("source_file", "unknown")
            page = chunk_meta.get("page")
            chunk_index = chunk_meta.get("chunk_index", int(_id))
        
        results.append({
            "id": int(_id),
            "score": float(score),
            "text": text,
            "source_file": source_file,
            "page": page,
            "chunk_index": chunk_index,
        })
    
    # Results are already sorted by score desc from FAISS
    return results


# ========= Convenience Functions =========

def get_index_stats(index: faiss.Index, meta: MetaStore) -> Dict[str, Any]:
    """Get statistics about the index."""
    source_files = set()
    for m in meta.values():
        if isinstance(m, dict):
            source_files.add(m.get("source_file", "unknown"))
    
    return {
        "total_vectors": index.ntotal,
        "total_chunks": len(meta),
        "source_files": list(source_files),
        "num_sources": len(source_files),
    }


def clear_index(
    index_path: str = INDEX_FILE_PATH,
    meta_path: str = META_FILE_PATH,
) -> None:
    """Delete index and metadata files."""
    if os.path.exists(index_path):
        os.remove(index_path)
    if os.path.exists(meta_path):
        os.remove(meta_path)


# ========= Demo Main =========

def main() -> None:
    """Demo: Build index from PDF and run a search query."""
    # Check if PDF exists
    if not os.path.exists(PDF_FILE_PATH):
        raise FileNotFoundError(f"PDF not found: {PDF_FILE_PATH}")

    # Load existing or create new index
    if os.path.exists(INDEX_FILE_PATH) and os.path.exists(META_FILE_PATH):
        print("加载已存在的 index...")
        index = load_index(INDEX_FILE_PATH)
        meta = load_meta(META_FILE_PATH)
    else:
        print("创建新的 index...")
        # PDF -> chunks (using legacy function for backward compatibility)
        texts = chunk_pdf_texts(PDF_FILE_PATH)
        if not texts:
            raise RuntimeError("No text chunks extracted from PDF.")

        # Create new index
        index = create_index(dim=EMBEDDING_DIM)
        meta: MetaStore = {}

        # Embed and add
        index, meta = add_texts(index=index, meta=meta, texts=texts)
        
        # Save
        os.makedirs(OUT_DIR, exist_ok=True)
        save_index(index, INDEX_FILE_PATH)
        save_meta(META_FILE_PATH, meta)

    # Search demo
    query = "统计语言模型"
    results = search(index=index, meta=meta, query=query, top_k=DEFAULT_TOP_K)

    print("== Build done ==")
    print(f"PDF: {PDF_FILE_PATH}")
    print(f"Index saved: {INDEX_FILE_PATH}")
    print(f"Meta saved: {META_FILE_PATH}")
    print(f"Indexed vectors: {index.ntotal}")
    print("")
    print("== Search demo ==")
    print(f"Query: {query}")
    for r in results:
        print(f"--------------------------------")
        print(f"Score: {r['score']:.4f} | Source: {r['source_file']} | Page: {r['page']}")
        print(f"{r['text']}\n")


if __name__ == "__main__":
    main()
