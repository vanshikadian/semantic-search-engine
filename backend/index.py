from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np
import orjson


INDEX_TYPE = "IndexFlatIP"


def normalize_embeddings(vectors: np.ndarray) -> np.ndarray:
    normalized = np.asarray(vectors, dtype="float32").copy()
    if normalized.size == 0:
        return normalized
    faiss.normalize_L2(normalized)
    return normalized


def build_exact_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    if vectors.ndim != 2 or vectors.shape[0] == 0:
        raise ValueError("Expected a non-empty 2D embeddings array.")

    normalized = normalize_embeddings(vectors)
    index = faiss.IndexFlatIP(normalized.shape[1])
    index.add(normalized)
    return index


def save_index(index: faiss.Index, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output_path))


def load_index(index_path: Path) -> faiss.Index:
    return faiss.read_index(str(index_path))


def load_metadata(metadata_path: Path) -> dict[str, dict]:
    return orjson.loads(metadata_path.read_bytes())


def save_metadata(metadata: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        str(position): item for position, item in enumerate(metadata)
    }
    output_path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))


def search_index(index: faiss.Index, query_vector: np.ndarray, k: int):
    query = normalize_embeddings(query_vector)
    return index.search(query, k)
