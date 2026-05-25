from __future__ import annotations

import os
from typing import Sequence

import numpy as np

from backend.hf_cache import SENTENCE_TRANSFORMERS_CACHE, configure_hf_cache


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_MODEL_CACHE: dict[str, SentenceTransformer] = {}

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
configure_hf_cache()

from sentence_transformers import SentenceTransformer


def get_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> SentenceTransformer:
    if model_name not in _MODEL_CACHE:
        try:
            _MODEL_CACHE[model_name] = SentenceTransformer(
                model_name,
                cache_folder=str(SENTENCE_TRANSFORMERS_CACHE),
            )
        except Exception:
            _MODEL_CACHE[model_name] = SentenceTransformer(
                model_name,
                cache_folder=str(SENTENCE_TRANSFORMERS_CACHE),
                local_files_only=True,
            )
    return _MODEL_CACHE[model_name]


def embed_batch(
    texts: Sequence[str],
    *,
    model: str = DEFAULT_EMBEDDING_MODEL,
) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype="float32")

    encoder = get_model(os.environ.get("EMBEDDING_MODEL", model))
    embeddings = encoder.encode(
        list(texts),
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False,
    )
    return np.asarray(embeddings, dtype="float32")
