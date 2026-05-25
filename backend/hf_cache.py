from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
HF_CACHE_ROOT = REPO_ROOT / ".hf_cache"
HF_HOME = HF_CACHE_ROOT / "home"
HF_DATASETS_CACHE = HF_CACHE_ROOT / "datasets"
HF_HUB_CACHE = HF_CACHE_ROOT / "hub"
SENTENCE_TRANSFORMERS_CACHE = HF_CACHE_ROOT / "sentence_transformers"


def configure_hf_cache() -> None:
    HF_HOME.mkdir(parents=True, exist_ok=True)
    HF_DATASETS_CACHE.mkdir(parents=True, exist_ok=True)
    HF_HUB_CACHE.mkdir(parents=True, exist_ok=True)
    SENTENCE_TRANSFORMERS_CACHE.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", str(HF_HOME))
    os.environ.setdefault("HF_DATASETS_CACHE", str(HF_DATASETS_CACHE))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_HUB_CACHE))
    os.environ.setdefault(
        "SENTENCE_TRANSFORMERS_HOME", str(SENTENCE_TRANSFORMERS_CACHE)
    )
