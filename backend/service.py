from __future__ import annotations

import os
from pathlib import Path

from backend.embed import DEFAULT_EMBEDDING_MODEL, embed_batch, get_model
from backend.index import INDEX_TYPE, load_index, load_metadata, search_index


BASE_DIR = Path(__file__).resolve().parent
INDEX_PATH = BASE_DIR / "faiss_index" / "index.faiss"
METADATA_PATH = BASE_DIR / "faiss_index" / "metadata.json"


class SemanticSearchService:
    def __init__(
        self,
        *,
        index_path: Path = INDEX_PATH,
        metadata_path: Path = METADATA_PATH,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.embedding_model = embedding_model
        self.index = None
        self.metadata: dict[str, dict] = {}
        self.model_loaded = False

    @property
    def ready(self) -> bool:
        return self.index is not None and bool(self.metadata)

    @property
    def artifacts_present(self) -> bool:
        return self.index_path.exists() and self.metadata_path.exists()

    def load_artifacts(self) -> None:
        if not self.artifacts_present:
            self.index = None
            self.metadata = {}
            return

        self.index = load_index(self.index_path)
        self.metadata = load_metadata(self.metadata_path)

    def load_model(self) -> None:
        get_model(os.environ.get("EMBEDDING_MODEL", self.embedding_model))
        self.model_loaded = True

    def warmup(self, *, preload_model: bool = False) -> None:
        self.load_artifacts()
        if preload_model:
            self.load_model()

    def search(self, query: str, *, k: int = 10) -> list[dict]:
        if not self.ready:
            self.load_artifacts()
        if not self.ready:
            raise RuntimeError("Search index is not ready. Build the FAISS artifacts first.")

        query_embedding = embed_batch([query], model=self.embedding_model)
        self.model_loaded = True
        scores, indices = search_index(self.index, query_embedding, k)

        results: list[dict] = []
        for score, faiss_id in zip(scores[0], indices[0]):
            if faiss_id < 0:
                continue

            record = self.metadata.get(str(int(faiss_id)))
            if not record:
                continue

            results.append(
                {
                    "id": int(record["id"]),
                    "text": str(record["text"]),
                    "score": float(score),
                }
            )

        return results

    def stats(self) -> dict:
        if not self.ready:
            self.load_artifacts()
        return {
            "total_passages": len(self.metadata),
            "embedding_model": self.embedding_model,
            "index_type": INDEX_TYPE,
            "index_loaded": self.ready,
            "artifacts_present": self.artifacts_present,
            "model_loaded": self.model_loaded,
        }
