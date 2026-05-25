from __future__ import annotations

import argparse
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import orjson
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.index import build_exact_index, save_index, save_metadata


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DATA_PATH = REPO_ROOT / "backend" / "data" / "passages.jsonl"
INDEX_DIR = REPO_ROOT / "backend" / "faiss_index"
INDEX_PATH = INDEX_DIR / "index.faiss"
METADATA_PATH = INDEX_DIR / "metadata.json"
STATE_PATH = INDEX_DIR / "build_state.json"
CHUNKS_DIR = INDEX_DIR / "chunks"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a FAISS index from passage embeddings."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DATA_PATH,
        help="Path to the source passages JSONL file.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Number of passages to embed per API call.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="Embedding model to use.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from saved chunk checkpoints if present.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Optional cap for local smoke tests.",
    )
    return parser.parse_args()


def load_records(input_path: Path, max_records: int | None = None) -> list[dict]:
    records: list[dict] = []
    with input_path.open("rb") as handle:
        for line in handle:
            records.append(orjson.loads(line))
            if max_records is not None and len(records) >= max_records:
                break
    return records


def load_state() -> dict:
    if not STATE_PATH.exists():
        return {"processed": 0, "chunks": [], "model": None, "batch_size": None}
    return orjson.loads(STATE_PATH.read_bytes())


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_bytes(orjson.dumps(state, option=orjson.OPT_INDENT_2))


def reset_state() -> None:
    if STATE_PATH.exists():
        STATE_PATH.unlink()


def clear_artifacts() -> None:
    if CHUNKS_DIR.exists():
        shutil.rmtree(CHUNKS_DIR)
    for artifact in (INDEX_PATH, METADATA_PATH, STATE_PATH):
        if artifact.exists():
            artifact.unlink()


def chunk_filename(chunk_index: int) -> str:
    return f"chunk_{chunk_index:05d}.npy"


def validate_resume_state(
    state: dict,
    *,
    batch_size: int,
    model: str,
) -> None:
    if state["model"] not in (None, model):
        raise RuntimeError(
            f"Checkpoint model mismatch: expected {model}, found {state['model']}"
        )
    if state["batch_size"] not in (None, batch_size):
        raise RuntimeError(
            f"Checkpoint batch size mismatch: expected {batch_size}, found {state['batch_size']}"
        )


def embed_and_checkpoint(records: list[dict], *, batch_size: int, model: str, resume: bool) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    if not resume:
        clear_artifacts()
        CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
        state = {
            "processed": 0,
            "chunks": [],
            "model": model,
            "batch_size": batch_size,
        }
    else:
        CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
        state = load_state()
        validate_resume_state(state, batch_size=batch_size, model=model)
        state["model"] = model
        state["batch_size"] = batch_size

    processed = int(state["processed"])
    total = len(records)
    if processed > total:
        raise RuntimeError(
            f"Checkpoint is ahead of input data: processed {processed}, total {total}"
        )
    total_batches = math.ceil(total / batch_size) if total else 0
    if processed >= total:
        return

    from backend.embed import embed_batch

    for start in tqdm(
        range(processed, total, batch_size),
        desc="Embedding batches",
        initial=processed // batch_size,
        total=total_batches,
    ):
        batch = records[start : start + batch_size]
        embeddings = embed_batch([item["text"] for item in batch], model=model)
        chunk_index = start // batch_size
        chunk_name = chunk_filename(chunk_index)
        np.save(CHUNKS_DIR / chunk_name, embeddings)

        state["processed"] = start + len(batch)
        if chunk_name not in state["chunks"]:
            state["chunks"].append(chunk_name)
        save_state(state)


def assemble_embeddings(records: list[dict], batch_size: int) -> np.ndarray:
    chunk_arrays: list[np.ndarray] = []
    expected_chunks = math.ceil(len(records) / batch_size) if records else 0

    for chunk_index in range(expected_chunks):
        chunk_path = CHUNKS_DIR / chunk_filename(chunk_index)
        if not chunk_path.exists():
            raise FileNotFoundError(f"Missing checkpoint chunk: {chunk_path}")
        chunk_arrays.append(np.load(chunk_path))

    if not chunk_arrays:
        raise RuntimeError("No checkpoint chunks found. Run embedding first.")

    return np.vstack(chunk_arrays).astype("float32")


def main() -> None:
    args = parse_args()
    records = load_records(args.input, max_records=args.max_records)
    if not records:
        raise RuntimeError(f"No records found in {args.input}")

    embed_and_checkpoint(
        records,
        batch_size=args.batch_size,
        model=args.model,
        resume=args.resume,
    )

    embeddings = assemble_embeddings(records, args.batch_size)
    if embeddings.shape[0] != len(records):
        raise RuntimeError(
            f"Embedding count mismatch: expected {len(records)}, found {embeddings.shape[0]}"
        )

    index = build_exact_index(embeddings)
    save_index(index, INDEX_PATH)
    save_metadata(
        [{"id": record["id"], "text": record["text"]} for record in records],
        METADATA_PATH,
    )

    print(
        f"Built index with {index.ntotal} passages using {args.model}. "
        f"Artifacts saved in {INDEX_DIR}"
    )


if __name__ == "__main__":
    main()
