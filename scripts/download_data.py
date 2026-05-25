from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import orjson
from datasets import load_dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.hf_cache import HF_DATASETS_CACHE, configure_hf_cache

configure_hf_cache()

DEFAULT_OUTPUT = REPO_ROOT / "backend" / "data" / "passages.jsonl"
DATASET_NAME = "microsoft/ms_marco"
DATASET_CONFIG = "v1.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a filtered MS MARCO passage subset."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100_000,
        help="Maximum number of passages to save after filtering.",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=20,
        help="Minimum character length for retained passages.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output JSONL path.",
    )
    return parser.parse_args()


def iter_passages(limit: int, min_length: int):
    dataset = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        split="train",
        cache_dir=str(HF_DATASETS_CACHE),
    )
    seen_hashes: set[str] = set()
    kept = 0

    for row in tqdm(dataset, desc="Scanning rows"):
        passages = row.get("passages") or {}
        texts = passages.get("passage_text") or []

        for text in texts:
            normalized = " ".join(str(text).split()).strip()
            if len(normalized) < min_length:
                continue

            passage_hash = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
            if passage_hash in seen_hashes:
                continue

            seen_hashes.add(passage_hash)
            synthetic_id = int(passage_hash[:12], 16)
            yield {"id": synthetic_id, "text": normalized}
            kept += 1

            if kept >= limit:
                return


def write_jsonl(records, output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with output_path.open("wb") as handle:
        for record in records:
            handle.write(orjson.dumps(record))
            handle.write(b"\n")
            count += 1

    return count


def main() -> None:
    args = parse_args()
    count = write_jsonl(
        iter_passages(limit=args.limit, min_length=args.min_length),
        args.output,
    )
    print(f"Wrote {count} passages to {args.output}")


if __name__ == "__main__":
    main()
