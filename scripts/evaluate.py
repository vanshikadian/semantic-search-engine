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

BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from backend.hf_cache import HF_DATASETS_CACHE, configure_hf_cache
from backend.service import SemanticSearchService

configure_hf_cache()

DEFAULT_OUTPUT = REPO_ROOT / "backend" / "faiss_index" / "eval_results.json"
DATASET_NAME = "microsoft/ms_marco"
DATASET_CONFIG = "v1.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate semantic search Recall@k on MS MARCO."
    )
    parser.add_argument(
        "--split",
        default="validation",
        help="MS MARCO split to evaluate against.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Maximum number of judged queries to evaluate.",
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[5, 10],
        help="Recall cutoffs to compute.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to write JSON evaluation results.",
    )
    return parser.parse_args()


def collect_eval_queries(split: str, limit: int) -> list[dict]:
    dataset = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        split=split,
        cache_dir=str(HF_DATASETS_CACHE),
    )
    queries: list[dict] = []

    for row in tqdm(dataset, desc="Collecting eval queries"):
        passages = row.get("passages") or {}
        passage_texts = passages.get("passage_text") or []
        is_selected = passages.get("is_selected") or []
        relevant_passages = []
        relevant_ids = set()
        for passage_text, selected in zip(passage_texts, is_selected):
            if int(selected) != 1:
                continue
            normalized = normalize_text(passage_text)
            if not normalized:
                continue
            relevant_passages.append(normalized)
            relevant_ids.add(synthetic_passage_id(normalized))

        query_text = str(row.get("query") or "").strip()
        if not query_text or not relevant_ids:
            continue

        queries.append(
            {
                "query_id": int(row["query_id"]),
                "query": query_text,
                "relevant_ids": sorted(relevant_ids),
                "relevant_passages": relevant_passages,
            }
        )
        if len(queries) >= limit:
            break

    return queries


def normalize_text(text: str) -> str:
    return " ".join(str(text).split()).strip()


def synthetic_passage_id(text: str) -> int:
    passage_hash = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return int(passage_hash[:12], 16)


def compute_recall_hits(results: list[dict], relevant_ids: set[int], k: int) -> int:
    returned_ids = {int(item["id"]) for item in results[:k]}
    return int(bool(returned_ids & relevant_ids))


def evaluate_queries(
    queries: list[dict],
    *,
    k_values: list[int],
    service: SemanticSearchService,
) -> dict:
    sorted_k = sorted(set(k_values))
    max_k = max(sorted_k)
    hit_counts = {k: 0 for k in sorted_k}
    per_query_results: list[dict] = []

    for item in tqdm(queries, desc="Evaluating queries"):
        relevant_ids = set(item["relevant_ids"])
        results = service.search(item["query"], k=max_k)

        recalls = {}
        for k in sorted_k:
            hit = compute_recall_hits(results, relevant_ids, k)
            hit_counts[k] += hit
            recalls[f"recall@{k}"] = hit

        per_query_results.append(
            {
                "query_id": item["query_id"],
                "query": item["query"],
                "relevant_ids": item["relevant_ids"],
                "top_results": results,
                **recalls,
            }
        )

    total = len(queries)
    aggregate = {
        f"recall@{k}": (hit_counts[k] / total if total else 0.0) for k in sorted_k
    }
    return {
        "evaluated_queries": total,
        "k_values": sorted_k,
        "aggregate": aggregate,
        "queries": per_query_results,
    }


def main() -> None:
    args = parse_args()
    service = SemanticSearchService()
    service.load_artifacts()
    if not service.ready:
        raise RuntimeError(
            "Search index artifacts are missing. Run scripts/build_index.py first."
        )

    queries = collect_eval_queries(args.split, args.limit)
    if not queries:
        raise RuntimeError("No evaluated queries found with judged relevant passages.")

    results = evaluate_queries(
        queries,
        k_values=args.k_values,
        service=service,
    )
    results["dataset"] = DATASET_NAME
    results["split"] = args.split
    results["stats"] = service.stats()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(orjson.dumps(results, option=orjson.OPT_INDENT_2))

    aggregate_summary = ", ".join(
        f"Recall@{k}={results['aggregate'][f'recall@{k}']:.3f}"
        for k in results["k_values"]
    )
    print(
        f"Evaluated {results['evaluated_queries']} queries on {args.split}. "
        f"{aggregate_summary}. Saved to {args.output}"
    )


if __name__ == "__main__":
    main()
