#!/usr/bin/env python3
"""Search the development knowledge base and print traceable evidence."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from common import add_project_to_import_path, configure_logging, project_config_path

PROJECT_ROOT = add_project_to_import_path()

from app.config import AppConfig  # noqa: E402
from app.reranker import Reranker  # noqa: E402
from app.retriever import Retriever  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=project_config_path())
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--population", default=None, help="Comma-separated controlled population values")
    parser.add_argument("--device-type", default=None, help="Comma-separated controlled device values")
    parser.add_argument("--source-type", default=None, help="Comma-separated source types")
    parser.add_argument("--evidence-tier", default=None)
    parser.add_argument("--review-status", default=None)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--no-filter", action="store_true")
    parser.add_argument("--no-dual", action="store_true", help="Use one vector route instead of clinical/research dual retrieval")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    return parser.parse_args()


def _values(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _print_result(item) -> None:
    print(f"[{item.rank}] {item.chunk_id} ({item.doc_id})")
    print(f"标题: {item.title}")
    print(f"年份: {item.year} | 来源类型: {item.source_type} | 证据等级: {item.evidence_tier}")
    print(f"人群: {','.join(item.population)} | 器械: {','.join(item.device_type)} | 主题: {','.join(item.topic)}")
    print(f"章节: {item.section} | 页码: {item.page_start}-{item.page_end} | page_verified={item.page_verified}")
    print(f"vector_score={item.vector_score:.6f} | reranker_score={item.reranker_score if item.reranker_score is not None else ''}")
    print(f"原文: {item.text}")
    print("-" * 80)


def main() -> int:
    args = parse_args()
    config = AppConfig.from_file(args.config)
    logger = configure_logging(config.logs_dir / "search.log", "mdrpi.search")
    population = _values(args.population)
    device_type = _values(args.device_type)
    source_type = _values(args.source_type)
    evidence_tier = _values(args.evidence_tier)
    review_status = _values(args.review_status)
    filters = {
        "population": population,
        "device_type": device_type,
        "source_type": source_type,
        "evidence_tier": evidence_tier,
        "review_status": review_status,
        "use_filters": not args.no_filter,
    }
    retriever = Retriever(config, logger=logger)
    top_k = int(args.top_k or config.get("retrieval", "vector_top_k", default=20))
    if args.no_dual:
        candidates = retriever.retrieve(args.query, top_k=top_k, **filters)
    else:
        candidates = retriever.dual_retrieve(args.query, top_k=top_k, **filters)
    results = candidates
    if not args.no_rerank:
        reranker = Reranker(config, logger=logger)
        results = reranker.rerank(args.query, candidates, top_k=int(args.top_k or config.get("reranker", "top_k", default=8)))
    print(f"Query: {args.query}")
    print(f"Applied filters: {json.dumps(filters, ensure_ascii=False)}")
    print(f"Collection: {config.qdrant_collection} | Results: {len(results)}")
    for result in results:
        _print_result(result)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "query": args.query,
            "filters": filters,
            "collection": config.qdrant_collection,
            "results": [item.to_dict() for item in results],
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"JSON: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
