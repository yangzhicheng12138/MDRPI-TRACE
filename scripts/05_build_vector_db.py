#!/usr/bin/env python3
"""Build the Qdrant development collection from audited chunks."""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from common import add_project_to_import_path, configure_logging, project_config_path, sha256_file

PROJECT_ROOT = add_project_to_import_path()

from app.config import AppConfig, ensure_output_dirs  # noqa: E402
from app.embedding import get_embedding_service  # noqa: E402
from app.vector_store import VectorStore  # noqa: E402
from app.corpus import AuditedCorpus, PAYLOAD_FIELDS, load_audited_chunks  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=project_config_path())
    parser.add_argument("--rebuild", action="store_true", help="Delete and recreate the development collection")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report-only", action="store_true", help="Refresh counts without embedding or upserting")
    parser.add_argument("--output-dir", type=Path, default=None, help="Separate build logs from preprocessing audit artifacts")
    return parser.parse_args()


def _payload(chunk: dict, audit: dict) -> dict:
    payload = {field: chunk.get(field) for field in PAYLOAD_FIELDS}
    payload["include_in_vector"] = True
    # ``include_in_retrieval`` is a stricter, reversible policy layered on top
    # of the original vector eligibility flag.  Older audit files do not have
    # it, so default to the historical behaviour for backward compatibility.
    payload["include_in_retrieval"] = str(audit.get("include_in_retrieval", "true")).lower() == "true"
    payload["audit_notes"] = audit.get("audit_notes", "")
    payload["content_type_suspected"] = audit.get("content_type_suspected", "")
    payload["retrieval_exclusion_reasons"] = audit.get("retrieval_exclusion_reasons", "")
    return payload


def _load_chunks(config: AppConfig) -> tuple[list[dict], dict[str, dict], int, int]:
    return load_audited_chunks(config)


def verify_index(store, corpus):
    seen, dimension = set(), None
    for point in store.iter_points(with_vectors=True):
        payload = dict(point.payload or {})
        corpus.verify_payload(payload)
        chunk_id = payload["chunk_id"]
        if chunk_id in seen or str(point.id) != store.point_id(chunk_id):
            raise ValueError(f"Unexpected index point identity: {chunk_id}")
        seen.add(chunk_id)
        vector = point.vector
        if not isinstance(vector, list) or not vector or not all(math.isfinite(value) for value in vector):
            raise ValueError(f"Invalid vector: {chunk_id}")
        dimension = dimension or len(vector)
        if len(vector) != dimension:
            raise ValueError("Inconsistent vector dimensions")
    if seen != set(corpus.chunks):
        raise ValueError("Index must contain exactly the admitted chunks; use an independent full rebuild")
    corpus.assert_current()
    return dimension


def build(config: AppConfig, args: argparse.Namespace, logger) -> dict:
    started = time.monotonic()
    if args.limit is not None and config.get("retrieval", "require_audited_index", default=False) and not args.dry_run:
        raise ValueError("Partial builds cannot be published as an audited index")
    all_chunks, audit_by_id, raw_total, excluded_by_audit = _load_chunks(config)
    total_selected = len(all_chunks)
    chunks = all_chunks[: args.limit] if args.limit is not None and args.limit >= 0 else all_chunks
    report = {
        "chunk_total": raw_total,
        "eligible_chunk_total": total_selected,
        "chunk_selected": len(chunks),
        "entered_vector_db": 0,
        "skipped": excluded_by_audit + total_selected - len(chunks),
        "excluded_by_audit": excluded_by_audit,
        "failed": 0,
        "embedding_model": str(config.get("embedding", "model", default="")),
        "vector_dimension": None,
        "batch_size": int(args.batch_size or config.get("embedding", "batch_size", default=16)),
        "device": str(config.get("embedding", "device", default="auto")),
        "collection": config.qdrant_collection,
        "dry_run": bool(args.dry_run),
        "elapsed_seconds": 0.0,
        "started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
    }
    if args.dry_run or args.report_only:
        if args.report_only:
            store = VectorStore(config, logger=logger)
            try:
                corpus = AuditedCorpus(config)
                report["vector_dimension"] = verify_index(store, corpus)
                corpus.verify_release(store)
                report.update(point_count=store.count(), verified_existing=True, verification_only=True)
            finally:
                store.close()
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        return report
    if not chunks:
        raise RuntimeError("No chunks selected for vectorization")
    embedder = get_embedding_service(config, logger=logger)
    store = VectorStore(config, logger=logger)
    batch_size = max(1, report["batch_size"])
    try:
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            vectors = embedder.embed_documents([str(item.get("text", "")) for item in batch], batch_size=batch_size)
            if report["vector_dimension"] is None:
                report["vector_dimension"] = len(vectors[0]) if vectors else 0
                store.ensure_collection(report["vector_dimension"], recreate=args.rebuild and start == 0)
            store.upsert(
                vectors=vectors,
                payloads=[_payload(item, audit_by_id.get(str(item.get("chunk_id", "")), {})) for item in batch],
                chunk_ids=[str(item.get("chunk_id", "")) for item in batch],
            )
            report["entered_vector_db"] += len(batch)
            logger.info("Upserted %s/%s chunks", report["entered_vector_db"], len(chunks))
        report["point_count"] = store.count()
        if len(chunks) == total_selected:
            corpus = AuditedCorpus(config)
            report["vector_dimension"] = verify_index(store, corpus)
            report["verified_complete"] = True
            release = {"identity": corpus.identity, "verified_complete": True,
                       "vector_dimension": report["vector_dimension"],
                       "built_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()}
            release_path = config.path("qdrant", "release_manifest", default=str(config.qdrant_path / "release.json"))
            from common import write_text_atomic
            write_text_atomic(release_path, json.dumps(release, ensure_ascii=False, indent=2))
    finally:
        store.close()
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    report["finished_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    return report


def main() -> int:
    args = parse_args()
    config = AppConfig.from_file(args.config)
    ensure_output_dirs(config)
    output_dir = args.output_dir or config.logs_dir
    logger = configure_logging(output_dir / "vector_build.log", "mdrpi.vector_build")
    report = build(config, args, logger)
    report_path = output_dir / "vector_build_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
