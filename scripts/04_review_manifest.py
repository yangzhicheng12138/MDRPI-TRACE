#!/usr/bin/env python3
"""Create a non-destructive, machine-assisted manifest review file."""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from collections import Counter
from pathlib import Path

from common import add_project_to_import_path, configure_logging, project_config_path

PROJECT_ROOT = add_project_to_import_path()

from app.config import AppConfig, ensure_output_dirs  # noqa: E402


DEFERRED = {
    "DOC002": ("garbled extracted text", "highest", "obtain a clean digital PDF or run OCR, then re-parse and review"),
    "DOC041": ("garbled extracted text", "medium", "obtain a clean digital PDF or run OCR, then re-parse and review"),
    "DOC053": ("scanned/empty pages; OCR required", "highest", "run OCR on the original PDF, then re-parse and review"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=project_config_path())
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--audit", type=Path, default=None)
    return parser.parse_args()


def _chunk_counts(config: AppConfig, audit_path: Path | None) -> tuple[Counter[str], Counter[str]]:
    total: Counter[str] = Counter()
    included: Counter[str] = Counter()
    path = audit_path or config.logs_dir / "chunk_audit.csv"
    if path.exists():
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                total[row.get("doc_id", "")] += 1
                if row.get("include_in_vector", "").lower() == "true":
                    included[row.get("doc_id", "")] += 1
        return total, included
    if config.chunks_path.exists():
        with config.chunks_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    total[str(row.get("doc_id", ""))] += 1
                    included[str(row.get("doc_id", ""))] += 1
    return total, included


def _review_row(row: dict[str, str], chunk_count: int, included_count: int) -> dict[str, str]:
    doc_id = row.get("doc_id", "")
    existing = (row.get("review_status") or "").strip().lower()
    reasons: list[str] = []
    notes = (row.get("notes") or "").strip()
    if notes:
        reasons.extend(item for item in notes.split(";") if item)
    for field in ("title", "source_type", "evidence_tier", "population", "device_type", "topic"):
        if not (row.get(field) or "").strip():
            reasons.append(f"{field}_missing_verify")
    placeholders = {"标题", "作者", "title", "author", "unknown", "unk"}
    if (row.get("title") or "").strip().casefold() in {item.casefold() for item in placeholders}:
        reasons.append("title_placeholder_verify")
    if (row.get("authors") or "").strip().casefold() in {item.casefold() for item in placeholders}:
        reasons.append("authors_placeholder_verify")
    if row.get("parse_status", "") not in {"ok", ""}:
        reasons.append(f"parse_status={row.get('parse_status')}")
    if existing in {"verified", "excluded"}:
        status = existing
    else:
        status = "needs_review" if reasons else "auto"
    row = dict(row)
    row["review_status"] = status
    row["review_reasons"] = ";".join(dict.fromkeys(reasons))
    row["chunk_count"] = str(chunk_count)
    row["vector_chunk_count"] = str(included_count)
    row["review_priority"] = "highest" if doc_id in {"DOC002", "DOC053"} else "medium" if doc_id == "DOC041" else ("high" if row.get("evidence_tier") == "A" else "normal")
    return row


def review_manifest(config: AppConfig, manifest_path: Path | None, audit_path: Path | None, logger) -> list[dict[str, str]]:
    source = manifest_path or config.manifest_path
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        base_fields = list(reader.fieldnames or [])
    totals, included = _chunk_counts(config, audit_path)
    output_rows = [_review_row(row, totals[row.get("doc_id", "")], included[row.get("doc_id", "")]) for row in rows]
    tier_order = {"A": 0, "B": 1, "C": 2, "D": 3}
    status_order = {"needs_review": 0, "auto": 1, "verified": 2, "excluded": 3}
    output_rows.sort(key=lambda row: (tier_order.get(row.get("evidence_tier", "D"), 9), status_order.get(row.get("review_status", "needs_review"), 9), row.get("doc_id", "")))
    fields = base_fields + [field for field in ("review_reasons", "chunk_count", "vector_chunk_count", "review_priority") if field not in base_fields]
    output_path = config.manifest_path.parent / "literature_manifest_review.csv"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=output_path.parent, delete=False) as temporary:
        writer = csv.DictWriter(temporary, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in output_rows)
        temporary_path = Path(temporary.name)
    temporary_path.replace(output_path)

    deferred_rows = []
    by_id = {row.get("doc_id", ""): row for row in rows}
    for doc_id, (problem, priority, action) in DEFERRED.items():
        row = by_id.get(doc_id, {})
        deferred_rows.append({"doc_id": doc_id, "filename": row.get("filename", ""), "problem": problem, "priority": priority, "recommended_action": action})
    deferred_path = config.logs_dir / "deferred_documents.csv"
    with deferred_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("doc_id", "filename", "problem", "priority", "recommended_action"))
        writer.writeheader()
        writer.writerows(deferred_rows)
    logger.info("Wrote manifest review for %s documents; deferred=%s", len(output_rows), len(deferred_rows))
    return output_rows


def main() -> int:
    args = parse_args()
    config = AppConfig.from_file(args.config)
    ensure_output_dirs(config)
    logger = configure_logging(config.logs_dir / "manifest_review.log", "mdrpi.manifest_review")
    rows = review_manifest(config, args.manifest, args.audit, logger)
    print(f"Reviewed {len(rows)} manifest rows")
    print(f"Manifest review: {config.manifest_path.parent / 'literature_manifest_review.csv'}")
    print(f"Deferred documents: {config.logs_dir / 'deferred_documents.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
