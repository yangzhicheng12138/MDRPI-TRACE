#!/usr/bin/env python3
"""Build deterministic, page-traceable JSONL chunks from parsed pages."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from common import add_project_to_import_path, configure_logging, project_config_path

PROJECT_ROOT = add_project_to_import_path()

from app.chunker import build_chunks  # noqa: E402
from app.config import AppConfig, ensure_output_dirs  # noqa: E402
from app.schemas import DocumentRecord, read_manifest  # noqa: E402


QUALITY_BLOCKED_STATUSES = {"needs_ocr", "garbled", "failed"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=project_config_path())
    parser.add_argument("--target-chars", type=int, default=None, help="Override configured Chinese chunk target")
    parser.add_argument("--overlap-chars", type=int, default=None, help="Override configured overlap")
    return parser.parse_args()


def _read_parsed(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_all(config: AppConfig, target_chars: int | None, overlap_chars: int | None, logger) -> list[dict[str, object]]:
    if config.get("preprocessing", "mode") == "clean":
        raise ValueError("Clean datasets must use scripts/08_build_clean_retrieval.py to apply source review and cleaning")
    records = read_manifest(config.manifest_path)
    configured_target = int(config.get("chunking", "target_chars_zh", default=700))
    configured_overlap = int(config.get("chunking", "overlap_chars_zh", default=100))
    target = target_chars or configured_target
    overlap = configured_overlap if overlap_chars is None else overlap_chars
    all_chunks = []
    report: list[dict[str, object]] = []
    for record in records:
        parsed_path = config.parsed_dir / f"{record.doc_id}.json"
        row: dict[str, object] = {"doc_id": record.doc_id, "filename": record.filename, "parse_status": record.parse_status, "chunks": 0, "characters": 0, "quality_pages_skipped": 0, "status": "skipped", "error": ""}
        try:
            if record.parse_status in QUALITY_BLOCKED_STATUSES:
                row["status"] = "skipped_quality"
                row["error"] = f"parse_status={record.parse_status}"
                logger.warning("skipping %s because parse status is %s", record.filename, record.parse_status)
                report.append(row)
                continue
            if not parsed_path.exists():
                raise FileNotFoundError(parsed_path)
            parsed = _read_parsed(parsed_path)
            pages = parsed.get("pages", [])
            if parsed.get("parse_status") in QUALITY_BLOCKED_STATUSES:
                raise ValueError("Parsed document is quality-blocked; refresh the manifest")
            usable_pages = [page for page in pages if page.get("quality_status") not in {"garbled", "blank", "needs_ocr"}]
            row["quality_pages_skipped"] = len(pages) - len(usable_pages)
            chunks = build_chunks(record, usable_pages, target_chars=target, overlap_chars=overlap)
            all_chunks.extend(chunks)
            row.update(chunks=len(chunks), characters=sum(chunk.char_count for chunk in chunks), status="ok" if chunks else "no_text")
            logger.info("chunked %s: %s chunks", record.filename, len(chunks))
        except Exception as exc:  # noqa: BLE001 - report one problematic document and continue
            row["error"] = f"{type(exc).__name__}: {exc}"
            logger.exception("failed to chunk %s", record.filename)
        report.append(row)

    temporary = config.chunks_path.with_name(f".{config.chunks_path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for chunk in all_chunks:
            handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(config.chunks_path)
    return report


def write_report(path: Path, rows: list[dict[str, object]]) -> None:
    fields = ("doc_id", "filename", "parse_status", "chunks", "characters", "quality_pages_skipped", "status", "error")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    config = AppConfig.from_file(args.config)
    ensure_output_dirs(config)
    logger = configure_logging(config.logs_dir / "build_chunks.log", "mdrpi.chunks")
    rows = build_all(config, args.target_chars, args.overlap_chars, logger)
    report_path = config.logs_dir / "chunk_report.csv"
    write_report(report_path, rows)
    chunk_count = sum(int(row["chunks"]) for row in rows)
    print(f"Built {chunk_count} chunks from {len(rows)} PDF records")
    print(f"Chunks: {config.chunks_path}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
