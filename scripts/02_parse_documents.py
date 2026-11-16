#!/usr/bin/env python3
"""Parse source PDFs page by page and write an OCR/quality report."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from common import add_project_to_import_path, configure_logging, project_config_path, sha256_file

PROJECT_ROOT = add_project_to_import_path()

from app.config import AppConfig, ensure_output_dirs  # noqa: E402
from app.metadata import detect_language  # noqa: E402
from app.pdf_parser import PARSER_VERSION, parse_pdf  # noqa: E402
from app.schemas import DocumentRecord, read_manifest, write_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=project_config_path())
    parser.add_argument("--force", action="store_true", help="Reparse PDFs even if a matching JSON output exists")
    return parser.parse_args()


def _source_path(config: AppConfig, record: DocumentRecord) -> Path:
    path = Path(record.source_path)
    return path if path.is_absolute() else config.project_root / path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def _append_note(record: DocumentRecord, note: str) -> None:
    """Append a review note once, keeping reruns deterministic."""

    notes = [item for item in record.notes.split(";") if item]
    if note not in notes:
        notes.append(note)
    record.notes = ";".join(notes)


def parse_all(config: AppConfig, force: bool, logger) -> list[dict[str, object]]:
    if config.get("preprocessing", "mode") == "clean":
        raise ValueError("Clean configurations require scripts/08_build_clean_retrieval.py so source review decisions are applied")
    records = read_manifest(config.manifest_path)
    report: list[dict[str, object]] = []
    changed: list[DocumentRecord] = []
    for record in records:
        source = _source_path(config, record)
        output = config.parsed_dir / f"{record.doc_id}.json"
        row: dict[str, object] = {
            "doc_id": record.doc_id,
            "filename": record.filename,
            "page_count": record.page_count,
            "total_characters": 0,
            "empty_pages": "",
            "ocr_pages": "",
            "garbled_pages": "",
            "quality_semantic_ratio": "",
            "quality_symbol_ratio": "",
            "quality_control_characters": "",
            "parse_method": "pymupdf",
            "status": "failed",
            "error": "",
        }
        try:
            if not source.exists():
                raise FileNotFoundError(source)
            digest = sha256_file(source)
            if output.exists() and not force:
                with output.open("r", encoding="utf-8") as handle:
                    cached = json.load(handle)
                if cached.get("sha256") == digest and cached.get("parser_version") == PARSER_VERSION:
                    result = cached
                    logger.info("resume: %s", record.filename)
                else:
                    result = parse_pdf(source, record.doc_id)
            else:
                result = parse_pdf(source, record.doc_id)
            result["sha256"] = digest
            result["kb_version"] = record.kb_version
            _write_json(output, result)
            row.update(
                page_count=result.get("page_count", 0),
                total_characters=result.get("total_characters", 0),
                empty_pages=json.dumps(result.get("empty_pages", []), ensure_ascii=False),
                ocr_pages=json.dumps(result.get("ocr_pages", []), ensure_ascii=False),
                garbled_pages=json.dumps(result.get("garbled_pages", []), ensure_ascii=False),
                quality_semantic_ratio=(result.get("text_quality") or {}).get("semantic_ratio", ""),
                quality_symbol_ratio=(result.get("text_quality") or {}).get("symbol_ratio", ""),
                quality_control_characters=(result.get("text_quality") or {}).get("control_characters", ""),
                status=result.get("parse_status", "ok"),
            )
            record.page_count = int(result.get("page_count", record.page_count))
            record.parse_status = str(result.get("parse_status", "ok"))
            full_text = "\n".join(str(page.get("text", "")) for page in result.get("pages", []))
            if full_text.strip():
                record.language = detect_language(full_text)
            if record.parse_status in {"needs_ocr", "garbled"}:
                # The extracted text is not reliable enough to assign a single
                # language; keep the field conservative until OCR/manual review.
                record.language = "Mixed"
                _append_note(record, "language_needs_review")
            if record.parse_status == "needs_ocr":
                _append_note(record, "needs_ocr_review")
            if record.parse_status == "garbled":
                _append_note(record, "garbled_text_review")
            elif result.get("garbled_pages"):
                _append_note(record, "garbled_pages_review")
            if result.get("empty_pages"):
                _append_note(record, "blank_pages_review")
            logger.info("parsed %s: %s pages, %s chars, status=%s", record.filename, row["page_count"], row["total_characters"], row["status"])
        except Exception as exc:  # noqa: BLE001 - retain a per-file failure report
            row["error"] = f"{type(exc).__name__}: {exc}"
            record.parse_status = "failed"
            _append_note(record, f"parse_failed:{type(exc).__name__}")
            logger.exception("failed to parse %s", record.filename)
        changed.append(record)
        report.append(row)
    write_manifest(config.manifest_path, changed)
    return report


def write_report(path: Path, rows: list[dict[str, object]]) -> None:
    fields = (
        "doc_id",
        "filename",
        "page_count",
        "total_characters",
        "empty_pages",
        "ocr_pages",
        "garbled_pages",
        "quality_semantic_ratio",
        "quality_symbol_ratio",
        "quality_control_characters",
        "parse_method",
        "status",
        "error",
    )
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def build_summary(rows: list[dict[str, object]]) -> dict[str, int]:
    """Summarise quality categories without hiding page-level warnings."""

    def pages(row: dict[str, object], key: str) -> list[object]:
        value = row.get(key, "[]")
        try:
            parsed = json.loads(str(value))
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []

    return {
        "total_documents": len(rows),
        "normal_documents": sum(str(row["status"]) == "ok" for row in rows),
        "suspected_scan_documents": sum(str(row["status"]) == "needs_ocr" for row in rows),
        "garbled_documents": sum(str(row["status"]) == "garbled" for row in rows),
        "failed_documents": sum(str(row["status"]) == "failed" for row in rows),
        "documents_with_blank_pages": sum(bool(pages(row, "empty_pages")) for row in rows),
        "blank_pages": sum(len(pages(row, "empty_pages")) for row in rows),
        "documents_with_garbled_pages": sum(bool(pages(row, "garbled_pages")) for row in rows),
        "garbled_pages": sum(len(pages(row, "garbled_pages")) for row in rows),
        "ocr_pages": sum(len(pages(row, "ocr_pages")) for row in rows),
    }


def main() -> int:
    args = parse_args()
    config = AppConfig.from_file(args.config)
    ensure_output_dirs(config)
    logger = configure_logging(config.logs_dir / "parse_documents.log", "mdrpi.parse")
    logger.info("starting PDF parse")
    rows = parse_all(config, args.force, logger)
    report_path = config.logs_dir / "parse_report.csv"
    write_report(report_path, rows)
    summary = build_summary(rows)
    summary_path = config.logs_dir / "parse_summary.json"
    _write_json(summary_path, summary)
    print(
        "Parsed {total_documents} PDF files: normal={normal_documents}, "
        "needs_ocr={suspected_scan_documents}, garbled={garbled_documents}, "
        "failed={failed_documents}, blank_pages={blank_pages}".format(**summary)
    )
    print(f"Report: {report_path}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
