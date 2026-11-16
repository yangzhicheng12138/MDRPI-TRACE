#!/usr/bin/env python3
"""Scan source PDFs and create a reviewable literature manifest."""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from collections import Counter
from pathlib import Path

from common import add_project_to_import_path, configure_logging, natural_sort_key, project_config_path, sha256_file

PROJECT_ROOT = add_project_to_import_path()

from app.config import AppConfig, ensure_output_dirs  # noqa: E402
from app.metadata import infer_metadata, validate_metadata  # noqa: E402
from app.pdf_parser import inspect_pdf  # noqa: E402
from app.schemas import DocumentRecord, read_manifest, write_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=project_config_path())
    parser.add_argument("--force", action="store_true", help="Reassign all document IDs instead of preserving an existing manifest")
    return parser.parse_args()


def scan(config: AppConfig, force: bool, logger: logging.Logger) -> list[DocumentRecord]:
    """Scan all PDFs, preserving existing IDs when a manifest already exists."""

    source_dir = config.source_dir
    files = sorted((path for path in source_dir.rglob("*") if path.is_file() and path.suffix.lower() == ".pdf"), key=lambda p: natural_sort_key(p.name))
    if not files:
        raise FileNotFoundError(f"No PDF files found under {source_dir}")

    existing: dict[str, DocumentRecord] = {}
    if config.manifest_path.exists() and not force:
        existing = {record.source_path: record for record in read_manifest(config.manifest_path)}
    filename_counts = Counter(path.name for path in files)
    old_filename_counts = Counter(record.filename for record in existing.values())
    by_filename = {record.filename: record for record in existing.values()}
    used_ids = {record.doc_id for record in existing.values()}
    next_id = max((int(record.doc_id.removeprefix("DOC")) for record in existing.values() if record.doc_id.startswith("DOC") and record.doc_id[3:].isdigit()), default=0) + 1
    records: list[DocumentRecord] = []

    for path in files:
        relative = path.relative_to(config.project_root).as_posix()
        filename = path.name
        old = existing.get(relative)
        if old is None and filename_counts[filename] == 1 and old_filename_counts[filename] == 1:
            old = by_filename[filename]
        if old is not None:
            doc_id = old.doc_id
        else:
            while f"DOC{next_id:03d}" in used_ids:
                next_id += 1
            doc_id = f"DOC{next_id:03d}"
            used_ids.add(doc_id)
            next_id += 1
        try:
            digest = sha256_file(path)
            if old is not None and old.sha256 == digest:
                # Preserve reviewed fields AND any pending manual corrections.
                records.append(replace(old, filename=filename, source_path=relative))
                continue
            inspected = inspect_pdf(path)
            if old is not None:
                # A replaced PDF invalidates inclusion and parsing, but must
                # not silently replace the user's bibliographic decisions.
                records.append(replace(
                    old, filename=filename, source_path=relative, sha256=digest,
                    page_count=int(inspected["page_count"]), parse_status="pending",
                    review_status="needs_review", include="no",
                    notes=old.notes + ";source_changed_review_required",
                ))
                continue
            inferred = infer_metadata(
                filename,
                text_sample=inspected["first_page_text"],
                pdf_title=inspected["title"],
                pdf_author=inspected["author"],
            )
            errors = validate_metadata(inferred.source_type, inferred.population, inferred.device_type, inferred.topic)
            notes = inferred.notes
            if errors:
                notes = ";".join(filter(None, [notes, "metadata_validation:" + "|".join(errors)]))
            record = DocumentRecord(
                doc_id=doc_id,
                filename=filename,
                source_path=relative,
                sha256=digest,
                title=inferred.title,
                year=inferred.year,
                authors=inferred.authors,
                organization=inferred.organization,
                source_type=inferred.source_type,
                evidence_tier=inferred.evidence_tier,
                population=inferred.population,
                device_type=inferred.device_type,
                topic=inferred.topic,
                language=inferred.language,
                include=inferred.include,
                review_status="needs_review",
                page_count=int(inspected["page_count"]),
                parse_status="pending",
                notes=notes,
                kb_version=config.version,
            )
            logger.info("scanned %s (%s pages)", filename, record.page_count)
        except Exception as exc:  # noqa: BLE001 - one corrupt PDF must not stop the inventory
            if old is not None:
                records.append(replace(
                    old, filename=filename, source_path=relative, parse_status="failed",
                    include="no", review_status="needs_review",
                    notes=old.notes + f";scan_failed:{type(exc).__name__}:{exc}",
                ))
                logger.exception("failed to inspect existing source %s", filename)
                continue
            record = DocumentRecord(
                doc_id=doc_id,
                filename=filename,
                source_path=relative,
                sha256=sha256_file(path),
                title=path.stem,
                source_type="other",
                evidence_tier="D",
                population=["general"],
                device_type=["general_mdrpi"],
                language="Mixed",
                include="no",
                review_status="needs_review",
                parse_status="failed",
                notes=f"scan_failed:{type(exc).__name__}:{exc}",
                kb_version=config.version,
            )
            logger.exception("failed to inspect %s", filename)
        records.append(record)
    records.sort(key=lambda item: natural_sort_key(item.filename))
    return records


def main() -> int:
    args = parse_args()
    config = AppConfig.from_file(args.config)
    ensure_output_dirs(config)
    logger = configure_logging(config.logs_dir / "scan_documents.log", "mdrpi.scan")
    logger.info("starting scan; project=%s version=%s", config.project_root, config.version)
    records = scan(config, args.force, logger)
    write_manifest(config.manifest_path, records)
    logger.info("manifest written: %s documents -> %s", len(records), config.manifest_path)
    print(f"Scanned {len(records)} PDF files")
    print(f"Manifest: {config.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
