"""Typed records shared by the document-processing pipeline."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar


def _json_list(value: Any) -> str:
    """Serialize list-like metadata losslessly for CSV storage."""

    if isinstance(value, str):
        return value
    return json.dumps(value or [], ensure_ascii=False)


def _parse_list(value: str | list[str] | None) -> list[str]:
    """Read metadata stored either as JSON or as a legacy comma list."""

    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except json.JSONDecodeError:
        pass
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass
class DocumentRecord:
    """One source PDF and its review/processing metadata."""

    doc_id: str
    filename: str
    source_path: str
    sha256: str
    title: str
    year: int | None = None
    authors: str = ""
    organization: str = ""
    source_type: str = "other"
    evidence_tier: str = "D"
    population: list[str] = field(default_factory=lambda: ["general"])
    device_type: list[str] = field(default_factory=lambda: ["general_mdrpi"])
    topic: list[str] = field(default_factory=list)
    language: str = "Mixed"
    include: str = "yes"
    review_status: str = "needs_review"
    page_count: int = 0
    parse_status: str = "pending"
    notes: str = ""
    kb_version: str = "MDRPI_KB_v1.0"

    CSV_FIELDS: ClassVar[tuple[str, ...]] = (
        "doc_id",
        "filename",
        "source_path",
        "sha256",
        "title",
        "year",
        "authors",
        "organization",
        "source_type",
        "evidence_tier",
        "population",
        "device_type",
        "topic",
        "language",
        "include",
        "review_status",
        "page_count",
        "parse_status",
        "notes",
        "kb_version",
    )

    def to_csv_row(self) -> dict[str, str]:
        """Return a CSV-safe representation with JSON list fields."""

        row = asdict(self)
        row["year"] = "" if self.year is None else str(self.year)
        row["page_count"] = str(self.page_count)
        for key in ("population", "device_type", "topic"):
            row[key] = _json_list(row[key])
        return {key: str(row.get(key, "")) for key in self.CSV_FIELDS}

    @classmethod
    def from_csv_row(cls, row: dict[str, str]) -> "DocumentRecord":
        """Build a record from a manifest row."""

        year_value = (row.get("year") or "").strip()
        page_value = (row.get("page_count") or "0").strip()
        return cls(
            doc_id=row.get("doc_id", ""),
            filename=row.get("filename", ""),
            source_path=row.get("source_path", ""),
            sha256=row.get("sha256", ""),
            title=row.get("title", ""),
            year=int(year_value) if year_value.isdigit() else None,
            authors=row.get("authors", ""),
            organization=row.get("organization", ""),
            source_type=row.get("source_type", "other"),
            evidence_tier=row.get("evidence_tier", "D"),
            population=_parse_list(row.get("population")),
            device_type=_parse_list(row.get("device_type")),
            topic=_parse_list(row.get("topic")),
            language=row.get("language", "Mixed"),
            include=row.get("include", "yes"),
            review_status=row.get("review_status", "needs_review"),
            page_count=int(page_value) if page_value.isdigit() else 0,
            parse_status=row.get("parse_status", "pending"),
            notes=row.get("notes", ""),
            kb_version=row.get("kb_version", "MDRPI_KB_v1.0"),
        )


@dataclass
class PageRecord:
    """Text extracted from one PDF page using a one-based PDF page number."""

    doc_id: str
    page: int
    text: str
    parse_method: str = "pymupdf"
    needs_ocr: bool = False
    character_count: int = 0
    page_verified: bool = True
    quality_status: str = "ok"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable page record."""

        return asdict(self)


@dataclass
class ChunkRecord:
    """A traceable retrieval unit derived from one or more PDF pages."""

    chunk_id: str
    doc_id: str
    text: str
    title: str
    authors: str
    year: int | None
    organization: str
    source_type: str
    evidence_tier: str
    population: list[str]
    device_type: list[str]
    topic: list[str]
    section: str
    page_start: int | None
    page_end: int | None
    page_verified: bool
    kb_version: str
    language: str
    include: str
    review_status: str
    filename: str = ""
    char_count: int = 0
    source_pages: list[int] = field(default_factory=list)
    source_pdf_sha256: str = ""
    preprocessing_warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable chunk record."""

        data = asdict(self)
        if data["char_count"] == 0:
            data["char_count"] = len(self.text)
        return data


def write_manifest(path: Any, records: list[DocumentRecord]) -> None:
    """Write a deterministic UTF-8 manifest."""

    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DocumentRecord.CSV_FIELDS)
        writer.writeheader()
        writer.writerows(record.to_csv_row() for record in records)


def read_manifest(path: Any) -> list[DocumentRecord]:
    """Read a manifest, returning records in file order."""

    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        return [DocumentRecord.from_csv_row(row) for row in csv.DictReader(handle)]
