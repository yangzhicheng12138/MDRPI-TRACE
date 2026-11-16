"""Explicit source-bound review decisions; never infer page deletion from a flag."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from .pdf_parser import ReferenceState, clean_retrieval_blocks, normalise_retrieval_text
from .schemas import DocumentRecord


def read_decisions(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("documents", {})


def decision_for(decisions: dict[str, Any], doc_id: str, digest: str) -> dict[str, Any]:
    decision = decisions.get(doc_id, {})
    if decision and decision.get("sha256") != digest:
        raise ValueError(f"Review decision source hash mismatch for {doc_id}; review the replacement PDF")
    return decision


def apply_metadata_decision(record: DocumentRecord, decision: dict[str, Any]) -> DocumentRecord:
    updates = decision.get("metadata", {})
    allowed = {"title", "authors", "year", "organization", "population", "device_type", "source_type",
               "evidence_tier", "language", "include"}
    if set(updates) - allowed:
        raise ValueError(f"Unsupported review fields: {sorted(set(updates) - allowed)}")
    if not updates:
        return record
    note = "source_bound_metadata_correction;remaining_fields_needs_review"
    notes = list(dict.fromkeys(filter(None, (record.notes + ";" + note).split(";"))))
    return replace(record, **updates, notes=";".join(notes), review_status="needs_review")


def reviewed_page_blocks(page: dict[str, Any], policy: dict[str, Any]) -> list[dict[str, Any]]:
    """Keep explicitly reviewed rectangles, rejecting intersected text blocks.

    Coordinates refer to the original PDF page in points. Raw page text and
    layout blocks remain in the parsed record for review and recovery.
    """
    if not policy.get("reason"):
        raise ValueError("Page review requires a reason")
    if policy.get("content_type") == "reference_list":
        return []
    regions = policy.get("keep_regions", [])
    if not regions or not page.get("layout_blocks"):
        raise ValueError("Reviewed regions require geometry from the current parser")
    kept = []
    for block in page["layout_blocks"]:
        x0, y0, x1, y1 = (float(block[key]) for key in ("x0", "y0", "x1", "y1"))
        for left, top, right, bottom in regions:
            intersects = x1 > left and x0 < right and y1 > top and y0 < bottom
            contained = x0 >= left and y0 >= top and x1 <= right and y1 <= bottom
            if intersects and not contained:
                raise ValueError(f"Review rectangle cuts text block {block['block_id']}")
            if contained:
                kept.append(block)
                break
    if not kept:
        raise ValueError("Reviewed rectangle contains no text")
    return kept


def reviewed_page_text(page: dict[str, Any], policy: dict[str, Any]) -> str:
    kept = reviewed_page_blocks(page, policy)
    text, _ = clean_retrieval_blocks(kept, reference_state=ReferenceState())
    return normalise_retrieval_text(text)
