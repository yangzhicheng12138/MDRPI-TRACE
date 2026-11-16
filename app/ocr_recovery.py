"""Source-bound OCR recovery; unreviewed or stale artifacts fail closed."""

from __future__ import annotations

import hashlib
import json
import math
import re
from copy import deepcopy
from pathlib import Path

from .pdf_parser import (
    ReferenceState, clean_retrieval_blocks, layout_review_reasons,
    normalise_retrieval_text, order_layout_blocks, text_quality,
)
from .preprocessing_review import reviewed_page_blocks


OCR_VERSION = "ocr-recovery-v1"
LOW_CONFIDENCE = 0.90


def _compact(text):
    return re.sub(r"\s+", "", text)


def expand_required_context(chunks, parsed, policy, max_chars=1500):
    """Keep reviewed clinical exceptions with their triggering recommendations.

    Expansion copies a contiguous span from the same cited pages. It never
    synthesizes a warning or appends an exception from another article.
    """
    pages = {p["page"]: p["retrieval_text"] for p in parsed["pages"]}
    for number, review in policy["pages"].items():
        number = int(number)
        page_text = _compact(pages[number])
        for context in review.get("required_context", []):
            start, end = _compact(context["start"]), _compact(context["end"])
            if (not context.get("reason") or not context.get("triggers") or not start or not end
                    or page_text.count(start) != 1 or page_text.count(end) != 1):
                raise ValueError("Required OCR context must have unique source anchors and a reason")
            span_start, span_end = page_text.index(start), page_text.index(end) + len(end)
            if span_start >= span_end:
                raise ValueError("Required OCR context anchors are reversed")
            for chunk in chunks:
                if number not in chunk["source_pages"] or not any(_compact(t) in _compact(chunk["text"]) for t in context["triggers"]):
                    continue
                source = "\n".join(pages[p] for p in chunk["source_pages"])
                positions = [i for i, char in enumerate(source) if not char.isspace()]
                compact_source, compact_chunk = _compact(source), _compact(chunk["text"])
                if compact_source.count(compact_chunk) != 1:
                    raise ValueError("Cannot locate OCR chunk uniquely for context expansion")
                offset = sum(len(_compact(pages[p])) for p in chunk["source_pages"] if p < number)
                chunk_start = compact_source.index(compact_chunk)
                lower = min(chunk_start, offset + span_start)
                upper = max(chunk_start + len(compact_chunk), offset + span_end)
                text = source[positions[lower]:positions[upper - 1] + 1]
                if len(text) > max_chars:
                    raise ValueError("Required clinical context exceeds audit chunk limit; review chunking")
                chunk.update(text=text, char_count=len(text))
                chunk.setdefault("context_expansions", []).append({"page": number, "reason": context["reason"]})
    return chunks


def merge_ocr_lines(blocks, page_width, column_aware=True):
    """Join adjacent heading/line pieces without joining opposing columns."""
    rows = []
    middle = page_width / 2
    for block in sorted(blocks, key=lambda b: ((b["y0"] + b["y1"]) / 2, b["x0"])):
        match = None
        for row in reversed(rows[-6:]):
            overlap = min(row["y1"], block["y1"]) - max(row["y0"], block["y0"])
            if overlap >= min(row["height"], block["height"]) * .6:
                left, right = sorted([row, block], key=lambda b: b["x0"])
                gutter = left["x1"] <= middle <= right["x0"]
                if not gutter and -min(row["height"], block["height"]) * .8 <= right["x0"] - left["x1"] < page_width * .065:
                    match = row
                    break
        if match is None:
            rows.append(dict(block, source_block_ids=[block["block_id"]]))
            continue
        left, right = sorted([match, block], key=lambda b: b["x0"])
        match["text"] = left["text"] + " " + right["text"]
        match["source_block_ids"].append(block["block_id"])
        for key, function in (("x0", min), ("y0", min), ("x1", max), ("y1", max)):
            match[key] = function(match[key], block[key])
        match["width"], match["height"] = match["x1"] - match["x0"], match["y1"] - match["y0"]
    return order_layout_blocks(rows, page_width) if column_aware else sorted(rows, key=lambda b: (b["y0"], b["x0"]))


def load_ocr_artifact(root, native, policy):
    if policy.get("status") != "source_checked" or not policy.get("review_scope"):
        raise ValueError("OCR requires explicit source review")
    path = root / policy["artifact"]
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != policy.get("artifact_sha256"):
        raise ValueError("OCR artifact hash mismatch; repeat source review")
    artifact = json.loads(payload)
    if (artifact.get("schema_version") != 1 or artifact.get("doc_id") != native["doc_id"]
            or artifact.get("source_sha256") != native["sha256"]):
        raise ValueError("OCR source identity mismatch")
    expected = list(range(1, native["page_count"] + 1))
    if (artifact.get("page_count") != len(expected)
            or [page.get("page") for page in artifact.get("pages", [])] != expected
            or [page.get("page") for page in native.get("pages", [])] != expected
            or set(policy.get("pages", {})) != {str(p) for p in expected}):
        raise ValueError("OCR missing pages or page reviews")
    engine = artifact.get("engine", {})
    if not engine.get("models") or not engine.get("version"):
        raise ValueError("OCR engine provenance missing")
    for page, original in zip(artifact["pages"], native["pages"]):
        if (page.get("source_sha256") != native["sha256"] or page.get("engine") != engine
                or page.get("doc_id") != native["doc_id"]):
            raise ValueError("OCR page provenance mismatch")
        for key in ("page_width", "page_height"):
            if not math.isclose(page[key], original[key], abs_tol=.01):
                raise ValueError("OCR page geometry mismatch")
        ids = set()
        for block in page["blocks"]:
            if block["block_id"] in ids:
                raise ValueError("Duplicate OCR block ID")
            ids.add(block["block_id"])
            coords = [block[key] for key in ("x0", "y0", "x1", "y1")]
            if (not all(math.isfinite(v) for v in coords)
                    or not 0 <= coords[0] < coords[2] <= page["page_width"] + .01
                    or not 0 <= coords[1] < coords[3] <= page["page_height"] + .01
                    or not 0 <= block["confidence"] <= 1):
                raise ValueError("Invalid OCR geometry or confidence")
    return artifact


def apply_ocr_recovery(native, policy, root: Path):
    artifact = load_ocr_artifact(root, native, policy)
    parsed = deepcopy(native)
    parsed["native_extraction"] = deepcopy(native)
    parsed["ocr_recovery"] = {"version": OCR_VERSION, "artifact": policy["artifact"],
                              "artifact_sha256": policy["artifact_sha256"],
                              "engine": artifact["engine"], "review_scope": policy["review_scope"],
                              "status": "source_checked", "clinical_validation": False}
    vocabulary = {w.casefold() for page in artifact["pages"] for w in re.findall(r"\b[A-Za-z]{3,}\b", page["text"])}
    for original, recovered in zip(parsed["pages"], artifact["pages"]):
        review = policy["pages"][str(original["page"])]
        if not review.get("reason") or not review.get("review_method"):
            raise ValueError("OCR page review requires method and reason")
        blocks = deepcopy(recovered["blocks"])
        by_id = {block["block_id"]: block for block in blocks}
        corrections = review.get("corrections", {})
        exclusions = review.get("exclude_blocks", {})
        confirmations = review.get("confirmed_blocks", {})
        if (set(corrections) | set(exclusions) | set(confirmations)) - set(by_id):
            raise ValueError("OCR review refers to unknown block")
        if any(not reason for reason in [*exclusions.values(), *confirmations.values()]):
            raise ValueError("OCR block review requires a reason")
        for key, correction in corrections.items():
            block = by_id[key]
            if block["text"] != correction.get("expected") or not correction.get("reason") or not correction.get("text"):
                raise ValueError("OCR correction does not match original text")
            block["ocr_text_before_correction"] = block["text"]
            block["text"] = correction["text"]
        original["ocr_raw_text"] = recovered["text"]
        original["text"] = "\n".join(block["text"] for block in blocks)
        original["layout_blocks"] = blocks
        original["layout_block_count"] = len(blocks)
        kept = reviewed_page_blocks(dict(original, layout_blocks=[b for b in blocks if b["block_id"] not in exclusions]), review)
        unresolved = [b["block_id"] for b in kept if b["confidence"] < LOW_CONFIDENCE
                      and b["block_id"] not in corrections and b["block_id"] not in confirmations]
        if review.get("reading_order") == "regions" and kept:
            ordered = []
            emitted = set()
            for region in review["keep_regions"]:
                selected = [b for b in kept if b["x0"] >= region[0] and b["y0"] >= region[1]
                            and b["x1"] <= region[2] and b["y1"] <= region[3]]
                if emitted.intersection(b["block_id"] for b in selected):
                    raise ValueError("OCR reading regions overlap")
                emitted.update(b["block_id"] for b in selected)
                ordered.extend(merge_ocr_lines(selected, original["page_width"], column_aware=False))
        else:
            ordered = merge_ocr_lines(kept, original["page_width"])
        text, removed = clean_retrieval_blocks(ordered, reference_state=ReferenceState())
        normalised = normalise_retrieval_text(text, vocabulary)
        warnings = list(dict.fromkeys(layout_review_reasons(ordered) + review.get("warnings", [])
                                     + (["ocr_low_confidence_needs_review"] if unresolved else [])))
        quality = text_quality(normalised)
        if quality["is_garbled"]:
            warnings.append("ocr_garbled_text_needs_review")
        for anchor in review.get("required_text", []):
            if re.sub(r"\s+", "", anchor) not in re.sub(r"\s+", "", normalised):
                raise ValueError(f"OCR reviewed text anchor missing: {native['doc_id']} page {original['page']}: {anchor}")
        original.update(
            retrieval_text=normalised, retrieval_text_before_normalisation=text,
            retrieval_layout_blocks=ordered, retrieval_removed_blocks=removed,
            parse_method=OCR_VERSION, needs_ocr=False, quality_status="ok", page_verified=True,
            character_count=len(original["text"]), preprocessing_warnings=warnings,
            reference_continuation_reset_needs_review=False, reference_section_active_at_end=False,
            ocr_review=review, ocr_low_confidence_blocks=unresolved,
            ocr_excluded_block_ids=[b["block_id"] for b in blocks if b["block_id"] not in {k["block_id"] for k in kept}],
        )
    parsed.update(parse_status="ocr_recovered", empty_pages=[], blank_pages=[], ocr_pages=[], garbled_pages=[],
                  total_characters=sum(page["character_count"] for page in parsed["pages"]))
    parsed["text_quality"] = text_quality("\n".join(page["text"] for page in parsed["pages"]))
    return parsed
