#!/usr/bin/env python3
"""Audit chunk quality without changing the source JSONL.

The audit is deliberately conservative: it records explainable flags and lets
the vector-build step exclude unsuitable chunks through ``include_in_vector``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
import tempfile
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from common import add_project_to_import_path, configure_logging, project_config_path, sha256_file

PROJECT_ROOT = add_project_to_import_path()

from app.config import AppConfig, ensure_output_dirs  # noqa: E402

try:  # Audit must also run in a minimal environment without the PDF parser extras.
    from app.pdf_parser import text_quality  # type: ignore  # noqa: E402
except ModuleNotFoundError:  # pragma: no cover - exercised on minimal deployments
    def text_quality(text: str) -> dict[str, int | float | bool]:
        non_whitespace = sum(not character.isspace() for character in text)
        chinese = len(re.findall(r"[\u4e00-\u9fff]", text))
        latin = len(re.findall(r"[A-Za-z]", text))
        digits = len(re.findall(r"\d", text))
        semantic = chinese + latin + digits
        controls = sum(unicodedata.category(character).startswith("C") and not character.isspace() for character in text)
        symbols = sum(unicodedata.category(character).startswith(("P", "S")) for character in text)
        semantic_ratio = semantic / non_whitespace if non_whitespace else 0.0
        symbol_ratio = symbols / non_whitespace if non_whitespace else 0.0
        control_ratio = controls / non_whitespace if non_whitespace else 0.0
        return {
            "is_garbled": bool(non_whitespace >= 400 and (symbol_ratio >= 0.40 and semantic_ratio < 0.65 or control_ratio >= 0.05 and semantic_ratio < 0.65 or semantic_ratio < 0.25)),
            "semantic_ratio": semantic_ratio,
            "symbol_ratio": symbol_ratio,
            "control_ratio": control_ratio,
        }


AUDIT_FIELDS = (
    "chunk_id", "doc_id", "length", "length_zh", "source_type", "evidence_tier",
    "population", "device_type", "topic", "section", "page_start", "page_end",
    "audit_too_short", "audit_too_long", "audit_low_information", "audit_garbled",
    "content_type_suspected", "duplicate_exact_group", "duplicate_near_group",
    "include_in_vector", "include_in_retrieval", "retrieval_exclusion_reasons", "audit_notes",
    "text_sha256",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=project_config_path())
    parser.add_argument("--input", type=Path, default=None, help="Override chunks JSONL")
    return parser.parse_args()


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().casefold()


def _json_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value or [], ensure_ascii=False)


def _percentile(values: list[int], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _stats(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": 0, "median": 0, "min": 0, "max": 0, "p25": 0, "p75": 0, "p90": 0, "p95": 0}
    return {
        "count": len(values),
        "mean": round(statistics.fmean(values), 2),
        "median": round(statistics.median(values), 2),
        "min": min(values),
        "max": max(values),
        "p25": round(_percentile(values, 0.25), 2),
        "p75": round(_percentile(values, 0.75), 2),
        "p90": round(_percentile(values, 0.90), 2),
        "p95": round(_percentile(values, 0.95), 2),
    }


def _content_types(text: str, repeated_lines: set[str]) -> list[str]:
    compact = _normalise(text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    result: list[str] = []
    reference_heading = any(re.fullmatch(r"(?:参考文献|references|bibliography)\s*[:：]?", line, re.I) for line in lines)
    inline_bibliography = bool(re.match(r"^(?:references|bibliography)\s+(?:doi\s*:|\[?1[.)\]])", compact))
    citation_entries = len(re.findall(r"\b(?:19|20)\d{2}\s*[;,]\s*\d|\[J\]|10\.\d{4,9}/", text))
    if reference_heading or inline_bibliography or citation_entries >= 2:
        result.append("reference_list")
    toc_lines = sum(bool(re.search(r"(?:\.{2,}|…{2,})\s*\d{1,3}\s*$", line)) for line in lines)
    if re.match(r"^(?:目录|contents|table of contents)(?:\s|$)", compact) or toc_lines >= 3:
        result.append("table_of_contents")
    if re.search(r"(?:copyright|版权所有|未经许可不得|all rights reserved)", compact):
        result.append("copyright")
    if re.match(r"^(?:作者单位|作者机构|author affiliation|department of|school of)", compact) and len(text) < 300:
        result.append("author_affiliation")
    if re.search(r"(?:致谢|acknowledg(?:e)?ments?)", compact):
        result.append("acknowledgement")
    header_chars = sum(len(line) for line in lines if _normalise(line) in repeated_lines
                       and re.search(r"\b(?:journal|vol(?:ume)?|issn|page)\b|杂志|第\s*\d+\s*卷", line, re.I))
    if header_chars >= len(text.strip()) * 0.8 and header_chars:
        result.append("header_footer")
    # A table-only fragment tends to have many separators/cells and little prose.
    semantic = len(re.findall(r"[\u4e00-\u9fffA-Za-z]", text))
    separators = len(re.findall(r"[|│┃┼+\\/\t]", text))
    if (separators >= 5 and semantic < max(80, len(text) * 0.45)) or (len(lines) >= 3 and sum(bool(re.search(r"\d+\s+\d+", line)) for line in lines) >= 3 and semantic < len(text) * 0.55):
        result.append("pure_table_fragment")
    if re.search(r"(?<!\w)(?:[-+\u2212]?\d+(?:\.\d+)?%?\s+){6,}[-+\u2212]?\d", text):
        result.append("pure_table_fragment")
    if re.match(r"^\s*(?:[图表]\s*\d+|(?:figure|fig\.?|table)\s+\d+)\b", text, re.I) and len(text) <= 260:
        result.append("figure_caption_only")
    return result


def _is_low_information(text: str, types: list[str]) -> tuple[bool, str]:
    stripped = text.strip()
    if not stripped:
        return True, "empty_text"
    semantic = len(re.findall(r"[\u4e00-\u9fffA-Za-z]", stripped))
    digits = len(re.findall(r"\d", stripped))
    non_space = len(re.sub(r"\s+", "", stripped))
    symbols = sum(unicodedata.category(char).startswith(("P", "S")) for char in stripped if not char.isspace())
    if semantic == 0:
        return True, "no_semantic_characters"
    if non_space and symbols / non_space >= 0.70 and semantic / non_space < 0.30:
        return True, "symbol_dominant"
    if non_space and digits / non_space >= 0.85 and semantic / non_space < 0.15:
        return True, "number_dominant"
    if "pure_table_fragment" in types and semantic < 30:
        return True, "table_fragment_low_information"
    return False, ""


def _duplicate_groups(rows: list[dict[str, Any]], threshold: float) -> tuple[dict[str, str], dict[str, str]]:
    exact_buckets: dict[str, list[int]] = defaultdict(list)
    normalised: list[str] = []
    for index, row in enumerate(rows):
        value = _normalise(str(row.get("text", "")))
        normalised.append(value)
        exact_buckets[hashlib.sha256(value.encode("utf-8")).hexdigest()].append(index)
    exact_labels: dict[str, str] = {}
    exact_index: dict[int, str] = {}
    for group_number, indices in enumerate(sorted((items for items in exact_buckets.values() if len(items) > 1), key=lambda x: x[0]), start=1):
        label = f"exact-{group_number:04d}"
        for index in indices:
            exact_index[index] = label
            exact_labels[str(rows[index].get("chunk_id", index))] = label

    # Compare only same-prefix/near-length candidates; this avoids an O(n^2)
    # comparison while still catching repeated headers and copied fragments.
    candidate_buckets: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, value in enumerate(normalised):
        if not value or index in exact_index:
            continue
        key = (re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", value)[:36], len(value) // 200)
        candidate_buckets[key].append(index)
    parent = list(range(len(rows)))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for candidates in candidate_buckets.values():
        for offset, left in enumerate(candidates):
            for right in candidates[offset + 1 : offset + 51]:
                a, b = normalised[left], normalised[right]
                if min(len(a), len(b)) < 80:
                    continue
                ratio = SequenceMatcher(None, a, b, autojunk=False).ratio()
                if ratio >= threshold:
                    union(left, right)
    near_members: dict[int, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        if index not in exact_index and normalised[index]:
            near_members[find(index)].append(index)
    near_labels: dict[str, str] = {}
    group_number = 0
    for indices in sorted((items for items in near_members.values() if len(items) > 1), key=lambda x: x[0]):
        group_number += 1
        label = f"near-{group_number:04d}"
        for index in indices:
            near_labels[str(rows[index].get("chunk_id", index))] = label
    return exact_labels, near_labels


def _distribution(rows: list[dict[str, Any]], field: str) -> dict[str, dict[str, float | int]]:
    total = len(rows)
    values: dict[str, set[str]] = defaultdict(set)
    counts: Counter[str] = Counter()
    for row in rows:
        raw = row.get(field, [])
        if not isinstance(raw, list):
            raw = [raw]
        seen: set[str] = set()
        for value in raw or [""]:
            label = str(value) or "(empty)"
            if label in seen:
                continue
            seen.add(label)
            counts[label] += 1
            values[label].add(str(row.get("doc_id", "")))
    return {
        key: {"chunks": count, "documents": len(values[key]), "percent_chunks": round(count / total * 100, 2) if total else 0.0, "percent_documents": round(len(values[key]) / len({str(row.get('doc_id', '')) for row in rows}) * 100, 2) if rows else 0.0}
        for key, count in sorted(counts.items())
    }


def audit(config: AppConfig, input_path: Path | None, logger) -> dict[str, Any]:
    source = input_path or config.chunks_path
    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("JSON value is not an object")
                rows.append(row)
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"Malformed JSONL line {line_number}: {exc}") from exc

    line_counter: Counter[str] = Counter()
    for row in rows:
        for line in str(row.get("text", "")).splitlines():
            normalised = _normalise(line)
            if 8 <= len(normalised) <= 180:
                line_counter[normalised] += 1
    repeated_lines = {line for line, count in line_counter.items() if count >= 5}
    exact_groups, near_groups = _duplicate_groups(rows, float(config.get("audit", "near_duplicate_ratio", default=0.92)))
    short_limit = int(config.get("audit", "short_chunk_chars", default=50))
    long_limit = int(config.get("audit", "long_chunk_chars", default=1500))
    exclude_suspected = bool(config.get("audit", "exclude_suspected_from_retrieval", default=False))
    configured_exclusions = config.get(
        "audit",
        "retrieval_exclude_types",
        default=[
            "reference_list",
            "table_of_contents",
            "header_footer",
            "copyright",
            "author_affiliation",
            "acknowledgement",
            "pure_table_fragment",
            "figure_caption_only",
        ],
    )
    retrieval_exclude_types = {str(item) for item in (configured_exclusions or [])}
    exclude_short = bool(config.get("audit", "exclude_short_from_retrieval", default=False))
    exclude_long = bool(config.get("audit", "exclude_long_from_retrieval", default=False))
    deduplicate = bool(config.get("audit", "deduplicate_within_document", default=False))
    canonical_chunks: dict[tuple[str, str], str] = {}
    audit_rows: list[dict[str, Any]] = []
    lengths: list[int] = []
    chinese_lengths: list[int] = []
    for row in rows:
        text = str(row.get("text", ""))
        length = len(text)
        length_zh = len(re.findall(r"[\u4e00-\u9fff]", text))
        types = _content_types(text, repeated_lines)
        # Older chunks can store the author/year prefix in `section` and only
        # the continuation in `text`. Inspect them together for bibliography.
        section = str(row.get("section", ""))
        if re.match(r"^\s*(?:\[\d+\]|\d+[.)])\s*[A-Z][a-z]+,?\s+[A-Z]{1,3}\b", section) and re.search(r"\b(?:19|20)\d{2}\s*[;,]|\bdoi\b|\bet\.?\s+al\b", section + " " + text, re.I):
            types = sorted(set(types) | {"reference_list"})
        low_information, low_reason = _is_low_information(text, types)
        quality = text_quality(text)
        garbled = (bool(quality["is_garbled"]) or "\ufffd" in text
                   or any(unicodedata.category(char) in {"Cc", "Cs"} and not char.isspace() for char in text))
        notes: list[str] = []
        if low_reason:
            notes.append(low_reason)
        if garbled:
            notes.append("quality_garbled")
        if types:
            notes.append("suspected=" + ",".join(types))
        original_include = str(row.get("include", "yes")).strip().lower()
        include_in_vector = original_include == "yes" and not garbled and not low_information and bool(text.strip())
        excluded_types = sorted(set(types) & retrieval_exclude_types) if exclude_suspected else []
        review = row.get("source_review") or {}
        if review.get("unit_id") and review.get("content_type") == "body_text" and "reference_list" in excluded_types:
            excluded_types.remove("reference_list")
            notes.append("source_review=body_text;reference_list_heuristic_dismissed")
        exclusion_reasons = list(excluded_types)
        if original_include != "yes":
            exclusion_reasons.append("document_excluded")
        if garbled:
            exclusion_reasons.append("quality_garbled")
        if low_information:
            exclusion_reasons.append(low_reason)
        if exclude_short and length < short_limit:
            exclusion_reasons.append("too_short")
        if exclude_long and length > long_limit:
            exclusion_reasons.append("too_long")
        exclusion_reasons.extend(row.get("preprocessing_warnings", []))
        if row.get("boundary_excluded"):
            exclusion_reasons.append("boundary_excluded")
        if row.get("page_verified") is False:
            exclusion_reasons.append("unverified_pdf_pages")
        include_in_retrieval = include_in_vector and not exclusion_reasons
        duplicate_key = (str(row.get("doc_id", "")), _normalise(text))
        if deduplicate and include_in_retrieval:
            if duplicate_key in canonical_chunks:
                exclusion_reasons.append("duplicate_within_document")
                notes.append("duplicate_of=" + canonical_chunks[duplicate_key])
                include_in_retrieval = False
            else:
                canonical_chunks[duplicate_key] = str(row.get("chunk_id", ""))
        audit_row = {
            "chunk_id": str(row.get("chunk_id", "")), "doc_id": str(row.get("doc_id", "")),
            "length": length, "length_zh": length_zh,
            "source_type": str(row.get("source_type", "")), "evidence_tier": str(row.get("evidence_tier", "")),
            "population": _json_value(row.get("population", [])), "device_type": _json_value(row.get("device_type", [])),
            "topic": _json_value(row.get("topic", [])), "section": str(row.get("section", "")),
            "page_start": row.get("page_start", ""), "page_end": row.get("page_end", ""),
            "audit_too_short": length < short_limit, "audit_too_long": length > long_limit,
            "audit_low_information": low_information, "audit_garbled": garbled,
            "content_type_suspected": ";".join(types),
            "duplicate_exact_group": exact_groups.get(str(row.get("chunk_id", "")), ""),
            "duplicate_near_group": near_groups.get(str(row.get("chunk_id", "")), ""),
            "include_in_vector": include_in_vector,
            "include_in_retrieval": include_in_retrieval,
            "retrieval_exclusion_reasons": ";".join(sorted(set(exclusion_reasons))),
            "audit_notes": ";".join(notes),
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        }
        if exclusion_reasons:
            audit_row["audit_notes"] = ";".join(notes + ["retrieval_excluded=" + ",".join(sorted(set(exclusion_reasons)))])
        audit_rows.append(audit_row)
        lengths.append(length)
        chinese_lengths.append(length_zh)

    audit_path = config.logs_dir / "chunk_audit.csv"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=config.logs_dir, delete=False) as temporary:
        writer = csv.DictWriter(temporary, fieldnames=AUDIT_FIELDS)
        writer.writeheader()
        writer.writerows(audit_rows)
        temporary_path = Path(temporary.name)
    temporary_path.replace(audit_path)

    documents = {str(row.get("doc_id", "")) for row in rows}
    summary: dict[str, Any] = {
        "input": str(source), "total_chunks": len(rows), "total_documents": len(documents),
        "input_sha256": sha256_file(source),
        "audit_csv_sha256": sha256_file(audit_path),
        "audit_config_sha256": hashlib.sha256(json.dumps(config.get("audit", default={}), sort_keys=True).encode("utf-8")).hexdigest(),
        "length_total": _stats(lengths), "length_chinese": _stats(chinese_lengths),
        "thresholds": {"too_short_lt_30": sum(value < 30 for value in lengths), "too_short_lt_50": sum(value < 50 for value in lengths), "too_short_lt_100": sum(value < 100 for value in lengths), "too_long_gt_1200": sum(value > 1200 for value in lengths), "too_long_gt_1500": sum(value > 1500 for value in lengths), "too_long_gt_2000": sum(value > 2000 for value in lengths)},
        "quality": {"empty": sum(not str(row.get("text", "")).strip() for row in rows), "low_information": sum(bool(item["audit_low_information"]) for item in audit_rows), "garbled": sum(bool(item["audit_garbled"]) for item in audit_rows), "suspected_content_chunks": sum(bool(item["content_type_suspected"]) for item in audit_rows), "include_in_vector": sum(bool(item["include_in_vector"]) for item in audit_rows), "excluded_from_vector": sum(not bool(item["include_in_vector"]) for item in audit_rows), "include_in_retrieval": sum(bool(item["include_in_retrieval"]) for item in audit_rows), "excluded_from_retrieval": sum(not bool(item["include_in_retrieval"]) for item in audit_rows)},
        "retrieval_policy": {"exclude_suspected_from_retrieval": exclude_suspected, "exclude_types": sorted(retrieval_exclude_types),
                             "exclude_short_from_retrieval": exclude_short, "exclude_long_from_retrieval": exclude_long,
                             "deduplicate_within_document": deduplicate},
        "retrieval_exclusion_counts": dict(Counter(reason for item in audit_rows for reason in item["retrieval_exclusion_reasons"].split(";") if reason)),
        "duplicates": {"exact_groups": len(set(exact_groups.values())), "exact_chunks": len(exact_groups), "near_groups": len(set(near_groups.values())), "near_chunks": len(near_groups)},
        "metadata_distribution": {field: _distribution(rows, field) for field in ("source_type", "evidence_tier", "population", "device_type", "topic", "language", "include", "review_status")},
        "repeated_line_count": len(repeated_lines),
    }
    summary_path = config.logs_dir / "chunk_audit_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Audited %s chunks (%s eligible for retrieval)", len(rows), summary["quality"]["include_in_retrieval"])
    return summary


def main() -> int:
    args = parse_args()
    config = AppConfig.from_file(args.config)
    ensure_output_dirs(config)
    logger = configure_logging(config.logs_dir / "chunk_audit.log", "mdrpi.chunk_audit")
    summary = audit(config, args.input, logger)
    print(f"Audited {summary['total_chunks']} chunks")
    print(f"include_in_vector=true: {summary['quality']['include_in_vector']}")
    print(f"include_in_retrieval=true: {summary['quality']['include_in_retrieval']}")
    print(f"Audit CSV: {config.logs_dir / 'chunk_audit.csv'}")
    print(f"Summary: {config.logs_dir / 'chunk_audit_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
