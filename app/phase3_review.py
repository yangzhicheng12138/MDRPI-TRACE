"""Source-bound Phase 3 triage. Classification never grants index admission."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
from datetime import date
import hashlib
import json
from pathlib import Path
import re

from .corpus import AuditedCorpus, sha256_file


RULES_VERSION = "phase3-triage-v3"
CATEGORIES = (
    "A_permanent_exclude", "B_review_recoverable", "C_rechunk_required",
    "D_source_page_review", "E_population_or_scope_exclude", "F_duplicate",
    "G_low_information", "H_unresolved",
)
PRIORITIES = ("P0", "P1", "P2", "P3")
KNOWN_PAGES = {("DOC041", 3), ("DOC053", 3), ("DOC053", 6), ("DOC053", 7), ("DOC053", 8), ("DOC069", 5)}
CHUNK_HUMAN_FIELDS = ("reviewer_decision", "reviewer_notes", "reviewer", "review_date")
PAGE_HUMAN_FIELDS = (
    "source_image_checked", "text_order_correct", "table_relation_correct",
    "number_verified", "unit_verified", "condition_verified", "exception_verified",
    "allow_retrieval", "human_review_completed", "reviewer", "review_date", "notes",
)
CHUNK_FIELDS = (
    "chunk_id", "doc_id", "title", "page_start", "page_end", "text", "current_include",
    "current_review_status", "exclusion_category", "suspected_reason", "source_type",
    "evidence_tier", "population", "device_type", "topic", "needs_page_review",
    "recoverable", "recommended_action", "reviewer_decision", "reviewer_notes",
    "reviewer", "review_date", "current_include_in_vector", "current_include_in_retrieval",
    "include_in_vector", "source_pdf_sha256", "text_sha256", "original_exclusion_reasons",
    "canonical_chunk_id", "rules_version",
)
PAGE_FIELDS = (
    "doc_id", "pdf_page", "review_priority", "priority_reason", "issue_type",
    "source_pdf_sha256", "page_text_sha256", "source_file", "indexed_chunk_count",
    "retrievable_chunk_count", "excluded_chunk_count", "containment", "known_issue",
    *PAGE_HUMAN_FIELDS,
)
CLINICAL = re.compile(
    r"\b(?:assess\w*|monitor\w*|prevent\w*|recommend\w*|should|must|reposition\w*|"
    r"securement|pressure|injur\w*|incidence|risk|patients?)\b|"
    r"\u8bc4\u4f30|\u89c2\u5bdf|\u9884\u9632|\u5efa\u8bae|\u63a8\u8350|\u62a4\u7406|\u635f\u4f24|\u60a3\u8005", re.I,
)
SAFETY = re.compile(
    r"\b(?:must|should|recommend\w*|contraindicat\w*|tight\w*|securement|reposition\w*|"
    r"unless|except|daily|hourly|per\s+shift|every\s+shift)\b|"
    r"\d+(?:\.\d+)?\s*(?:%|\uff05|mm\s*Hg|mg|mcg|kg|kcal|cal|m[lL]|cm|mm|"
    r"hours?\b|minutes?\b|h\b|days?\b|times?\b|\u5c0f\u65f6|\u5206\u949f|\u6b21)|"
    r"\u5efa\u8bae|\u63a8\u8350|\u7981\u5fcc|\u677e\u7d27|\u56fa\u5b9a|\u4f8b\u5916|"
    r"\u9608\u503c|\u6bcf\u73ed|\u6bcf\u65e5|\u5fc5\u987b|\u5242\u91cf", re.I,
)
PAGE_RISKS = {
    "unstructured_table_needs_review", "source_energy_unit_ambiguity_needs_clinical_review",
    "ocr_figure_layout_needs_review", "ocr_low_confidence_needs_review",
    "fragmented_word_layout_needs_review", "quality_garbled", "pure_table_fragment",
    "figure_caption_only", "table_fragment_low_information",
}


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def is_true(value):
    return value is True or (isinstance(value, str) and value.lower() == "true")


def tags(value):
    if isinstance(value, list):
        return set(value)
    if not value:
        return set()
    if isinstance(value, str) and value.startswith("["):
        parsed = json.loads(value)
        if not isinstance(parsed, list):
            raise ValueError("Expected a list of metadata tags")
        return set(parsed)
    return {value}


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def classify(chunk, audit, canonical=""):
    """Return a primary category; explicit source risks outrank easy recovery."""
    text = chunk["text"].strip()
    reasons = set(filter(None, audit.get("retrieval_exclusion_reasons", "").split(";")))
    types = set(filter(None, audit.get("content_type_suspected", "").split(";")))
    populations = tags(chunk.get("population"))
    if populations and populations <= {"pediatric", "neonatal"}:
        return "E_population_or_scope_exclude", "exclusively_nonadult_population", "retain_outside_adult_index"
    if chunk.get("boundary_excluded"):
        return "E_population_or_scope_exclude", "reviewed_document_boundary", "retain_outside_target_article"
    if canonical:
        return "F_duplicate", "same_document_exact_text_with_admitted_canonical", "retain_canonical_only"
    if not re.search(r"[A-Za-z\u4e00-\u9fff]", text):
        return "A_permanent_exclude", "no_semantic_text", "keep_excluded"
    # Statistical prose was previously mistaken for references. Multiple actual
    # bibliographic entries, without a body-text boundary, are required here.
    entries = len(re.findall(r"\b(?:19|20)\d{2}\s*[;:]\s*\d+|\[J\]|\b10\.\d{4,9}/", text))
    recommendation = re.search(r"\b(?:should|must|recommend|unless)\b|\u5e94\u5f53|\u5efa\u8bae", text, re.I)
    numbered_entries = len(re.findall(r"(?:^|\s)(?:\[\d+\]|\d+\.)\s+[A-Z\u4e00-\u9fff]", text))
    if "reference_list" in types | reasons and entries >= 2 and numbered_entries >= 2 and not recommendation and "\n" not in text:
        return "A_permanent_exclude", "multiple_bibliographic_entries_without_body_boundary", "keep_excluded"
    if "copyright" in types | reasons and re.search(r"all rights reserved|no part of this publication", text, re.I):
        if not re.search(r"\b(?:patients?|assess|reposition|should|must|recommend|airway|ventilation|tube|tracheostomy|treatment)\b", text, re.I):
            return "A_permanent_exclude", "explicit_publication_rights_only", "keep_excluded"
    risks = sorted(reasons & PAGE_RISKS)
    if risks:
        return "D_source_page_review", ";".join(risks), "verify_source_layout_numbers_conditions_before_recovery"
    if "document_excluded" in reasons or chunk.get("include") != "yes":
        return "H_unresolved", "document_exclusion_requires_scope_and_source_appraisal", "review_document_admission_reason"
    if audit.get("duplicate_exact_group") or audit.get("duplicate_near_group"):
        return "H_unresolved", "duplicate_without_admitted_exact_canonical", "verify_canonical_and_context"
    if "reference_list" in types | reasons and "\n" in text and entries:
        return "C_rechunk_required", "mixed_bibliography_and_body_boundary", "split_after_source_boundary_check"
    if is_true(audit.get("audit_too_long")):
        return "C_rechunk_required", "chunk_exceeds_audited_length", "rechunk_preserving_conditions"
    if is_true(audit.get("audit_too_short")) or is_true(audit.get("audit_low_information")):
        heading = bool(re.fullmatch(
            r"[\d.\s]*(?:.*(?:\u63a8\u8350\u610f\u89c1\u8981\u70b9|\u9884\u9632\u6307\u5357)|"
            r"\u53c2\u8003\u6587\u732e|Introduction|Methods|Results|Discussion|Conclusion|References|Acknowledgements)", text, re.I))
        if heading or re.fullmatch(r"(?:see|refer to)\s+(?:table|figure)\s*\d+[.\s]*", text, re.I):
            return "G_low_information", "standalone_heading_or_cross_reference", "retain_with_parent_context_only"
        if CLINICAL.search(text) or len(text) >= 20:
            return "C_rechunk_required", "short_fragment_requires_adjacent_context", "rechunk_with_original_neighbors"
        return "G_low_information", "isolated_low_information_fragment", "keep_excluded"
    if CLINICAL.search(text):
        return "B_review_recoverable", "substantive_text_without_explicit_layout_risk", "human_check_then_separate_index_build"
    return "H_unresolved", "insufficient_evidence_for_automatic_classification", "manual_review"


def preserve_reviews(rows, path, key_fields, identity_fields, human_fields):
    """Do not carry a review onto changed text or silently discard a review."""
    if not Path(path).exists():
        return
    old = {}
    for row in read_csv(path):
        key = tuple(str(row[k]) for k in key_fields)
        if key in old:
            raise ValueError(f"Duplicate review key in {path}: {key}")
        old[key] = row
    new_keys = set()
    for row in rows:
        key = tuple(str(row[k]) for k in key_fields)
        new_keys.add(key)
        previous = old.get(key, {})
        reviewed = any(previous.get(k, "") not in ("", "False", "false") for k in human_fields)
        if reviewed and any(previous.get(k) != str(row[k]) for k in identity_fields):
            raise ValueError(f"Review identity changed or is unbound: {key}; migrate the review explicitly")
        for field in human_fields:
            if field in previous:
                row[field] = previous[field]
    for key, row in old.items():
        if key not in new_keys and any(row.get(k, "") not in ("", "False", "false") for k in human_fields):
            raise ValueError(f"Refusing to drop reviewed record {key}")


def page_priority(doc_id, page, text, reasons):
    signal = SAFETY.search(text)
    if (doc_id, page) in KNOWN_PAGES:
        return "P0", "known_safety_or_numeric_source_issue"
    if signal:
        return "P0", "safety_text:" + signal.group(0)
    if re.search(r"number|unit|energy|interval|frequency|dose|recommend|safety", reasons, re.I):
        return "P0", "safety_preprocessing_flag"
    if not text.strip() or re.search(r"table|figure|column|layout|cross_page|garbled|ocr|blank", reasons, re.I):
        return "P1", "layout_or_unreadable_source_cannot_be_assumed_low_risk"
    if re.search(r"risk|incidence|prevalence|mechanism|background|discussion", text, re.I):
        return "P2", "background_or_risk_discussion"
    if re.search(r"reference|content_loss", reasons):
        return "P2", "possible_body_text_loss_requires_check"
    return "P3", "no_high_risk_signal_detected"


def human_page_review_complete(row):
    if not (is_true(row.get("human_review_completed")) and is_true(row.get("source_image_checked"))):
        return False
    if not all(row.get(k) for k in ("reviewer", "review_date", "notes", "source_pdf_sha256", "page_text_sha256")):
        return False
    if str(row["reviewer"]).strip().lower() in {"codex", "assistant", "coding assistant", "ai"}:
        return False
    try:
        date.fromisoformat(row["review_date"])
    except (TypeError, ValueError):
        return False
    if str(row.get("allow_retrieval", "")).lower() not in {"true", "false"}:
        return False
    checks = ("text_order_correct", "table_relation_correct", "number_verified", "unit_verified", "condition_verified", "exception_verified")
    allowed = {"true", "not_applicable"} if is_true(row["allow_retrieval"]) else {"true", "false", "not_applicable", "unresolved"}
    return all(str(row.get(k, "")).lower() in allowed for k in checks)


def build_triage(config, review_dir, summary_path):
    corpus = AuditedCorpus(config)
    with config.chunks_path.open(encoding="utf-8") as handle:
        chunks = {c["chunk_id"]: c for c in map(json.loads, filter(str.strip, handle))}
    canonical_by_text = {}
    for cid, chunk in sorted(corpus.chunks.items()):
        if cid not in corpus.blocked_chunk_ids:
            canonical_by_text.setdefault((chunk["doc_id"], re.sub(r"\s+", " ", chunk["text"]).strip()), cid)
    rows = []
    excluded_ids = set(chunks) - set(corpus.chunks)
    for cid in sorted(excluded_ids):
        chunk, audit = chunks[cid], corpus.audit[cid]
        canonical = canonical_by_text.get((chunk["doc_id"], re.sub(r"\s+", " ", chunk["text"]).strip()), "")
        category, reason, action = classify(chunk, audit, canonical)
        row = {k: chunk.get(k, "") for k in ("chunk_id", "doc_id", "title", "page_start", "page_end", "text", "source_type", "evidence_tier", "source_pdf_sha256")}
        row.update({k: json.dumps(chunk.get(k, []), ensure_ascii=False) for k in ("population", "device_type", "topic")})
        row.update({k: "" for k in CHUNK_HUMAN_FIELDS})
        row.update(current_include=chunk.get("include", ""), current_review_status=chunk.get("review_status", ""),
                   exclusion_category=category, suspected_reason=reason, recommended_action=action,
                   needs_page_review=category[0] in "CDH", recoverable="candidate_only" if category[0] in "BCD" else "undetermined" if category[0] == "H" else "false",
                   current_include_in_vector=audit["include_in_vector"], current_include_in_retrieval=audit["include_in_retrieval"],
                   include_in_vector=False, text_sha256=audit["text_sha256"], original_exclusion_reasons=audit["retrieval_exclusion_reasons"],
                   canonical_chunk_id=canonical if category[0] == "F" else "", rules_version=RULES_VERSION)
        rows.append(row)
    candidate_path = config.logs_dir / "pages_to_review.csv"
    candidates_by_key = {}
    for row in read_csv(candidate_path):
        key = row["doc_id"], int(row["page"])
        if key in candidates_by_key:
            raise ValueError(f"Duplicate candidate page: {key}")
        candidates_by_key[key] = row["reasons"]
    page_chunks = defaultdict(set)
    for chunk in chunks.values():
        for page in chunk["source_pages"]:
            page_chunks[(chunk["doc_id"], page)].add(chunk["chunk_id"])
    documents, page_rows, supplemental = {}, [], []
    keys = {**dict.fromkeys(KNOWN_PAGES, "known_issue_supplement"), **candidates_by_key}
    for (doc_id, number), reasons in sorted(keys.items()):
        if doc_id not in documents:
            document = json.loads((config.parsed_dir / f"{doc_id}.json").read_text(encoding="utf-8"))
            if sha256_file(Path(document["source_file"])) != document["sha256"]:
                raise ValueError(f"Source changed: {doc_id}")
            documents[doc_id] = document
        document = documents[doc_id]
        page = next(p for p in document["pages"] if p["page"] == number)
        text = page.get("text", "") + "\n" + page.get("retrieval_text", "")
        priority, reason = page_priority(doc_id, number, text, reasons)
        ids = page_chunks[(doc_id, number)]
        admitted = ids & corpus.chunks.keys()
        available = admitted - corpus.blocked_chunk_ids
        row = dict(doc_id=doc_id, pdf_page=number, review_priority=priority, priority_reason=reason,
                   issue_type=reasons, source_pdf_sha256=document["sha256"], page_text_sha256=text_hash(text),
                   source_file=document["source_file"], indexed_chunk_count=len(admitted), retrievable_chunk_count=len(available),
                   excluded_chunk_count=len(ids & excluded_ids), containment="not_retrievable" if not available else "retrievable_requires_review",
                   known_issue=(doc_id, number) in KNOWN_PAGES)
        row.update({k: "" for k in PAGE_HUMAN_FIELDS})
        (page_rows if (doc_id, number) in candidates_by_key else supplemental).append(row)
    review_dir = Path(review_dir)
    preserve_reviews(rows, review_dir / "excluded_chunk_review.csv", ("chunk_id",), ("source_pdf_sha256", "text_sha256"), CHUNK_HUMAN_FIELDS)
    for filename, group in (("page_review_priorities.csv", page_rows), ("supplemental_page_review.csv", supplemental)):
        preserve_reviews(group, review_dir / filename, ("doc_id", "pdf_page"), ("source_pdf_sha256", "page_text_sha256"), PAGE_HUMAN_FIELDS)
    counts = Counter(r["exclusion_category"] for r in rows)
    summary = dict(rules_version=RULES_VERSION, status="triaged_not_human_approved", total_chunks=corpus.total,
                   indexed_chunks=len(corpus.chunks), runtime_quarantined_chunk_ids=sorted(corpus.blocked_chunk_ids),
                   current_retrievable_chunks=len(corpus.chunks) - len(corpus.blocked_chunk_ids), total_excluded=len(rows),
                   category_counts={k: counts[k] for k in CATEGORIES}, category_proportions={k: counts[k] / len(rows) if rows else 0 for k in CATEGORIES},
                   candidate_recovery_chunks=sum(counts[k] for k in CATEGORIES if k[0] in "BCD"), confirmed_recovered_chunks=0,
                   page_count=len(page_rows), page_counts={p: sum(r["review_priority"] == p for r in page_rows) for p in PRIORITIES},
                   p0_human_completed=sum(r["review_priority"] == "P0" and human_page_review_complete(r) for r in page_rows),
                   p0_retrievable_pages=sum(r["review_priority"] == "P0" and r["retrievable_chunk_count"] > 0 for r in page_rows),
                   supplemental_page_count=len(supplemental), supplemental_pages=[{k: r[k] for k in ("doc_id", "pdf_page", "review_priority", "retrievable_chunk_count")} for r in supplemental])
    summary["exclusion_distributions"] = {}
    for field in ("doc_id", "source_type", "evidence_tier", "device_type", "population"):
        totals, excluded = Counter(), Counter()
        for cid, chunk in chunks.items():
            for value in tags(chunk.get(field)) or {"unknown"}:
                totals[value] += 1
                excluded[value] += cid in excluded_ids
        summary["exclusion_distributions"][field] = {value: {"total": n, "excluded": excluded[value], "excluded_ratio": excluded[value] / n} for value, n in sorted(totals.items())}
    summary["recovery_focus"] = {}
    for label in ("tier_A", "niv_mask", "nasogastric_tube", "endotracheal_tube", "tracheostomy", "spo2_probe"):
        selected = [r for r in rows if (r["evidence_tier"] == "A" if label == "tier_A" else label in tags(r["device_type"]))]
        summary["recovery_focus"][label] = dict(excluded=len(selected), categories=dict(Counter(r["exclusion_category"] for r in selected)), documents=sorted({r["doc_id"] for r in selected}))
    summary["inputs"] = {str(p): sha256_file(p) for p in (config.chunks_path, config.logs_dir / "chunk_audit.csv", candidate_path, config.manifest_path)}
    write_csv(review_dir / "excluded_chunk_review.csv", rows, CHUNK_FIELDS)
    for filename, group in (("page_review_priorities.csv", page_rows), ("supplemental_page_review.csv", supplemental)):
        write_csv(review_dir / filename, sorted(group, key=lambda r: (r["review_priority"], r["doc_id"], r["pdf_page"])), PAGE_FIELDS)
    summary["outputs"] = {name: sha256_file(review_dir / name) for name in ("excluded_chunk_review.csv", "page_review_priorities.csv", "supplemental_page_review.csv")}
    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary
