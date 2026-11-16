"""Fail-closed release gates for the Phase 3 research Evidence Layer."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from .corpus import AuditedCorpus, json_digest, sha256_file
from .phase3_review import (
    CATEGORIES, KNOWN_PAGES, RULES_VERSION, human_page_review_complete,
    is_true, page_priority, read_csv, text_hash,
)
from .provenance import code_identity, runtime_identity
from .rag import SYSTEM_PROMPT
from .retriever import Retriever


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_review_gate(config, corpus, review_dir):
    review_dir = Path(review_dir)
    with config.chunks_path.open(encoding="utf-8") as handle:
        chunks = {c["chunk_id"]: c for c in map(json.loads, filter(str.strip, handle))}
    rows = read_csv(review_dir / "excluded_chunk_review.csv")
    expected = set(chunks) - set(corpus.chunks)
    if len(rows) != len(expected) or {r["chunk_id"] for r in rows} != expected:
        raise ValueError("Excluded-chunk review must cover exactly the excluded corpus")
    for row in rows:
        chunk = chunks[row["chunk_id"]]
        if (row.get("exclusion_category") not in CATEGORIES or row.get("rules_version") != RULES_VERSION
                or row.get("text") != chunk["text"] or row.get("text_sha256") != text_hash(chunk["text"])
                or row.get("source_pdf_sha256") != chunk["source_pdf_sha256"] or is_true(row.get("include_in_vector"))):
            raise ValueError(f"Invalid or stale excluded-chunk review: {row['chunk_id']}")
    base = read_csv(review_dir / "page_review_priorities.csv")
    supplement = read_csv(review_dir / "supplemental_page_review.csv")
    candidates = {(r["doc_id"], int(r["page"])): r["reasons"] for r in read_csv(config.logs_dir / "pages_to_review.csv")}
    expected_pages = set(candidates) | KNOWN_PAGES
    pages = base + supplement
    if (len(base) != len(candidates) or {(r["doc_id"], int(r["pdf_page"])) for r in base} != set(candidates)
            or len(pages) != len(expected_pages) or {(r["doc_id"], int(r["pdf_page"])) for r in pages} != expected_pages):
        raise ValueError("Page review coverage is incomplete or duplicates known pages")
    available_by_page = defaultdict(set)
    for cid, chunk in corpus.chunks.items():
        if cid not in corpus.blocked_chunk_ids:
            for page in chunk["source_pages"]:
                available_by_page[(chunk["doc_id"], page)].add(cid)
    documents, blockers = {}, []
    complete = Counter()
    unreviewed_available = []
    for row in pages:
        key = row["doc_id"], int(row["pdf_page"])
        if key[0] not in documents:
            documents[key[0]] = read_json(config.parsed_dir / (key[0] + ".json"))
        document = documents[key[0]]
        source_page = next(p for p in document["pages"] if p["page"] == key[1])
        text = source_page.get("text", "") + "\n" + source_page.get("retrieval_text", "")
        required_priority, _ = page_priority(*key, text, candidates.get(key, "known_issue_supplement"))
        if (row.get("source_pdf_sha256") != document["sha256"] or row.get("page_text_sha256") != text_hash(text)
                or row.get("review_priority") not in {"P0", "P1", "P2", "P3"}
                or row["review_priority"] > required_priority):
            raise ValueError(f"Stale page review or downgraded risk: {key}")
        approved = human_page_review_complete(row)
        if approved:
            complete[row["review_priority"]] += 1
        available = sorted(available_by_page[key])
        if row["review_priority"] == "P0" and not approved:
            blockers.append({"gate": "human_P0_review", "doc_id": key[0], "pdf_page": key[1]})
        if row["review_priority"] == "P1" and not approved and available:
            blockers.append({"gate": "P1_unreviewed_retrievable", "doc_id": key[0], "pdf_page": key[1]})
        if row["review_priority"] in {"P0", "P1"} and not approved and available:
            unreviewed_available.append({"doc_id": key[0], "pdf_page": key[1], "chunk_ids": available})
        if approved and not is_true(row["allow_retrieval"]) and available:
            blockers.append({"gate": "denied_page_still_retrievable", "doc_id": key[0], "pdf_page": key[1], "chunk_ids": available})
    return {"passed": not blockers, "excluded_chunks": len(rows), "categories": dict(Counter(r["exclusion_category"] for r in rows)),
            "original_page_count": len(base), "original_priority_counts": dict(Counter(r["review_priority"] for r in base)),
            "supplemental_page_count": len(supplement), "human_completed": dict(complete),
            "unreviewed_retrievable_pages": unreviewed_available, "blockers": blockers}


def validate_verification_gate(config, corpus, manifest_path):
    manifest = read_json(manifest_path)
    root = config.project_root
    if (manifest.get("passed") is not True or manifest.get("inputs_unchanged") is not True
            or manifest.get("code_identity") != code_identity(root)
            or manifest.get("runtime_config_sha256") != json_digest(config.values)
            or manifest.get("corpus_identity") != corpus.identity):
        raise ValueError("Verification failed or used different code, configuration or corpus")
    for section in ("input_hashes", "artifact_hashes"):
        if not manifest.get(section):
            raise ValueError(f"Missing verification {section}")
        for filename, digest in manifest[section].items():
            if sha256_file(Path(filename)) != digest:
                raise ValueError(f"Verification artifact changed: {filename}")
    reports = manifest.get("reports", {})
    if set(reports) != {"tests", "model_contract", "regression", "crosslingual"}:
        raise ValueError("Required verification reports are missing")
    for path in reports.values():
        if str(Path(path).resolve()) not in manifest["artifact_hashes"]:
            raise ValueError("Verification report is not bound to the manifest")
    tests = Counter()
    for suite in ET.parse(reports["tests"]).getroot().iter("testsuite"):
        for field in ("tests", "failures", "errors", "skipped"):
            tests[field] += int(suite.get(field, 0))
    if tests["tests"] < 160 or any(tests[k] for k in ("failures", "errors", "skipped")):
        raise ValueError("Full automated tests did not pass")
    contract = read_json(reports["model_contract"])
    expected_contract = {"supported", "missing_number", "opposing_sources", "untrusted_source_instruction", "no_sources", "reported_setting", "unreported_setting"}
    contract_rows = contract.get("results", [])
    if (contract.get("passed") is not True or {r["id"] for r in contract_rows} != expected_contract
            or len(contract_rows) != 7 or not all(r.get("passed") is True for r in contract_rows)
            or contract.get("system_prompt") != SYSTEM_PROMPT.replace("MAX_SELECTED", str(config.get("rag", "max_selected_evidence", default=4)))):
        raise ValueError("Seven model contracts, including reported_setting, must pass")
    cross = read_json(reports["crosslingual"])
    cross_dataset = root / "data/eval/crosslingual_contract_tests.json"
    expected_cross = read_json(cross_dataset)
    cross_rows = cross.get("results", [])
    metrics = cross.get("metrics", {})
    if (cross.get("passed") is not True or len(expected_cross) < 30 or len(cross_rows) != len(expected_cross)
            or {r["id"] for r in cross_rows} != {r["id"] for r in expected_cross}
            or cross.get("dataset_sha256") != sha256_file(cross_dataset)
            or not all(r.get("passed") is True for r in cross_rows)
            or metrics.get("false_support_rate") != 0 or metrics.get("false_refusal_rate", 1) >= 0.05
            or metrics.get("error_rate", 1) != 0):
        raise ValueError("Complete current crosslingual contracts must pass with no false support")
    regression = read_json(reports["regression"])
    dataset_path = root / "data/retrieval_tests/rag_cases_v1.json"
    cases = read_json(dataset_path)["cases"]
    rows = regression.get("rows", [])
    if (regression.get("passed") is not True or regression.get("retrieve_only") is not False
            or len(rows) != len(cases) or len(cases) != 40 or {r["id"] for r in rows} != {r["id"] for r in cases}
            or regression.get("cases_sha256") != sha256_file(dataset_path)
            or not all(r.get("passed") is True for r in rows)
            or regression.get("source_anchor_hit_at_8") != 1 or regression.get("source_anchor_citation_hit") != 1
            or regression.get("refusal_accuracy") != 1 or regression.get("all_citations_exact_and_traceable") is not True):
        raise ValueError("Current 40-case regression, retrieval and citation checks must pass")
    for report in (contract, cross, regression):
        if report.get("runtime_code_sha256") != runtime_identity()["sha256"] or report.get("runtime_config_sha256") != json_digest(config.values):
            raise ValueError("Reports were produced with different runtime code or settings")
    return {"passed": True, "tests": dict(tests), "model_contract_count": len(contract_rows),
            "crosslingual_count": len(cross_rows), "crosslingual_metrics": metrics,
            "regression_count": len(rows), "citation_count": regression["citation_count"],
            "source_anchor_mrr_at_8": regression["source_anchor_mrr_at_8"], "manifest": str(Path(manifest_path).resolve()),
            "manifest_sha256": sha256_file(Path(manifest_path))}


def readiness(config, review_dir, verification=None):
    corpus = AuditedCorpus(config)
    release_path = config.qdrant_path / "release.json"
    index_release = read_json(release_path)
    if index_release.get("identity") != corpus.identity or index_release.get("verified_complete") is not True:
        raise ValueError("Existing index release is incomplete or has a different identity")
    documents = {}
    manifest = read_csv(config.manifest_path)
    for row in manifest:
        path = (config.source_dir / row["filename"]).resolve()
        if path.parent != config.source_dir.resolve() or sha256_file(path) != row["sha256"]:
            raise ValueError(f"Source PDF changed: {row['doc_id']}")
        documents[row["doc_id"]] = {"doc_id": row["doc_id"], "filename": row["filename"], "sha256": row["sha256"], "title": row["title"]}
    review = validate_review_gate(config, corpus, review_dir)
    try:
        validation = validate_verification_gate(config, corpus, verification) if verification else {"passed": False, "error": "Current verification manifest required"}
    except (OSError, ValueError, KeyError, TypeError, ET.ParseError) as exc:
        validation = {"passed": False, "error": str(exc)}
    blockers = list(review["blockers"])
    if not validation["passed"]:
        blockers.append({"gate": "engineering_verification", "reason": validation["error"]})
    return {"release_id": "evidence_layer_v1", "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "ready_to_freeze" if not blockers else "blocked_not_frozen", "release_authorized": not blockers,
            "blockers": blockers, "review": review, "verification": validation,
            "source_document_count": len(documents), "documents": list(documents.values()),
            "corpus_identity": corpus.identity, "runtime_quarantined_chunk_ids": sorted(corpus.blocked_chunk_ids),
            "retrievable_chunks": len(corpus.chunks) - len(corpus.blocked_chunk_ids),
            "embedding": config.get("embedding"), "reranker": config.get("reranker"),
            "qdrant": {**config.get("qdrant"), "release_manifest_sha256": sha256_file(release_path)},
            "chunk_audit_rules": config.get("audit"), "review_rules_version": RULES_VERSION,
            "metadata_rules_sha256": sha256_file(config.project_root / "app/metadata.py"),
            "retriever": {**config.get("retrieval"), "clinical_source_types": Retriever.CLINICAL_TYPES, "evidence_source_types": Retriever.EVIDENCE_TYPES},
            "runtime_filters": config.get("filters"), "rag": config.get("rag"),
            "citation_validator_sha256": sha256_file(config.project_root / "app/rag.py"),
            "code_identity": code_identity(config.project_root), "runtime_config_sha256": json_digest(config.values),
            "review_artifacts": {name: sha256_file(Path(review_dir) / name) for name in ("excluded_chunk_review.csv", "page_review_priorities.csv", "supplemental_page_review.csv")},
            "formal_cases_loaded": False, "clinical_validity_established": False}


def write_release(report, path, freeze=False):
    if freeze and report.get("status") != "ready_to_freeze":
        raise ValueError("Evidence freeze blocked: human source review and current verification must pass")
    result = {**report, "status": "frozen" if freeze else report["status"]}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return result
