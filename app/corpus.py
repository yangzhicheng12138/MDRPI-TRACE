"""Audited corpus identity shared by indexing and evidence consumers."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


PAYLOAD_FIELDS = (
    "chunk_id", "doc_id", "text", "title", "authors", "year", "organization",
    "source_type", "evidence_tier", "population", "device_type", "topic",
    "section", "page_start", "page_end", "page_verified", "language", "include",
    "review_status", "kb_version", "filename", "source_pages", "source_pdf_sha256",
    "source_page_text_sha256", "retrieval_source", "preprocessing_warnings",
    "ocr_artifact_sha256", "ocr_review_scope",
    "journal", "volume", "issue", "pages", "doi", "source_review",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()


def embedding_identity(config) -> dict:
    return {
        "model": str(config.get("embedding", "model", default="BAAI/bge-m3")),
        "revision": config.get("embedding", "revision"),
        "normalize": bool(config.get("embedding", "normalize", default=True)),
        "max_seq_length": int(config.get("embedding", "max_seq_length", default=8192)),
    }


def load_audited_chunks(config) -> tuple[list[dict], dict[str, dict], int, int]:
    audit_path = config.logs_dir / "chunk_audit.csv"
    summary_path = config.logs_dir / "chunk_audit_summary.json"
    if not audit_path.exists() or not summary_path.exists():
        raise ValueError("Current chunk audit is required; run scripts/04_audit_chunks.py with the same config")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (summary.get("input_sha256") != sha256_file(config.chunks_path)
            or summary.get("audit_csv_sha256") != sha256_file(audit_path)
            or summary.get("audit_config_sha256") != json_digest(config.get("audit", default={}))):
        raise ValueError("Stale or modified chunk audit; rerun scripts/04_audit_chunks.py with the same config")
    audit_by_id = {}
    with audit_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            chunk_id = row.get("chunk_id", "")
            if not chunk_id or chunk_id in audit_by_id:
                raise ValueError("Missing or duplicate chunk ID in audit")
            audit_by_id[chunk_id] = row
    chunks, seen, excluded, total = [], set(), 0, 0
    with config.chunks_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            chunk = json.loads(line)
            total += 1
            chunk_id = str(chunk.get("chunk_id", ""))
            if not chunk_id or chunk_id in seen:
                raise ValueError("Missing or duplicate chunk ID in dataset")
            seen.add(chunk_id)
            audit = audit_by_id.get(chunk_id)
            if audit is None or audit.get("text_sha256") != hashlib.sha256(chunk["text"].encode("utf-8")).hexdigest():
                raise ValueError(f"Missing or mismatched audit for {chunk_id}")
            eligible = audit.get("include_in_retrieval")
            if eligible in (None, ""):
                eligible = audit.get("include_in_vector", "")
            if (str(chunk.get("include", "no")).lower() != "yes" or chunk.get("preprocessing_warnings")
                    or chunk.get("boundary_excluded") or chunk.get("page_verified") is False
                    or str(eligible).lower() != "true"):
                excluded += 1
            else:
                chunks.append(chunk)
    if seen != set(audit_by_id) or total != summary.get("total_chunks"):
        raise ValueError("Chunk audit does not cover exactly the current dataset")
    return chunks, audit_by_id, total, excluded


class AuditedCorpus:
    def __init__(self, config):
        self.config = config
        chunks, self.audit, self.total, self.excluded = load_audited_chunks(config)
        self.chunks = {item["chunk_id"]: item for item in chunks}
        self.identity = {
            "chunks_sha256": sha256_file(config.chunks_path),
            "audit_sha256": sha256_file(config.logs_dir / "chunk_audit.csv"),
            "audit_config_sha256": json_digest(config.get("audit", default={})),
            "embedding": embedding_identity(config),
            "collection": config.qdrant_collection,
            "point_count": len(chunks),
        }
        self._watched = [config.chunks_path, config.logs_dir / "chunk_audit.csv",
                         config.logs_dir / "chunk_audit_summary.json"]
        if config.get("preprocessing", "mode") == "clean":
            build_path = config.logs_dir / "clean_build_report.json"
            build = json.loads(build_path.read_text(encoding="utf-8"))
            decisions = config.path("preprocessing", "review_decisions")
            if (build.get("chunks_sha256") != self.identity["chunks_sha256"]
                    or build.get("manifest_sha256") != sha256_file(config.manifest_path)
                    or build.get("decisions_sha256") != sha256_file(decisions)
                    or build.get("documents_failed") != 0):
                raise ValueError("Preprocessing source decisions changed or the clean build is incomplete")
            self._watched.extend([build_path, config.manifest_path, decisions])
        self._stats = {path: self._stat(path) for path in self._watched}
        self._source_stats = {}
        self.blocked_chunk_ids = set()
        for decision in config.get("retrieval", "quarantined_pages", default=[]):
            matched = [item for item in chunks if item["doc_id"] == decision["doc_id"]]
            if not matched or any(item["source_pdf_sha256"] != decision["source_pdf_sha256"] for item in matched):
                raise ValueError("Runtime page quarantine does not match the indexed source")
            if not decision.get("reason") or not decision.get("pages"):
                raise ValueError("Page quarantine requires explicit pages and a reason")
            self.blocked_chunk_ids.update(item["chunk_id"] for item in matched
                                          if set(item["source_pages"]) & set(decision["pages"]))

    @staticmethod
    def _stat(path):
        stat = path.stat()
        return stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size

    def assert_current(self):
        if any(self._stat(path) != stamp for path, stamp in self._stats.items()):
            raise ValueError("Corpus or audit changed; rebuild the index and restart the consumer")

    def verify_source(self, chunk):
        source = (self.config.source_dir / chunk["filename"]).resolve()
        if source.parent != self.config.source_dir.resolve():
            raise ValueError("Source filename escapes the PDF directory")
        stamp = self._stat(source)
        if self._source_stats.get(source) != stamp:
            if sha256_file(source) != chunk.get("source_pdf_sha256"):
                raise ValueError(f"Source PDF changed: {chunk['doc_id']}")
            self._source_stats[source] = stamp

    def verify_payload(self, payload):
        chunk_id = str(payload.get("chunk_id", ""))
        expected = self.chunks.get(chunk_id)
        if expected is None:
            raise ValueError(f"Unaudited chunk in index: {chunk_id}")
        if any(payload.get(field) != expected.get(field) for field in PAYLOAD_FIELDS):
            raise ValueError(f"Indexed evidence differs from audited source: {chunk_id}")
        if payload.get("include_in_retrieval") is not True:
            raise ValueError(f"Chunk is not admitted to retrieval: {chunk_id}")
        self.verify_source(expected)

    def verify_release(self, store):
        path = self.config.path("qdrant", "release_manifest", default=str(self.config.qdrant_path / "release.json"))
        if not path.is_file():
            raise ValueError("No complete index release; run scripts/05_build_vector_db.py first")
        release = json.loads(path.read_text(encoding="utf-8"))
        if release.get("identity") != self.identity or not release.get("verified_complete"):
            raise ValueError("Index release does not match the current corpus, audit or embedding model")
        if store.count() != len(self.chunks):
            raise ValueError("Index point count does not match the complete audited corpus")
        return release
