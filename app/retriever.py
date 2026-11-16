"""Audited, metadata-aware dense or BM25/RRF retrieval with two source routes."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Any
from types import SimpleNamespace
from threading import RLock
import time

from .embedding import get_embedding_service
from .vector_store import VectorStore
from .metadata import DEVICE_TYPES, POPULATIONS, SOURCE_TYPES
from .population_isolation import (isolated_request,
                                  population_decision, strict_population_isolation)


@dataclass
class SearchResult:
    """A traceable chunk returned from vector search."""

    rank: int
    chunk_id: str
    doc_id: str
    text: str
    title: str
    year: int | None
    authors: str
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
    language: str
    include: str
    review_status: str
    kb_version: str
    vector_score: float | None
    reranker_score: float | None = None
    filename: str = ""
    source_pages: list[int] = field(default_factory=list)
    source_pdf_sha256: str = ""
    source_page_text_sha256: str = ""
    retrieval_source: str = ""
    ocr_artifact_sha256: str = ""
    ocr_review_scope: str = ""
    journal: str = ""
    volume: str = ""
    issue: str = ""
    pages: str = ""
    doi: str = ""
    source_review: dict = field(default_factory=dict)
    bm25_score: float | None = None
    fusion_score: float | None = None
    dense_rank: int | None = None
    lexical_rank: int | None = None
    retrieval_channels: list[str] = field(default_factory=list)
    retrieval_routes: list[str] = field(default_factory=list)
    reranker_query_scores: dict[str, float] = field(default_factory=dict)
    population_isolation: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _condition(key: str, values: list[str]):
    from qdrant_client.models import FieldCondition, MatchAny
    return FieldCondition(key=key, match=MatchAny(any=values))


class Retriever:
    """Run filtered vector search and a guideline/research dual route."""

    CLINICAL_TYPES = ["guideline", "consensus", "standard"]
    # Reviewed educational/safety material remains its original source type;
    # admission and population filters apply just as they do to research papers.
    EVIDENCE_TYPES = ["systematic_review", "meta_analysis", "evidence_summary", "scoping_review", "integrative_review", "original_research",
                      "poster", "patient_safety_alert", "other"]
    DEVICE_EXPANSION = {
        "niv_mask": ["niv_mask", "general_mdrpi"],
        "nasogastric_tube": ["nasogastric_tube", "general_mdrpi"],
        "endotracheal_tube": ["endotracheal_tube", "tracheostomy", "general_mdrpi"],
        "tracheostomy": ["tracheostomy", "endotracheal_tube", "general_mdrpi"],
        "spo2_probe": ["spo2_probe", "general_mdrpi"],
    }

    def __init__(self, config, logger: logging.Logger | None = None, store: VectorStore | None = None) -> None:
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.store = store or VectorStore(config, logger=self.logger)
        self.embedder = get_embedding_service(config, logger=self.logger)
        self.corpus = None
        self.require_audited_index = bool(config.get("retrieval", "require_audited_index", default=False))
        self.lexical_index = None
        self._index_lock = RLock()
        self._index_ready = None
        self._index_checked_at = None

    def _filter(self, population, device_type, source_type, evidence_tier, review_status, doc_id=None):
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        conditions = []
        must_not = []
        strict_population = strict_population_isolation(self.config)
        if strict_population:
            population, reason = isolated_request(population)
            if reason:
                raise ValueError(reason)
        if population:
            conditions.append(_condition("population", list(population)))
            if strict_population:
                must_not.append(_condition("population", sorted(POPULATIONS - set(population))))
            elif not set(population) & {"pediatric", "neonatal"}:
                must_not.append(_condition("population", ["pediatric", "neonatal"]))
            elif "adult" not in population:
                must_not.append(_condition("population", ["adult"]))
        if device_type:
            strict = bool(self.config.get("retrieval", "strict_device", default=False))
            fallback = bool(self.config.get("retrieval", "allow_general_fallback", default=True))
            allowed = set(device_type)
            if not strict:
                for item in device_type:
                    allowed.update(value for value in self.DEVICE_EXPANSION.get(item, [item]) if value != "general_mdrpi")
            if strict and fallback and "general_mdrpi" not in allowed:
                # A general tag must not turn an unrelated device-specific paper
                # into evidence for the requested device.
                conditions.append(Filter(should=[
                    _condition("device_type", sorted(allowed)),
                    Filter(must=[_condition("device_type", ["general_mdrpi"])],
                           must_not=[_condition("device_type", sorted(DEVICE_TYPES - {"general_mdrpi"}))]),
                ]))
            else:
                if fallback:
                    allowed.add("general_mdrpi")
                conditions.append(_condition("device_type", sorted(allowed)))
        if source_type:
            conditions.append(_condition("source_type", list(source_type)))
        if evidence_tier:
            conditions.append(_condition("evidence_tier", list(evidence_tier)))
        if review_status:
            conditions.append(_condition("review_status", list(review_status)))
        if doc_id:
            conditions.append(_condition("doc_id", list(doc_id)))
        if self.config.get("retrieval", "require_audited_index", default=False):
            conditions.extend([
                FieldCondition(key="include", match=MatchValue(value="yes")),
                FieldCondition(key="include_in_retrieval", match=MatchValue(value=True)),
                FieldCondition(key="page_verified", match=MatchValue(value=True)),
            ])
        corpus = getattr(self, "corpus", None)
        if strict_population and corpus is not None:
            # Restrict before top-k, so a wrongly labelled high-scoring passage
            # cannot crowd compatible evidence out of either retrieval route.
            cache = getattr(self, "_population_ids", None)
            if cache is None:
                cache = self._population_ids = {}
            lane = tuple(population)
            if lane not in cache:
                cache[lane] = sorted(cid for cid, item in corpus.chunks.items()
                                     if population_decision(item, population)["eligible"])
            conditions.append(_condition("chunk_id", cache[lane]))
        if corpus and corpus.blocked_chunk_ids:
            must_not.append(_condition("chunk_id", sorted(corpus.blocked_chunk_ids)))
        return Filter(must=conditions, must_not=must_not or None) if conditions or must_not else None

    def check_index(self):
        with self._index_lock:
            try:
                if self.require_audited_index and self.corpus is None:
                    from .corpus import AuditedCorpus
                    corpus = AuditedCorpus(self.config)
                    corpus.verify_release(self.store)
                    self.corpus = corpus
                if self.corpus:
                    self.corpus.assert_current()
                self._index_ready = True
            except (ValueError, RuntimeError, OSError):
                self._index_ready = False
                raise
            finally:
                self._index_checked_at = time.time()

    def index_readiness(self):
        """Health must not wait behind an index load or a generation request."""
        acquired = self._index_lock.acquire(blocking=False)
        if acquired:
            try:
                try:
                    self.check_index()
                except (ValueError, RuntimeError, OSError):
                    pass  # check_index records the actual failed observation.
            finally:
                self._index_lock.release()
        return {"ready": self._index_ready, "checked_at": self._index_checked_at,
                "cached_while_busy": not acquired}

    def close(self):
        self.store.close()

    @staticmethod
    def _result(point: Any, rank: int) -> SearchResult:
        payload = dict(point.payload or {})
        def integer(value):
            try:
                return int(value) if value not in (None, "") else None
            except (TypeError, ValueError):
                return None
        def boolean(value):
            return value if isinstance(value, bool) else str(value).lower() == "true"
        return SearchResult(
            rank=rank,
            chunk_id=str(payload.get("chunk_id", "")),
            doc_id=str(payload.get("doc_id", "")),
            text=str(payload.get("text", "")),
            title=str(payload.get("title", "")),
            year=integer(payload.get("year")),
            authors=str(payload.get("authors", "")),
            organization=str(payload.get("organization", "")),
            source_type=str(payload.get("source_type", "")),
            evidence_tier=str(payload.get("evidence_tier", "")),
            population=list(payload.get("population", []) or []),
            device_type=list(payload.get("device_type", []) or []),
            topic=list(payload.get("topic", []) or []),
            section=str(payload.get("section", "")),
            page_start=integer(payload.get("page_start")),
            page_end=integer(payload.get("page_end")),
            page_verified=boolean(payload.get("page_verified")),
            language=str(payload.get("language", "")),
            include=str(payload.get("include", "")),
            review_status=str(payload.get("review_status", "")),
            kb_version=str(payload.get("kb_version", "")),
            vector_score=float(getattr(point, "score", 0.0)),
            filename=str(payload.get("filename") or ""),
            source_pages=list(payload.get("source_pages") or []),
            source_pdf_sha256=str(payload.get("source_pdf_sha256") or ""),
            source_page_text_sha256=str(payload.get("source_page_text_sha256") or ""),
            retrieval_source=str(payload.get("retrieval_source") or ""),
            ocr_artifact_sha256=str(payload.get("ocr_artifact_sha256") or ""),
            ocr_review_scope=str(payload.get("ocr_review_scope") or ""),
            journal=str(payload.get("journal") or ""),
            volume=str(payload.get("volume") or ""),
            issue=str(payload.get("issue") or ""),
            pages=str(payload.get("pages") or ""),
            doi=str(payload.get("doi") or ""),
            source_review=dict(payload.get("source_review") or {}),
        )

    def retrieve(self, query: str, top_k: int = 10, population: list[str] | None = None,
                 device_type: list[str] | None = None, source_type: list[str] | None = None,
                 evidence_tier: list[str] | None = None, review_status: list[str] | None = None,
                 use_filters: bool = True, doc_id: list[str] | None = None) -> list[SearchResult]:
        """Retrieve chunks with optional controlled-vocabulary filters."""

        if not query.strip():
            raise ValueError("query must not be empty")
        if top_k < 1 or top_k > 200:
            raise ValueError("top_k must be between 1 and 200")
        for name, values, vocabulary in (("population", population, POPULATIONS),
                                          ("device_type", device_type, DEVICE_TYPES),
                                          ("source_type", source_type, SOURCE_TYPES)):
            if values is not None and (not values or not set(values) <= vocabulary):
                raise ValueError(f"Invalid {name} filter")
        self.check_index()
        strict_population = strict_population_isolation(self.config)
        if strict_population:
            from .query_policy import inferred_population
            population, reason = isolated_request(population, inferred_population(query))
            if reason:
                raise ValueError(reason)
        elif use_filters and population is None:
            population = list(self.config.get("filters", "adult_population", default=["adult", "general", "mixed"]))
        vector = self.embedder.embed_query(query)
        query_filter = (self._filter(population, device_type, source_type, evidence_tier, review_status, doc_id)
                        if use_filters else self._filter(population if strict_population else None,
                                                         None, None, None, None))
        points = self.store.search(vector, limit=top_k, query_filter=query_filter)
        if self.corpus:
            for point in points:
                self.corpus.verify_payload(dict(point.payload or {}))
        results = []
        for point in points:
            decision = population_decision(dict(point.payload or {}), population) if strict_population else {}
            if strict_population and not decision["eligible"]:
                continue
            results.append(replace(self._result(point, len(results) + 1), population_isolation=decision))
        return results

    def dual_retrieve(self, query: str, top_k: int | None = None, **filters) -> list[SearchResult]:
        """Merge clinical-guideline and research-evidence routes by chunk ID."""

        strategy = self.config.get("retrieval", "strategy", default="dense")
        if strategy not in {"dense", "hybrid"}:
            raise ValueError("retrieval.strategy must be dense/hybrid")
        limit = int(top_k or self.config.get("retrieval", "vector_top_k", default=20))
        if not 1 <= limit <= 200:
            raise ValueError("top_k must be between 1 and 200")
        clinical_k = int(self.config.get("retrieval", "clinical_top_k", default=10))
        evidence_k = int(self.config.get("retrieval", "evidence_top_k", default=10))
        requested_sources = filters.pop("source_type", None)
        if requested_sources is not None and (not requested_sources or not set(requested_sources) <= SOURCE_TYPES):
            raise ValueError("Invalid source_type filter")
        clinical_types = [item for item in self.CLINICAL_TYPES if not requested_sources or item in requested_sources]
        evidence_types = [item for item in self.EVIDENCE_TYPES if not requested_sources or item in requested_sources]
        clinical = self.retrieve(query, top_k=clinical_k, source_type=clinical_types, **filters) if clinical_types else []
        evidence = self.retrieve(query, top_k=evidence_k, source_type=evidence_types, **filters) if evidence_types else []
        merged: dict[str, SearchResult] = {}
        for item in clinical + evidence:
            current = merged.get(item.chunk_id)
            if current is None or item.vector_score > current.vector_score:
                merged[item.chunk_id] = item
        ordered = sorted(merged.values(), key=lambda item: item.vector_score, reverse=True)
        if strategy == "hybrid":
            from .lexical import reciprocal_rank_fusion
            routes = {item.chunk_id: ["clinical"] for item in clinical}
            routes.update({item.chunk_id: ["research"] for item in evidence})
            ordered = [replace(item, retrieval_routes=routes[item.chunk_id]) for item in ordered]
            lexical = []
            for name, types in (("clinical", clinical_types), ("research", evidence_types)):
                if types:
                    lexical.extend(replace(item, retrieval_routes=[name]) for item in
                                   self.lexical_retrieve(query, source_type=types, **filters))
            lexical.sort(key=lambda item: (-item.bm25_score, item.chunk_id))
            return reciprocal_rank_fusion(ordered, lexical, top_k=limit,
                    k=int(self.config.get("retrieval", "rrf_k", default=60)))
        return [replace(item, rank=rank) for rank, item in enumerate(ordered[:limit], start=1)]

    def lexical_retrieve(self, query, top_k=None, *, population=None, device_type=None,
                         source_type=None, evidence_tier=None, review_status=None,
                         use_filters=True, doc_id=None):
        from .lexical import BM25Index, metadata_matches
        if not self.require_audited_index:
            raise ValueError("Lexical retrieval requires the audited main corpus")
        for name, values, vocabulary in (("population", population, POPULATIONS),
                ("device_type", device_type, DEVICE_TYPES), ("source_type", source_type, SOURCE_TYPES)):
            if values is not None and (not values or not set(values) <= vocabulary):
                raise ValueError(f"Invalid {name} filter")
        self.check_index()
        if self.lexical_index is None:
            # Audit flags are derived from the current audited corpus, never from user input.
            chunks = [{**item, "include_in_retrieval": True} for item in self.corpus.chunks.values()
                      if item["chunk_id"] not in self.corpus.blocked_chunk_ids]
            self.lexical_index = BM25Index(chunks,
                k1=float(self.config.get("retrieval", "bm25_k1", default=1.5)),
                b=float(self.config.get("retrieval", "bm25_b", default=.75)))
        strict_population = strict_population_isolation(self.config)
        if strict_population:
            from .query_policy import inferred_population
            population, reason = isolated_request(population, inferred_population(query))
            if reason:
                raise ValueError(reason)
        elif use_filters and population is None:
            population = self.config.get("filters", "adult_population", default=["adult", "general", "mixed"])
        active = dict(population=population, device_type=device_type, source_type=source_type,
                      evidence_tier=evidence_tier, review_status=review_status, doc_id=doc_id) if use_filters else (
                          {"population": population} if strict_population else {})
        hits = self.lexical_index.search(query, top_k=int(top_k or self.config.get("retrieval", "lexical_top_k", default=20)),
            predicate=lambda payload: metadata_matches(payload, self.config,
                blocked_ids=self.corpus.blocked_chunk_ids, **active))
        results = []
        for rank, (payload, score) in enumerate(hits, 1):
            self.corpus.verify_payload(payload)
            item = self._result(SimpleNamespace(payload=payload, score=0.0), rank)
            decision = population_decision(payload, population) if strict_population else {}
            results.append(replace(item, vector_score=None, bm25_score=score, retrieval_channels=["bm25"],
                                   population_isolation=decision))
        return results
