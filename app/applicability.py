"""Expose passage-scoped source cues with exact character offsets, without inferring scope."""

import hashlib
import re

from .lexical import DEVICE_ALIASES


POPULATION_CUES = {
    "adult": r"\badults?\b|成人|成年人",
    "pediatric": r"\b(?:pediatric|paediatric|children|child|infants?)\b|儿童|儿科|患儿|婴儿",
    "neonatal": r"\bneonat\w*\b|\bnewborn\w*\b|\bpreterm\b|\bNICU\b|新生儿|早产儿",
}
SETTING_CUES = {
    "surgical_icu": r"surgical\s+(?:icu|intensive care)|外科\s*ICU|外科重症",
    "icu": r"\bicu\b|intensive care|重症监护",
    "ward": r"\bwards?\b|普通病房|病区",
    "acute_care": r"acute care|急性护理",
}
CONDITION_CUE = re.compile(r"\b(?:if|unless|except|when|provided|contraindicat\w*|physician|orders?|compromise)\b|如果|若|除非|医嘱|前提|禁忌|不影响|不能|不得", re.I)
NUMERIC_CUE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:hours?|hrs?|h|minutes?|mins?|days?|d|weeks?|w|times?|%|mm\s*hg)(?!\w)|every shift|each shift|每班|\d+\s*(?:小时|分钟|次|天)", re.I)


def evidence_scope(item):
    text = item["text"]
    def cues(patterns):
        return [{"label": label, "quote": match.group(), "start": match.start(), "end": match.end()}
                for label, pattern in patterns.items() for match in re.finditer(pattern, text, re.I)]
    conditions, numeric = [], []
    for sentence in re.finditer(r".+?(?:(?<=[.!?。！？])(?=\s)|\n|$)", text, re.S):
        start, end = sentence.start(), sentence.end()
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        quote = text[start:end]
        span = {"quote": quote, "start": start, "end": end}
        if CONDITION_CUE.search(quote):
            conditions.append(span)
        if NUMERIC_CUE.search(quote):
            numeric.append({**span, "setting_cues": cues_for_span(SETTING_CUES, quote, start)})
    review = item.get("source_review") or {}
    return {
        "schema_version": "passage_scope_v1",
        "method": "exact_source_cues_not_semantic_applicability_verdict",
        "document_scope": {"population": item.get("population", []), "device_type": item.get("device_type", [])},
        "passage_cues": {"population": cues(POPULATION_CUES), "device_type": cues(DEVICE_ALIASES),
                         "setting": cues(SETTING_CUES), "conditions": conditions, "numeric_statements": numeric},
        "scope_limit": "Document tags and source-review notes do not extend any passage to a different population, device or setting. Missing cues mean unknown, not unrestricted applicability.",
        "source_location": {"chunk_id": item["chunk_id"], "source_pages": item.get("source_pages", []),
                            "section": item.get("section", ""), "excerpt_start": 0, "excerpt_end": len(text),
                            "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
                            "source_pdf_sha256": item.get("source_pdf_sha256", ""),
                            "source_page_text_sha256": item.get("source_page_text_sha256", ""),
                            "unit_id": review.get("unit_id"), "part": review.get("part"), "review_id": review.get("review_id")},
        "source_review_scope": {key: review.get(key) for key in
                               ("review_type", "trust_scope", "human_review_completed", "clinical_validity", "applicability")},
    }


def cues_for_span(patterns, text, offset):
    return [{"label": label, "quote": match.group(), "start": offset + match.start(), "end": offset + match.end()}
            for label, pattern in patterns.items() for match in re.finditer(pattern, text, re.I)]
