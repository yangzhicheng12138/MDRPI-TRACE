"""Conservative, deterministic population boundaries for reviewed RAG passages.

Document labels establish the retrieval lane. Opposing population cues quarantine
the whole passage; they never relabel it or edit its source text. This is a scope
guard, not an automated clinical applicability review.
"""

from __future__ import annotations

import re


CLINICAL_POPULATIONS = frozenset({"adult", "pediatric", "neonatal"})
POLICY_VERSION = "exclusive_document_scope_and_passage_guard_v1"
POPULATION_PATTERNS = {
    "adult": re.compile(r"(?<!未)成年人|(?<!未)成人|\badults?\b", re.I),
    "pediatric": re.compile(
        r"儿童|幼儿|婴(?:幼)?儿|儿科|患儿|青少年|未成年|"
        r"\bp(?:a)?ediatric\w*\b|\bchildren\b|\bchild\b|\binfants?\b|"
        r"\bbabies\b|\bbaby\b|\btoddlers?\b|\badolescen\w*\b|\bteenagers?\b", re.I),
    "neonatal": re.compile(
        r"新生儿|早产儿|极低出生体重|超低出生体重|"
        r"\bneonat\w*\b|\bnewborn\w*\b|\bpreterm\b|\bNICU\b|"
        r"\b(?:very|extremely)\s+low\s+birth\s*weight\b|\b(?:VLBW|ELBW)\b|"
        r"\bpremature\s+(?:infants?|babies|neonates?)\b", re.I),
    # A newborn is also an infant. Generic pediatric/infant terminology alone
    # does not contradict a reviewed neonatal document's population label.
    "older_child": re.compile(
        r"幼儿|学龄|青少年|\bolder\s+children\b|\btoddlers?\b|"
        r"\badolescen\w*\b|\bteenagers?\b|\bschool[- ]aged?\b", re.I),
}
PASSAGE_AGE = re.compile(
    r"(?<![\d.])(\d{1,3}(?:\.\d+)?)\s*"
    r"(岁|[- ]?years?[- ]old|[- ]?months?[- ]old|个月龄|月龄|[- ]?days?[- ]old|日龄)", re.I)


def strict_population_isolation(config) -> bool:
    return bool(config.get("filters", "strict_population_isolation", default=False))


def isolated_request(population=None, inferred=()):
    """Narrow a request to one clinical lane, without general/mixed fallback."""
    known = set(inferred) & CLINICAL_POPULATIONS
    explicit = set(population or ()) & CLINICAL_POPULATIONS
    if len(known) > 1:
        return [], "population_scope_ambiguous"
    if known:
        if population is not None and not known <= explicit:
            return [], "query_filter_mismatch"
        return sorted(known), ""
    if len(explicit) > 1:
        return [], "population_scope_ambiguous"
    if explicit:
        return sorted(explicit), ""
    if population is not None:
        return [], "population_scope_unverified"
    # Preserve the project's adult default for population-unspecified queries.
    return ["adult"], ""


def population_decision(payload, population):
    """Return an auditable decision using existing labels and exact source cues."""
    from .query_policy import CASE_AGE
    target = set(population or ())
    actual = set(payload.get("population") or ())
    result = {"policy": POLICY_VERSION, "target": sorted(target),
              "document_population": sorted(actual), "eligible": False,
              "reason": "document_population_not_exclusive", "opposing_cues": []}
    if len(target) != 1 or not target <= CLINICAL_POPULATIONS:
        result["reason"] = "population_scope_ambiguous"
        return result
    if actual != target:
        return result
    lane = next(iter(target))
    opposing = {"adult": ("pediatric", "neonatal"),
                "pediatric": ("adult", "neonatal"),
                "neonatal": ("adult", "older_child")}[lane]
    # Section headings are source context, not the article-wide title or an
    # applicability note. Retain separate offsets so every cue is verifiable.
    for field in ("text", "section"):
        text = str(payload.get(field) or "")
        for label in opposing:
            for match in POPULATION_PATTERNS[label].finditer(text):
                result["opposing_cues"].append({"population": label, "field": field,
                    "quote": match.group(), "start": match.start(), "end": match.end()})
        age_matches = { (m.start(), m.end()): m for pattern in (PASSAGE_AGE, CASE_AGE)
                        for m in pattern.finditer(text) }
        for match in age_matches.values():
            age, unit = float(match[1]), match[2]
            if re.search(r"days?|日龄|天", unit, re.I):
                age_lane = "neonatal" if age < 28 else "pediatric"
            elif re.search(r"months?|月龄|个月", unit, re.I):
                age_lane = "adult" if age >= 216 else "pediatric"
            else:
                age_lane = "adult" if age >= 18 else "pediatric"
            if age_lane and age_lane != lane:
                result["opposing_cues"].append({"population": age_lane, "field": field,
                    "quote": match.group(), "start": match.start(), "end": match.end(),
                    "cue_type": "explicit_age"})
    if result["opposing_cues"]:
        result["reason"] = "opposing_population_in_passage"
        return result
    result.update(eligible=True, reason="exclusive_reviewed_document_scope")
    return result
