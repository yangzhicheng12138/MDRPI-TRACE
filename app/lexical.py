"""Small, deterministic BM25 index for the *admitted* corpus, without a second DB."""

from __future__ import annotations

from collections import Counter, defaultdict
import math
import re
import unicodedata

from .metadata import DEVICE_TYPES
from .population_isolation import isolated_request, population_decision, strict_population_isolation


DEVICE_ALIASES = {
    "niv_mask": r"\bniv\b|non[- ]invasive ventilation|无创(?:通气|呼吸)|无创面罩",
    "nasogastric_tube": r"\bngt\b|nasogastric|鼻胃管|鼻饲管|胃管",
    "endotracheal_tube": r"\bett\b|endotracheal|气管插管",
    "tracheostomy": r"tracheostom\w*|气管切开|气切",
    "spo2_probe": r"\bspo2\b|pulse oximet\w*|血氧(?:探头|传感器)",
    "cervical_collar": r"cervical collar|颈托",
}
SETTING_ALIASES = {
    "surgical_icu": r"surgical\s+(?:icu|intensive care)|外科\s*ICU|外科重症",
    "icu": r"\bicu\b|intensive care|重症监护",
}


def english_search_terms(query):
    """Controlled query vocabulary, not an LLM translation or a source-conditioned answer."""
    english_devices = {"niv_mask": "NIV non-invasive ventilation mask", "nasogastric_tube": "nasogastric tube NGT",
                       "endotracheal_tube": "endotracheal tube", "tracheostomy": "tracheostomy",
                       "spo2_probe": "pulse oximetry sensor", "cervical_collar": "cervical collar"}
    terms = []
    # Keep study acronyms actually present in the user's query, never from gold labels.
    terms.extend(re.findall(r"\b[A-Z][A-Z0-9-]{2,}\b", re.sub(r"\bDOC\d+\b", "", query)))
    terms.extend(english_devices[name] for name, pattern in DEVICE_ALIASES.items() if re.search(pattern, query, re.I))
    terms.extend(label.replace("_", " ").replace("icu", "ICU") for label, pattern in SETTING_ALIASES.items()
                 if re.search(pattern, query, re.I))
    concepts = {
        "assessment inspection": r"评估|检查|assess|inspect",
        "securement resecurement": r"重新固定|固定|resecure|securement",
        "frequency interval": r"多久|多长时间|频次|间隔|frequency|interval",
        "physician order exception": r"医嘱|physician|orders?",
        "skin mucosal staging": r"皮肤|黏膜|分期|mucosa|staging",
        "risk factors": r"风险|危险因素|risk factors",
        "study denominator sample": r"分母|根因分析|denominator",
        "replacement duration manufacturer": r"留置时间|更换|manufacturer|replacement",
    }
    terms.extend(term for term, pattern in concepts.items() if re.search(pattern, query, re.I))
    return " ".join(dict.fromkeys(terms)) if terms else ""


def tokenize(text: str) -> list[str]:
    """Preserve numbers/acronyms and Chinese bigrams; share explicit device aliases."""
    text = unicodedata.normalize("NFKC", text).lower()
    output = re.findall(r"[a-z][a-z0-9_]*|\d+(?:\.\d+)?", text)
    for sequence in re.findall(r"[\u3400-\u9fff]+", text):
        output.extend(sequence)
        output.extend(sequence[i:i + 2] for i in range(len(sequence) - 1))
    output.extend("device:" + name for name, pattern in DEVICE_ALIASES.items()
                  if re.search(pattern, text))
    output.extend("setting:" + name for name, pattern in SETTING_ALIASES.items()
                  if re.search(pattern, text, re.I))
    return output


def metadata_matches(payload, config, *, population=None, device_type=None, source_type=None,
                     evidence_tier=None, review_status=None, doc_id=None, blocked_ids=()):
    """Mirror Retriever._filter, including strict general fallback and adult exclusions."""
    if payload.get("chunk_id") in blocked_ids:
        return False
    if config.get("retrieval", "require_audited_index", default=False) and (
            payload.get("include") != "yes" or payload.get("include_in_retrieval") is not True
            or payload.get("page_verified") is not True):
        return False
    actual_people = set(payload.get("population") or [])
    if strict_population_isolation(config):
        population, reason = isolated_request(population)
        if reason or not population_decision(payload, population)["eligible"]:
            return False
    if population and (not actual_people.intersection(population) or (
            not set(population).intersection({"pediatric", "neonatal"})
            and actual_people.intersection({"pediatric", "neonatal"}))):
        return False
    if population and set(population) & {"pediatric", "neonatal"} and "adult" not in population and "adult" in actual_people:
        return False
    if device_type:
        # Import only at call time to avoid a retriever/lexical import cycle.
        from .retriever import Retriever
        allowed = set(device_type)
        strict = config.get("retrieval", "strict_device", default=False)
        fallback = config.get("retrieval", "allow_general_fallback", default=True)
        if not strict:
            for item in device_type:
                allowed.update(v for v in Retriever.DEVICE_EXPANSION.get(item, [item])
                               if v != "general_mdrpi")
        actual = set(payload.get("device_type") or [])
        if strict and fallback and "general_mdrpi" not in allowed:
            matches = bool(actual.intersection(allowed)) or (
                "general_mdrpi" in actual and not actual.intersection(DEVICE_TYPES - {"general_mdrpi"}))
        else:
            if fallback:
                allowed.add("general_mdrpi")
            matches = bool(actual.intersection(allowed))
        if not matches:
            return False
    for key, values in (("source_type", source_type), ("evidence_tier", evidence_tier),
                        ("review_status", review_status), ("doc_id", doc_id)):
        if values and payload.get(key) not in values:
            return False
    return True


class BM25Index:
    """Inverted index; callers supply exactly the verified corpus, never upload data."""

    def __init__(self, chunks, *, k1=1.5, b=0.75):
        if not math.isfinite(k1) or k1 <= 0 or not math.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("Invalid BM25 parameters")
        self.k1, self.b = k1, b
        self.payloads = {item["chunk_id"]: item for item in chunks}
        self.lengths, self.postings = {}, defaultdict(dict)
        for cid, item in sorted(self.payloads.items()):
            # Text dominates; titles/sections also retain rare study names and acronyms.
            counts = Counter(tokenize(" ".join(str(item.get(k) or "") for k in ("title", "section", "text"))))
            self.lengths[cid] = sum(counts.values())
            for token, count in counts.items():
                self.postings[token][cid] = count
        self.average_length = sum(self.lengths.values()) / max(1, len(self.lengths)) or 1.0

    def search(self, query, *, top_k=20, predicate=None):
        if not query.strip() or not 1 <= top_k <= 200:
            raise ValueError("BM25 requires a query and top_k between 1 and 200")
        scores = defaultdict(float)
        size = len(self.payloads)
        for token in sorted(set(tokenize(query))):
            postings = self.postings.get(token, {})
            inverse = math.log(1 + (size - len(postings) + .5) / (len(postings) + .5))
            for cid, frequency in postings.items():
                norm = self.k1 * (1 - self.b + self.b * self.lengths[cid] / self.average_length)
                scores[cid] += inverse * frequency * (self.k1 + 1) / (frequency + norm)
        ranked = sorted(scores, key=lambda cid: (-scores[cid], cid))
        return [(self.payloads[cid], scores[cid]) for cid in ranked
                if scores[cid] > 0 and (predicate is None or predicate(self.payloads[cid]))][:top_k]


def reciprocal_rank_fusion(dense, lexical, *, k=60, top_k=40):
    """Fuse rankings, not incomparable cosine/BM25 score scales. Ties use chunk IDs."""
    from dataclasses import replace
    if k < 1 or not 1 <= top_k <= 200:
        raise ValueError("Invalid RRF parameters")
    items, scores, ranks, channels = {}, defaultdict(float), defaultdict(dict), defaultdict(list)
    for channel, ranking in (("dense", dense), ("bm25", lexical)):
        seen = set()
        for rank, item in enumerate(ranking, 1):
            if item.chunk_id in seen:
                continue
            seen.add(item.chunk_id)
            scores[item.chunk_id] += 1 / (k + rank)
            ranks[item.chunk_id][channel] = rank
            channels[item.chunk_id].append(channel)
            previous = items.get(item.chunk_id)
            if previous is None:
                items[item.chunk_id] = item
            else:
                items[item.chunk_id] = replace(previous, bm25_score=item.bm25_score if channel == "bm25" else previous.bm25_score,
                                               retrieval_routes=sorted(set(previous.retrieval_routes + item.retrieval_routes)))
    ordered = sorted(items, key=lambda cid: (-scores[cid], cid))[:top_k]
    return [replace(items[cid], rank=rank, fusion_score=scores[cid],
                    dense_rank=ranks[cid].get("dense"), lexical_rank=ranks[cid].get("bm25"),
                    retrieval_channels=channels[cid]) for rank, cid in enumerate(ordered, 1)]
