"""Population-aware query scope, enabled only for reviewed evidence releases."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from .metadata import DEVICE_TYPES, POPULATIONS
from .population_isolation import isolated_request, strict_population_isolation


DEVICE_PATTERNS = {
    "niv_mask": r"无创(?:通气|呼吸)|面罩|鼻罩|\bnasal masks?\b|\b(?:bi)?nasal prongs?\b|(?<![A-Za-z])niv(?![A-Za-z])|non.?invasive|\bbipap\b",
    # Nasal feeding tubes share the existing audited device family. This does
    # not make gastric position checks applicable to a jejunal tube.
    "nasogastric_tube": r"鼻(?:胃|饲|空肠|肠)管|胃管|nasogastric|naso[- ]?(?:jejunal|enteric|intestinal|duodenal)|\bngt?\b|\bnjt\b|\bnj\s+tubes?\b",
    "endotracheal_tube": r"气管(?:插|(?:内)?导)管|经口插管|endotracheal|\btracheal tubes?\b|\bett\b",
    "tracheostomy": r"气管切开|气切|tracheostom",
    # SpO2 alone is a physiological measurement, not a second device in a NIV
    # or orthopedic case. Require probe/sensor wording for that abbreviation.
    "spo2_probe": r"血氧|氧饱和度探头|(?<![A-Za-z])spo\s*2\s*(?:探头|传感器|probes?\b|sensors?\b)|(?:指夹(?:式)?|粘贴(?:式)?|反射(?:式)?)\s*探头|oximet",
    "blood_pressure_cuff": r"血压袖带|测压袖带|充气袖带|blood.?pressure cuff|sphygmomanometer cuff",
    "vascular_access": r"血管通路|静脉留置针|静脉导管|输液接头|无针接头|(?<![A-Za-z])picc(?![A-Za-z])|vascular access|(?:peripheral|central) (?:intravenous|venous) catheter",
    "cervical_collar": r"颈托|颈圈|cervical collar",
    "orthopedic_device": r"石膏|夹板|牵引|矫形|骨科|支具|orthop[ae]edic|splint|\bcast\b|\borthos(?:is|es)\b|\btls[oa]\b|\b(?:hip|knee|ankle|spinal|thoracolumbar(?:[- ]sacral)?)\s+(?:abduction\s+)?braces?\b",
    "cpap": r"(?<![A-Za-z])n?cpap(?![A-Za-z])|continuous positive airway pressure|持续气道正压",
}
DOMAIN = re.compile(r"压力性损伤|压力损伤|压疮|压伤|器械相关|器械(?:下|接触|压迫)|\bmdrpi\b|\bdrpi\b|pressure (?:injur|ulcer)|device.related pressure|(?<![A-Za-z])Braden\s*[- ]?Q(?:D)?(?![A-Za-z])|(?<![A-Za-z])(?:Glamorgan|NSRAS|PPUPET)(?![A-Za-z])", re.I)
NASAL_VENTILATION = re.compile(r"nasal injur.{0,160}(?:positive pressure ventilation|respiratory support)", re.I)
CHILD = re.compile(r"新生儿|早产儿|婴儿|儿童|儿科|患儿|幼儿|孩子|未成年|青少年|neonat|p(?:a)?ediatric|\binfants?\b|\bchildren\b|\bchild\b|\badolescen\w*\b|\bteenagers?\b", re.I)
ADULT = re.compile(r"(?<!未)成年人|(?<!未)成人|\badults?\b", re.I)
NEONATAL = re.compile(r"新生儿|早产儿|\bneonat(?:al|es?)\b|(?<![A-Za-z])nicu(?![A-Za-z])|\bnewborns?\b|\b(?:preterm|premature)\s+(?:infants?|neonates?|babies)\b", re.I)
CASE_AGE = re.compile(r"(?:患者|患儿|patient|年龄|age|生后|出生后)\s*[,，:：]?\s*(?:(?:男性|女性|男|女|male|female)\s*[,，:：]?\s*)?(\d{1,3}(?:\.\d+)?)\s*(岁|[- ]?years?[- ]old|个月|月龄|[- ]?months?[- ]old|天|日龄|[- ]?days?[- ]old)(?!前|后|内|以来)", re.I)
CHILD_AGE = re.compile(r"(?<![\d.-])(\d{1,3}(?:\.\d+)?)\s*(岁|[- ]?years?[- ]old|[- ]?months?[- ]old|个月(?:龄|大)|月龄|[- ]?days?[- ]old|天(?:龄|大)|日龄)", re.I)
INJECTION = re.compile(r"ignore (?:all |the )?(?:previous|system) (?:instructions|prompt)|忽略.{0,12}(?:系统|指令)|泄露.{0,8}(?:提示词|密钥)|system prompt", re.I)


@dataclass(frozen=True)
class QueryScope:
    devices: list[str]
    population: list[str]
    reason: str = ""


class QueryTokenBudgetError(ValueError):
    """A query cannot fit without dropping its detected current question."""


def recognition_text(query):
    """Exclude explicit negative evidence requests from recognition, not input.

    This does not change passage admission or infer another population from an
    exclusion. Positive comparisons remain ambiguous, and explicit case age
    still wins. Keep offsets by masking whole negative request clauses.
    """
    text = unicodedata.normalize("NFKC", query)
    negative = re.compile(
        r"(?:不要|不能|不应|不得|避免|排除|别|而非|而不是|不包括|不包含|不适用|无需)"
        r"[^，,。.;；\n!?！？]{0,100}|"
        r"\b(?:exclude|excluding|without using|rather than|instead of|do not use|don't use|"
        r"not applicable to|no adult evidence|no pediatric evidence|no neonatal evidence)\b"
        r"[^,.;\n!?]{0,160}", re.I)
    def mask(match):
        clause = match[0]
        pivot=re.search(r"而是|但是|但要|只要|仅需|请(?:给|找|查)|需要|想要|希望|"
                        r"重点(?:找|关注)|\bbut\b|\binstead\s+(?:find|use)\b",clause,re.I)
        excluded=clause[:pivot.start()] if pivot else clause
        # 'Do not exclude adults' and 'not only adults' request inclusion.
        if re.match(r"(?:不要|不能|别)(?:忽略|排除|漏掉|漏检)|"
                    r"\b(?:do not|don't)\s+(?:exclude|ignore)|\bnot only\b", clause, re.I):
            return clause
        evidence_request = re.search(r"证据|研究|文献|资料|课件|综述|指南|论文|来源|"
            r"\b(?:evidence|studies|research|articles?|literature|sources?)\b|"
            r"(?:套用|外推|照搬|只给|仅给)",excluded,re.I)
        contrast = re.match(r"排除|而非|而不是|不包括|不包含|不适用|"
                            r"\b(?:exclude|excluding|rather than|instead of|not applicable to)\b",clause,re.I)
        if not evidence_request and not contrast:
            # A clinical condition such as '不能摘掉无创通气面罩' is
            # about the actual device, not an excluded evidence population.
            return clause
        if (ADULT.search(excluded) or CHILD.search(excluded) or NEONATAL.search(excluded)
                or any(re.search(p, excluded, re.I) for p in DEVICE_PATTERNS.values())):
            return " " * len(excluded) + clause[len(excluded):]
        return clause
    return negative.sub(mask, text)


def query_token_count(tokenizer, text, *, pair=None, add_special_tokens=True):
    """Count actual configured tokenizer IDs, without tokenizer truncation."""
    args = (text,) if pair is None else (text, pair)
    try:
        encoded = tokenizer.encode(*args, add_special_tokens=add_special_tokens, truncation=False)
    except TypeError:
        # The cached Rust tokenizer is also useful for independent CPU audits.
        encoded = tokenizer.encode(*args, add_special_tokens=add_special_tokens)
    return len(encoded.ids if hasattr(encoded, "ids") else encoded)


def prepare_token_bounded_query(query, tokenizer, token_budget):
    """Keep the beginning and current-question tail under an actual token cap.

    Special tokens and the omission marker count toward the cap. A detected
    current question that cannot itself fit is refused, rather than cut off.
    """
    token_budget = int(token_budget)
    if token_budget < 32:
        raise QueryTokenBudgetError("资料检索的实际 token 预算不足，请缩短问题或拆分资料。")
    initial_tokens = query_token_count(tokenizer, query)
    base = {"input_characters": len(query), "input_tokens": initial_tokens,
            "token_budget": token_budget, "special_tokens_included": True}
    if initial_tokens <= token_budget:
        return {"query": query, **base, "actual_characters": len(query), "actual_tokens": initial_tokens,
                "shortened": False, "reason": "within_token_budget", "strategy": "full_input"}
    marker = "\n【中间背景已省略；保留开头和末尾当前问题】\n"
    labels = list(re.finditer(r"(?:当前问题|当前提问|current\s+question)\s*[:：]", query, re.I))
    last_line = query.rstrip().rfind("\n")
    required_tail = len(query) - labels[-1].start() if labels else len(query) - last_line - 1 if last_line >= 0 else 0
    boundary = "explicit_current_question" if labels else "last_line" if last_line >= 0 else "suffix_window"

    def candidate(kept):
        tail = max(required_tail, kept - kept // 3)
        head = max(0, kept - tail)
        return query[:head] + marker + (query[-tail:] if tail else "")

    # Binary search is only an efficiency choice. Every accepted candidate is
    # measured; correctness does not assume a tokens-per-character ratio.
    best = candidate(required_tail)
    if query_token_count(tokenizer, best) > token_budget:
        raise QueryTokenBudgetError("末尾当前问题超过资料检索的实际 token 预算，请缩短当前问题后重试。")
    low, high = required_tail, len(query) - 1
    while low <= high:
        middle = (low + high) // 2
        value = candidate(middle)
        if query_token_count(tokenizer, value) <= token_budget:
            best, low = value, middle + 1
        else:
            high = middle - 1
    measured = query_token_count(tokenizer, best)
    if measured > token_budget:
        raise QueryTokenBudgetError("资料检索输入未通过 token 预算核查，请拆分后重试。")
    return {"query": best, **base, "actual_characters": len(best), "actual_tokens": measured,
            "shortened": True, "reason": "retrieval_token_budget", "strategy": "head_and_current_tail",
            "current_question_boundary": boundary,
            "required_tail_characters": required_tail, "required_tail_preserved": True}


def inferred_population(query: str) -> list[str]:
    """Keep neonatal scope separate from older children and adult evidence."""
    query = recognition_text(query)
    case_age = CASE_AGE.search(query)
    if case_age:
        return _age_population(case_age)
    if ADULT.search(query) and CHILD.search(query):
        return ["adult", "neonatal" if NEONATAL.search(query) else "pediatric"]
    if NEONATAL.search(query):
        return ["neonatal"]
    if ADULT.search(query) and not CHILD.search(query):
        return ["adult"]
    for match in CHILD_AGE.finditer(query):
        return _age_population(match)
    if CHILD.search(query):
        return ["pediatric"]
    return []


def _age_population(match):
    age, unit = float(match[1]), match[2]
    if re.search(r"days?|天|日龄", unit, re.I):
        return ["neonatal" if age < 28 else "pediatric"]
    if re.search(r"months?|个月|月龄", unit, re.I):
        return ["adult" if age >= 216 else "pediatric"]
    return ["adult" if age >= 18 else "pediatric"]


def context_population(context):
    """Carry the case population through external search planning and admission."""
    return list(context.get("population") or inferred_population(context.get("question", "")) or ["adult"])


def population_prompt(prompt, query):
    """Use the reviewed case scope; leave all adult prompt text unchanged."""
    people = inferred_population(query)
    if "neonatal" in people:
        return prompt.replace("成人MDRPI", "新生儿MDRPI").replace("adult MDRPI", "neonatal MDRPI")
    if "pediatric" in people:
        return prompt.replace("成人MDRPI", "儿童MDRPI").replace("adult MDRPI", "pediatric MDRPI")
    return prompt


def query_scope(query: str, config, device_type=None, population=None, *, max_query_chars=None) -> QueryScope:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must not be empty")
    limit = int(config.get("rag", "max_query_chars", default=1000) if max_query_chars is None else max_query_chars)
    if len(query) > limit:
        raise ValueError("query exceeds max_query_chars")
    for field, values, vocabulary in (("device_type", device_type, DEVICE_TYPES),
                                       ("population", population, POPULATIONS)):
        if values is not None and (not values or not set(values) <= vocabulary):
            raise ValueError(f"Invalid {field} filter")
    # Normalize only recognition text; retrieval and generation retain the
    # original question. NFKC handles SpO₂ and full-width clinical abbreviations.
    scope_query = recognition_text(query)
    inferred = [key for key, pattern in DEVICE_PATTERNS.items() if re.search(pattern, scope_query, re.I)]
    devices = list(device_type if device_type is not None else inferred)
    requested_people = inferred_population(scope_query)
    child_request = set(requested_people) & {"pediatric", "neonatal"}
    people = list(population if population is not None else (requested_people if child_request else []) or
                  config.get("filters", "adult_population", default=["adult", "general", "mixed"]))
    validated = set(config.get("filters", "validated_populations", default=["adult"]))
    nonadult = (set(people) | set(requested_people)) & {"pediatric", "neonatal"}
    reason = ""
    if INJECTION.search(scope_query):
        reason = "instruction_override_request"
    elif not nonadult <= validated:
        reason = "pediatric_evidence_not_validated"
    elif child_request and population is not None and not child_request <= set(people):
        reason = "query_filter_mismatch"
    elif requested_people == ["adult"] and population is not None and set(people) <= {"pediatric", "neonatal"}:
        reason = "query_filter_mismatch"
    elif inferred and device_type is not None and not set(inferred) <= set(devices):
        reason = "query_filter_mismatch"
    elif not inferred and not DOMAIN.search(scope_query) and not (child_request and NASAL_VENTILATION.search(scope_query)):
        reason = "outside_mdrpi_scope"
    if child_request and not reason:
        # An explicit broad filter cannot bring adult evidence into a child case.
        people = requested_people
    elif requested_people == ["adult"] and not reason:
        people = [p for p in people if p not in {"pediatric", "neonatal"}]
    elif not reason and set(people) & {"pediatric", "neonatal"}:
        # Explicit child filters must not admit general/adult fallback material.
        people = [p for p in people if p in {"pediatric", "neonatal"}]
    if strict_population_isolation(config) and not reason:
        people, reason = isolated_request(population, requested_people)
    return QueryScope(devices=devices, population=people, reason=reason)


def _author_reference_pattern(surname: str, year: str) -> str:
    # A surname and a year somewhere else in the question are not a citation:
    # e.g. "2022 ... how long" must not select an article by Long.
    year_pattern = re.escape(year) if year else r"(?:19|20)\d{2}"
    return (r"\b" + re.escape(surname) + r"\b(?:['’]s)?"
            r"(?:\s+(?:et\s+al\.?|and\s+colleagues))?[\s,(（]*"
            r"(?=" + year_pattern + r"(?!\d))")


def referenced_documents(query: str, chunks: dict) -> list[str]:
    """Resolve explicit IDs, full titles, or author/year references, never topics."""
    documents = {item["doc_id"]: item for item in chunks.values()}
    ids = re.findall(r"\bDOC\d{3}\b", query, re.I)
    if ids:
        return sorted({item.upper() for item in ids})
    compact = re.sub(r"[\W_]", "", query).casefold()
    years = set(re.findall(r"\b(?:19|20)\d{2}\b|(?<!\d)(?:19|20)\d{2}(?=年|版)", query))
    matches = []
    for doc_id, document in documents.items():
        if years and str(document.get("year")) not in years:
            continue
        title = re.sub(r"[\W_]", "", str(document.get("title", ""))).casefold()
        title_without_year = re.sub(r"(?:19|20)\d{2}版", "", title)
        title_match = len(title_without_year) >= 12 and title_without_year in compact
        authors = str(document.get("authors", ""))
        surnames = [part.strip().split()[-1] for part in authors.split(";") if part.strip()]
        author_match = bool(years) and any(re.fullmatch(r"[A-Za-z]{4,}", surname)
                                         and re.search(_author_reference_pattern(surname, str(document.get('year'))), query, re.I)
                                         for surname in surnames)
        if title_match or author_match:
            matches.append(doc_id)
    return sorted(matches)


def focused_evidence_query(query: str, chunks: dict, doc_ids: list[str]) -> str:
    """Remove resolved bibliographic labels, retaining the user's factual question."""
    if not doc_ids:
        return query
    documents = {item["doc_id"]: item for item in chunks.values() if item["doc_id"] in doc_ids}
    focused = re.sub(r"\bDOC\d{3}\b", " ", query, flags=re.I)
    for document in documents.values():
        original_title = str(document.get("title", ""))
        title = re.sub(r"[\W_]", "", original_title)
        title = re.sub(r"(?:19|20)\d{2}版", "", title)
        if len(title) >= 12:
            pattern = r"[\W_]*".join(re.escape(character) for character in title)
            # A title such as "Medical device-related pressure ulcers: A systematic
            # review..." may contain the question's only clinical subject. Retain
            # that subject while dropping the bibliographic subtitle.
            subject, separator, _ = original_title.partition(":")
            replacement = subject if separator and DOMAIN.search(subject) else " "
            focused = re.sub(pattern, lambda _: replacement, focused, flags=re.I)
        for author in str(document.get("authors", "")).split(";"):
            surname = author.strip().split()[-1] if author.strip() else ""
            if re.fullmatch(r"[A-Za-z]{4,}", surname):
                focused = re.sub(_author_reference_pattern(surname, str(document.get('year') or '')),
                                 " ", focused, flags=re.I)
    # Keep dates and numbers: they may be the outcome or interval being asked about.
    focused = re.sub(r"\s+", " ", focused).strip()
    return focused if len(re.sub(r"[\W_]", "", focused)) >= 6 else query
