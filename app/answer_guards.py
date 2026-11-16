"""Conservative checks with exact case/source spans, not clinical adjudication."""
from copy import deepcopy
from decimal import Decimal
import re

from .applicability import POPULATION_CUES, SETTING_CUES
from .evidence_guard import _number, _same_quantity, quantities
from .lexical import DEVICE_ALIASES
from .six_step_schema import all_claims


BOUNDARY = re.compile(r"[。！？!?；;]|(?<!\d)\.(?!\d)|(?:\r?\n){2,}|[—-]{2,}")
FINGERS = re.compile(r"(?P<number>\d+(?:\.\d+)?|[一二两三四五六七八九十]+)\s*横指")
DEVICES = {**DEVICE_ALIASES,
    "niv_mask": DEVICE_ALIASES["niv_mask"] + r"|(?<![A-Za-z])NIV(?![A-Za-z])",
    "nasogastric_tube": DEVICE_ALIASES["nasogastric_tube"] + r"|(?<![A-Za-z])NGT(?![A-Za-z])",
    "endotracheal_tube": DEVICE_ALIASES["endotracheal_tube"] + r"|气管导管|(?<![A-Za-z])ETT(?![A-Za-z])",
    "spo2_probe": DEVICE_ALIASES["spo2_probe"] + r"|SpO[₂2].{0,8}探头|血氧饱和度.{0,8}探头|SpO[₂2]监测(?:波形)?",
    "orthosis": r"\bTLSO\b|胸腰骶支具|胸腰椎支具",
    "cast": r"\bcasts?\b|石膏",
}
ACTIVITIES = {
    "assessment": r"评估|复评|检查|观察|\bassess\w*|\binspect\w*|\bobserv\w*|\bmonitor\w*",
    "securement": r"重新固定|再固定|重固定|\bresecure\w*|\breapply\b.{0,60}securement",
    "repositioning": r"换位|轮换|更换.{0,12}(?:位置|部位)|重新定位|\breposition\w*|\bre[- ]?site\b",
    "tube_replacement": r"换管|整管更换|更换(?:整根|整条|整个)?(?:鼻胃管|胃管|气管导管)(?!固定)|\breplac\w*\s+(?:the\s+)?(?:whole\s+)?(?:nasogastric\s+)?tube\b(?!\s+securement)",
}
DIRECTIVE = re.compile(r"应|建议|调整|确保|保证|间隙|频率|每|\brecommend\w*|\bmust\b|\bsecure\w*", re.I)
NEGATED_LIMIT = re.compile(r"不应|不宜|不能|不得|不以|不要|不作为|不可作为|无法达到|错误|不合理|\b(?:not|never|invalid|incorrect|impossible)\b", re.I)
CASE_PAIN_SCALE = re.compile(r"疼痛[^。；\n]{0,100}?[（(](\d+(?:\.\d+)?)\s*[—–\-~～]\s*(\d+(?:\.\d+)?)\s*分?[）)]")
PAIN_LIMIT = re.compile(r"疼痛(?:评分)?[^。；\n]{0,24}?(?P<operator>超过|大于|高于|>=|≥|>)(?P<value>\d+(?:\.\d+)?)\s*分(?!钟)")
PAIN_ABOVE_MAXIMUM = re.compile(r"疼痛(?:评分)?[^。；\n]{0,24}?(?:超过|大于|高于|>)\s*(?:该|本)?(?:量表(?:的)?上限|量表最高分|评分上限|满分|最高分)")
IMPOSSIBLE_EXAMPLE = re.compile(
    r"(?P<lead>疼痛(?:评分)?(?:持续|进一步|明显|进行性)?(?:升高|加重))"
    r"(?P<example>[（(](?:如|例如)\s*(?P<operator>超过|大于|高于|>=|≥|>)\s*"
    r"(?P<limit>\d+(?:\.\d+)?\s*分|量表(?:的)?上限|量表最高分|评分上限|满分|最高分)\s*[）)])")


def _labels(patterns, text):
    return sorted(k for k, pattern in patterns.items() if re.search(pattern, text, re.I))


def _clauses(text):
    start = 0
    for boundary in [*BOUNDARY.finditer(text), None]:
        end = boundary.start() if boundary else len(text)
        left, right = start, end
        while left < right and text[left].isspace():
            left += 1
        while right > left and text[right - 1].isspace():
            right -= 1
        if left < right:
            yield {"quote": text[left:right], "start": left, "end": right}
        start = boundary.end() if boundary else len(text)


def _measurements(text):
    values = list(quantities(text))
    for match in FINGERS.finditer(text):
        value = str(_number(match["number"]))
        values.append({"quote": match.group(), "start": match.start(), "end": match.end(),
                       "dimension": "finger_width", "low": value, "high": value})
    return sorted(values, key=lambda q: q["start"])


def source_clause_bindings(text):
    result = []
    for clause in _clauses(text):
        values = _measurements(clause["quote"])
        if not values:
            continue
        result.append({**clause, "devices": _labels(DEVICES, clause["quote"]),
            "activities": _labels(ACTIVITIES, clause["quote"]),
            "population": _labels(POPULATION_CUES, clause["quote"]),
            "setting": _labels(SETTING_CUES, clause["quote"]),
            "measurements": [{**q, "start": clause["start"] + q["start"],
                              "end": clause["start"] + q["end"]} for q in values]})
    return result


def with_clause_bindings(source):
    """Annotate ambiguous numeric passages without changing source text or IDs."""
    text = source.get("text", "")
    clauses = source_clause_bindings(text)
    devices = {d for c in clauses for d in c["devices"]}
    activities = {a for c in clauses for a in c["activities"]}
    # Avoid duplicating every ordinary sentence in already long contexts.
    if len(devices) < 2 and len(activities) < 2 and not any(len(c["devices"]) > 1 for c in clauses):
        # Device names may occur in another non-numeric clause of a mixed chunk.
        if len(_labels(DEVICES, text)) < 2:
            return source
    selected = [c for c in clauses if len(c["quote"]) <= 500][:6]
    if not selected:
        return source
    result = {**source, "evidence_scope": deepcopy(source.get("evidence_scope") or {})}
    result["evidence_scope"]["clause_bindings"] = {
        "method": "exact_explicit_cues_not_clinical_applicability_verdict",
        "offsets_relative_to": "text", "complete": len(selected) == len(clauses),
        "limit": "Missing cues remain unknown; different activities and devices cannot inherit each other's quantities.",
        "clauses": selected,
    }
    return result


def _record(claim, suffix, reason, proof):
    return {"check_id": f"program:{suffix}:" + claim["claim_id"], "check_kind": "evidence",
        "target_id": claim["claim_id"], "fact_ids": claim.get("fact_ids", []),
        "evidence_ids": claim.get("evidence_ids", []), "verdict": "issue",
        "issue_type": "numeric_error" if suffix == "case_scale" else "applicability_mismatch",
        "severity": "major", "route": "correction", "reason": reason,
        "needed": reason + "；仅核对并修改该处，保留病例原有观察和其他正确内容，不另编替代参数。",
        "origin": "deterministic_case_and_clause_guard", "proof": proof,
        "support_scope": "case_scale_or_explicit_source_cue_conflict_not_clinical_certification"}


def _case_scale_checks(query, claim):
    scales = list(CASE_PAIN_SCALE.finditer(query))
    if len(scales) != 1:
        return []
    low, high = map(Decimal, scales[0].groups())
    if low >= high:
        return []
    text = claim["text"]
    failures = []
    for clause in _clauses(text):
        if NEGATED_LIMIT.search(clause["quote"]):
            continue
        for match in PAIN_LIMIT.finditer(clause["quote"]):
            value = Decimal(match["value"])
            if value < high or value == high and match["operator"] in (">=", "≥"):
                continue
            failures.append({"case_quote": scales[0].group(), "case_start": scales[0].start(),
                "case_end": scales[0].end(), "scale_maximum": str(high),
                "claim_quote": match.group(), "claim_start": clause["start"] + match.start(),
                "claim_end": clause["start"] + match.end()})
        for match in PAIN_ABOVE_MAXIMUM.finditer(clause["quote"]):
            failures.append({"case_quote": scales[0].group(), "case_start": scales[0].start(),
                "case_end": scales[0].end(), "scale_maximum": str(high),
                "claim_quote": match.group(), "claim_start": clause["start"] + match.start(),
                "claim_end": clause["start"] + match.end(), "symbolic_upper_bound": True})
    if not failures:
        return []
    return [_record(claim, "case_scale", "升级条件超出病例明确给出的疼痛量表上界，无法在该量表内达到", failures)]


def repair_case_scale_examples(query, steps):
    """Remove only an impossible parenthetical example after an existing trend.

    This never invents a threshold, replaces a main condition, or changes case
    observations. Anything outside this narrow pattern remains for review.
    """
    scales = list(CASE_PAIN_SCALE.finditer(query))
    if len(scales) != 1:
        return steps, []
    low, high = map(Decimal, scales[0].groups())
    if low >= high:
        return steps, []
    result, edits = deepcopy(steps), []
    for claim in all_claims(result).values():
        issues = _case_scale_checks(query, claim)
        if claim["claim_id"].startswith("recognize.") or not issues:
            continue
        before = claim["text"]
        spans = []
        for match in IMPOSSIBLE_EXAMPLE.finditer(before):
            if not any(p["claim_start"] <= match.start("lead") < p["claim_end"] for r in issues for p in r["proof"]):
                continue
            symbolic = not match["limit"][0].isdigit()
            value = high if symbolic else Decimal(re.match(r"\d+(?:\.\d+)?", match["limit"])[0])
            if value < high or value == high and match["operator"] in (">=", "≥"):
                continue
            spans.append({"start": match.start("example"), "end": match.end("example"),
                          "quote": match["example"]})
        for span in reversed(spans):
            claim["text"] = claim["text"][:span["start"]] + claim["text"][span["end"]:]
        if spans:
            edits.append({"claim_id": claim["claim_id"], "before": before, "after": claim["text"],
                "removed_spans": spans, "case_scale_quote": scales[0].group(),
                "method": "delete_unreachable_example_only_no_new_medical_content"})
    return (result, edits) if edits else (steps, [])


def _source_scope_checks(query, claim, evidence, bindings):
    if claim["claim_id"].split(".")[0] == "recognize":
        return []  # Observations are not recommendations, even if kind is wrong.
    case_devices = _labels(DEVICES, query)
    failures = []
    for clause in _clauses(claim["text"]):
        if not DIRECTIVE.search(clause["quote"]) or NEGATED_LIMIT.search(clause["quote"]):
            continue
        wanted = _labels(DEVICES, clause["quote"]) or case_devices
        activity = _labels(ACTIVITIES, clause["quote"])
        for quantity in _measurements(clause["quote"]):
            matches = []
            for eid in claim.get("evidence_ids", []):
                source = evidence.get(eid, {})
                for original in bindings.get(eid, []):
                    if not any(_same_quantity(quantity, q) for q in original["measurements"]):
                        continue
                    reason = ""
                    if len(wanted) == len(original["devices"]) == 1 and wanted != original["devices"]:
                        reason = "器械不同"
                    elif (activity == ["tube_replacement"] and original["activities"]
                          and "tube_replacement" not in original["activities"]):
                        reason = "整管更换与评估/重新固定等操作不同"
                    matches.append({"source_id": eid, "source_span": original,
                        "source_location": source.get("location", ""), "mismatch": reason})
            # Only a positive conflict is actionable. Missing/unknown scope or
            # another compatible occurrence cannot establish an error here.
            if matches and all(m["mismatch"] for m in matches):
                failures.append({"quantity": {**quantity, "start": clause["start"] + quantity["start"],
                                              "end": clause["start"] + quantity["end"]}, "claim_clause": clause,
                    "claim_device_cues": wanted, "claim_activity_cues": activity, "matches": matches})
    if not failures:
        return []
    return [_record(claim, "clause_scope", "所引数值的原文条款仅定位到其他器械或其他操作，不能直接作为本句的适用标准", failures)]


def check_answer_guards(query, steps, evidence, *, case_scales=False, clause_scope=False):
    records = []
    bindings = {eid: source_clause_bindings(source.get("text", "")) for eid, source in evidence.items()} if clause_scope else {}
    for claim in all_claims(steps).values():
        if case_scales:
            records.extend(_case_scale_checks(query, claim))
        if clause_scope:
            records.extend(_source_scope_checks(query, claim, evidence, bindings))
    return records
