"""Locate quantities in cited passages; never infer clinical validity from a match."""

from decimal import Decimal
import re
import unicodedata

from .applicability import POPULATION_CUES, SETTING_CUES
from .lexical import DEVICE_ALIASES
from .six_step_schema import all_claims


NUMBER = r"(?:\d+(?:,\d{3})*(?:\.\d+)?|[零〇一二两三四五六七八九十百]+)"
QUANTITY = re.compile(
    r"(?<![0-9A-Za-z_./])(?:q(?=\d))?(?P<low>" + NUMBER + r")"
    r"(?:\s*(?:[-–—~～至]|到)\s*(?P<high>" + NUMBER + r"))?\s*"
    r"(?P<unit>mm\s*Hg|cm\s*H[₂2]O|mm\s*H[₂2]O|kPa|%|％|百分比|"
    r"小时|分钟|秒钟|秒|天|日|周|毫米|厘米|摄氏度|°\s*C|"
    r"seconds?|secs?|minutes?|mins?|hours?|hrs?|days?|weeks?|mm|cm|h)(?![A-Za-z])", re.I)
UNITS = {
    "小时": ("time", 3600), "hour": ("time", 3600), "hours": ("time", 3600),
    "hr": ("time", 3600), "hrs": ("time", 3600), "h": ("time", 3600),
    "分钟": ("time", 60), "minute": ("time", 60), "minutes": ("time", 60),
    "min": ("time", 60), "mins": ("time", 60), "秒": ("time", 1), "秒钟": ("time", 1),
    "second": ("time", 1), "seconds": ("time", 1), "sec": ("time", 1), "secs": ("time", 1),
    "天": ("time", 86400), "日": ("time", 86400), "day": ("time", 86400), "days": ("time", 86400),
    "周": ("time", 604800), "week": ("time", 604800), "weeks": ("time", 604800),
    "%": ("percent", 1), "百分比": ("percent", 1),
    "mm": ("length", 1), "毫米": ("length", 1), "cm": ("length", 10), "厘米": ("length", 10),
    "mmhg": ("pressure_mmhg", 1), "cmh2o": ("pressure_cmh2o", 1),
    "mmh2o": ("pressure_cmh2o", Decimal("0.1")), "kpa": ("pressure_kpa", 1),
    "摄氏度": ("temperature_c", 1), "°c": ("temperature_c", 1),
}
DIGITS = {c: i for i, chars in enumerate(["零〇", "一", "二两", "三", "四", "五", "六", "七", "八", "九"])
          for c in chars}
EXAMPLE = re.compile(r"(?:记录格式|格式示例|非病例实测|record(?:ing)? format|example record)[^。；;\n\"”’]{0,80}$|"
                     r"(?:记录|documentation)[^。；;\n]{0,25}(?:如|例如|example)[^。；;\n]{0,10}[\"“‘][^\"”’]*$", re.I)
GENERAL = re.compile(r"所有患者|全部患者|通用标准|统一标准|一般成人标准|all patients|universal", re.I)
DIRECTIVE = re.compile(r"每|频率|周期|阈值|升级条件|建议|应当|设定|目标|[<>≤≥]|every|threshold|target|recommend", re.I)
ACTIVITIES = {
    "securement": r"重新固定|再固定|重固定|resecure\w*|securement|reapply",
    "repositioning": r"重新定位|移位|reposition\w*",
    "assessment": r"评估|复评|检查|观察|assess\w*|inspect\w*|observ\w*|monitor\w*",
    "replacement": r"更换(?:面罩|器械|设备)|更换频率|replac\w*",
}


def _number(value):
    if value[0].isdigit():
        return Decimal(value.replace(",", ""))
    total, digit = 0, 0
    for char in value:
        if char in DIGITS:
            digit = DIGITS[char]
        elif char in "十百":
            total += (digit or 1) * (10 if char == "十" else 100)
            digit = 0
    return Decimal(total + digit)


def quantities(text):
    # Normalize matching without changing offsets in the original passage.
    dates = [(m.start(), m.end()) for m in re.finditer(r"(?:\d{4}\s*年\s*)?\d{1,2}\s*月\s*\d{1,2}\s*日", text)]
    for match in QUANTITY.finditer(text):
        if any(start <= match.start() < end for start, end in dates):
            continue
        unit = re.sub(r"\s+", "", unicodedata.normalize("NFKC", match["unit"])).lower()
        dimension, scale = UNITS[unit]
        low, high = _number(match["low"]), _number(match["high"] or match["low"])
        yield {"quote": match.group(), "start": match.start(), "end": match.end(),
               "dimension": dimension, "low": str(low * scale), "high": str(high * scale)}


def _sentence(text, start, end):
    boundaries = list(re.finditer(r"[。！？!?；;\n]|(?<!\d)\.(?!\d)", text))
    left = max([0, *[m.end() for m in boundaries if m.end() <= start]])
    right = min([len(text), *[m.end() for m in boundaries if m.start() >= end]])
    return {"quote": text[left:right], "start": left, "end": right}


def _labels(patterns, text):
    return {label for label, pattern in patterns.items() if re.search(pattern, text, re.I)}


def _same_quantity(left, right):
    if left["dimension"] != right["dimension"]:
        return False
    if (Decimal(left["low"]), Decimal(left["high"])) == (Decimal(right["low"]), Decimal(right["high"])):
        return True
    # A value inside a reported range is locatable, but does not establish a
    # recommended frequency or threshold. Model review still checks its use.
    return (left["low"] == left["high"] and
            Decimal(right["low"]) <= Decimal(left["low"]) <= Decimal(right["high"]))


def _mismatch(claim_text, source_text):
    for label, patterns in (("器械", DEVICE_ALIASES), ("人群", POPULATION_CUES),
                            ("场景", SETTING_CUES), ("活动", ACTIVITIES)):
        claimed, source = _labels(patterns, claim_text), _labels(patterns, source_text)
        # Broad/general labels and ICU subtypes are not contradictory.
        claimed.discard("general_mdrpi")
        source.discard("general_mdrpi")
        if label == "场景" and "icu" in claimed and "icu" in source:
            continue
        if label == "场景":
            if "surgical_icu" in claimed:
                claimed.discard("icu")
            if "surgical_icu" in source:
                source.discard("icu")
        if len(claimed) == len(source) == 1 and not claimed & source:
            return label + "不一致"
    if GENERAL.search(claim_text) and _labels(SETTING_CUES, source_text):
        return "特定场景的数值被写成一般标准"
    return ""


def check_evidence_quantities(steps, facts, evidence):
    records = []
    fact_index = {f["fact_id"]: f for f in facts.get("facts", [])}
    for cid, claim in all_claims(steps).items():
        kind = claim["kind"]
        if kind == "uncertainty":
            continue
        observations, missing, mismatches = [], [], []
        for field in ("text", "conditions", "rationale"):
            text = claim.get(field, "")
            for quantity in quantities(text):
                sentence = _sentence(text, quantity["start"], quantity["end"])
                prefix = text[sentence["start"]:quantity["start"]]
                if EXAMPLE.search(prefix) and not re.search(r"(?:每|every\s*)$", prefix, re.I):
                    continue
                candidates = []
                if kind == "case_fact":
                    candidates = [(fid, fact_index[fid]["quote"], {}) for fid in claim["fact_ids"] if fid in fact_index]
                else:
                    candidates = [(eid, evidence[eid]["text"], evidence[eid])
                                  for eid in claim["evidence_ids"] if eid in evidence]
                    if kind == "case_inference" and not DIRECTIVE.search(sentence["quote"]):
                        candidates += [(fid, fact_index[fid]["quote"], {"location": "病例原文"})
                                       for fid in claim["fact_ids"] if fid in fact_index]
                matches, wrong_scope = [], []
                for identifier, source_text, source in candidates:
                    for original in quantities(source_text):
                        if not _same_quantity(quantity, original):
                            continue
                        span = _sentence(source_text, original["start"], original["end"])
                        mismatch = "" if kind == "case_fact" or identifier in fact_index else _mismatch(sentence["quote"], span["quote"])
                        entry = {"source_id": identifier, "quantity": original, "sentence": span,
                                 "location": source.get("location", ""),
                                 "source_pages": source.get("source_pages", []), "scope_mismatch": mismatch}
                        (wrong_scope if mismatch else matches).append(entry)
                observations.append({"field": field, "quantity": quantity, "matches": matches,
                                     "scope_mismatches": wrong_scope})
                if not matches:
                    (mismatches if wrong_scope else missing).append(quantity["quote"])
        if not observations:
            continue
        cited = kind == "case_fact" or bool(claim["evidence_ids"])
        # Uncited model knowledge remains available. Deterministic checks do
        # not claim that absence from this corpus makes that knowledge false.
        issue = cited and bool(missing or mismatches)
        reason = ("所引原文未定位到相同数值与单位：" + "、".join(missing) if missing else
                  "数值所在原句与主张的适用条件不同：" + "、".join(mismatches) if mismatches else
                  "仅核对数值定位；匹配不代表结论或临床适用性已验证。")
        if not cited and missing:
            reason = "一般推断的数值没有给定来源；本程序不判断其临床正确性。"
        records.append({"check_id": "program:quantity:" + cid,
            "check_kind": "case" if kind == "case_fact" else "evidence", "target_id": cid,
            "fact_ids": claim["fact_ids"], "evidence_ids": claim["evidence_ids"],
            "verdict": "issue" if issue else "pass",
            "issue_type": "numeric_error" if issue and missing else "applicability_mismatch" if issue else "none",
            "severity": "major" if issue else "none", "route": "correction" if issue else "none",
            "reason": reason, "needed": reason + "；核对具体出处，修订错误归因或无依据的确定参数，保留有据部分。" if issue else "无需按数值定位规则修改",
            "origin": "deterministic_quantity_locator", "measurements": observations,
            "support_scope": "quantity_location_only_not_semantic_or_clinical_validation"})
    return records
