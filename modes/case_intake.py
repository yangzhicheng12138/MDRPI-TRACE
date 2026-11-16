"""Validate practice specifications and redact identifiers before persistence."""
from copy import deepcopy
from datetime import datetime
import re

POPULATIONS = {"adult":"成人", "pediatric":"儿童", "neonatal":"新生儿"}
DIFFICULTIES = {"基础", "进阶", "复杂"}
DEVICE_TYPES = {"niv_mask", "nasogastric_tube", "endotracheal_tube", "tracheostomy_tube", "pulse_oximeter", "cervical_collar", "cast", "nasal_cannula", "nasojejunal_tube", "nasal_prong_cpap", "bite_block", "general_mdrpi"}


def case_scope_clues(text):
    """Recognize explicit labels only; missing or mixed scope remains unknown."""
    populations = []
    if re.search(r"新生儿|早产儿|neonat\w*|newborn|preterm\s+(?:infant|baby)", text, re.I):
        populations.append("neonatal")
    if re.search(r"儿童|小儿|幼儿|p[ae]diatric|\bchild(?:ren)?\b", text, re.I):
        populations.append("pediatric")
    if re.search(r"成人|成年人|\badult\b", text, re.I):
        populations.append("adult")
    patterns = {
        "nasal_prong_cpap": r"(?:鼻塞|nasal\s+prongs?).{0,20}(?:CPAP|持续气道正压)|(?:CPAP|持续气道正压).{0,20}(?:鼻塞|nasal\s+prongs?)",
        "nasogastric_tube": r"鼻胃管|nasogastric",
        "nasojejunal_tube": r"鼻空肠管|nasojejunal",
        "endotracheal_tube": r"气管插管|气管内管|endotracheal",
        "tracheostomy_tube": r"气管切开|气切套管|tracheostom",
        "pulse_oximeter": r"血氧探头|脉搏血氧|pulse\s+oximet",
        "cervical_collar": r"颈托|cervical\s+collar",
        "cast": r"石膏|\bcast\b",
        "nasal_cannula": r"鼻导管|nasal\s+cannula",
        "niv_mask": r"面罩|\bmask\b",
        "bite_block": r"咬口|牙垫|bite\s+block",
    }
    devices = [name for name, pattern in patterns.items() if re.search(pattern, text, re.I)]
    return {"populations": populations, "device_types": devices}


def validate_custom_spec(value=None):
    if value is not None and not isinstance(value,dict): raise ValueError("案例选项需为对象。")
    value = deepcopy(value or {})
    result = {"topic":str(value.get("topic") or "MDRPI风险与预防").strip(),
              "population":str(value.get("population") or "adult"),
              "device_type":str(value.get("device_type") or "niv_mask"),
              "difficulty":str(value.get("difficulty") or "基础"),
              "focus":str(value.get("focus") or "六步临床推理").strip()}
    if result["difficulty"] == "综合": result["difficulty"] = "进阶"
    if result["population"] not in POPULATIONS: raise ValueError("请选择成人、儿童或新生儿。")
    if result["difficulty"] not in DIFFICULTIES: raise ValueError("案例难度需为基础、进阶或复杂。")
    if result["device_type"] not in DEVICE_TYPES: raise ValueError("请选择支持的医疗器械。")
    if not result["topic"] or len(result["topic"]) > 300 or len(result["focus"]) > 600:
        raise ValueError("主题需为1–300字，练习重点不超过600字。")
    if any(inspect_import(result[field])["redaction_count"] for field in ("topic","focus")):
        raise ValueError("主题或练习重点含可能的身份信息，请脱敏后再提交。")
    return result


def inspect_import(text, spec=None):
    """Only replacement counts leave this function; original identifiers never do.

    Detection follows span detection then replacement, as in Presidio. This
    small Chinese clinical-input boundary requires user confirmation as well.
    """
    if not isinstance(text,str) or not text.strip(): raise ValueError("请先填写脱敏后的案例描述。")
    findings = []
    patterns = [
        ("身份证号",r"(?<!\d)\d{17}[\dXx](?!\d)"),
        ("手机号",r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)"),
        ("电子邮箱",r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
        ("姓名",r"(?:患者姓名|姓名|patient\s*name|name)\s*[:：]\s*([\u4e00-\u9fff]{2,8}|[A-Za-z][A-Za-z .'-]{1,60})"),
        ("住院/病历号",r"(?:住院号|病历号|门诊号|就诊号|医疗记录号|MRN)\s*[:：]\s*[A-Za-z0-9-]{3,30}"),
        ("具体地址",r"(?:家庭住址|家庭地址|详细地址|住址|address)\s*[:：]\s*[^\n；;。]{3,100}"),
        ("床位",r"(?:床号|床位号|bed\s*(?:number|no\.?))\s*[:：]\s*[A-Za-z0-9-]{1,15}"),
        ("姓名",r"(?<![\u4e00-\u9fff])([\u4e00-\u9fff]{2,4})(?:先生|女士)(?![\u4e00-\u9fff])"),
    ]
    for kind,pattern in patterns:
        for match in re.finditer(pattern,text,re.I):
            start,end = match.span(1) if match.lastindex else match.span()
            findings.append((start,end,kind))
    # Longest spans win overlaps (an address containing an email is one span).
    selected = []
    for row in sorted(findings,key=lambda x:(-(x[1]-x[0]),x[0])):
        if not any(row[0] < other[1] and other[0] < row[1] for other in selected): selected.append(row)
    sanitized = text
    for start,end,kind in sorted(selected,reverse=True): sanitized = sanitized[:start]+f"[已隐藏{kind}]"+sanitized[end:]
    counts = {kind:sum(row[2]==kind for row in selected) for kind in dict.fromkeys(row[2] for row in selected)}
    missing = []
    if not re.search(r"岁|月龄|日龄|成人|儿童|新生儿|早产|adult|child|neonat|year.old",sanitized,re.I): missing.append("年龄或人群")
    if not re.search(r"面罩|管|探头|颈托|石膏|支具|氧疗|呼吸|mask|tube|probe|collar|device",sanitized,re.I): missing.append("器械种类")
    if not re.search(r"皮肤|黏膜|红斑|红|破损|水肿|不适|疼痛|skin|mucosa|red|pain|oedema|edema",sanitized,re.I): missing.append("局部皮肤/黏膜表现")
    if not re.search(r"\d+\s*(?:小时|分钟|天|hours?|minutes?|days?)|(?:持续时间|使用时长)[：:]\s*\d+",sanitized,re.I): missing.append("使用时间或变化过程")
    clues = case_scope_clues(sanitized)
    chosen = validate_custom_spec(spec) if spec else {}
    # Defaults used by generated cases are not patient facts for an import.
    for field in ("population", "device_type"):
        if not (spec or {}).get(field):
            chosen.pop(field, None)
    conflicts = []
    for field, key in (("population", "populations"), ("device_type", "device_types")):
        candidates = clues[key]
        if len(candidates) == 1:
            if chosen.get(field) and chosen[field] != candidates[0] and chosen[field] != "general_mdrpi":
                conflicts.append({"field": field, "selected": chosen[field], "explicit_case_value": candidates[0]})
            elif not chosen.get(field):
                chosen[field] = candidates[0]
    return {"sanitized_text":sanitized,"redactions":counts,"redaction_count":len(selected),"missing_fields":missing,
            "uncertainties":["缺失信息保持未知，请根据教学需要补充。"] if missing else [],
            "case_spec": chosen, "scope_clues": clues, "scope_conflicts": conflicts,
            "review_status":"identifiers_checked_pending_user_confirmation"}


def spec_request(spec):
    spec = validate_custom_spec(spec)
    return (f"生成一个MDRPI教学练习案例。主题：{spec['topic']}；人群：{POPULATIONS[spec['population']]}；"
            f"器械：{spec['device_type']}；难度：{spec['difficulty']}；练习重点：{spec['focus']}。"
            "仅生成病例题面，不提前给出护理方案或六步答案；符合所选人群和器械，保留必要的不确定信息。")
