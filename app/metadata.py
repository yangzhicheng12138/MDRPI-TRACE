"""Conservative metadata inference and controlled vocabulary validation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


SOURCE_TYPES = {
    "guideline",
    "consensus",
    "standard",
    "systematic_review",
    "meta_analysis",
    "evidence_summary",
    "scoping_review",
    "integrative_review",
    "original_research",
    "patient_safety_alert",
    "poster",
    "other",
}
POPULATIONS = {"adult", "pediatric", "neonatal", "general", "mixed"}
DEVICE_TYPES = {
    "general_mdrpi",
    "niv_mask",
    "nasogastric_tube",
    "endotracheal_tube",
    "tracheostomy",
    "spo2_probe",
    "blood_pressure_cuff",
    "vascular_access",
    "cervical_collar",
    "orthopedic_device",
    "cpap",
    "other",
}
TOPICS = {
    "definition",
    "mechanism",
    "incidence",
    "prevalence",
    "risk_factor",
    "risk_assessment",
    "skin_assessment",
    "mucosal_assessment",
    "device_assessment",
    "device_fixation",
    "device_reposition",
    "pressure_relief",
    "repositioning",
    "skin_protection",
    "prophylactic_dressing",
    "moisture_management",
    "nutrition",
    "prevention",
    "priority",
    "safety_boundary",
    "monitoring",
}


@dataclass(frozen=True)
class InferredMetadata:
    """Metadata inferred from a filename and optional extracted text sample."""

    title: str
    year: int | None
    authors: str
    organization: str
    source_type: str
    evidence_tier: str
    population: list[str]
    device_type: list[str]
    topic: list[str]
    language: str
    include: str
    notes: str


def _normalise_stem(filename: str) -> str:
    stem = Path(filename).stem
    stem = re.sub(r"^\d+_(?:19|20)\d{2}_", "", stem)
    stem = re.sub(r"\.{2,}", " ", stem)
    return re.sub(r"\s+", " ", stem.replace("_", " ")).strip()


def _year(filename: str) -> int | None:
    match = re.search(r"(?:^|_)((?:19|20)\d{2})(?:_|\.)", filename)
    return int(match.group(1)) if match else None


def _organisation(text: str) -> str:
    known = (
        "NPIAP",
        "EPUAP",
        "PPPIA",
        "中华护理学会",
        "甘肃省护理学会",
        "AARC",
        "RNAO",
        "NICE",
        "WOCN",
        "WoundsInternational",
        "WoundsUK",
        "WUWHS",
        "The Joint Commission",
        "HSE",
        "Cochrane",
    )
    found = [item for item in known if item.lower() in text.lower()]
    return "; ".join(found)


def _has_any(text: str, keys: tuple[str, ...]) -> bool:
    lower = text.lower()
    return any(key.lower() in lower for key in keys)


def _source_type(filename: str, pdf_title: str = "", text_sample: str = "") -> str:
    """Classify from bibliographic signals, keeping body text as a fallback only.

    A full article often mentions several study designs in its background and
    references.  Those mentions must not override the document's own filename or
    PDF title, which are the strongest locally available classification signals.
    """

    bibliographic = f"{Path(filename).stem} {pdf_title}".strip()
    if _has_any(bibliographic, ("最佳证据总结", "最佳实践总结", "循证摘要", "evidence summary", "best practice summary")):
        return "evidence_summary"
    if _has_any(bibliographic, ("范围综述", "scoping review", "scoping_review")):
        return "scoping_review"
    if _has_any(bibliographic, ("整合性", "整合综述", "integrative review", "integrative_review")):
        return "integrative_review"
    if _has_any(bibliographic, ("网络meta", "meta分析", "meta_", "meta ", "meta-analysis", "meta analysis")):
        return "meta_analysis"
    if _has_any(bibliographic, ("系统评价", "系统综述", "systematic review", "systematic_review")):
        return "systematic_review"
    # Consensus/Delphi must be checked before the generic word "standard".
    if _has_any(bibliographic, ("共识", "delphi", "consensus")):
        return "consensus"
    if _has_any(bibliographic, ("团标", "t/cnas", "cnas", "标准", "standard")):
        return "standard"
    if _has_any(bibliographic, ("患者安全提醒", "患者安全补充", "patient safety", "safety alert")):
        return "patient_safety_alert"
    if _has_any(bibliographic, ("海报", "poster")):
        return "poster"
    if _has_any(bibliographic, ("指南", "guideline", "practice recommendation")):
        return "guideline"
    # Only use a short first-page sample when bibliographic metadata is silent.
    sample = text_sample[:1200]
    if _has_any(sample, ("最佳证据", "最佳实践总结", "循证摘要", "evidence summary", "best practice summary")):
        return "evidence_summary"
    if _has_any(sample, ("范围综述", "scoping review")):
        return "scoping_review"
    if _has_any(sample, ("整合性", "integrative review")):
        return "integrative_review"
    if _has_any(sample, ("网络meta", "meta-analysis", "meta analysis")):
        return "meta_analysis"
    if _has_any(sample, ("系统评价", "系统综述", "systematic review")):
        return "systematic_review"
    if _has_any(sample, ("共识", "delphi", "consensus")):
        return "consensus"
    if _has_any(sample, ("指南", "guideline")):
        return "guideline"
    return "other"


def _population(filename: str, pdf_title: str = "") -> tuple[list[str], bool]:
    """Infer population only from the filename/PDF title, never article body text."""

    text = f"{Path(filename).stem} {pdf_title}".lower()
    if _has_any(text, ("mixed", "成人和儿童", "儿童和成人", "adult and pediatric", "adult and paediatric")):
        return ["mixed"], False
    flags = {
        "neonatal": _has_any(text, ("新生儿", "早产儿", "nicu", "neonatal", "preterm", "premature infant")),
        "pediatric": _has_any(text, ("儿童", "儿科", "患儿", "pediatric", "paediatric", "children")),
        "adult": _has_any(text, ("成人", "adult", "adults")),
    }
    # NICU/early-preterm markers are a neonatal population signal even when
    # Chinese titles also contain the generic word "患儿" (child/patient).
    if flags["neonatal"] and flags["adult"]:
        return ["mixed"], True
    if flags["neonatal"] and not flags["adult"]:
        return ["neonatal"], False
    selected = [name for name in ("pediatric", "adult") if flags[name]]
    if len(selected) == 1:
        return selected, False
    if len(selected) > 1:
        return ["mixed"], True
    return ["general"], True


def _device_types(text: str) -> list[str]:
    types: list[str] = []
    checks = (
        (("气管切开", "气切", "tracheostomy"), "tracheostomy"),
        (("气管插管", "经口气管", "endotracheal", "intubation"), "endotracheal_tube"),
        (("鼻饲", "胃管", "nasogastric"), "nasogastric_tube"),
        (("脉氧", "SpO2", "脉搏血氧", "血氧探头"), "spo2_probe"),
        (("颈托", "cervical collar"), "cervical_collar"),
        (("骨科", "orthopedic"), "orthopedic_device"),
        (("nCPAP", "CPAP", "cpap"), "cpap"),
        (("无创通气", "无创正压", "NIV", "面罩压力", "面罩相关"), "niv_mask"),
    )
    for keys, device in checks:
        if any(key in text for key in keys) and device not in types:
            types.append(device)
    if "general_mdrpi" not in types:
        types.append("general_mdrpi")
    return types


def _topics(text: str) -> list[str]:
    rules = (
        (("定义", "definition"), "definition"),
        (("机制", "mechanism"), "mechanism"),
        (("发生率",), "incidence"),
        (("患病率",), "prevalence"),
        (("危险因素", "risk factor"), "risk_factor"),
        (("风险评估", "risk assessment"), "risk_assessment"),
        (("皮肤评估", "皮肤观察", "skin assessment"), "skin_assessment"),
        (("黏膜", "mucosal"), "mucosal_assessment"),
        (("设备评估", "device assessment"), "device_assessment"),
        (("固定", "fixation"), "device_fixation"),
        (("轮换", "重新定位", "reposition"), "device_reposition"),
        (("减压", "pressure relief"), "pressure_relief"),
        (("翻身", "体位", "repositioning"), "repositioning"),
        (("皮肤保护", "skin protection"), "skin_protection"),
        (("敷料", "dressing"), "prophylactic_dressing"),
        (("潮湿", "湿度", "moisture"), "moisture_management"),
        (("营养", "nutrition"), "nutrition"),
        (("预防", "prevention"), "prevention"),
        (("优先", "priority"), "priority"),
        (("安全", "safety"), "safety_boundary"),
        (("监测", "monitoring"), "monitoring"),
    )
    output: list[str] = []
    for keys, topic in rules:
        if any(key.lower() in text.lower() for key in keys) and topic not in output:
            output.append(topic)
    return output


def detect_language(text: str) -> str:
    chinese = len(re.findall(r"[\u4e00-\u9fff]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    # Bilingual abstracts and standards should remain explicit rather than
    # being forced into the language with the larger raw character count.
    if chinese >= 100 and latin >= 100:
        return "Mixed"
    if chinese and chinese >= latin:
        return "Chinese"
    if latin and latin > chinese * 2:
        return "English"
    return "Mixed"


def infer_metadata(filename: str, text_sample: str = "", pdf_title: str = "", pdf_author: str = "") -> InferredMetadata:
    """Infer controlled fields conservatively and document what needs review."""

    filename_text = Path(filename).stem
    bibliographic = f"{filename_text} {pdf_title}".strip()
    # Body text is intentionally limited to the first page for auxiliary device,
    # topic and language hints; it cannot determine source type or population.
    sample = text_sample[:1200]
    combined = f"{bibliographic} {sample}".strip()
    title = pdf_title.strip() or _normalise_stem(filename)
    authors = pdf_author.strip()
    source_type = _source_type(filename, pdf_title, sample)
    population, population_needs_review = _population(filename, pdf_title)
    device_type = _device_types(combined)
    topic = _topics(combined)
    organization = _organisation(f"{bibliographic} {pdf_author}")
    year = _year(filename)
    evidence_tier = {
        "guideline": "A",
        "consensus": "A",
        "standard": "A",
        "systematic_review": "B",
        "meta_analysis": "B",
        "evidence_summary": "B",
        "scoping_review": "C",
        "integrative_review": "C",
        "original_research": "C",
    }.get(source_type, "D")
    notes: list[str] = []
    if not pdf_title.strip():
        notes.append("title_from_filename_verify")
    if not authors:
        notes.append("authors_not_available_verify")
    if not organization:
        notes.append("organization_not_available_verify")
    if year is None:
        notes.append("year_not_parsed_verify")
    if population_needs_review:
        notes.append("population_needs_review")
    if population[0] in {"pediatric", "neonatal"}:
        evidence_tier = "D"
        notes.append("non_target_population_review_before_adult_generation")
    if year is not None and year < 2017:
        evidence_tier = "D"
        notes.append("older_source_review_before_generation")
    if source_type in {"poster", "patient_safety_alert", "other"}:
        notes.append("lower_priority_source_review")
    # Flag a strong filename/content contradiction without silently changing the
    # bibliographic classification.  DOC010 is a known example of this pattern.
    if _has_any(filename_text, ("气管切开", "tracheostomy")) and _has_any(sample, ("children aged", "paediatric", "pediatric", "儿童", "儿科")) and not _has_any(sample, ("tracheostomy", "气管切开")):
        notes.append("filename_content_mismatch_review")
    include = "no" if evidence_tier == "D" or "filename_content_mismatch_review" in notes else "yes"
    if include == "no":
        notes.append("parsed_and_retained_but_excluded_from_default_generation")
    return InferredMetadata(
        title=title,
        year=year,
        authors=authors,
        organization=organization,
        source_type=source_type,
        evidence_tier=evidence_tier,
        population=population,
        device_type=device_type,
        topic=topic,
        language=detect_language(f"{title} {text_sample}"),
        include=include,
        notes=";".join(notes),
    )


def validate_metadata(source_type: str, population: list[str], device_type: list[str], topic: list[str]) -> list[str]:
    """Return validation errors instead of silently correcting metadata."""

    errors: list[str] = []
    if source_type not in SOURCE_TYPES:
        errors.append(f"invalid source_type: {source_type}")
    errors.extend(f"invalid population: {item}" for item in population if item not in POPULATIONS)
    errors.extend(f"invalid device_type: {item}" for item in device_type if item not in DEVICE_TYPES)
    errors.extend(f"invalid topic: {item}" for item in topic if item not in TOPICS)
    return errors
