"""Import the supplied case narratives without inventing an answer key."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import zipfile
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[1]
BANK_PATH = ROOT / "data/teaching_bank/standardized_cases.json"
SOURCE_PATH = ROOT / "data/private_research/all_cases_source.json"
DOCX_PATH = ROOT / "data/private_research/cases_v2.docx"
PREVIOUS_DOCX_PATH = ROOT / "data/private_research/cases_v1.docx"
VERSION = "mdrpi_standardized_cases_v2_20261009"
SOURCE_VERSION = "v2_scope_supplement_20261005"
CASE_IDS = tuple(f"{group}-{number:02d}" for group in ("NIV", "NG", "AIR", "SPO2", "ORTH") for number in range(1, 7))
DEVICE_MAP = {
    **{f"NIV-{number:02d}": ["niv_mask"] for number in range(1, 7)},
    "NG-01": ["nasogastric_tube"], "NG-02": ["nasogastric_tube"],
    "NG-03": ["nasogastric_tube", "nasal_cannula"],
    "NG-04": ["nasogastric_tube", "nasal_cannula"],
    "NG-05": ["nasojejunal_tube", "nasal_tube_fixation_device"],
    "NG-06": ["nasogastric_tube"],
    "AIR-01": ["endotracheal_tube", "bite_block"],
    "AIR-02": ["endotracheal_tube", "bite_block"],
    "AIR-03": ["endotracheal_tube", "bite_block"],
    "AIR-04": ["tracheostomy_tube"], "AIR-05": ["endotracheal_tube"],
    "AIR-06": ["tracheostomy_tube"],
    **{f"SPO2-{number:02d}": ["pulse_oximeter"] for number in range(1, 7)},
    "ORTH-01": ["cervical_collar"], "ORTH-02": ["cast"],
    "ORTH-03": ["cervical_collar"], "ORTH-04": ["thoracolumbosacral_brace"],
    "ORTH-05": ["cast"], "ORTH-06": ["hip_abduction_brace"],
}


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def case_content_sha256(item):
    """Same immutable clinical-content boundary used by the browsing taxonomy."""
    content = {"case": item.get("case"), "case_zh": item.get("case_zh"), "scope": item.get("scope")}
    return hashlib.sha256(json.dumps(content, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def docx_cases(path):
    """Read Word paragraphs, joining runs so highlighted revisions stay intact."""
    with zipfile.ZipFile(path) as archive:
        document = ElementTree.fromstring(archive.read("word/document.xml"))
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    cases, current = {}, None
    for paragraph in document.findall(".//w:body/w:p", namespace):
        text = "".join(node.text or "" for node in paragraph.findall(".//w:t", namespace)).strip()
        heading = re.fullmatch(r"((?:NIV|NG|AIR|SPO2|ORTH)-\d{2})\s+(.+)", text)
        if heading:
            current = heading.group(1)
            if current in cases:
                raise ValueError("Duplicate original case number: " + current)
            cases[current] = {"title": heading.group(2), "paragraphs": []}
        elif current and text and not re.match(r"^[A-E]\s", text):
            cases[current]["paragraphs"].append(text)
    return cases


def build_standardized_bank(source_path=SOURCE_PATH, docx_path=DOCX_PATH, previous_docx_path=PREVIOUS_DOCX_PATH):
    """Validate the frozen extraction against every supplied v2 Word paragraph."""
    source = json.loads(source_path.read_text(encoding="utf-8"))
    originals = docx_cases(docx_path)
    if tuple(row.get("id") for row in source) != CASE_IDS or set(originals) != set(CASE_IDS):
        raise ValueError("Expected the original 30 ordered case numbers.")
    if Counter(row["difficulty"] for row in source) != Counter({"基础": 10, "综合": 10, "复杂": 10}):
        raise ValueError("Original case difficulty counts changed.")
    if Counter(row["population"] for row in source) != Counter({"adult": 28, "pediatric": 1, "neonatal": 1}):
        raise ValueError("Original case population counts changed.")
    document_hash = sha256_file(docx_path)
    extraction_hash = sha256_file(source_path)
    items = []
    for row in source:
        raw_id = row["id"]
        if originals[raw_id] != {"title": row["title"], "paragraphs": row["paragraphs"]}:
            raise ValueError("Frozen extraction differs from supplied Word case: " + raw_id)
        if hashlib.sha256(row["query"].encode()).hexdigest() != row["query_sha256"]:
            raise ValueError("Frozen query hash differs: " + raw_id)
        text = "\n\n".join(row["paragraphs"])
        if not row["query"].startswith(text + "\n\n"):
            raise ValueError("Frozen query contains a different clinical narrative: " + raw_id)
        title = re.sub(r"^(?:基础|综合|复杂)\s+", "", row["title"])
        reference = {"source_file": docx_path.name, "source_sha256": document_hash,
                     "source_version": SOURCE_VERSION, "original_id": raw_id,
                     "original_title": row["title"], "snapshot_file": str(SOURCE_PATH.relative_to(ROOT)),
                     "snapshot_sha256": extraction_hash,
                     "case_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                     "original_query_sha256": row["query_sha256"]}
        item = {"id": "standard-" + raw_id, "case_id": raw_id, "original_id": raw_id,
                "title": title, "original_title": row["title"], "level": row["difficulty"],
                "source_difficulty": row["difficulty"],
                "difficulty": "进阶" if row["difficulty"] == "综合" else row["difficulty"],
                "category": "mdrpi", "kind": "standardized_case", "has_reference_answer": False,
                "answer_origin": "no_supplied_answer_key", "source_type": "provided_standardized_case",
                "source_reference": reference, "source": reference, "version": VERSION,
                "review_status": "source_imported_pending_expert_review", "supported_languages": ["zh"],
                "population": row["population"], "device_type": DEVICE_MAP[raw_id][0],
                "device_types": DEVICE_MAP[raw_id], "device_group": row["device_group"],
                "case_text": text, "case": text, "case_zh": text, "paragraphs": row["paragraphs"],
                "six_step_rubric": None, "must_answer_points": [],
                "missing_fields": ["six_step_rubric", "must_answer_points", "expert_review"],
                "scope": {"population": row["population"], "devices": DEVICE_MAP[raw_id]},
                "review_note": "原件为案例文本修订稿；未提供可核验的专家审核结论、六步评分细则或标准答案。",
                "classification": {"populations": [row["population"]], "devices": DEVICE_MAP[raw_id],
                                   "basis": f"{docx_path.name}:{raw_id}; 人群来自封存源数据，器械逐例对照题面"}}
        item["classification"]["source_sha256"] = case_content_sha256(item)
        items.append(item)
    return {"version": VERSION, "source_version": SOURCE_VERSION,
            "source_reference": {"source_file": docx_path.name, "source_sha256": document_hash,
                                 "snapshot_file": str(SOURCE_PATH.relative_to(ROOT)), "snapshot_sha256": extraction_hash,
                                 "previous_source_file": previous_docx_path.name,
                                 "previous_source_sha256": sha256_file(previous_docx_path)},
            "import_checks": {"case_count": 30, "word_paragraphs_match_snapshot": True,
                              "population_counts": dict(Counter(row["population"] for row in source)),
                              "source_difficulty_counts": dict(Counter(row["difficulty"] for row in source))},
            "items": items}


def load_standardized_cases(path=BANK_PATH):
    # Formal study cases are optional private assets, not part of this release.
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data["items"]
    if tuple(item.get("case_id") for item in items) != CASE_IDS:
        raise ValueError("Standardized bank must contain the original 30 ordered case numbers.")
    bank = {}
    for item in items:
        if item["id"] != "standard-" + item["case_id"] or item["id"] in bank:
            raise ValueError("Duplicate or unstable standardized case ID.")
        if item["case"] != item["case_text"] or item["case_zh"] != item["case_text"]:
            raise ValueError("Imported standardized narrative changed: " + item["id"])
        if item["source_reference"]["case_text_sha256"] != hashlib.sha256(item["case_text"].encode()).hexdigest():
            raise ValueError("Imported standardized narrative hash changed: " + item["id"])
        if item["has_reference_answer"] or item["six_step_rubric"] is not None or item["must_answer_points"]:
            raise ValueError("Supplied narratives do not include an authoritative answer key or rubric.")
        bank[item["id"]] = item
    return bank
