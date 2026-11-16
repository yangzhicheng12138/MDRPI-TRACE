"""Pinned, reviewed development materials for generation; never formal cases.

Project-authored teaching references are distinct from primary-source excerpts.
Every selected item is rechecked against its review audit and source PDFs before
it is supplied to either generation agent. Nothing here publishes a bank item.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .bank_review import approved_items, content_fingerprint, load_review_queue

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "data/question_generation/development_materials.json"
SOURCE_PATH = "data/teaching_bank/expansion_250.json"
SCHEMA_VERSION = "question_generation_development_materials_v1"
POPULATIONS = {"adult", "pediatric", "neonatal"}
_PUBLIC_FIELDS = ("id", "title", "population", "populations", "device_types",
                  "source_question_id", "review_status", "reviewer_kind",
                  "independent_clinical_validation", "usage", "source_path",
                  "license", "origin")


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _path(value):
    if not isinstance(value, str) or not value:
        raise ValueError("Material source path is missing.")
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT.resolve()) or not path.is_file():
        raise ValueError("Material source is unavailable or outside the project.")
    return path


def _anchors(item):
    return [{"id": point["id"], "step": step["id"],
             "text": point["criterion"]["zh"],
             "acceptable": point["acceptable"]["zh"],
             "reference_ids": deepcopy(point["reference_ids"])}
            for step in item["six_step_rubric"]["steps"]
            for point in step["required"]]


def _registry():
    # A fresh public checkout has no private PDF-backed development registry.
    # Once supplied, all original provenance/approval checks still apply.
    if not REGISTRY_PATH.exists():
        return []
    data = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    if (data.get("schema_version") != SCHEMA_VERSION
            or data.get("usage") != "development_only"
            or not isinstance(data.get("items"), list) or not data["items"]):
        raise ValueError("Unsupported development material registry.")
    ids = [row.get("id") for row in data["items"]]
    if any(not isinstance(value, str) or not value.startswith("dev-exp250-") for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("Development material IDs must be distinct approved expansion IDs.")
    return data["items"]


def _scope(population, device_type):
    if population is not None and population not in POPULATIONS:
        raise ValueError("Unsupported material population.")
    if device_type is not None:
        devices = json.loads((ROOT / "data/teaching_bank/catalog_taxonomy.json").read_text(encoding="utf-8"))["devices"]
        if device_type not in devices:
            raise ValueError("Unsupported material device type.")


def _verified(rows):
    queues, verified_pdfs = {}, set()
    for row in rows:
        prefix = "Development material " + row["id"] + ": "
        if (row.get("source_path") != SOURCE_PATH
                or row.get("source_question_id") != row["id"].removeprefix("dev-")
                or row.get("usage") != "development_only"
                or row.get("review_status") != "approved_ai"
                or row.get("reviewer_kind") != "ai"
                or row.get("independent_clinical_validation") is not False):
            raise ValueError(prefix + "source identity or review scope is invalid.")
        source_path = _path(row["source_path"])
        if row["source_path"] not in queues:
            if _sha(source_path) != row.get("source_sha256"):
                raise ValueError(prefix + "source fingerprint is stale; review and repin the material.")
            queue = load_review_queue(source_path)
            queues[row["source_path"]] = (_sha(source_path), {
                item["id"]: item for item in approved_items(queue)})
        source_sha, approved = queues[row["source_path"]]
        if source_sha != row.get("source_sha256"):
            raise ValueError(prefix + "source fingerprint is stale; review and repin the material.")
        item = approved.get(row["source_question_id"])
        if item is None:
            raise ValueError(prefix + "current reviewed approval is unavailable.")
        if item.get("kind") == "standardized_case":
            raise ValueError(prefix + "formal standardized cases cannot be generation materials.")
        if (content_fingerprint(item) != row.get("source_item_fingerprint")
                or item["review_history"][-1]["event_hash"] != row.get("source_review_event_hash")):
            raise ValueError(prefix + "item or approval fingerprint is stale.")
        expected = {
            "title": item["title"], "populations": item["populations"],
            "population": item["populations"][0], "device_types": item["device_types"],
            "case_facts": item["case_text"], "excerpt": item["answer"]["zh"],
            "teaching_anchors": _anchors(item), "references": item["source"]["references"],
            "license": item["license"], "license_url": item["license_url"],
            "origin": item["content_origin"],
            "learning_objectives": item["learning_objectives"]["zh"],
        }
        if any(row.get(key) != value for key, value in expected.items()):
            raise ValueError(prefix + "registered text or applicability differs from reviewed content.")
        if (row.get("excerpt_kind") != "project_authored_reviewed_teaching_reference"
                or row.get("case_facts_kind") != "existing_fictional_development_case"):
            raise ValueError(prefix + "teaching-reference provenance is missing.")
        for reference in row["references"]:
            if not set(row["populations"]) <= set(reference.get("populations", [])):
                raise ValueError(prefix + "reference population does not cover the material.")
            pdf = _path(reference["local_source"])
            identity = (str(pdf), reference["source_pdf_sha256"])
            if identity not in verified_pdfs:
                if _sha(pdf) != reference["source_pdf_sha256"]:
                    raise ValueError(prefix + "original evidence PDF fingerprint is stale.")
                verified_pdfs.add(identity)
    return deepcopy(rows)


def materials(population=None, device_type=None):
    """Return safe selection metadata; stale source approvals fail closed."""
    _scope(population, device_type)
    rows = [row for row in _registry()
            if (population is None or population in row.get("populations", []))
            and (device_type is None or device_type in row.get("device_types", []))]
    return [{key: deepcopy(row[key]) for key in _PUBLIC_FIELDS}
            for row in _verified(rows)]


def load_materials(ids, population, device_type):
    """Load explicit approved development IDs within one requested scope.

    Raises ValueError on unknown IDs, changed reviewed content, changed PDFs,
    or crossed population/device scope; no silent partial material selection.
    """
    _scope(population, device_type)
    if (not isinstance(ids, list) or not ids or len(ids) > 5
            or not all(isinstance(value, str) for value in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError("Select one to five distinct development material IDs.")
    registry = {row["id"]: row for row in _registry()}
    if any(value not in registry for value in ids):
        raise ValueError("Unknown development material ID; formal cases cannot be selected.")
    rows = [registry[value] for value in ids]
    if any(population not in row.get("populations", []) or device_type not in row.get("device_types", []) for row in rows):
        raise ValueError("Selected material does not cover the requested population and device.")
    return _verified(rows)
