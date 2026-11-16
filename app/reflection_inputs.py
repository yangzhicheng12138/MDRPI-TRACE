"""Conservative input projection for reviews.

This module does not summarize medical text or alter claims. Original inputs
remain in the run log. Catalog omission is gated by the assigned review task.
"""
from copy import deepcopy

AUDIT_HASH_FIELDS = frozenset({"content_sha256", "source_pdf_sha256", "source_page_text_sha256"})
TARGET_REVIEWS = frozenset({"audit_consolidated", "audit_focused"})

def project_review_input(payload, name):
    """Remove only known audit hashes and a provably unassigned title catalog."""
    result, changes = deepcopy(payload), []
    if name not in TARGET_REVIEWS:
        return result, changes
    for collection in ("sources", "verification_sources"):
        sources = result.get(collection, [])
        if not isinstance(sources, list):
            continue
        for index, source in enumerate(sources):
            if not isinstance(source, dict):
                continue
            scope = source.get("evidence_scope")
            location = scope.get("source_location") if isinstance(scope, dict) else None
            if isinstance(location, dict):
                for field in sorted(AUDIT_HASH_FIELDS & location.keys()):
                    location.pop(field)
                    changes.append({"path": f"/{collection}/{index}/evidence_scope/source_location/{field}",
                                    "reason": "audit_hash_retained_in_original_log"})
    task, targets = result.get("research_task"), result.get("targets")
    if (isinstance(task, dict) and task.get("literature_requested") is False
            and task.get("assigned_steps") == [] and isinstance(targets, dict)
            and targets and all(not key.endswith(".evidence") and isinstance(row, dict)
                and not str(row.get("target_id", "")).endswith(".evidence")
                for key, row in targets.items())
            and isinstance(result.get("available_evidence"), list) and result["available_evidence"]):
        changes.append({"path": "/available_evidence", "reason": "no_assigned_literature_task",
                        "catalog_entries": len(result["available_evidence"])})
        result["available_evidence"] = []
    return result, changes
