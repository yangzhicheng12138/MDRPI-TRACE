"""Targeted retrieval reuses corpus admission and only emits professional queries."""

import hashlib

from .six_step_schema import STEPS


DEVICE_TERMS = {
    "niv_mask": "non-invasive ventilation mask", "cpap": "CPAP mask",
    "nasogastric_tube": "nasogastric tube", "endotracheal_tube": "endotracheal tube",
    "tracheostomy": "tracheostomy", "spo2_probe": "pulse oximetry sensor",
    "blood_pressure_cuff": "blood pressure cuff", "vascular_access": "vascular access catheter",
    "cervical_collar": "cervical collar", "orthopedic_device": "orthopedic device",
}
TOPICS = {
    "recognize": "risk factors assessment", "analyze": "assessment differential skin mucosal injury",
    "prioritize": "assessment priority treatment safety", "solutions": "prevention management device function",
    "action": "management repositioning safety monitoring", "evaluate": "reassessment monitoring escalation",
}


def professional_query(devices, step, population=None):
    if step not in STEPS or any(d not in DEVICE_TERMS for d in devices):
        raise ValueError("Unrecognized professional search vocabulary")
    people = set(population or ["adult"])
    prefix = "neonatal" if "neonatal" in people else "pediatric" if "pediatric" in people else "adult"
    return prefix + " medical device related pressure injury " + " ".join(DEVICE_TERMS[d] for d in devices) + " " + TOPICS[step]


def retrieval_routes(issues, *, research_mode=False):
    patient, local, corrections = [], set(), []
    for issue in issues:
        known_errors = {"case_fact_error", "risk_misclassification", "numeric_error", "missing_condition",
                        "applicability_mismatch", "unsupported_priority", "safety_issue"}
        if research_mode:
            known_errors.discard("missing_condition")
        patient_gap = (issue["route"] == "patient" and issue.get("issue_type") == "missing_condition"
                            and (research_mode or issue["target_id"].startswith("F")
                                 or issue["target_id"] == "case.missing_information"))
        if patient_gap:
            patient.append(issue["needed"])
        elif issue.get("issue_type") in known_errors:
            corrections.append(issue["check_id"])
        elif issue["route"] == "patient":
            patient.append(issue["needed"])
        elif issue["route"] in {"knowledge", "conflict"}:
            step = issue["target_id"].split(".")[0]
            if step in STEPS:
                local.add(step)
        elif issue["route"] == "correction":
            corrections.append(issue["check_id"])
    return {"questions": list(dict.fromkeys(patient)), "steps": sorted(local, key=list(STEPS).index),
            "corrections": corrections}


def local_batch(engine, queries, devices, people, budget, log, *, original_question=None):
    budget.consume("local_batches")
    found = {}
    for query in dict.fromkeys(queries):
        budget.remaining()
        result = engine.retrieve(query, device_type=devices or None, population=people)
        log.write("local_retrieval", result)
        if result["status"] == "index_unavailable":
            raise RuntimeError("index_or_source_unavailable")
        for item in result.get("evidence", []):
            if original_question:
                from .evidence_constraints import requested_years, year_decision
                date = year_decision(item, requested_years(original_question))
                if not date["eligible"]:
                    log.write("local_constraint_rejection", {"chunk_id": item["chunk_id"], **date})
                    continue
            corpus = engine.retriever.corpus
            if corpus is not None:
                if item["chunk_id"] in corpus.blocked_chunk_ids:
                    raise ValueError("Quarantined source in retrieval")
                source = corpus.chunks[item["chunk_id"]]
                corpus.verify_source(source)
                if item["text"] != source["text"]:
                    raise ValueError("Source text changed")
            eid = item["chunk_id"]
            found[eid] = {**item, "evidence_id": eid, "origin": "local", "material_level": "full_text_excerpt",
                          "location": f"PDF pages {item['source_pages']}; {item.get('section', '')}",
                          "source_url": f"/v1/sources/{item['doc_id']}#page={item['page_start']}",
                          "content_sha256": hashlib.sha256(item["text"].encode()).hexdigest()}
    return found


def verify_evidence(engine, evidence):
    engine.retriever.check_index()
    corpus = engine.retriever.corpus
    for eid, item in evidence.items():
        if hashlib.sha256(item["text"].encode()).hexdigest() != item["content_sha256"]:
            raise ValueError("Evidence content hash mismatch")
        if item["origin"] == "local" and corpus is not None:
            corpus.assert_current()
            if eid in corpus.blocked_chunk_ids or corpus.chunks[eid]["text"] != item["text"]:
                raise ValueError("Local source no longer admitted")
            corpus.verify_source(corpus.chunks[eid])
