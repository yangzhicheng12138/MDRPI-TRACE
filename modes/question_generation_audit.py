"""Traceable full reviews for planned development questions.

The controller verifies source identities, never clinical truth. New reviews
may retain the reviewer's faithful summaries instead of verbatim excerpts.
An absent reference is distinct from a medical error. Legacy jobs keep their
original review contract and approval bindings.
"""
from copy import deepcopy
import re

from jsonschema import Draft202012Validator, ValidationError

from .question_generation_schema import review_schema, validate_review


def _object(properties):
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


def _text(maximum=240, minimum=1):
    return {"type": "string", "minLength": minimum, "maxLength": maximum}


def _status(values):
    return _object({"status": {"type": "string", "enum": list(values)}, "note": _text()})


def evidence_text_field(evidence_mode):
    if evidence_mode not in {"verbatim", "semantic"}:
        raise ValueError("Unsupported review evidence mode")
    return "summary" if evidence_mode == "semantic" else "quote"


def audited_review_schema(evidence_ids, question, *, evidence_mode="verbatim"):
    """Keep the wire schema stable across 1/2/3-point questions in one batch.

    The validator below binds this shared schema to the current criterion IDs
    and its complete citation set. Current item shape stays out of the shared
    model prefix, allowing genuine prefix reuse for differently shaped items.
    """
    field = evidence_text_field(evidence_mode)
    schema = review_schema(evidence_ids)
    support = _object({
        "criterion_id": {"type": "string", "enum": ["R1", "R2", "R3"]},
        "status": {"type": "string", "enum": ["supported", "limited", "unsupported", "uncertain"]},
        "sources": {"type": "array", "minItems": 0, "maxItems": 16,
                    "items": _object({"evidence_id": {"type": "string", "enum": list(evidence_ids)},
                                      "status": {"type": "string", "enum": ["supported", "limited", "unsupported", "uncertain"]},
                                      field: ({"type": "string"} if evidence_mode == "semantic"
                                              else _text(360, 0))})},
        "rationale": _text(300),
    })
    schema["properties"]["quality_audit"] = _object({
        "case_facts": _status(["consistent", "unsupported", "uncertain"]),
        "assessment_alignment": _status(["aligned", "misaligned", "uncertain"]),
        "distinctness": _status(["distinct", "overlap", "not_applicable", "uncertain"]),
        "evidence_support": {"type": "array", "items": support,
                             "minItems": 1, "maxItems": 3},
    })
    schema["required"].append("quality_audit")
    return schema


def _normalized(value):
    # Readers preserve tables/HTML. Do not rewrite their clinical content or
    # make a paraphrased quotation appear to be verbatim.
    return re.sub(r"\s+", "", value)


def validate_audited_review(value, evidence, question, *, duplicate_candidates=False,
                            evidence_mode="verbatim"):
    """Validate source bindings and consistency of an acceptance.

    Legacy mode also checks verbatim excerpts. Semantic mode retains model
    summaries without assigning them a substring-based truth judgment. Neither
    mode proves that the reviewer interpreted the source correctly.
    """
    field = evidence_text_field(evidence_mode)
    sources = {row["evidence_id"]: row for row in evidence["sources"]}
    Draft202012Validator(audited_review_schema(sources, question, evidence_mode=evidence_mode)).validate(value)
    audit = value["quality_audit"]
    criteria = {row["id"]: row for row in question["scoring_points"]}
    audited_ids = [row["criterion_id"] for row in audit["evidence_support"]]
    if len(set(audited_ids)) != len(audited_ids) or set(audited_ids) != set(criteria):
        raise ValidationError("Every scoring point must have exactly one support check")
    for check in audit["evidence_support"]:
        cited = criteria[check["criterion_id"]]["evidence_ids"]
        if check["status"] == "supported" and not check["sources"]:
            raise ValidationError("Supported criteria must expose at least one actual source excerpt" if field == "quote"
                                  else "Supported criteria must explain at least one cited source")
        checked_ids = [quote["evidence_id"] for quote in check["sources"]]
        if len(set(checked_ids)) != len(checked_ids):
            raise ValidationError("A scoring point's source cannot be audited twice")
        for quote in check["sources"]:
            identity = quote["evidence_id"]
            if identity not in cited:
                raise ValidationError("A support check must use that scoring point's cited sources")
            excerpt = _normalized(quote[field])
            if quote["status"] == "supported" and not excerpt:
                raise ValidationError("A supported source must expose an actual source excerpt" if field == "quote"
                                      else "A supported source must explain its evidence basis")
            # Semantic support is the reviewer's judgment over the full source
            # packet. Do not substitute a substring or similarity threshold.
            if evidence_mode == "verbatim" and excerpt and excerpt not in _normalized(sources[identity]["text"]):
                raise ValidationError("Support quotation must be a verbatim excerpt of the cited source")
        if set(checked_ids) != set(cited):
            raise ValidationError("Every cited source for that scoring point must be audited exactly once")
    if value["verdict"] == "accept":
        # Do not hide a known limitation in an overall accept. The reviewer
        # chooses a revision or more evidence; the code assigns no medical label.
        consistent = (audit["case_facts"]["status"] == "consistent"
                      and audit["assessment_alignment"]["status"] == "aligned"
                      and all(row["status"] == "supported"
                              and all(source["status"] == "supported" for source in row["sources"])
                              for row in audit["evidence_support"]))
        distinct = audit["distinctness"]["status"]
        if not consistent or distinct in {"overlap", "uncertain"}:
            raise ValidationError("An acceptance cannot contain unresolved quality checks")
        if duplicate_candidates and distinct == "not_applicable":
            raise ValidationError("Detected overlap candidates require an explicit distinctness check")
    # The base contract remains authoritative for verdict/issue consistency.
    core = {key: item for key, item in value.items() if key != "quality_audit"}
    validate_review(core, list(sources))
    return deepcopy(value)
