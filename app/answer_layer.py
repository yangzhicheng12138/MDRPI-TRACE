"""Evidence-bound structured Answer and one-pass Reflection contracts."""
ANSWER_SCHEMA = {"status":"supported","risk_factors":{"patient":[],"device":[],"care_process":[]},"observation_sites":[],"interventions":[],"ongoing_monitoring":[],"uncertainty":[],"citations":[]}
REFLECTION_SCHEMA = {"missing_risks":[],"missing_sites":[],"missing_interventions":[],"unsupported_claims":[],"population_extrapolation":[],"device_extrapolation":[],"safety_issues":[],"citation_issues":[],"priority_issue":False,"needs_revision":False}

def build_answer(status, citations=(), conflict_pairs=(), reason=""):
    """Project verified extracts into the research Answer schema.

    The projection deliberately carries source text as evidence-bound rationale;
    it does not invent a clinical paraphrase. A later model Answer Layer may
    populate the semantic fields only when each item retains these IDs.
    """
    ids=[c.get('chunk_id') for c in citations if c.get('chunk_id')]
    return {"status": "supported" if status == "answered" else "conflict" if status == "possible_conflict" else "insufficient" if status in ("insufficient_evidence","out_of_scope") else status,
            "risk_factors":{"patient":[],"device":[],"care_process":[]},"observation_sites":[],"interventions":[],"ongoing_monitoring":[],
            "uncertainty":[reason] if reason else [],"citations":ids,"conflict_pairs":list(conflict_pairs)}

def validate_answer(answer, evidence_ids):
    """Validate structure and citation coverage, not clinical entailment.

    Every clinical item uses action, rationale and evidence_ids. Interventions
    additionally require priority; unknown means evidence does not rank urgency.
    """
    if not isinstance(answer, dict):
        return False
    required = set(ANSWER_SCHEMA)
    if not required <= set(answer) or set(answer) - required - {'conflict_pairs'}:
        return False
    if answer['status'] not in ('supported', 'insufficient', 'conflict'):
        return False

    def unique_ids(value):
        return (isinstance(value, list) and all(isinstance(x, str) and x.strip() for x in value)
                and len(value) == len(set(value)))

    if not isinstance(evidence_ids, (list, tuple, set)) or not all(isinstance(x, str) for x in evidence_ids):
        return False
    ids = set(evidence_ids)
    if not unique_ids(answer['citations']) or not set(answer['citations']) <= ids:
        return False
    risks = answer['risk_factors']
    if not isinstance(risks, dict) or set(risks) != {'patient', 'device', 'care_process'}:
        return False
    if (not isinstance(answer['uncertainty'], list)
            or not all(isinstance(x, str) and x.strip() for x in answer['uncertainty'])):
        return False
    groups = [(v, False) for v in risks.values()]
    groups += [(answer[k], k == 'interventions') for k in
               ('observation_sites', 'interventions', 'ongoing_monitoring')]
    used = set()
    count = 0
    for items, intervention in groups:
        if not isinstance(items, list):
            return False
        for item in items:
            fields = {'action', 'rationale', 'evidence_ids'} | ({'priority'} if intervention else set())
            if not isinstance(item, dict) or set(item) != fields:
                return False
            if not all(isinstance(item[k], str) and item[k].strip() for k in ('action', 'rationale')):
                return False
            if not unique_ids(item['evidence_ids']) or not item['evidence_ids']:
                return False
            if not set(item['evidence_ids']) <= ids:
                return False
            if intervention and item['priority'] not in ('immediate', 'routine', 'conditional', 'unknown'):
                return False
            used.update(item['evidence_ids'])
            count += 1
    pairs = answer.get('conflict_pairs', [])
    if not isinstance(pairs, list):
        return False
    for pair in pairs:
        if (not isinstance(pair, dict) or set(pair) != {'left_id', 'right_id'}
                or not all(isinstance(v, str) for v in pair.values())
                or pair['left_id'] == pair['right_id']
                or not set(pair.values()) <= ids):
            return False
        used.update(pair.values())
    if used != set(answer['citations']):
        return False
    if answer['status'] == 'supported':
        return count > 0 and not pairs
    if answer['status'] == 'conflict':
        return count == 0 and bool(pairs) and bool(answer['uncertainty'])
    return count == 0 and not pairs and not used and bool(answer['uncertainty'])

def reflection(answer, evidence_ids):
    """Deterministic safety/citation audit; does not add evidence or knowledge."""
    out={k:(v.copy() if isinstance(v,list) else v) for k,v in REFLECTION_SCHEMA.items()}
    if not validate_answer(answer,evidence_ids): out['citation_issues'].append('invalid_or_unavailable_evidence_id')
    # This helper is a structural audit only; a semantic model review is pending.
    out['audit_scope'] = 'structure_and_citations_only'
    out['clinical_review_completed'] = False
    out['unsupported_claims'] = []
    out['needs_revision']=bool(out['citation_issues'] or out['unsupported_claims'] or out['safety_issues'])
    return out
