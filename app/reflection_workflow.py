"""Bounded generation, independent audits, targeted repairs and publication."""

from copy import deepcopy
import hashlib
import time

from .case_facts import FACT_PROMPT, candidate_clauses, extraction_schema, extract_located
from .corpus import json_digest
from .evidence_router import local_batch, professional_query, retrieval_routes, verify_evidence
from .evidence_snapshot import BudgetExceeded, ModelRunner, RunBudget, RunLog
from .external_evidence import ADMISSION_PROMPT, EuropePMC, ExternalFailure
from .external_query import QUERY_PROMPT
from .web_search import BraveSearch, DDGSSearch, BaiduSearch, WEB_QUERY_PROMPT, SEARCH_PROVIDERS, create_web_search, search_provider
from .web_reader import Crawl4AIReader
from .local_llm import InvalidModelOutput, ModelUnavailable
from .query_policy import query_scope
from .reflection_checks import FOCUSED_RESEARCH_PROMPT, PROMPTS, run_checks
from .six_step_answer import (GENERATION_PROMPT, REVISION_PROMPT, RESEARCH_GENERATION_PROMPT,
                              RESEARCH_REVISION_PROMPT, RESEARCH_RECOGNIZE_FOCUS, GROUPED_RESEARCH_PROMPT, LEAN_RESEARCH_PROMPT,
                              RESEARCH_RESPONSE_GUIDE,
                              RESEARCH_PLACEMENT_PROMPT, RESEARCH_SOURCE_PROMPT,
                              change_set, complete_evidence_task, evidence_context, generate_research_steps, generate_step)
from .six_step_schema import SCHEMA_VERSION, STEPS, all_claims, block_claims, empty_steps, render_answer, validate_steps


def prompt_identity():
    from .revision_scope_review import SCOPE_REVIEW_PROMPT
    return json_digest({"facts": FACT_PROMPT, "generate": GENERATION_PROMPT, "revise": REVISION_PROMPT,
                        "revision_scope_review": SCOPE_REVIEW_PROMPT,
                        "research_generate": RESEARCH_GENERATION_PROMPT, "research_revise": RESEARCH_REVISION_PROMPT,
                        "research_recognize": RESEARCH_RECOGNIZE_FOCUS,
                        "research_grouped": GROUPED_RESEARCH_PROMPT, "research_focused_audit": FOCUSED_RESEARCH_PROMPT,
                        "research_lean": LEAN_RESEARCH_PROMPT,
                        "research_response_guide": RESEARCH_RESPONSE_GUIDE,
                        "research_claim_placement": RESEARCH_PLACEMENT_PROMPT,
                        "research_source_synthesis": RESEARCH_SOURCE_PROMPT,
                        "audits": PROMPTS, "admission": ADMISSION_PROMPT,
                        "external_query": QUERY_PROMPT, "web_query": WEB_QUERY_PROMPT, "requirements": STEPS})


def audit_issues(audits):
    return [item for item in audits if item["verdict"] == "issue"]


FACT_EXTRACTION_ISSUE_TYPES = {"case_fact_error", "risk_misclassification"}


def fact_extraction_issues(issues):
    """Return only findings that require rebuilding case facts.

    A finding on a fact ID is not enough: ``content_omission`` means the
    answer failed to use an already extracted fact and must be repaired in
    the answer.  Only classification/state errors on the extracted facts or
    the missing-information contract may invoke ``repair_case_facts``.
    """
    return [
        issue for issue in issues
        if issue.get("check_kind", "case") == "case"
        and issue.get("issue_type") in FACT_EXTRACTION_ISSUE_TYPES
        and (issue.get("target_id", "").startswith("F")
             or issue.get("target_id") == "case.missing_information")
    ]


def preserve_missing_information(previous, repaired, *, replace=False):
    """Keep known patient-data gaps while allowing an explicit gap-list repair."""
    merged = deepcopy(repaired)
    if not replace:
        merged["missing_information"] = list(dict.fromkeys(
            [*previous.get("missing_information", []), *repaired.get("missing_information", [])]
        ))
    return merged


def revision_input_issues(issues, *, research_mode=False):
    # Retain the full review in its audit log. Both fact repair and step
    # revision rebuild from findings, not unverified replacement prose.
    if research_mode:
        return [{key: item[key] for key in ("check_kind", "target_id", "issue_type", "severity", "reason")}
                for item in issues]
    return issues


def affected_content(steps, issues, *, propagate=True):
    claims, affected, requirements = all_claims(steps), set(), set()
    for issue in issues:
        target = issue["target_id"]
        if target in claims:
            affected.add(target)
        elif target.startswith("F"):
            affected.update(cid for cid, c in claims.items() if target in c["fact_ids"])
        elif target == "case.missing_information":
            requirements |= {"recognize.unknowns", "recognize.questions", "analyze.information"}
        elif target.endswith(".coverage") and target.split(".")[0] in steps:
            step = target.split(".")[0]
            requirements.update(f"{step}.{key}" for key in STEPS[step][1])
            affected.update(c["claim_id"] for c in steps[step]["claims"])
        elif "." in target and target.split(".")[0] in steps:
            requirements.add(target)
    for target in requirements:
        step, key = target.split(".", 1)
        row = steps[step]["coverage"][key]
        affected.update(row["claim_ids"])
        if row["status"] == "missing" and not row["claim_ids"]:
            # A missing task can already have an unmapped placeholder with a
            # stable ID. Permit its completion, while mapped claims stay locked.
            mapped = {cid for value in steps[step]["coverage"].values() for cid in value["claim_ids"]}
            affected.update(c["claim_id"] for c in steps[step]["claims"] if c["claim_id"] not in mapped)
    while propagate:
        extra = {cid for cid, c in claims.items() if set(c["depends_on"]) & affected}
        if extra <= affected:
            break
        affected |= extra
    for step, value in steps.items():
        for key, row in value["coverage"].items():
            if set(row["claim_ids"]) & affected:
                requirements.add(f"{step}.{key}")
    return {step: {"affected_claim_ids": sorted(cid for cid in affected if cid.startswith(step + ".")),
                   "affected_requirements": [key for key in STEPS[step][1] if f"{step}.{key}" in requirements]}
            for step in steps if any(t.startswith(step + ".") for t in affected | requirements)}


def publication_candidate(steps, facts, audits):
    issues, claims = audit_issues(audits), all_claims(steps)
    bad_facts = {i["target_id"] for i in fact_extraction_issues(issues)
                 if i["target_id"].startswith("F")}
    blocked = {cid for cid, c in claims.items() if set(c["fact_ids"]) & bad_facts}
    for issue in issues:
        target = issue["target_id"]
        if target in claims:
            blocked.add(target)
        elif issue["issue_type"] not in {"content_omission", "unclear_time_role"} and "." in target:
            step, key = target.split(".", 1)
            if step in steps and key in steps[step]["coverage"]:
                blocked.update(steps[step]["coverage"][key]["claim_ids"])
    candidate, removed = block_claims(steps, blocked, "相关内容未通过核查，待补充依据或病例资料")
    for issue in issues:
        if "." in issue["target_id"]:
            step, key = issue["target_id"].split(".", 1)
            if step in candidate and key in candidate[step]["coverage"]:
                row = candidate[step]["coverage"][key]
                row.update(status="partial" if row["claim_ids"] else "missing", reason=issue["reason"])
    public_facts = deepcopy(facts)
    public_facts["facts"] = [f for f in public_facts["facts"] if f["fact_id"] not in bad_facts]
    public_facts["case_conflicts"] = [c for c in public_facts["case_conflicts"] if not set(c["fact_ids"]) & bad_facts]
    return candidate, public_facts, removed


def safe_audited_version(steps, audits):
    """Only honestly labelled coverage gaps may remain in a public answer."""
    claims = all_claims(steps)
    for issue in audit_issues(audits):
        target = issue["target_id"]
        if target in claims or target.startswith("F") or target == "case.missing_information":
            return False
        if issue["issue_type"] != "content_omission":
            return False
        step, key = target.split(".", 1)
        if key == "coverage":
            return False
        if steps[step]["coverage"][key]["status"] not in {"partial", "missing", "conflict"}:
            return False
    return True


def _annotate_research_claims(steps, audits=()):
    """Label provenance and observed support without changing the answer text."""
    result = deepcopy(steps)
    support_checks = [row for row in audits if row["check_kind"] in {"case", "evidence"}]
    passed = {row["check_id"] for row in support_checks if row["verdict"] == "pass"}
    disputed = {row["target_id"] for row in support_checks if row["verdict"] == "issue"}
    for value in result.values():
        for claim in value["claims"]:
            if claim["kind"] == "model_inference":
                claim["support_type"] = "model_knowledge_or_inference"
                claim["evidence_status"] = "not_directly_supported"
            elif claim["kind"] == "case_fact":
                claim["support_type"] = "case_input"
                claim["evidence_status"] = "supported_by_case_text"
            elif claim["kind"] == "source_statement":
                claim["support_type"] = "retrieved_evidence"
                claim["evidence_status"] = "direct_or_case_supported"
            elif claim["kind"] == "case_inference":
                claim["support_type"] = "case_inference"
                claim["evidence_status"] = "direct_or_case_supported"
            else:
                claim["support_type"] = "uncertainty"
                claim["evidence_status"] = "unresolved"
            if claim["kind"] in {"case_fact", "source_statement", "case_inference"}:
                check_kind = "case" if claim["kind"] == "case_fact" else "evidence"
                if (f"{check_kind}:{claim['claim_id']}" not in passed
                        or claim["claim_id"] in disputed or set(claim["fact_ids"]) & disputed):
                    claim["evidence_status"] = "unresolved"
    return result


def reflective_answer(engine, query, *, device_type=None, population=None, web_mode="off", replay_run_id=None,
                      budget_seconds=None, reflection=True, max_revision_rounds=None, research_mode=False,
                      web_strategy="on_demand", frozen_external_evidence=None):
    scope = query_scope(query, engine.config, device_type, population)
    if web_mode not in {"off", "live", "replay"} or (web_mode == "replay" and not replay_run_id):
        raise ValueError("web_mode must be off/live/replay; replay requires replay_run_id")
    if web_strategy not in {"on_demand", "verify"}:
        raise ValueError("web_strategy must be on_demand/verify")
    frozen_external = deepcopy(frozen_external_evidence or [])
    if frozen_external and web_mode != "off":
        raise ValueError("Frozen evaluation evidence cannot be combined with live/replay discovery")
    frozen_ids = [source.get("evidence_id") for source in frozen_external]
    if len(set(frozen_ids)) != len(frozen_ids):
        raise ValueError("Duplicate frozen external evidence ID")
    for source in frozen_external:
        if (not source.get("evidence_id") or source.get("origin") != "external"
                or not isinstance(source.get("text"), str) or not source.get("location")
                or not source.get("source_url", "").startswith("https://")
                or source.get("content_sha256") != hashlib.sha256(source["text"].encode()).hexdigest()):
            raise ValueError("Invalid frozen external evidence")
    gap_web_search = bool(reflection and engine.config.get("reflection", "web_gap_search", default=False))
    general_web = web_strategy == "verify" or gap_web_search
    web_provider = search_provider(engine.config) if general_web else engine.config.get("web_search", "provider", default="ddgs")
    if gap_web_search and web_provider != "deepseek":
        raise ValueError("reflection.web_gap_search requires web_search.provider=deepseek")
    postdraft_web = web_mode != "off" and (web_strategy == "on_demand" or gap_web_search)
    seconds = budget_seconds or engine.config.get("reflection", "budget_seconds", default=600)
    # Research mode keeps the same finite retry/revision limits but allows a
    # longer wall-clock budget for a real local model whose six structured
    # generations and audits can exceed the release-oriented default.
    if research_mode and budget_seconds is None:
        seconds = max(seconds, engine.config.get("reflection", "research_budget_seconds", default=3600))
    rounds_limit = min(2, max(0, max_revision_rounds if max_revision_rounds is not None else
                              engine.config.get("reflection", "max_revision_rounds", default=2)))
    compact = research_mode and engine.config.get("reflection", "research_compact", default=False)
    lean = compact and engine.config.get("reflection", "research_lean_output", default=False)
    consolidated = lean and engine.config.get("reflection", "research_consolidated_review", default=False)
    program_checks = research_mode and engine.config.get("reflection", "program_checks", default=False)
    protected_revision = research_mode and engine.config.get("reflection", "protect_revision_scope", default=False)
    reviewed_revision_scope = protected_revision and reflection and engine.config.get(
        "reflection", "review_out_of_scope_revisions", default=False)
    shared_web_sources = research_mode and reflection and engine.config.get("reflection", "shared_web_sources", default=False)
    case_scale_checks = research_mode and reflection and engine.config.get("reflection", "case_scale_checks", default=False)
    clause_scope_checks = research_mode and reflection and engine.config.get("reflection", "clause_scope_checks", default=False)
    repair_scale_examples = case_scale_checks and engine.config.get("reflection", "repair_impossible_scale_examples", default=False)

    def annotated_sources(items):
        if not clause_scope_checks:
            return items
        from .answer_guards import with_clause_bindings
        return {eid: with_clause_bindings(source) for eid, source in items.items()}

    def repaired_scale_examples(current, stage):
        if not repair_scale_examples:
            return current
        from .answer_guards import repair_case_scale_examples
        repaired, edits = repair_case_scale_examples(query, current)
        if edits:
            log.write("case_scale_example_repairs", {"stage": stage, "edits": edits, "extra_model_calls": 0})
        return repaired
    correction_rounds = rounds_limit
    if research_mode and max_revision_rounds is None:
        correction_rounds = min(rounds_limit, max(0, engine.config.get(
            "reflection", "research_max_revision_rounds", default=rounds_limit)))
        # Retain one existing evidence follow-up opportunity when new local
        # evidence fails to close a gap. This is not another general rewrite.
        rounds_limit = min(rounds_limit, correction_rounds + int(postdraft_web and correction_rounds > 0))
    audit_cache = {}
    log = RunLog(engine.config.path("reflection", "log_dir", default="logs/reflection_v2"))
    budget = RunBudget(seconds)
    runner = ModelRunner(engine.llm, budget, log)
    if reflection:
        from .reflection_backend import reflection_runner
        runner = reflection_runner(engine.config, runner)
    result = {"schema_version": SCHEMA_VERSION, "run_id": log.run_id, "query": query,
              "answer_mode": "research_six_steps" if research_mode else "reflective_six_steps", "status": "technical_failure", "six_steps": empty_steps(),
              "facts": {"facts": [], "missing_information": [], "case_conflicts": []}, "citations": [],
              "coverage": {}, "audit_summary": {}, "resolved_issues": [], "unresolved_issues": [],
              "questions": [], "web": {"mode": web_mode, "strategy": web_strategy,
                  "provider": SEARCH_PROVIDERS[web_provider] if general_web else "EuropePMC", "events": [],
                  "reflection_gap_search": gap_web_search},
              "stop_reason": "", "errors": [], "timing": {}, "research_status": "development_unfrozen_not_clinically_reviewed",
              "completeness_meaning": "automatic_requirement_coverage_not_clinical_correctness",
              "program_validation": {"quantity_checks": bool(program_checks),
                                     "revision_scope_protection": bool(protected_revision),
                                     "out_of_scope_revision_review": bool(reviewed_revision_scope),
                                     "shared_web_sources": bool(shared_web_sources),
                                     "case_scale_checks": bool(case_scale_checks),
                                     "clause_scope_checks": bool(clause_scope_checks),
                                     "repair_impossible_scale_examples": bool(repair_scale_examples),
                                     "clinical_correctness_verified": False},
              "model": engine.llm.model, "runtime_code_sha256": engine.runtime_code_sha256,
              "runtime_config_sha256": json_digest(engine.config.values), "prompt_sha256": prompt_identity(),
              "kb_version": engine.config.version, "collection": engine.config.qdrant_collection,
              "local_corpus_identity": getattr(engine.retriever.corpus, "identity", None)}
    if getattr(runner, "backend", None) is not None:
        result["reflection_backend"] = {key: runner.options[key] for key in
            ("provider", "model", "context_mode", "thinking", "reasoning_effort")}
    evidence, steps, audits, issues = {}, {}, [], []
    facts, last_verified, completed_rounds, web = result["facts"], None, 0, None
    research_fallback = None
    verification_ids = set()
    draft_text = {}
    log.write("input_config", {"query": query, "device_type": device_type, "population": population,
                               "web_mode": web_mode, "web_strategy": web_strategy,
                               "replay_run_id": replay_run_id, "configuration": engine.config.values,
                               "frozen_external_evidence_ids": frozen_ids,
                               "identity": {k: result[k] for k in ("model", "runtime_code_sha256", "prompt_sha256", "local_corpus_identity")},
                               "limits": {"seconds": seconds, "revision_rounds": rounds_limit,
                                          "correction_rounds": correction_rounds}, "reflection": reflection,
                               "review_strategy": "consolidated" if consolidated else "existing",
                               "research_mode": research_mode})
    try:
        if scope.reason:
            result.update(status="insufficient", stop_reason=scope.reason)
        else:
            browser_python = engine.config.get("web_search", "browser_python")
            web_options = {"max_search_rounds": engine.config.get("web_search", "max_search_rounds", default=2),
                           "browser_reader": Crawl4AIReader(engine.config.path("web_search", "browser_python"),
                               timeout=engine.config.get("web_search", "browser_timeout_seconds", default=30))
                               if browser_python else None}
            if general_web:
                web = create_web_search(engine.config, budget, log, runner, mode=web_mode, replay_run_id=replay_run_id,
                    factories={"ddgs": DDGSSearch, "brave": BraveSearch, "baidu": BaiduSearch}, **web_options)
                result["web"].update(provider=web.provider, search_strategy=engine.config.get("web_search", "strategy", default="fallback"))
            else:
                from .search_planner import planning_runner
                web = EuropePMC(budget, log, planning_runner(engine.config,runner), mode=web_mode, replay_run_id=replay_run_id)
            evidence = local_batch(engine, [query, professional_query(scope.devices, "solutions", scope.population)],
                                   scope.devices, scope.population, budget, log, original_question=query)
            if frozen_external:
                if set(frozen_ids) & set(evidence):
                    raise InvalidModelOutput("Frozen external evidence ID collides with local evidence")
                evidence.update({source["evidence_id"]: source for source in frozen_external})
                verification_ids = set(frozen_ids)
                log.write("frozen_external_evidence", frozen_external)
            log.write("initial_evidence", list(evidence.values()))
            fact_raw = runner.call(FACT_PROMPT, {"case": query, "candidate_clauses": candidate_clauses(query)},
                                   extraction_schema(query), "case_facts", max_tokens=1400)
            facts = extract_located(query, fact_raw)
            log.write("facts", facts)
            if reflection and web_strategy == "verify" and web_mode != "off":
                log.write("web_trigger", {"round": 0, "step": "answer_verification", "mode": web_mode,
                    "reason": "pre_draft_web_retrieval", "local_evidence_ids": sorted(evidence)})
                external = web.search(query, context={"question": query, "step": "answer_verification",
                    "phase": "pre_draft", "population": scope.population, "local_sources": evidence_context(evidence.values())})
                verification_ids = set(external)
                evidence.update(external)
                log.write("web_result", {"round": 0, "admitted_evidence_ids": sorted(external),
                    "changed_candidate_evidence": bool(external), "support_verified": False})
                log.write("verification_evidence", list(external.values()))
            evidence = annotated_sources(evidence)
            if compact:
                steps = generate_research_steps(runner, query, facts, evidence, lean=lean,
                                                verification_ids=verification_ids)
                for step in STEPS:
                    log.write("draft_" + step, steps[step])
            else:
                for step in STEPS:
                    steps[step] = generate_step(runner, query, facts, evidence, steps, step,
                                                research_mode=research_mode, verification_ids=verification_ids)
                    log.write("draft_" + step, steps[step])
            steps = repaired_scale_examples(steps, "initial_draft")
            validate_steps(steps, facts, evidence, research_mode=research_mode)
            log.write("draft", steps)
            draft_text = {cid: c["text"] for cid, c in all_claims(steps).items()}
            if research_mode:
                research_fallback = (deepcopy(steps), deepcopy(facts), [])
            if not reflection:
                result.update(status="insufficient", stop_reason="unreviewed_evaluation_baseline")
            else:
                audits = run_checks(runner, query, facts, steps, evidence, research_mode=research_mode,
                                    compact=compact, cache=audit_cache, scoped=lean, consolidated=consolidated,
                                    verification_ids=verification_ids, program_checks=program_checks,
                                    shared_web_sources=shared_web_sources, case_scale_checks=case_scale_checks,
                                    clause_scope_checks=clause_scope_checks)
                log.write("audits_0", audits)
                issues = audit_issues(audits)
                if research_mode:
                    research_fallback = (deepcopy(steps), deepcopy(facts), deepcopy(audits))
                if safe_audited_version(steps, audits):
                    last_verified = (deepcopy(steps), deepcopy(facts), deepcopy(audits))
                signatures = set()
                local_attempted_steps = set()
                gap_web_attempted_steps = set()
                for round_no in range(1, rounds_limit + 1):
                    if not issues:
                        result["stop_reason"] = "checks_complete"
                        break
                    signature = json_digest({"issues": [(i["check_id"], i["issue_type"], i["route"]) for i in issues],
                                             "coverage": [r["status"] for s in steps.values() for r in s["coverage"].values()],
                                             "evidence_ids": sorted(evidence),
                                             "step_context_ids": {s: sorted(ids) for s, ids in runner.context_ids.items()}})
                    if signature in signatures:
                        result["stop_reason"] = "repeated_issues_without_improvement"
                        break
                    signatures.add(signature)
                    routes = retrieval_routes(issues, research_mode=research_mode)
                    if completed_rounds >= correction_rounds and not (postdraft_web and routes["steps"]):
                        result["stop_reason"] = "revision_limit"
                        break
                    if research_mode and routes["questions"]:
                        # Review replacement prose is retained in the audit,
                        # not presented as a question or instruction to a user.
                        routes["questions"] = list(dict.fromkeys(
                            ["待确认：" + f["quote"] for f in facts["facts"] if f["state"] == "unknown"]
                            + facts["missing_information"])) or ["请补充尚未记录的患者观察资料。"]
                    log.write(f"routing_{round_no}", {
                        "round": round_no,
                        "issues": [{"check_id": i["check_id"], "target_id": i["target_id"],
                                    "issue_type": i["issue_type"], "route": i["route"],
                                    "needed": i.get("needed", "")} for i in issues],
                        "patient_questions": routes["questions"],
                        "local_steps": routes["steps"],
                        "corrections": routes["corrections"],
                        "web_mode": web_mode,
                    })
                    result["questions"] = list(dict.fromkeys(result["questions"] + routes["questions"]))
                    actionable = [i for i in issues if i["route"] != "patient" or i["check_id"] in routes["corrections"]]
                    if not actionable:
                        result["stop_reason"] = "patient_information_required"
                        break
                    if compact and completed_rounds and not (
                        any(i["severity"] in {"major", "critical"} for i in actionable)
                        or (postdraft_web and routes["steps"])
                    ):
                        # Keep the complete reviewed answer and minor findings.
                        # Another rewrite is reserved for material errors or
                        # the existing live/replay evidence-supplement loop.
                        result["stop_reason"] = "minor_issues_after_revision"
                        break
                    old_ids = set(evidence)
                    supplemental = {}
                    if routes["steps"]:
                        supplemental = local_batch(engine, [professional_query(scope.devices, s, scope.population) for s in routes["steps"]],
                                                   scope.devices, scope.population, budget, log, original_question=query)
                        evidence.update(supplemental)
                    unseen = set().union(*(set(supplemental) - runner.context_ids.get(s, set()) for s in routes["steps"])) if routes["steps"] else set()
                    new_ids = (set(evidence) - old_ids) | unseen
                    # A persistent knowledge/conflict finding has already been
                    # reviewed after one local supplement. Rotating a larger
                    # local result set through the bounded context must not
                    # postpone external retrieval until the revision limit.
                    web_steps = [s for s in routes["steps"] if not new_ids or s in local_attempted_steps]
                    if gap_web_search:
                        # The reviewer has already inspected the initial local
                        # sources. A knowledge/conflict route may supplement
                        # them immediately, without waiting for another rewrite.
                        # Each step gets one bounded search session; that session
                        # can refine its query within the shared request budget.
                        web_steps = [s for s in routes["steps"] if s not in gap_web_attempted_steps]
                    if postdraft_web:
                        for step in web_steps[:2]:
                            if gap_web_search and budget.available("search_requests") == 0:
                                log.write("reflection_search_skipped", {"step": step, "reason": "shared_search_budget_exhausted"})
                                break
                            external_query = professional_query(scope.devices, step, scope.population)
                            before_ids = set(evidence)
                            gap_web_attempted_steps.add(step)
                            log.write("web_trigger", {
                                "round": round_no,
                                "step": step,
                                "reason": "reflection_evidence_gap" if gap_web_search else "local_evidence_insufficient_or_conflicting",
                                "local_supplement_previously_tried": step in local_attempted_steps,
                                "route": "conflict" if any(i["route"] == "conflict" and
                                                            i["target_id"].startswith(step + ".")
                                                            for i in actionable) else "knowledge",
                                "issue_ids": [i["check_id"] for i in actionable
                                              if i["target_id"].startswith(step + ".")],
                                "query": external_query,
                                "local_evidence_ids": sorted(before_ids),
                                "mode": web_mode,
                            })
                            search_context = {
                                "question": query, "step": step, "devices": scope.devices, "population": scope.population,
                                "gaps": [{key: i.get(key, "") for key in
                                          (("target_id", "issue_type", "route", "assessment", "needed")
                                           + (("reason",) if gap_web_search else ()))}
                                         for i in actionable if i["target_id"].startswith(step + ".")
                                         and i["route"] in {"knowledge", "conflict"}],
                            }
                            if gap_web_search:
                                gap_claims = steps[step]["claims"]
                                cited_ids = {eid for c in gap_claims for eid in c["evidence_ids"]}
                                search_context.update(phase="reflection_gap", draft_claims=deepcopy(gap_claims),
                                    local_sources=evidence_context(evidence[eid] for eid in sorted(cited_ids)))
                            external = web.search(external_query, context=search_context)
                            evidence.update(external)
                            if gap_web_search:
                                # Uncited new pages must reach the follow-up
                                # audit too; discovery itself is not support.
                                verification_ids.update(external)
                            log.write("web_result", {
                                "round": round_no,
                                "step": step,
                                "query": external_query,
                                "admitted_evidence_ids": sorted(external),
                                "new_evidence_ids": sorted(set(evidence) - before_ids),
                                "changed_candidate_evidence": bool(set(evidence) - before_ids),
                            })
                    local_attempted_steps.update(routes["steps"])
                    new_ids = (set(evidence) - old_ids) | unseen
                    if verification_ids:
                        affected_steps = affected_content(steps, actionable)
                        new_ids.update(set().union(*(verification_ids - runner.context_ids.get(s, set())
                                                    for s in affected_steps)))
                    if not new_ids and not routes["corrections"]:
                        result["stop_reason"] = "no_new_usable_evidence"
                        break
                    evidence = annotated_sources(evidence)
                    log.write(f"evidence_{round_no}", list(evidence.values()))
                    candidate, changes = deepcopy(steps), []
                    affected = affected_content(steps, actionable, propagate=not protected_revision)
                    fact_issues = fact_extraction_issues(actionable)
                    if fact_issues:
                        raw = runner.call(FACT_PROMPT, {"case": query, "candidate_clauses": candidate_clauses(query), "previous": facts,
                                          "issues": revision_input_issues(fact_issues, research_mode=research_mode)},
                                          extraction_schema(query), "repair_case_facts", max_tokens=1400)
                        repaired = preserve_missing_information(
                            facts, extract_located(query, raw),
                            replace=any(i["target_id"] == "case.missing_information" for i in fact_issues))
                        stable = {(f["start"], f["end"]): f["fact_id"] for f in facts["facts"]}
                        next_id = max([int(f["fact_id"][1:]) for f in facts["facts"]] or [0]) + 1
                        translation = {}
                        for f in repaired["facts"]:
                            old, fid = f["fact_id"], stable.get((f["start"], f["end"]))
                            if fid is None:
                                fid, next_id = f"F{next_id:03}", next_id + 1
                            f["fact_id"], translation[old] = fid, fid
                        for conflict in repaired["case_conflicts"]:
                            conflict["fact_ids"] = [translation[fid] for fid in conflict["fact_ids"]]
                        facts = repaired
                        affected = {s: {"affected_claim_ids": [c["claim_id"] for c in candidate[s]["claims"]],
                                        "affected_requirements": list(STEPS[s][1])} for s in STEPS}
                        log.write(f"facts_{round_no}", facts)
                    external_ids = {eid for eid in new_ids if evidence[eid]["origin"] == "external"}
                    # Newly admitted external passages must reach the bounded
                    # revision context; unchanged citations are retained by
                    # generate_step. The model still decides whether to use them.
                    ordered = {eid: evidence[eid] for eid in sorted(
                        evidence, key=lambda x: (x not in external_ids, x not in new_ids))}
                    revisions = {}
                    for step, scope_change in affected.items():
                        step_issues = [i for i in actionable if step in affected_content(steps, [i])
                                       or i["target_id"].startswith("F")
                                       or i["target_id"] == "case.missing_information"]
                        step_issues = revision_input_issues(step_issues, research_mode=research_mode)
                        revisions[step] = {"previous_step": candidate[step], "issues": step_issues, **scope_change}
                    grouped = generate_research_steps(runner, query, facts, ordered, candidate,
                                                       revisions=revisions, lean=lean,
                                                       verification_ids=verification_ids,
                                                       compact_revision_provenance=engine.config.get(
                                                           "reflection", "compact_revision_provenance", default=False)
                                                       ) if compact else None
                    task = audit_cache.get("research_task_scope", {}).get("value", {})
                    if lean and external_ids and task.get("literature_requested") and web_strategy == "on_demand":
                        assigned = task["assigned_steps"]
                        for step in routes["steps"]:
                            if step in affected and "evidence" in STEPS[step][1] and (
                                not assigned or step in assigned or "unspecified" in assigned
                            ):
                                grouped[step] = complete_evidence_task(runner, task["requirement_quote"] or query,
                                    {eid: ordered[eid] for eid in ordered if eid in external_ids}, grouped, step)
                                log.write("research_evidence_completion", {
                                    "round": round_no, "step": step, "new_evidence_ids": sorted(external_ids),
                                    "updated_step": grouped[step]})
                    protection = []
                    scope_review_attempts = budget.counts["model_calls"]
                    if reviewed_revision_scope:
                        from .revision_scope_review import merge_with_scope_review
                        proposed = {step: grouped[step] if compact else generate_step(
                            runner, query, facts, ordered, candidate, step, revision=revisions[step],
                            research_mode=research_mode, verification_ids=verification_ids) for step in affected}
                        # Non-compact generation belongs to the existing revision,
                        # not the additional change-review call count.
                        scope_review_attempts = budget.counts["model_calls"]
                        candidate, affected, protection = merge_with_scope_review(
                            runner, query, facts, steps, proposed, affected, evidence)
                    scope_review_attempts = budget.counts["model_calls"] - scope_review_attempts
                    for step, scope_change in (() if reviewed_revision_scope else affected.items()):
                        updated = grouped[step] if compact else generate_step(
                            runner, query, facts, ordered, candidate, step, revision=revisions[step],
                            research_mode=research_mode, verification_ids=verification_ids)
                        if protected_revision:
                            from .six_step_answer import merge_scoped_revision
                            updated, rejected = merge_scoped_revision(candidate[step], updated,
                                scope_change["affected_claim_ids"], scope_change["affected_requirements"])
                            protection.extend({"step": step, **row} for row in rejected)
                        candidate[step] = updated
                    if protected_revision:
                        from .six_step_answer import preserve_revision_dependencies
                        candidate, rejected = preserve_revision_dependencies(steps, candidate)
                        protection.extend(rejected)
                        log.write("revision_protection", {"round": round_no, "scope": affected,
                                  "rejected_edits": protection, "extra_model_calls": scope_review_attempts,
                                  "scope_review_enabled": bool(reviewed_revision_scope),
                                  "scope_review_model_attempts": scope_review_attempts})
                    candidate = repaired_scale_examples(candidate, f"revision_{round_no}")
                    for step, scope_change in affected.items():
                        changes.extend({"step": step, **c} for c in change_set(steps[step], candidate[step],
                                       scope_change["affected_claim_ids"], scope_change["affected_requirements"],
                                       strict=not research_mode or protected_revision))
                    if not changes:
                        result["stop_reason"] = "no_revision_change"
                        break
                    validate_steps(candidate, facts, evidence, research_mode=research_mode)
                    log.write(f"changes_{round_no}", changes)
                    log.write(f"candidate_{round_no}", candidate)
                    candidate_audits = run_checks(runner, query, facts, candidate, evidence, research_mode=research_mode,
                                                  compact=compact, cache=audit_cache, scoped=lean, consolidated=consolidated,
                                                  verification_ids=verification_ids, program_checks=program_checks,
                                                  shared_web_sources=shared_web_sources, case_scale_checks=case_scale_checks,
                                                  clause_scope_checks=clause_scope_checks)
                    log.write(f"audits_{round_no}", candidate_audits)
                    old_keys = {i["check_id"] for i in issues}
                    new_issues = audit_issues(candidate_audits)
                    result["resolved_issues"] += sorted(old_keys - {i["check_id"] for i in new_issues})
                    steps, audits, issues = candidate, candidate_audits, new_issues
                    if research_mode:
                        research_fallback = (deepcopy(steps), deepcopy(facts), deepcopy(audits))
                    completed_rounds = round_no
                    if safe_audited_version(steps, audits):
                        last_verified = (deepcopy(steps), deepcopy(facts), deepcopy(audits))
                if research_mode and steps:
                    # Research output remains available after review issues; the
                    # issues are retained as annotations instead of deleting the
                    # model's exploratory answer.
                    last_verified = (deepcopy(steps), deepcopy(facts), deepcopy(audits))
                elif not last_verified or not safe_audited_version(steps, audits):
                    candidate, public_facts, removed = publication_candidate(steps, facts, audits)
                    log.write("publication_pruning", {"removed_ids": removed, "steps": candidate, "facts": public_facts})
                    validate_steps(candidate, public_facts, evidence)
                    final_audits = run_checks(runner, query, public_facts, candidate, evidence,
                                              verification_ids=verification_ids)
                    log.write("publication_audits", final_audits)
                    if safe_audited_version(candidate, final_audits):
                        last_verified = (candidate, public_facts, final_audits)
                    else:
                        result["stop_reason"] = "final_publication_checks_failed"
                result["stop_reason"] = result["stop_reason"] or ("revision_limit" if issues else "checks_complete")
    except Exception as exc:
        category = ("budget" if isinstance(exc, BudgetExceeded) else "model" if isinstance(exc, ModelUnavailable)
                    else "format" if isinstance(exc, InvalidModelOutput) else "network" if isinstance(exc, ExternalFailure)
                    else "index" if isinstance(exc, (RuntimeError, OSError, ValueError)) else "internal")
        error = {"category": category, "type": type(exc).__name__, "message": str(exc)}
        result["errors"].append(error)
        result["stop_reason"] = category + "_failure"
        log.write("failure", error)
        if research_mode and research_fallback:
            # Keep the completed answer and its matching review. An interrupted
            # revision must not overwrite it with a partly generated candidate.
            last_verified = research_fallback
    if last_verified:
        verified_steps, verified_facts, verified_audits = last_verified
        try:
            used = {eid for c in all_claims(verified_steps).values() for eid in c["evidence_ids"]}
            # Research mode is exploratory: do not turn provenance hash
            # checks into an answer gate. Evidence is already captured in the
            # run log; strict source validation remains on the normal RAG path.
            if not research_mode:
                verify_evidence(engine, {eid: evidence[eid] for eid in used})
            result["last_verified_fallback"] = bool(result["errors"]) or verified_steps != steps
            if research_mode:
                verified_steps = _annotate_research_claims(verified_steps, verified_audits)
            result.update(six_steps=verified_steps, facts=verified_facts)
            result["citations"] = [evidence[eid] for eid in sorted(used)]
            rows = [row for step in verified_steps.values() for row in step["coverage"].values()]
            complete = (all(row["status"] in {"supported", "not_applicable"} for row in rows)
                        and not audit_issues(verified_audits) and not result["errors"])
            conflict = any(i["issue_type"] == "source_conflict" for i in issues) or bool(verified_facts["case_conflicts"])
            result["status"] = "conflict" if conflict else "complete" if complete else "partial" if all_claims(verified_steps) else "insufficient"
            result["audit_summary"] = {"check_count": len(verified_audits), "passed": sum(i["verdict"] == "pass" for i in verified_audits),
                                       "issues": audit_issues(verified_audits), "three_checks_completed": bool(verified_audits),
                                       "model_check_count": sum(not i.get("origin", "").startswith("deterministic_") for i in verified_audits),
                                       "program_check_count": sum(i.get("origin", "").startswith("deterministic_") for i in verified_audits),
                                       "targets_by_check": {kind: sum(i["check_kind"] == kind for i in verified_audits)
                                                            for kind in ("case", "evidence", "safety")}}
        except (ValueError, OSError, RuntimeError, KeyError) as exc:
            result["errors"].append({"category": "index", "type": type(exc).__name__, "message": str(exc)})
            result.update(status="technical_failure", stop_reason="source_changed_before_publication",
                          six_steps=empty_steps("来源一致性校验失败"), citations=[])
    elif not scope.reason and reflection:
        result["status"] = "technical_failure" if result["errors"] else "conflict" if any(i["issue_type"] == "source_conflict" for i in issues) else "insufficient"
    result["unresolved_issues"] = issues
    if web:
        result["web"]["events"] = web.events
        for event in web.events:
            if event["status"] in {"technical_failure", "replay_miss"}:
                result["errors"].append({"category": event.get("category", "network"), "type": event["status"], "message": event.get("error", event["status"])})
    if result["errors"] and not all_claims(result["six_steps"]) and result["status"] != "conflict":
        result["status"] = "technical_failure"
    elif result["errors"] and result["status"] == "complete":
        result["status"] = "partial"
    if general_web:
        result["web"]["verification_evidence_ids"] = sorted(verification_ids)
        result["web"]["used_in_final"] = sorted(verification_ids & {s["evidence_id"] for s in result["citations"]})
        result["web"]["answer_text_changed"] = draft_text != {cid: c["text"] for cid, c in all_claims(result["six_steps"]).items()}
    result["coverage"] = {f"{s}.{k}": row["status"] for s, value in result["six_steps"].items() for k, row in value["coverage"].items()}
    result["timing"] = {"seconds": round(time.monotonic() - budget.started, 3), **budget.counts, "revision_rounds": completed_rounds,
                        "model_stage_seconds": {name: round(seconds, 3) for name, seconds in budget.stage_seconds.items()}}
    result["answer"] = render_answer(result)
    log.write("final", result)
    return result
