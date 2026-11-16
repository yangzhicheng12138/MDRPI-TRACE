"""Six-step formative practice, independent of the legacy short-question flow.

The requirements below describe educational tasks. They are project-authored
and pending clinical review; they are not expert answer keys for the 30 cases.
Clinical meaning is assessed by the model against learner quotations. This
module controls progression, quotation provenance and observable help use.
"""
from copy import deepcopy
from datetime import datetime, timezone
import re

from .language import instruction, localized
from .learner_evidence import checked_quotes, quote_catalog, resolve_evidence, enforce_status_contract
from .rubric import ground_review
from .tasks import check_cancelled


VERSION = "guided_v1"
LEGACY_SCORING_VERSION = "project_formative_v1"
SCORING_VERSION = "project_formative_v2_20261012"
RUBRIC_ORIGIN = "project_formative_pending_clinical_review"
STEP_IDS = ("recognize", "analyze", "prioritize", "solutions", "action", "evaluate")
STEP_TITLES = {
    "recognize": ("识别线索", "Recognize cues"),
    "analyze": ("分析线索", "Analyze cues"),
    "prioritize": ("确定优先问题", "Prioritize problems"),
    "solutions": ("制定解决方案", "Generate solutions"),
    "action": ("采取行动", "Take action"),
    "evaluate": ("评价结果", "Evaluate outcomes"),
}

# Each tuple contains a task and two task requirements, in both languages.
TASKS = {
    "recognize": (
        ("请从病例中找出与器械接触和组织情况有关的关键线索；区分已观察到的事实与仍需评估的信息。无需先提出完整护理方案。",
         "Identify the case findings relevant to device contact and tissue condition. Distinguish observed facts from information that still needs assessment; a full care plan is not needed yet."),
        ("准确提取病例中与当前问题有关的具体观察或器械使用情况。",
         "Identify specific supplied observations or device-use facts relevant to the problem."),
        ("基于至少一条已有线索指出一项需要关注或继续评估的内容；题面未知信息仍保持未知。",
         "Use at least one supplied cue to identify one relevant assessment target; keep unavailable case information unknown.")),
    "analyze": (
        ("选取刚才识别的一条关键线索，说明它与可能的组织问题或器械因素有什么联系，以及判断依据；尚不能确定的内容请说明。",
         "Connect a key cue to a possible tissue problem or device factor and explain your reasoning. State any uncertainty in that interpretation."),
        ("将一个具体病例线索与合理的临床解释联系起来，并说明依据。",
         "Connect a specific case cue to a reasonable clinical interpretation and explain the basis."),
        ("区分已支持的判断与仍待确认的推测，不把未知结果或风险当成已经发生。",
         "Distinguish supported interpretation from unconfirmed inference; do not present an unknown outcome or a risk as an established event.")),
    "prioritize": (
        ("现在最应优先处理或确认哪个问题？结合病例线索说明理由，并考虑相关治疗或器械使用的安全。",
         "Which problem should be addressed or clarified first? Explain the priority using case findings and the safety of the associated treatment or device."),
        ("提出与病例实际情况相符的优先问题。",
         "Identify a priority problem consistent with the supplied case."),
        ("根据现有表现和处理的安全性说明优先理由，而非只罗列风险名称。",
         "Explain the priority using existing findings and safe management rather than listing risk labels alone.")),
    "solutions": (
        ("针对优先问题提出一个或几个护理措施，并说明每项措施要改善什么；需要哪些条件或进一步确认？",
         "Propose care measures for the priority problem and explain what each should improve. Identify any conditions or checks needed before using them."),
        ("提出与已识别问题相对应、在病例情境中合理的措施及目的。",
         "Propose measures and intended benefits that address the identified problem in this case."),
        ("说明措施实施的相关条件或安全考虑，不补造固定频率、剂量或未提供的禁忌。",
         "Describe relevant conditions or safety considerations without inventing fixed intervals, doses or unspecified contraindications.")),
    "action": (
        ("把方案写成可执行的处理步骤：先做什么、如何实施、需要谁配合，以及过程中监测什么。请保持必要治疗和器械使用的安全。",
         "Turn the plan into an actionable sequence: what to do first, how to carry it out, who should assist when needed, and what to monitor while maintaining safe treatment and device use."),
        ("描述一个能够执行的操作或处理顺序，与所提方案一致。",
         "Describe an actionable intervention or sequence consistent with the proposed plan."),
        ("说明相关的安全监测或必要协作，不预设措施已经执行或患者已经改善。",
         "Describe relevant safety monitoring or coordination without claiming that the intervention or improvement has already occurred.")),
    "evaluate": (
        ("实施后如何判断效果？说明要比较的具体观察指标，以及未改善或恶化时如何重新评估和调整。病例未给出的结果请保留为未知。",
         "How would you evaluate the intervention? Specify observations for comparison and what you would reassess or change if the problem persists or worsens. Keep outcomes absent from the case unknown."),
        ("提出与最初问题及措施对应的具体效果观察或前后比较指标。",
         "Identify specific outcome observations or before-and-after comparisons linked to the original problem and intervention."),
        ("说明未改善或恶化时的再评估与调整思路，不宣称题目未提供的治疗结果。",
         "Explain reassessment or adjustment if improvement is absent or deterioration occurs, without inventing an outcome.")),
}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _step(flow):
    if flow.get("version") != VERSION or len(flow.get("steps", [])) != 6:
        raise ValueError("六步训练状态无效。")
    index = flow.get("current_step")
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < 6:
        raise ValueError("当前步骤无效。")
    return flow["steps"][index]


def start_flow(case, item=None, language="zh"):
    if not isinstance(case, str) or not case.strip():
        raise ValueError("请先提供病例。")
    reviewed = deepcopy((item or {}).get("six_step_rubric")) if (item or {}).get("review_status") == "approved_ai" else None
    return {"version": VERSION, "current_step": 0, "status": "in_progress",
            "rubric_origin": RUBRIC_ORIGIN, "scoring_version": SCORING_VERSION,
            **({"rubric_snapshot": reviewed, "rubric_origin": reviewed["origin"], "scoring_version": reviewed["version"],
                "content_review_fingerprint": item["content_review"]["review_fingerprint"]} if reviewed else {}),
            **({"task_rubric_snapshot": {"version": SCORING_VERSION,
                 "steps": {key: _project_rubric(key, updated=True) for key in STEP_IDS}}} if not reviewed else {}),
            "case_id": (item or {}).get("case_id") or (item or {}).get("id", ""),
            "case_text": case, "language": language,
            "steps": [{"id": key, "status": "active" if index == 0 else "pending",
                       "answers": [], "reviews": [], "criterion_checks": [],
                       "hints": [], "completed_at": ""}
                      for index, key in enumerate(STEP_IDS)]}


def current_question(flow, language="zh"):
    step = _step(flow)
    reviewed = next((row for row in flow.get("rubric_snapshot", {}).get("steps", []) if row["id"] == step["id"]), None)
    task_snapshot = flow.get("task_rubric_snapshot", {}).get("steps", {}).get(step["id"], {})
    question = (reviewed["question"][language] if reviewed else
                task_snapshot["task_question"][language] if task_snapshot.get("task_question") else
                localized(language, *TASKS[step["id"]][0]))
    return localized(language, f"第 {flow['current_step'] + 1}/6 步 · {STEP_TITLES[step['id']][0]}",
                     f"Step {flow['current_step'] + 1}/6 · {STEP_TITLES[step['id']][1]}") + "\n\n" + question


def current_rubric(flow, language="zh"):
    """A bilingual project task rubric; no case-specific gold answer is invented."""
    step = _step(flow)
    reviewed = next((row for row in flow.get("rubric_snapshot", {}).get("steps", []) if row["id"] == step["id"]), None)
    if reviewed:
        return {"version": flow["scoring_version"] + ":" + step["id"], "rubric_origin": flow["rubric_origin"],
                "review_status": "approved_ai", "basis_question": reviewed["question"]["en"],
                "basis_answer": (reviewed["reference_answer"].get(language)
                                 or reviewed["reference_answer"].get("zh")
                                 or reviewed["reference_answer"].get("en", "")),
                "required": deepcopy(reviewed["required"]),
                "optional_extensions": {"zh": "只评价当前步骤的两项要求，参考答案的其他细节为可选说明。", "en": "Assess the two requirements of this step only; other reference details are optional elaboration."}}
    snapshot = flow.get("task_rubric_snapshot", {}).get("steps", {}).get(step["id"])
    if snapshot:
        return deepcopy(snapshot)
    # A saved pre-snapshot project flow keeps the contract used when it began.
    # New equivalent-expression guidance is snapshotted only for new flows.
    return _project_rubric(step["id"], updated=False)


def _project_rubric(step_id, *, updated):
    rows = []
    for index, texts in enumerate(TASKS[step_id][1:], 1):
        acceptable = {"zh": "接受符合题目事实和要求的等效表达。学生可用自然语言作答，无需固定格式；此前当前步骤的正确内容继续有效。",
                      "en": "Accept equivalent meaning consistent with the supplied facts and task. No fixed format is required; correct content from earlier answers to this step remains available."}
        if step_id == "recognize" and index == 2:
            acceptable = {"zh":"根据一条线索指出一项需关注或进一步评估的内容，并明确未记录信息仍未知，即覆盖本项。已指名的未知变量与已有器械或组织线索形成关注目标也可接受，不要求MDRPI术语。例如管路牵拉→需评估固定或受力变化。无需详细说明压力分布、暴露时长、病理机制、护理措施或所有评估项目；不得把后续分析步骤的要求加到本项。",
                          "en":"One sensible assessment target linked to a supplied cue, with unrecorded information kept unknown, satisfies this criterion. A named unknown variable linked to device or tissue cues can express that target; the term MDRPI is not required. For example, tubing traction calls for checking fixation or changes in force. Do not require detailed pressure distribution, exposure duration, mechanisms, interventions or an exhaustive assessment; those exceed this cue-recognition task."}
            if not updated:
                acceptable = {"zh":"根据一条线索指出一项需关注或进一步评估的内容，并明确未记录信息仍未知，即覆盖本项。例如管路牵拉→需评估固定或受力变化。无需详细说明压力分布、暴露时长、病理机制、护理措施或所有评估项目；不得把后续分析步骤的要求加到本项。",
                              "en":"One sensible assessment target linked to a supplied cue, with unrecorded information kept unknown, satisfies this criterion. For example, tubing traction calls for checking fixation or changes in force. Do not require detailed pressure distribution, exposure duration, mechanisms, interventions or an exhaustive assessment; those exceed this cue-recognition task."}
        elif updated and step_id == "solutions" and index == 2:
            acceptable = {"zh": "一项与当前措施关联的安全考虑或实施前核查即可，例如核对必要支持、适配或许可/团队复核。无需指定产品、固定时间或唯一警句；不能仅重复措施而没有安全考虑。",
                          "en": "One safety consideration or pre-implementation check linked to the current intervention suffices, such as necessary support, fit, authorization or team review. No product, fixed interval or exact warning is required; an intervention without a safety consideration is insufficient."}
        elif updated and step_id == "action" and index == 2:
            acceptable = {"zh": "指出与本措施有关的具体监测目标，或明确协作/升级边界中的一项即可，例如向负责团队报告或按应急流程升级。无需指名团队成员或同时列全监测与协作；不能只写无监测或协作含义的操作措施。",
                          "en": "A relevant specific monitoring target OR an explicit collaboration/escalation boundary suffices, such as notifying the responsible team or escalating under an emergency pathway. Naming a team member or listing both categories is not required; an action with neither monitoring nor collaboration is insufficient."}
        rows.append({"id": f"{step_id}_R{index}",
                     "criterion": dict(zip(("zh", "en"), texts)),
                     "acceptable": acceptable,
                     "incorrect": {"zh": "仅在学生实际陈述与病例事实、该任务或安全处理要求冲突时判为错误；遗漏或未知信息不等于错误。",
                                   "en": "Mark incorrect only for an actual learner statement conflicting with case facts, the task or safe management. Omitted or unavailable information is not itself an error."},
                     "follow_up": dict(zip(("zh", "en"), texts))})
    return {"version": (SCORING_VERSION if updated else LEGACY_SCORING_VERSION) + ":" + step_id, "rubric_origin": RUBRIC_ORIGIN,
            "review_status": "pending_clinical_review", "basis_question": TASKS[step_id][0][1],
            "task_question": dict(zip(("zh", "en"), TASKS[step_id][0])),
            "basis_answer": "", "required": rows,
            "optional_extensions": {"zh": "其他步骤的完整答案、题目未提供的数值和项目未确认的专家答案不是当前步骤必答内容。",
                                    "en": "Complete answers for other steps, unavailable measurements and unconfirmed expert answer keys are outside this step's requirements."}}


def append_answer(flow, text):
    result = deepcopy(flow)
    step = _step(result)
    if result["status"] != "in_progress" or step["status"] != "active":
        raise ValueError("请进入当前待作答步骤后再提交答案。")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("请输入作答内容。")
    step["answers"].append(text.strip())
    return result


def assess_step(backend, state, log):
    """Initial assessment plus at most two deadline-bound scoped checks.

    Evidence attempt numbers are local to this step, never global session turns.
    The result does not update a checkpoint or advance the flow.
    """
    from app.evidence_snapshot import ModelRunner, RunBudget
    from .backend import source_context
    from . import prompts
    from .question_bank import rubric_context

    flow = state["clinical_flow"]
    check_cancelled()
    step = _step(flow)
    if not step["answers"]:
        raise ValueError("当前步骤尚无学生回答，不执行评价。")
    language = state.get("language", flow.get("language", "zh"))
    rubric = current_rubric(flow, language)
    catalog = quote_catalog(step["answers"])
    teaching_events = sorted(step["hints"] + step.get("reference_views", []) + step.get("teaching_dialogue_events", []),
                             key=lambda event: event.get("at", ""))
    quote_ids = [row["id"] for row in catalog]
    schema = prompts.rubric_assessment_schema(rubric, len(step["answers"]), quote_ids)
    # An explicit semantic decision can replace a previously covered point.
    # A missing-only result is not enough to erase its retained learner evidence.
    for variant in schema["properties"]["criteria"]["properties"].values():
        for choice in variant["anyOf"]:
            choice["properties"]["supersedes_previous"] = {"type": "boolean"}
            if "supersedes_previous" not in choice["required"]:
                choice["required"].append("supersedes_previous")
    payload = {"case": state.get("case", flow["case_text"]), "current_task": current_question(flow, language),
               "current_step": step["id"], "output_language": language,
               "latest_answer": step["answers"][-1], "earlier_answers": step["answers"][:-1],
               "student_answers": [{"attempt": index, "text": text} for index, text in enumerate(step["answers"], 1)],
               "quote_catalog": catalog, "grading_rubric": rubric_context(rubric, language),
               "rubric_origin": flow.get("rubric_origin", RUBRIC_ORIGIN),
               "previous_criterion_checks": [{key: row[key] for key in ("id", "status", "evidence")}
                                              for row in step["criterion_checks"]],
               "tutor_context": [{"role": "assistant", "content": row["text"]} for row in teaching_events],
               "sources": source_context(state.get("sources", [])), "reference_answer": rubric.get("basis_answer") or None}
    prompt = instruction(language) + prompts.RUBRIC_ASSESS + "\n" + (
        "This is ONE step in formative clinical reasoning, not an expert-validated examination. "
        "Any supplied reference is AI-reviewed educational guidance, not a patient outcome or evidence of learner performance. "
        "Grade only this current task and its explicit acceptable statements. Never impose other steps. "
        "Do not mark partial or missing for an optional elaboration, a clinician's additional assessment checklist, or a fact absent from the case. "
        "Set supersedes_previous=true only when current learner wording explicitly changes, retracts or contradicts a previous supported criterion. "
        "A short supplement need not repeat earlier correct content. Cite any overriding learner wording. "
        "Do not infer an actual patient outcome, mandatory interval or clinical mastery.")
    llm = backend._llm()
    try:
        raw = ModelRunner(llm, RunBudget(600), log).call(prompt, payload, schema,
            "teaching_guided_" + step["id"], max_tokens=1500)
        raw = enforce_status_contract(raw, rubric, catalog, payload["case"], language)
        # Extra checks share one deadline, and their failure cannot discard the
        # grounded initial assessment or leave the whole turn awaiting a check.
        verification_budget = RunBudget(180)
        outstanding = [criterion for criterion in rubric["required"]
                       if raw["criteria"][criterion["id"]]["status"] != "met"]
        for criterion in sorted(outstanding,
                                key=lambda row: raw["criteria"][row["id"]]["status"] != "incorrect")[:2]:
            original = raw["criteria"][criterion["id"]]
            if original["status"] not in {"partial", "missing", "incorrect", "uncertain"}:
                continue
            isolated = {**rubric,"required":[criterion]}
            checked_payload = {"case":payload["case"],"output_language":language,
                "grading_rubric":rubric_context(isolated,language),"quote_catalog":catalog,
                "latest_answer":step["answers"][-1],"checking_explicit_correction":False,
                "current_task": payload["current_task"], "reference_answer": payload["reference_answer"],
                "student_answers": payload["student_answers"]}
            try:
                check_cancelled()
                checked = ModelRunner(llm,verification_budget,log).call(
                    instruction(language)+prompts.CRITERION_VERIFY+"\nCheck only this exact requirement and its explicit acceptable statements. "
                    "Do not demand other cues, an exhaustive assessment or later-step details. The former verdict is deliberately not supplied.",
                    checked_payload,prompts.rubric_assessment_schema(isolated,len(step["answers"]),quote_ids),
                    "guided_verify_"+criterion["id"],max_tokens=600)
            except Exception as exc:
                check_cancelled()
                event = {"criterion": criterion["id"], "error_type": type(exc).__name__,
                         "message": str(exc), "action": "retained_grounded_initial_assessment"}
                raw.setdefault("verification_unavailable", []).append(event)
                log.write("guided_verification_unavailable", event)
                break
            check_cancelled()
            checked = enforce_status_contract(checked, isolated, catalog, payload["case"], language)
            raw.setdefault("status_contract_warnings", []).extend(checked.get("status_contract_warnings", []))
            revised = checked["criteria"][criterion["id"]]
            log.write("guided_criterion_verification",{"id":criterion["id"],"initial":original,"verified":revised})
            raw["criteria"][criterion["id"]] = {**revised,"supersedes_previous":original.get("supersedes_previous",False)}
        check_cancelled()
        result = resolve_evidence(raw, catalog)
        log.write("guided_step_assessment_raw", result)
        return result
    finally:
        llm.close()


def apply_review(flow, raw, language="zh"):
    """Return a new flow with current-step review, without auto-advancing.

    Missing-only regressions retain checked earlier evidence. A supported new
    incorrect judgment or explicit supersession can revoke a covered criterion.
    """
    result = deepcopy(flow)
    step = _step(result)
    if result["status"] != "in_progress" or step["status"] not in {"active", "awaiting_next"}:
        raise ValueError("当前步骤不能重新评价。")
    if not step["answers"]:
        raise ValueError("当前步骤尚无学生回答。")
    rubric = current_rubric(result, language)
    source = resolve_evidence(raw, quote_catalog(step["answers"])) if any(
        "quote_id" in evidence for row in raw.get("criteria", {}).values() for evidence in row.get("evidence", [])) else deepcopy(raw)
    source.setdefault("reflection_explained", False)
    for row in rubric["required"]:
        source.setdefault("criteria", {}).setdefault(row["id"], {
            "status": "uncertain", "evidence": [], "explanation": localized(language, "本项评价未返回，请重新核对。", "This criterion was not returned; request a recheck.")})
    source = enforce_status_contract(source, rubric, quote_catalog(step["answers"]), result["case_text"], language)
    before = {row["id"]: row for row in step["criterion_checks"]}
    for key, row in source["criteria"].items():
        previous = before.get(key)
        if not previous or previous["status"] not in {"met", "incorrect"} or row["status"] not in {"missing", "partial", "uncertain"} or row.get("supersedes_previous"):
            continue
        retained, rejected = checked_quotes(previous["evidence"], step["answers"])
        if retained and not rejected:
            row.update(status=previous["status"], evidence=retained,
                       explanation=(localized(language, "保留此前当前步骤已覆盖且有原话支持的要点，无需重复作答。", "The earlier quote-backed point for this step remains covered; repetition is unnecessary.") if previous["status"] == "met" else
                                    localized(language, "此前有原话支持的错误尚未看到明确纠正，不能因本次未重复提及而消除。", "The previously quote-backed error has not been explicitly corrected; omission in a supplement does not erase it.")),
                       decision_origin="retained_step_evidence")
            for field in ("decision_basis", "error_quote", "conflict_rule_quote"):
                if field in previous:
                    row[field] = previous[field]
    # Retained old errors need the same actual-word/rule provenance as a new
    # error. Earlier review history stays intact if that basis is unavailable.
    source = enforce_status_contract(source, rubric, quote_catalog(step["answers"]), result["case_text"], language)
    review = ground_review(source, rubric, step["answers"], step["reviews"], language)
    review.update(step_id=step["id"], rubric_origin=rubric["rubric_origin"], review_status=rubric["review_status"],
                  grading_mode="guided_formative", answer_count=len(step["answers"]))
    same_answers = bool(step["reviews"] and step["reviews"][-1].get("answer_count") == len(step["answers"]))
    references = step.get("reference_views", [])
    explanations = step.get("teaching_dialogue_events", [])
    teaching_events = step["hints"] + references + explanations
    for check in review["criterion_checks"]:
        previous = before.get(check["id"])
        relevant = [event for event in teaching_events if event.get("focus", "all") in {check["id"], "all", "none"}]
        assisted = any(evidence["attempt"] > event["answer_count"]
                       for evidence in check["evidence"] for event in relevant)
        if previous and previous["status"] == "met" and previous.get("attribution") == "student":
            assisted = False
        supported = bool(check["evidence"]) and check["status"] in {"met", "partial", "incorrect"}
        check["attribution"] = "teacher_help" if supported and assisted else "student" if supported else "not_demonstrated"
        after_reference = assisted and any(evidence["attempt"] > event["answer_count"]
            for evidence in check["evidence"] for event in references if event in relevant)
        after_explanation = assisted and any(evidence["attempt"] > event["answer_count"]
            for evidence in check["evidence"] for event in explanations if event in relevant)
        check["answer_context"] = ("after_reference" if supported and after_reference else "after_explanation" if supported and after_explanation
                                   else "after_hint" if supported and assisted else "without_requested_hint" if supported else "not_demonstrated")
        if same_answers:
            check["newly_demonstrated"] = False
            check["judgment_revised"] = bool(previous and previous["status"] != check["status"])
    if same_answers:
        review["improvements"] = []
    review["review_origin"] = "recheck" if same_answers else "answer"
    review["assistance"] = {"hints_seen": len(step["hints"]), "reference_answers_seen": len(references), "explanations_seen": len(explanations),
        "covered_without_requested_hint": sum(row["status"] == "met" and row["attribution"] == "student" for row in review["criterion_checks"]),
        "covered_after_hint": sum(row["status"] == "met" and row["attribution"] == "teacher_help" for row in review["criterion_checks"])}
    step["reviews"].append(review)
    step["criterion_checks"] = deepcopy(review["criterion_checks"])
    if review["ready_for_summary"] and not review["material_errors"]:
        step["status"] = "awaiting_next" if result["current_step"] < 5 else "passed"
        step["completed_at"] = step["completed_at"] or _now()
        if result["current_step"] == 5:
            result["status"] = "completed"
    else:
        step["status"], step["completed_at"] = "active", ""
    return result


def advance_step(flow, expected_step=None):
    result = deepcopy(flow)
    step = _step(result)
    if expected_step is not None and expected_step != result["current_step"]:
        raise ValueError("步骤已改变，请刷新后继续；不会重复进入下一步。")
    if result["status"] != "in_progress" or step["status"] != "awaiting_next":
        raise ValueError("当前步骤尚未通过，不能进入下一步。")
    step["status"] = "passed"
    result["current_step"] += 1
    _step(result)["status"] = "active"
    return result


def interrupt_flow(flow):
    result = deepcopy(flow)
    _step(result)
    if result["status"] != "completed":
        result["status"] = "interrupted"
    return result


def classify_intent(text):
    """Classify standalone controls; clinical mixed utterances stay answers."""
    value = (text or "").strip().strip("。.!！?？").lower()
    if not value:
        return "ambiguous"
    from .teaching import is_recheck_control
    if is_recheck_control(text):
        return "recheck"
    if re.fullmatch(r"(?:请|我想|现在|就)?(?:结束(?:本次)?(?:学习|练习|训练)?(?:并总结)?|停止(?:本次)?(?:学习|练习|训练)|总结(?:一下|一下吧)?|不练了)|(?:please )?(?:finish|end|stop)(?: (?:the |this )?(?:exercise|practice|session))?(?: and summari[sz]e)?|(?:please )?summari[sz]e", value):
        return "end"
    if re.fullmatch(r"(?:请|我想|现在)?(?:进入|继续|开始)?下一步|(?:please )?(?:next step|continue to (?:the )?next step|advance)", value):
        return "advance"
    if re.fullmatch(r"(?:(?:我|这题我|这个我)?(?:还|也|真的)?(?:不知道|不会|不懂|答不出来|不清楚)(?:怎么(?:答|回答|分析))?(?:了|啊|呢)?|i (?:do not|don't|dont) know(?: (?:how to answer|the answer))?|i(?:'m| am) (?:not sure|stuck)|no idea)(?:[，,]\s*(?:请)?(?:给(?:我)?(?:点|一些)?提示|帮(?:帮)?我|help me))?", value):
        return "unknown"
    if re.fullmatch(r"(?:请|能否|可以)?(?:给我|给|提供)?(?:一些|一点|一个|点)?提示(?:呢|吗)?|(?:请)?帮(?:帮)?我|(?:please )?(?:give me )?(?:a |another )?hint|help me", value):
        return "hint"
    if re.fullmatch(r"(?:什么意思|请(?:解释|澄清)(?:一下)?(?:题目|问题|这一步)?|这(?:一步|个问题|题)(?:是什么意思|要求什么|怎么理解))|(?:what does (?:this|the question) mean|please clarify(?: (?:the question|this step))?)", value):
        return "clarify"
    if re.fullmatch(r"(?:好的?|嗯|明白了|知道了|谢谢|是的|对|我同意(?:你说的)?)|(?:ok(?:ay)?|yes|thanks|understood|i agree|got it)", value):
        return "ambiguous"
    # Short interrogatives without an actual first-person action/interpretation
    # are clarification requests, not negative clinical claims.
    if (text or "").rstrip().endswith(("?", "？")) and re.match(r"^(?:为什么|怎么理解|这是什么意思|能否解释|what does\b|why does\b|can you explain\b|could you explain\b)", value, re.I):
        return "clarify"
    if re.match(r"^(?:补充[：:]?|另外|还有|再补充|我补充|also\b|in addition\b|to add\b)", value):
        return "partial_answer"
    if len(value) <= 3 and not re.search(r"[a-z0-9]|皮肤|红斑|疼痛|减压|面罩", value):
        return "ambiguous"
    return "answer"


HINT_DIRECTIONS = {
    "recognize": ("先把题面中可直接观察到的内容和推测分开。", "Separate directly observed facts from assumptions."),
    "analyze": ("沿着“线索 → 可能解释 → 判断依据”组织思路。", "Use a cue → possible explanation → supporting reason sequence."),
    "prioritize": ("比较问题的紧迫性、现有证据和处理的安全条件。", "Compare urgency, existing evidence and safe management conditions."),
    "solutions": ("让措施对应已识别的问题，并解释措施的目的。", "Link each intervention to an identified problem and its purpose."),
    "action": ("把建议改写为可以按顺序实施的动作。", "Turn suggestions into actions that can be carried out in sequence."),
    "evaluate": ("把效果观察对应回最初的问题，并考虑后续调整。", "Link outcome observations to the initial problem and consider further adjustment."),
}


def hint(flow, case=None, level=None, language="zh"):
    """Return {flow,event,text,exhausted}; help never becomes an answer.

    A supplied level must be the next unseen level. Three levels are a ceiling,
    not a requirement before answering. Cases are quoted, never extended.
    """
    result = deepcopy(flow)
    step = _step(result)
    if result["status"] != "in_progress" or step["status"] != "active":
        text = localized(language, "当前步骤已通过或训练已结束，无需继续提示。", "This step has passed or training has ended; no further hint is needed.")
        return {"flow": result, "event": None, "text": text, "exhausted": True}
    next_level = len(step["hints"]) + 1
    if next_level > 3:
        return {"flow": result, "event": None, "text": localized(language, "当前步骤的三级提示已给出，可继续作答或结束并总结。", "All three hints for this step have been shown. Continue answering or finish with a summary."), "exhausted": True}
    if level is None:
        level = next_level
    if isinstance(level, bool) or level != next_level:
        raise ValueError("请按方向、聚焦、部分答案顺序查看下一条提示。")
    key = step["id"]
    case_text = case or result["case_text"]
    if level == 1:
        body = localized(language, *HINT_DIRECTIONS[key])
    elif level == 2:
        # Quote a real complete case clause rather than inventing a clinical
        # finding. Prefer a tissue/device clause when the source supplies one.
        clauses = [part.strip() for part in re.split(r"(?<=[。.!?！？])\s*|\n+", case_text) if part.strip()]
        focus = next((part for part in clauses if re.search(r"鼻梁|红斑|皮肤|黏膜|疼痛|受压|固定带|牵拉|skin|redness|pain|pressure|mucosa|strap", part, re.I)), clauses[0] if clauses else case_text)
        body = localized(language, "聚焦题面中的这条原始线索：\n> ", "Focus on this original case observation:\n> ") + focus.replace("\n", "\n> ")
        body += "\n\n" + localized(language, "把它与当前步骤的要求联系起来，只使用原题给出的事实。", "Relate it to the current task using only supplied facts.")
    else:
        partial = {
            "recognize": ("一个可用的起点是准确写出器械接触部位的一项组织表现；对尚未记录的情况，用“需要进一步评估”表达。", "One useful starting point is to name a supplied tissue observation at the device contact site. Describe unrecorded findings as needing assessment."),
            "analyze": ("先建立一条解释链：器械接触或受力情况可能影响局部组织；是否已经造成损伤仍要结合题面的实际组织表现判断。", "Begin with one interpretation: device contact or force can affect local tissue; whether injury is established must be judged from the actual tissue findings."),
            "prioritize": ("优先问题应同时考虑局部组织情况和正在进行治疗的安全；不能仅因局部受压就假定可以直接停止必要治疗。", "The priority should consider both local tissue findings and safety of ongoing treatment; local pressure alone does not establish that necessary treatment can simply stop."),
            "solutions": ("一个方案方向是处理题面明确提示的器械接触或受力因素，并把拟采取的调整与要改善的局部问题对应；具体办法仍需你结合病例选择。", "One plan direction is to address a device-contact or force factor explicitly supplied in the case and link the adjustment to the local problem; choose the case-appropriate measure yourself."),
            "action": ("可先写一项动作和一项安全观察，例如在符合治疗条件的配合下检查接触部位，同时关注器械功能和患者耐受；继续补充适合本例的实施顺序。", "Start with an action and safety observation, such as inspecting a contact site with appropriate treatment support while monitoring device function and patient tolerance; add the sequence appropriate to this case."),
            "evaluate": ("一项效果观察可以来自干预前后的局部组织表现；还应核对器械治疗是否有效。题面没有干预后结果时，描述复评计划而非宣称已经改善。", "One outcome comparison can use local tissue findings before and after intervention, alongside treatment effectiveness. If no post-intervention result is supplied, describe a reassessment plan rather than claiming improvement."),
        }[key]
        body = localized(language, *partial)
    label = localized(language, ("一级 · 方向提示", "二级 · 聚焦提示", "三级 · 部分答案提示")[level - 1],
                      ("Level 1 · Direction", "Level 2 · Focus", "Level 3 · Partial answer")[level - 1])
    text = "**" + label + "**\n\n" + body
    event = {"level": level, "step_id": key, "text": text,
             "answer_count": len(step["answers"]), "at": _now(),
             "origin": "project_formative_scaffold", "language": language}
    step["hints"].append(event)
    return {"flow": result, "event": deepcopy(event), "text": text, "exhausted": False}


def feedback(flow, language="zh"):
    """Four fixed sections; no fabricated marks or competence claims."""
    step = _step(flow)
    checks = step["criterion_checks"]
    groups = (("已经达到的内容", "What you have covered", {"met"}),
              ("可能遗漏的内容", "What may be missing", {"partial", "missing"}),
              ("需要修正的内容", "What needs correction or clarification", {"incorrect", "uncertain"}))
    parts = []
    for zh, en, statuses in groups:
        rows = [row for row in checks if row["status"] in statuses]
        body = []
        for row in rows:
            text = "- " + row["criterion"][language]
            for evidence in row["evidence"]:
                text += "\n\n" + localized(language, f"当前步骤第{evidence['attempt']}次作答原话：", f"Your words in this step's answer {evidence['attempt']}: ") + "\n> " + evidence["quote"].replace("\n", "\n> ")
            if row.get("attribution") == "teacher_help" and row["status"] == "met":
                text += "\n\n" + (localized(language, "该要点在查看参考答案后补充，学习帮助单独记录。", "This point was added after viewing the reference; help use is recorded separately.")
                    if row.get("answer_context") == "after_reference" else localized(language, "该要点在教学讲解后补充；讲解不占提示次数。", "This point was added after a teaching explanation, which does not use a hint opportunity.")
                    if row.get("answer_context") == "after_explanation" else localized(language, "该要点在查看提示后补充，提示使用单独记录。", "This point was added after a hint; help use is recorded separately."))
            if row["status"] != "met":
                latest = step["reviews"][-1] if step["reviews"] else {}
                if latest.get("language") == language and row.get("explanation"):
                    text += "\n\n" + row["explanation"]
            if row.get("judgment_revised"):
                text += "\n\n" + localized(language, "重新核对改变了判定，本次没有新增学生作答。",
                                          "Rechecking changed the judgement; no new student answer was submitted.")
            body.append(text)
        empty = localized(language, "尚无当前步骤的评价。", "This step has not been assessed yet.") if not checks else localized(language, "本次未发现此类内容。", "None identified in this assessment.")
        parts.append("**" + localized(language, zh, en) + "**\n\n" + ("\n\n".join(body) or empty))
    if flow["status"] == "completed":
        next_text = localized(language, "本轮六步必答点已覆盖，可以查看总结与参考要点。", "The six formative tasks are covered. View the summary and reference points.")
    elif flow["status"] == "interrupted":
        next_text = localized(language, "本次训练已提前结束，未完成步骤保留在记录中。可查看总结后重新练习。", "Training ended early; unfinished steps remain in the record. View the summary and practise again.")
    elif step["status"] == "awaiting_next":
        next_text = localized(language, "当前步骤要求已覆盖，无需重写；请选择“进入下一步”。", "This step's requirements are covered; repetition is unnecessary. Select Next step.")
    else:
        target = next((row for row in checks if row["status"] != "met"), None)
        next_text = (target["criterion"][language] if target else current_question(flow, language)) + "\n\n" + localized(language, "可直接补充这一点，此前正确内容继续保留；需要时获取提示。", "You may supplement this point alone; earlier correct content is retained. Request a hint when needed.")
    if step["reviews"] and step["reviews"][-1].get("verification_unavailable"):
        next_text += "\n\n" + localized(language, "额外核对未完成，已保留本次初评与原话；可重新核对已有作答。",
                                       "The additional check did not finish. The initial assessment and original words are retained; you can recheck your existing answer.")
    parts.append("**" + localized(language, "继续思考", "Continue thinking") + "**\n\n" + next_text)
    return "\n\n".join(parts)


def review_summary(flow):
    """Saved observations for an archive/radar; this does not create a score."""
    _step(flow)
    steps = [{"id": step["id"], "status": step["status"],
              "answer_count": len(step["answers"]), "hint_count": len(step["hints"]),
              "hint_levels": [event["level"] for event in step["hints"]],
              "covered": sum(row["status"] == "met" for row in step["criterion_checks"]),
              "required": len(current_rubric({**flow, "current_step": index})["required"]),
              "criterion_checks": deepcopy(step["criterion_checks"]), "completed_at": step["completed_at"]}
             for index, step in enumerate(flow["steps"])]
    return {"version": VERSION, "rubric_origin": flow.get("rubric_origin", RUBRIC_ORIGIN),
            "scoring_version": flow.get("scoring_version") or (SCORING_VERSION if flow.get("task_rubric_snapshot") else LEGACY_SCORING_VERSION),
            "status": flow["status"], "completed_steps": sum(step["status"] in {"passed", "awaiting_next"} for step in flow["steps"]),
            "total_steps": 6, "hint_count": sum(step["hint_count"] for step in steps),
            "answer_count": sum(step["answer_count"] for step in steps), "steps": steps}


def learning_record(flow, language="zh"):
    """Render actual original answers independently of AI teacher narrative."""
    parts = [localized(language, "### 本轮作答记录（原话与判定分开展示）",
                       "### Your answer record (original words and judgements)")]
    from .rubric import STATUS_LABELS
    for index, step in enumerate(flow["steps"], 1):
        parts.append(f"**{index}. {localized(language, *STEP_TITLES[step['id']])}**")
        for attempt, answer in enumerate(step["answers"], 1):
            parts.append(localized(language, f"实际第{attempt}次作答：", f"Actual answer {attempt}: ")
                         + "\n> " + answer.replace("\n", "\n> "))
        if not step["answers"]:
            parts.append(localized(language, "本步骤没有学生作答。", "No student answer was submitted for this step."))
        for row in step["criterion_checks"]:
            parts.append("- " + row["criterion"][language] + "：" + localized(language, *STATUS_LABELS[row["status"]]))
        rechecks = sum(review.get("review_origin") == "recheck" for review in step["reviews"])
        if rechecks:
            parts.append(localized(language, f"同原答重新核对{rechecks}次；这不是新增作答或学习提升的证据。",
                                   f"The same answer was rechecked {rechecks} time(s); this is not a new answer or evidence of improvement."))
    return "\n\n".join(parts)
