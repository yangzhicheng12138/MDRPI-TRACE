"""Deterministic development-question blueprints, without retrieval/model calls.

The caller still verifies material approval and evidence. This planner binds one
unchanged case per item, spreads reviewed assessment foci across a small batch,
and reports shortages instead of inventing a new patient or teaching target.
"""
from copy import deepcopy
import hashlib
import json
import re

from .question_generation_schema import validate_blueprint


VERSION = "source_bound_batch_plan_v1"
STEPS = ("recognize", "analyze", "prioritize", "solutions", "action", "evaluate")
STEP_LABELS = {
    "recognize": "识别线索与评估目标", "analyze": "解释与鉴别",
    "prioritize": "优先问题与理由", "solutions": "措施与实施条件",
    "action": "行动与协作边界", "evaluate": "复评与调整",
}
_TEMPLATES = {
    "recognize": (
        "指出与本题焦点有关的已知线索，并用题面信息说明其与目标的关系。",
        "基于已给线索，指出需要继续观察或核实的内容，并说明对应的病例依据。",
    ),
    "analyze": (
        "解释与本题焦点有关的线索关系，并说明题面依据；不扩大为完整护理计划。",
        "区分本题已有资料支持的判断与仍待核实的内容，并说明依据。",
    ),
    "prioritize": (
        "指出当前最优先的问题，并结合题面线索说明理由。",
        "围绕当前最优先问题说明优先理由，保留与本例有关的安全边界。",
    ),
    "solutions": (
        "提出与本题目标对应的可执行措施，并说明目的。",
        "针对与本题目标相符的措施，说明相关实施条件或安全核查及其病例依据。",
    ),
    "action": (
        "描述与本题目标直接相关的立即行动或先后安排，并说明关键依据。",
        "指出相关监测目标、必要协作或操作权限边界中的合理内容，并说明依据；不强制三者全部出现。",
    ),
    "evaluate": (
        "提出能反映本题目标的效果观察或前后比较指标，并说明比较目的。",
        "说明提示需要再评估或升级的变化及对应调整思路，不预设已经改善。",
    ),
}


def canonical_sha256(value):
    """Shared canonical encoding for input pins and readable plan integrity."""
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def content_sha256(value):
    """Hash a plan/item excluding only its own content_sha256 field."""
    return canonical_sha256({key: item for key, item in value.items() if key != "content_sha256"})


def _terms(text):
    result = set(re.findall(r"[a-z]+", text.casefold()))
    for segment in re.findall(r"[\u4e00-\u9fff]+", text):
        result.update(segment[index:index + 2] for index in range(len(segment) - 1))
    return result


def _goal_steps(objective):
    patterns = {
        "recognize": r"识别|提取|辨认|找出|评估目标|继续观察|待核实项目|\b(?:recogniz\w*|recognis\w*|identify|identifying|extract\w*)\b",
        "analyze": r"分析|解释|鉴别|区分|推断|判断|\b(?:analy[sz]\w*|interpret\w*|differentiat\w*|explain\w*)\b",
        "prioritize": r"优先|排序|首要|\b(?:prioriti[sz]\w*|priority|rank\w*)\b",
        "solutions": r"方案|措施|干预|护理计划|解决办法|\b(?:solution\w*|intervention\w*|care plan)\b",
        "action": r"行动|立即处理|先后顺序|操作顺序|执行顺序|角色分工|协作安排|落实|执行|实施|\b(?:action\w*|implement\w*)\b",
        "evaluate": r"复评|复查|复测|再评估|评价|效果观察|疗效|\b(?:evaluat\w*|reassess\w*|outcome\w*)\b",
    }
    text = re.sub(r"(?:不要求|无需|不必|不做|不涉及|不考查|不要)[^，。；;,.]*", "", objective)
    text = re.sub(r"(?:而)?不(?:分析|解释|鉴别|推断|排序|生成方案|采取行动|复评)[^，。；;,.]*", "", text)
    text = re.sub(r"可执行|可实施|实施条件|实施前核查|实施限制", "", text, flags=re.I)
    matches = {step: re.search(pattern, text, re.I) for step, pattern in patterns.items()}
    if matches["prioritize"]:
        if not re.search(r"线索|因素|观察|资料|信息|cues?|clues?|findings?|factors?|observations?", text, re.I):
            matches["recognize"] = None
        if not re.search(r"分析|鉴别|区分|推断|病因|机制|关联|关系|analy[sz]|differentiat|interpret", text, re.I):
            matches["analyze"] = None
    return [step for step, found in sorted(matches.items(), key=lambda row: row[1].start() if row[1] else len(text) + 1) if found]


def _minimum_targets(blueprint):
    # A number of assessment/solution targets is different from the number of
    # scoring dimensions. Default to one; do not manufacture a two-target task.
    match = re.search(r"(?:至少|提出|给出|列出|识别|说明|选择).{0,4}?([一二两三四五1-5])(?:个|项|条|种)",
                      blueprint["learning_objective"] + " " + blueprint["requirements"])
    if not match:
        return 1
    value = match[1]
    return int(value) if value.isdigit() else {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5}[value]


def _material_inputs(blueprint, materials):
    if not isinstance(materials, list) or not all(isinstance(row, dict) for row in materials):
        raise ValueError("A reviewed development material list is required")
    if any(not isinstance(row.get("id"), str) or not row["id"] for row in materials):
        raise ValueError("Every material must have a stable ID")
    by_id = {row.get("id"): row for row in materials}
    if len(by_id) != len(materials) or set(by_id) != set(blueprint["material_ids"]):
        raise ValueError("The supplied materials must match the selected IDs exactly")
    ordered = [deepcopy(by_id[identity]) for identity in blueprint["material_ids"]]
    for row in ordered:
        if (row.get("kind") == "standardized_case" or row.get("usage") != "development_only"
                or row.get("review_status") not in {"approved_ai", "approved_human", "approved"}):
            raise ValueError("Only reviewed development materials can enter a batch plan")
        if not isinstance(row.get("case_facts"), str) or not row["case_facts"].strip():
            raise ValueError("Selected material must contain an immutable case body")
        if blueprint["population"] not in row.get("populations", [row.get("population")]):
            raise ValueError("Material population does not cover the requested scope")
        if blueprint["device_type"] not in row.get("device_types", []):
            raise ValueError("Material device does not cover the requested scope")
    return ordered


def _anchors(material, warnings):
    rows, seen = [], set()
    for original in material.get("teaching_anchors", []):
        if not isinstance(original, dict):
            continue
        identity = original.get("id")
        text = original.get("text") or original.get("criterion")
        references = original.get("reference_ids")
        if (not isinstance(identity, str) or not identity or identity in seen
                or original.get("step") not in STEPS or not isinstance(text, str) or not text.strip()
                or not isinstance(references, list) or not references
                or not all(isinstance(value, str) and value for value in references)):
            warnings.append(material["id"] + "存在缺少有效步骤、原文标准或来源的锚点，未作必答依据。")
            continue
        seen.add(identity)
        rows.append({
            "id": identity, "material_id": material["id"], "step": original["step"],
            "criterion": text, "acceptable": deepcopy(original.get("acceptable", "")),
            "reference_ids": deepcopy(references), "review_status": material["review_status"],
            "source_question_id": material.get("source_question_id", ""),
            "source_review_event_hash": material.get("source_review_event_hash", ""),
            "source_item_fingerprint": material.get("source_item_fingerprint", ""),
            "reviewer_kind": material.get("reviewer_kind", ""),
            "independent_clinical_validation": material.get("independent_clinical_validation", False),
        })
    return rows


def _scope(blueprint, steps, minimum):
    single = blueprint["question_type"] == "single_choice"
    return {
        "min_targets": 1 if single else minimum,
        "content_targets_requested": minimum,
        "scoring_points": {"min": 1, "max": 1} if single else {"min": 2, "max": 3},
        "grading_mode": "correct_option_only" if single else "answer_and_case_basis",
        "required_dimensions": ["选择唯一最合适选项"] if single else ["与本题焦点相符的回答或目标", "该回答与原病例线索的对应依据"],
        "allow_equivalent_answers": not single,
        "not_required": ["未明确要求的两个独立评估目标", "固定患者/器械/护理/组织分类",
                         "题外相邻步骤或完整六步护理计划", "材料未支持的固定频率、阈值或新诊断"],
        "steps": list(steps),
        "boundary": "两个评分维度可以评价同一个回答及其依据，不等于必须给出两个独立目标。"
                    "单选多个判断可体现在选项与解析中，不形成额外加分项。",
    }


def build_batch_plan(blueprint, materials, teaching_plan):
    """Return a JSON-only, pinned small-batch plan; inputs are never mutated.

    plan_id is bp_ plus the first 24 hex characters of the full input_sha256.
    Input identity includes VERSION, the normalized blueprint, ordered complete
    material snapshots and the original teaching plan. Each content_sha256 uses
    content_sha256(value), including child hashes but excluding its own field.
    """
    blueprint = validate_blueprint(blueprint)
    selected = _material_inputs(blueprint, materials)
    if not isinstance(teaching_plan, dict):
        raise ValueError("The reviewed teaching-plan snapshot must be an object")
    teaching_plan = deepcopy(teaching_plan)
    pinned = {"version": VERSION, "blueprint": blueprint, "materials": selected, "teaching_plan": teaching_plan}
    input_sha = canonical_sha256(pinned)
    plan_id, warnings = "bp_" + input_sha[:24], []
    anchors = {row["id"]: _anchors(row, warnings) for row in selected}
    steps = _goal_steps(blueprint["learning_objective"])
    if not steps:
        warnings.append("学习目标未明确指向可核对的考核步骤；只提供候选，不把全套六步默认为必答。")
    minimum = _minimum_targets(blueprint)
    items, use_counts, signatures, repeats = [], {}, {}, []
    all_available = []
    for row in selected:
        all_available.extend(row["id"] + ":" + anchor["id"] for anchor in anchors[row["id"]] if anchor["step"] in steps)
    for index in range(blueprint["count"]):
        material = selected[index % len(selected)]
        if blueprint["count"] == 1:
            item_steps = list(steps)
        elif steps:
            item_steps = [steps[(index // len(selected) + index % len(selected)) % len(steps)]]
        else:
            item_steps = []
        chosen = []
        missing_steps = []
        for step in item_steps:
            group = [anchor for anchor in anchors[material["id"]] if anchor["step"] == step]
            target = _terms(blueprint["learning_objective"])
            group.sort(key=lambda anchor: -len(target & _terms(anchor["criterion"])))
            key = (material["id"], step)
            offset = use_counts.get(key, 0)
            use_counts[key] = offset + 1
            if group:
                chosen.append(group[offset % len(group)])
            else:
                missing_steps.append(step)
        anchor_ids = [row["id"] for row in chosen]
        ready = bool(item_steps) and not missing_steps and len(item_steps) <= 3
        focus = "；".join(row["criterion"] for row in chosen) or blueprint["learning_objective"]
        scope = _scope(blueprint, item_steps, minimum)
        if not item_steps:
            guidance = "先明确本题要考查的单一能力，再按学习目标出题；候选锚点不能自动转成必答，不默认完整六步。"
        elif len(item_steps) > 1:
            guidance = "仅围绕明确要求的" + "、".join(STEP_LABELS[step] for step in item_steps) + "设计一个连贯任务，不扩展到未要求的步骤。"
        else:
            variants = _TEMPLATES[item_steps[0]]
            selected_anchor = chosen[0] if chosen else None
            variant = 1 if selected_anchor and selected_anchor["id"].endswith("R2") else 0
            guidance = variants[variant]
        if blueprint["question_type"] == "single_choice":
            if item_steps:
                guidance = "本题仅考查" + "、".join(STEP_LABELS[step] for step in item_steps) + "；单选任务只要求从选项中选择唯一最合适项。"
            guidance += " 选项保持平行，必要判断和理由放在选项与解析中，不要求学生另写解释，唯一判分由correct_options决定。"
            if minimum > 1:
                guidance += f" 学习目标明确要求的{minimum}项内容应在选项中体现，学生仍只选一项，不形成多个加分点。"
        else:
            guidance += f" 本题目标数量下限为{minimum}项，评分围绕回答与病例依据对齐；不按固定分类强切，也不额外要求多个独立目标。"
        item_warnings = []
        if missing_steps:
            item_warnings.append("所选病例没有支持目标步骤的已审核锚点：" + "、".join(missing_steps) + "；不得借其他步骤凑题。")
        if len(item_steps) > 3:
            item_warnings.append("单题目标覆盖超过三个步骤，需要确认与既定评分点数量是否相容。")
        signature = (material["id"], tuple(item_steps), tuple(anchor_ids))
        if signature in signatures:
            earlier = signatures[signature]
            repeats.append({"index": index + 1, "earlier_index": earlier, "material_id": material["id"]})
            item_warnings.append(f"与第{earlier}题使用相同病例、步骤和锚点，存在重复风险；不能靠换措辞假装新覆盖。")
        else:
            signatures[signature] = index + 1
        identity = {"plan_id": plan_id, "index": index + 1, "material_id": material["id"],
                    "steps": item_steps, "anchor_ids": anchor_ids, "focus": focus}
        item = {"id": "qi_" + canonical_sha256(identity)[:24], "plan_id": plan_id, "index": index + 1,
                "material_id": material["id"], "material_title": material.get("title", ""),
                "case_facts": material["case_facts"],
                "case_facts_sha256": hashlib.sha256(material["case_facts"].encode("utf-8")).hexdigest(),
                "focus": focus, "steps": item_steps, "anchor_ids": anchor_ids, "anchors": deepcopy(chosen),
                "candidate_anchor_ids": [row["id"] for row in anchors[material["id"]]][:3] if not item_steps else [],
                "question_type": blueprint["question_type"], "difficulty": blueprint["difficulty"],
                "task_guidance": guidance, "answer_scope": scope,
                "planning_status": "ready" if ready else "needs_goal_review", "warnings": item_warnings}
        item["content_sha256"] = content_sha256(item)
        items.append(item)
        warnings.extend("第" + str(index + 1) + "题：" + warning for warning in item_warnings)
    covered_materials = list(dict.fromkeys(item["material_id"] for item in items))
    covered_steps = list(dict.fromkeys(anchor["step"] for item in items for anchor in item["anchors"]
                                      if anchor["step"] in item["steps"]))
    used_anchors = list(dict.fromkeys(item["material_id"] + ":" + anchor for item in items for anchor in item["anchor_ids"]))
    uncovered_materials = [row["id"] for row in selected if row["id"] not in covered_materials]
    uncovered_steps = [step for step in steps if step not in covered_steps]
    if uncovered_materials:
        warnings.append("题量不足以覆盖全部所选病例：" + "、".join(uncovered_materials) + "。")
    if uncovered_steps:
        warnings.append("缺少已审核依据或题量不足，尚未覆盖步骤：" + "、".join(uncovered_steps) + "。")
    plan = {"version": VERSION, "plan_id": plan_id, "input_sha256": input_sha,
            "learning_objective": blueprint["learning_objective"], "requested_count": blueprint["count"],
            "items": items,
            "coverage": {"selected_material_ids": [row["id"] for row in selected], "covered_material_ids": covered_materials,
                         "uncovered_material_ids": uncovered_materials, "requested_steps": steps, "covered_steps": covered_steps,
                         "uncovered_steps": uncovered_steps, "available_anchor_ids": all_available,
                         "covered_anchor_ids": used_anchors, "unused_anchor_ids": [value for value in all_available if value not in used_anchors],
                         "unique_focus_count": len(signatures), "repeated_focus_count": len(repeats),
                         "repeat_risks": repeats, "needs_goal_review_count": sum(item["planning_status"] != "ready" for item in items)},
            "warnings": list(dict.fromkeys(warnings)), "development_only": True, "formal_allowed": False}
    plan["content_sha256"] = content_sha256(plan)
    return plan
