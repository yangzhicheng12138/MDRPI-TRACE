"""Explainable practice selection from actual saved evidence and coverage.

These are transparent MVP rules, not a fitted mastery estimator. No model call
or catalogue mutation occurs here; expert approval remains a formal-use gate.
"""
from __future__ import annotations

from collections import Counter

from .learning_analytics import aggregate_records, DIFFICULTIES, normalize_difficulty, STEP_LABELS


SCHEMA_VERSION = "learning_recommendations_v1"
RULE_VERSION = "coverage_hint_progression_v1"
_EXPERT_APPROVED = {"expert_approved", "clinical_expert_approved", "approved_by_expert", "expert_review_approved"}


def _values(value):
    if isinstance(value, str):
        return [value] if value else []
    return [item for item in value or [] if isinstance(item, str) and item]


def _expert_approved(item):
    review = item.get("expert_review") or {}
    independent = item.get("independent_expert_review") is True or (
        isinstance(review, dict) and review.get("independent_expert_review") is True)
    return independent and (item.get("review_status") in _EXPERT_APPROVED
            or isinstance(review, dict) and review.get("status") == "approved"
            and bool(review.get("reviewer") or review.get("reviewed_by")))


def _eligible(item, mode):
    if (not item.get("id") or item.get("practice_allowed") is False or item.get("enabled") is False
            or item.get("review_status") in {"rejected", "withdrawn", "disabled", "retired"}):
        return False
    if mode == "formal":
        return bool((item.get("formal_allowed") is True or item.get("formal_eligible") is True)
                    and _expert_approved(item))
    return True


def _difficulty(item):
    return normalize_difficulty(item.get("difficulty") or item.get("level") or item.get("source_difficulty"))


def _step_ids(item):
    direct = _values(item.get("step_ids") or item.get("dimensions") or item.get("weak_step_ids"))
    mapped = [value for value in direct if value in STEP_LABELS]
    if not mapped and (item.get("kind") == "standardized_case" or item.get("training_module") == "specialty"):
        # Full guided cases practise all six dimensions. We do not invent a
        # specific clinical skill mapping for unannotated concept cards.
        mapped = list(STEP_LABELS)
    return mapped


def _progression(history):
    target = "基础"
    earned = []
    for current, following in zip(DIFFICULTIES, DIFFICULTIES[1:]):
        comparable = [row for row in history if row["difficulty"] == current]
        latest = comparable[-2:]
        if (len(latest) == 2 and all(row["low_hint_mastery"] for row in latest)
                and len({row["case_id"] for row in latest}) == 2):
            target = following
            earned.append(current)
        else:
            break
    # A newly demonstrated difficulty at the next level prevents jumping
    # beyond its evidence when higher-difficulty attempts remain uncertain.
    recent = history[-2:]
    consolidate = any(row["guided"] and row["completion_status"] == "interrupted" for row in recent)
    if recent and any(row["guided"] and (row["hint_count"] >= 3 or
                                          any(step["incorrect"] or step["value"] is not None and step["value"] < 60
                                              for step in row["six_steps"])) for row in recent):
        consolidate = True
    if consolidate:
        target = "基础"
    return {"target_difficulty": target, "earned_levels": earned,
            "basis": "最近两次不同案例的六步完整作答、必答点全部有原话支持且提示不超过一次，才允许进阶。",
            "consolidate": consolidate}


def recommend(records, catalog, current_id="", limit=3, *, mode="practice"):
    """Return distinct eligible items, reasons, and an explicit exhausted state.

    ``mode='formal'`` additionally requires an affirmative formal-use flag and
    an independent expert approval. Source-pair checking alone never qualifies.
    """
    if mode not in {"practice", "formal"}:
        raise ValueError("推荐模式需为 practice 或 formal。")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 10:
        raise ValueError("推荐数量需为 1 到 10。")
    analytics = aggregate_records(records)
    history = analytics["history"]
    progression = _progression(history)
    target = progression["target_difficulty"]
    frequencies = Counter(row["case_id"] for row in history)
    recent_ids = {row["case_id"] for row in history[-2:]}
    if current_id:
        recent_ids.add(current_id)
    populations = {item["id"] for item in analytics["coverage"]["populations"]}
    devices = {item["id"] for item in analytics["coverage"]["device_types"]}
    weak_axes = {axis["id"] for axis in analytics["six_steps"]
                 if axis["value"] is not None and axis["value"] < 80 or axis["hint_count"] >= 3}
    weak_axes |= {step["id"] for row in history[-2:] for step in row["six_steps"]
                  if step["incorrect"] or step["hint_count"] >= 3}
    # Deduplicate catalogue IDs before ranking; the caller can supply several
    # device memberships for the same case without generating duplicate cards.
    eligible = {}
    for item in catalog:
        if isinstance(item, dict) and _eligible(item, mode):
            eligible.setdefault(item["id"], item)
    candidates = [item for item in eligible.values() if item["id"] not in recent_ids]
    # Do not recommend harder cases before the evidence-based progression
    # gate. Unknown difficulties cannot silently count as suitable beginners.
    candidates = [item for item in candidates if _difficulty(item) and
                  DIFFICULTIES.index(_difficulty(item)) <= DIFFICULTIES.index(target)]

    def features(item):
        matches = sorted(set(_step_ids(item)) & weak_axes, key=lambda value: list(STEP_LABELS).index(value))
        new_people = sorted(set(_values(item.get("populations") or item.get("population"))) - populations)
        new_devices = sorted(set(_values(item.get("device_types") or item.get("devices") or item.get("device_type"))) - devices)
        return matches, new_people, new_devices

    def priority(item):
        matches, new_people, new_devices = features(item)
        return (0 if _difficulty(item) == target else 1,
                0 if matches else 1, 0 if frequencies[item["id"]] == 0 else 1,
                -len(matches), -(len(new_people) + len(new_devices)), frequencies[item["id"]], item["id"])

    items = []
    for item in sorted(candidates, key=priority)[:limit]:
        identity, difficulty = item["id"], _difficulty(item)
        matches, new_people, new_devices = features(item)
        codes, reasons = [], []
        if not history:
            codes.append("cold_start_basic")
            reasons.append("还没有已保存训练记录，先从基础案例开始。")
        if progression["consolidate"]:
            codes.append("consolidate_recent_difficulty")
            reasons.append("近期有遗漏、错误或较多提示使用，先用基础案例巩固。")
        if matches:
            codes.append("weak_step_practice")
            reasons.append("练习近期覆盖不足或提示依赖较多的步骤：" + "、".join(STEP_LABELS[value] for value in matches) + "。")
        if difficulty != "基础" and difficulty == target:
            codes.append("evidence_based_progression")
            reasons.append("前一难度已有两个不同案例低提示完成六步，建议尝试" + difficulty + "案例。")
        if new_people or new_devices:
            codes.append("coverage_expansion")
            reasons.append("补充尚未练习的人群或器械情境。")
        if frequencies[identity] == 0:
            codes.append("untried_case")
            reasons.append("这道案例尚未出现在你的保存记录中。")
        else:
            codes.append("spaced_case_return")
            reasons.append("近期两轮未练习此题，可回到旧案例检验判断。")
        items.append({"case_id": identity, "title": str(item.get("title", identity)).split(" / ")[0],
                      "difficulty": difficulty, "reason": "".join(reasons), "reason_codes": codes,
                      "weak_step_ids": matches, "new_populations": new_people,
                      "new_device_types": new_devices, "previous_rounds": frequencies[identity],
                      "rule_version": RULE_VERSION})
    exhausted = ("当前条件下没有合适的新推荐；近期两轮及当前题已排除，可切换题目分类或由你主动选择复习。"
                 if eligible else "当前没有符合练习许可与审核要求的题目。")
    return {"schema_version": SCHEMA_VERSION, "rule_version": RULE_VERSION, "mode": mode,
            "items": items, "eligible_count": len(eligible), "candidate_count": len(candidates),
            "no_match_reason": "" if items else exhausted, "cooldown_rounds": 2,
            "progression": progression, "history_rounds": len(history),
            "basis": "真实保存的必答点覆盖、错误、提示次数及人群/器械覆盖；规则建议，尚未验证教学效果。"}
