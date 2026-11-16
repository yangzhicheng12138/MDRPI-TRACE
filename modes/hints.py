"""Explanatory help, separate from assessment questions and grading."""
import re

from .language import localized


def question_shaped(text):
    return bool(re.search(r"[?？]|(?:^|[\n.!。！]\s*)(?:what|why|how|when|where|which|who|can you|could you|do you)\b|^(?:你认为|你会|为什么|如何|哪些|怎么)", text, re.I))


def learning_scaffold(language, focus):
    first = {
        "recognize": ("先区分题目已给出的观察和仍未提供的资料。", "只有已给出的内容才能作为病例事实；缺失资料可以列为待评估项。",
                      "Separate the observations given in the case from information not provided.", "Use the supplied observations as facts; missing details can be listed for assessment."),
        "analyze": ("把一个病例线索与一个可能的问题联系起来。", "用“这个线索提示……，因此需要确认……”说明联系；推测要与已确认事实分开。",
                    "Connect one observation in the case with a possible problem.", "Explain the connection as 'this finding suggests ... so I would check ...'; keep an inference separate from a confirmed fact."),
        "prioritize": ("先考虑哪些问题最需要及时处理。", "依据题目中的现有表现说明优先顺序，不要仅因某种风险可能存在就当作已经发生。",
                       "Consider which problem needs attention first.", "Explain the priority using the actual findings; a possible risk is not an established complication."),
        "solutions": ("把拟定措施与前面识别的具体问题对应。", "先解释措施想改变什么，再补充执行它所需的条件，避免只罗列护理措施。",
                      "Match a proposed measure to a specific problem you identified.", "Explain what the measure should change and the conditions needed to use it, rather than listing unrelated interventions."),
        "action": ("把护理建议写成一个可执行的步骤。", "说明动作、需要配合的人员以及执行时要观察的变化；不要补造题目没有提供的结果。",
                   "Turn a proposed intervention into an actionable step.", "Describe the action, any assistance needed, and observations during it; do not invent the outcome."),
        "evaluate": ("回到最初发现的问题，选择能用于比较的观察。", "用干预前后的变化说明效果；仍未解决的问题需要继续评估，而不是预设已经改善。",
                     "Return to the initial problem and select an observation for comparison.", "Explain how a before-and-after comparison would show the effect; do not assume improvement has already occurred."),
    }.get(focus)
    if first is None:
        first = ("先围绕题目明确要求回答的范围整理线索。", "将病例已知事实与推测分开，并说明一个判断的依据。",
                 "Start with the scope of the question and its stated observations.", "Separate supplied facts from assumptions and explain the basis for one judgement.")
    offset = 0 if language == "zh" else 2
    return [{"cue": first[offset], "explanation": first[offset + 1]},
            {"cue": localized(language, "把你的答案写成“观察 → 判断 → 理由”的简短链条。", "Use a short observation → judgement → reason sequence in your revision."),
             "explanation": localized(language, "每个判断尽量对应题目中的一个事实；如果理由依赖未知资料，就标明需要补充评估。", "Link each judgement to a case fact. If the reasoning depends on missing information, identify the assessment needed.")}]


def prepare_hints(raw, language, focus):
    """One bounded model call, then usable learning scaffolds, never rewrite loops."""
    items = raw.get("hints", [])
    reasons = []
    if not items:
        reasons.append("empty_hints")
    for item in items[:2]:
        if not isinstance(item, dict) or not all(isinstance(item.get(k), str) and item[k].strip() for k in ("cue", "explanation")):
            reasons.append("missing_cue_or_explanation")
            continue
        text = item["cue"] + "\n" + item["explanation"]
        if question_shaped(text):
            reasons.append("question_in_hint")
        if language == "en" and re.search(r"[\u4e00-\u9fff]", text):
            reasons.append("language_mismatch")
        if language == "zh" and any(not re.search(r"[\u4e00-\u9fff]", item[k]) for k in ("cue", "explanation")):
            reasons.append("language_mismatch")
    if reasons:
        items = learning_scaffold(language, focus)
    formatted = [localized(language, "**线索：** ", "**Cue:** ") + item["cue"] + "\n\n" +
                 localized(language, "**解释：** ", "**Explanation:** ") + item["explanation"] for item in items[:2]]
    return formatted, list(dict.fromkeys(reasons))
