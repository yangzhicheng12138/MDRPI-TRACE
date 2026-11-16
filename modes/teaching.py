"""Small original practice library, independent of formal evaluation cases."""
import re

from .language import localized
from .question_bank import load_bank

CASES = {
    "mask": {"title": "面罩与管路牵拉 / Mask and tubing", "level": "基础",
        "case": "Fictional practice case: An alert adult receiving non-invasive ventilation reports mild discomfort over the nasal bridge. The skin is intact with a blanchable indentation after briefly lifting the mask while respiratory support is maintained. The tubing pulls on the mask during turning. No persistent redness or broken skin is documented. What would you assess, change, and reassess, and why?"},
    "tube": {"title": "鼻胃管固定 / Nasogastric tube", "level": "基础",
        "case": "Fictional practice case: An adult with a nasogastric tube reports discomfort at the nostril. The fixation tape is damp, and the tube rests against the nasal rim. The insertion record is available but the current skin and mucosal findings are not documented. How would you assess the site and fixation, protect the tissue, and evaluate your actions?"},
    "complex": {"title": "多器械与沟通困难 / Multiple devices", "level": "进阶",
        "case": "Fictional practice case: A sedated adult in intensive care has an endotracheal tube and a cervical collar. Facial oedema has increased since the previous shift. No recent assessment of skin under the devices is documented, and the patient cannot report pain. Explain your priorities, how you would obtain missing information and coordinate safe inspection, and how you would judge whether your plan helped."},
}

WELCOME = "可以使用自己的案例，也可以选一个练习案例，或告诉我想练习的主题、难度和语言，由我出题。你先尝试回答，我再追问或提示，最后给六步总结。"
WELCOME_EN = "Use your own case, choose a practice case, or tell me a topic and level. Try an answer first; I will explain what you did well, offer help where needed, and finish with a six-step summary."

CASES["mask"]["case_zh"] = "虚构练习案例：一名意识清楚、接受无创通气的成人诉鼻梁轻微不适。在维持呼吸支持的条件下短暂移开面罩，鼻梁皮肤完整，有可褪色的浅压痕。患者翻身时管路会牵拉面罩。未记录持续性红斑或皮肤破损。你会评估哪些内容、作哪些调整，以及如何复评？请说明理由。"
CASES["tube"]["case_zh"] = "虚构练习案例：一名留置鼻胃管的成人诉鼻孔处不适。固定胶带潮湿，管路贴靠鼻翼边缘。可以查到置管记录，但当前皮肤和黏膜情况尚未记录。你会如何评估接触部位和固定情况、保护局部组织，并评价措施的效果？"
CASES["complex"]["case_zh"] = "虚构练习案例：一名镇静中的重症成人同时使用气管插管和颈托。与上一班相比，面部水肿加重。近期没有记录器械下皮肤的评估，患者也无法表达疼痛。请说明你的处理重点、如何获取缺失信息并协调安全检查，以及如何判断措施是否有效。"

for _case in CASES.values():
    _case.update(category="original", kind="fictional_case", has_reference_answer=False)
from .content_review import apply_review, load_reviews
_reviews = load_reviews()
CASES = {key: apply_review({**value, "id": key}, ledger=_reviews) for key, value in CASES.items()}
CASES.update(load_bank())


def practice_catalog(category=None, *, module=None, population=None, device_type=None, difficulty=None, question_type=None, topic=None, learning_topic=None):
    """The browsing catalog contains no answer keys or hints."""
    from .catalog import catalog_entry, filter_catalog
    rows = [catalog_entry(key, value) for key, value in CASES.items() if category is None or value["category"] == category]
    return filter_catalog(rows, module=module, population=population, device_type=device_type, difficulty=difficulty, question_type=question_type, topic=topic, learning_topic=learning_topic)


def practice_case(case_id, language):
    item = CASES[case_id]
    return item["case_zh"] if language == "zh" else item["case"]


def practice_title(item, language):
    titles = item["title"].split(" / ")
    return titles[1] if language == "en" and len(titles) > 1 else titles[0]


def invitation(case, language=None):
    return ("请先写出你的初步评估、护理方案及理由。可以自然作答，不必强行写成六个标题。"
            if (language == "zh" if language else any("\u4e00" <= c <= "\u9fff" for c in case)) else
            "Please give your initial assessment and care plan, with your reasoning. You can answer naturally; six headings are optional.")


def help_intent(text):
    """Only short standalone help controls; clinical answers remain answers."""
    from .subjective_help import learning_help_intent
    support = learning_help_intent(text)
    if support:
        return support
    value = text.strip().strip("。.!！?？").lower()
    if is_recheck_control(text):
        return "recheck"
    if re.fullmatch(r"(?:请|能否|可以)?(?:给我|给|提供)?(?:一些|一点|一个|点)?提示(?:呢|吗)?|(?:(?:please )?(?:give me )?(?:a |another )?hint|help me)", value):
        return "hint"
    if re.fullmatch(r"(?:我(?:的)?(?:回答|答得|答的)(?:怎么|怎样|如何)(?:样)?(?:了)?|(?:请)?(?:评价|点评)(?:一下)?(?:我的)?回答)|how did i do|(?:please )?(?:evaluate|review) my answer", value):
        return "feedback"
    if re.fullmatch(r"(?:请)?(?:重新核对|重新评价|重新评估)(?:一下)?(?:我的)?(?:回答|作答|已有回答)?|(?:这(?:一点|点)我已经说了[，,]?)(?:请)?重新核对|(?:please )?recheck(?: my (?:answer|answers))?", value):
        return "recheck"
    if re.fullmatch(r"(?:请|现在)?(?:总结(?:一下|一下吧)?|结束(?:本次)?练习(?:并总结)?)|(?:please )?(?:summari[sz]e|finish(?: the exercise)?)", value):
        return "summary"
    return "send"


def is_recheck_control(text):
    """Standalone requests to reread an existing answer, never clinical actions."""
    value = (text or "").strip().strip("。.!！?？").lower()
    target = r"(?:我(?:的)?(?:已有|已经提交的|之前的|前面的|刚才的|此前的)?(?:回答|答案|作答)|(?:已有|之前|前面|刚才|此前)的?(?:回答|答案|作答)|回答|答案|作答)"
    courtesy = r"(?:请|麻烦(?:你)?|能否|可以|帮我)?"
    action = r"(?:重新|再|再次)?(?:核对|检查|复核|评价|评估|看看)(?:一下)?"
    if re.fullmatch(courtesy + r"(?:重新|再|再次)(?:核对|检查|复核|评价|评估|看看)(?:一下)?(?:" + target + r")?", value):
        return True
    if re.fullmatch(courtesy + action + target, value):
        return True
    if re.fullmatch(courtesy + r"(?:根据|对照|重读)" + target + r"[，,]?" + courtesy + action, value):
        return True
    if re.fullmatch(r"(?:这(?:一)?点|这一项|这项|这些|我)?(?:我)?(?:已|已经)(?:说过|说了|说明了|回答了)[，,]?" + courtesy + action, value):
        return True
    return bool(re.fullmatch(r"(?:please\s+)?(?:recheck|re-check|check again|review again|reread|read again)"
                             r"(?:\s+(?:my\s+)?(?:existing\s+|previous\s+|earlier\s+)?(?:answer|answers|response))?", value))


def student_answers(state):
    """Keep accepted routing decisions stable; interpret unmarked old records."""
    accepted = {(event["answer_index"], event["text"])
                for event in state.get("answer_events", [])
                if isinstance(event, dict) and isinstance(event.get("answer_index"), int)
                and not isinstance(event["answer_index"], bool)
                and isinstance(event.get("text"), str)}
    # Indices refer to the unchanged attempts array, including historical help
    # that had mistakenly been recorded there. Never reclassify a new answer
    # after the router already accepted it, even if it resembles a control.
    return [text for index, text in enumerate(state.get("attempts", []), 1)
            if (index, text) in accepted or help_intent(text) == "send"]


def is_student_message(message, state):
    """Identify actual answer messages without re-routing their saved text."""
    if message.get("role") != "user" or message.get("kind") != "student":
        return False
    # Mixed turns retain the complete user message for display while the
    # grader receives only the verbatim answer fragment recorded separately.
    answer_text = message.get("answer_text")
    if isinstance(answer_text, str) and answer_text.strip():
        return True
    content = message.get("content", "")
    if any(isinstance(event, dict) and event.get("text") == content
           and event.get("turn_id") is not None
           and event.get("turn_id") == message.get("turn_id")
           for event in state.get("answer_events", [])):
        return True
    return help_intent(content) == "send"


def tutor_context(state):
    """Actual dialogue preceding this answer, including any hints the learner saw."""
    messages = state.get("messages", [])
    answers = [i for i, message in enumerate(messages) if is_student_message(message, state)]
    if not answers:
        return []
    end = answers[-1]
    start = answers[-2] + 1 if len(answers) > 1 else next(
        (i for i in range(end - 1, -1, -1) if messages[i].get("kind") in {"case", "invitation"}
         and messages[i]["role"] == "assistant"), 0)
    return [{"role": m["role"], "content": m["content"]} for m in messages[start:end]
            if (m["role"] == "assistant" and m.get("kind") != "welcome")
            or (m["role"] == "user" and not is_student_message(m, state))]


def normalize_review(review, language):
    result = dict(review)
    empty = {"", "none", "none.", "n/a", "无", "无。", "没有", "无实质错误", "no material errors", "no material errors."}
    for key in ["strengths", "improvements", "gaps", "material_errors", "hints"]:
        result[key] = [s for s in result.get(key, []) if s.strip().lower() not in empty]
    result["feedback"] = re.sub(r"^\s*phase\s+(?:answer|probe|hint|reflect)\s*:\s*", "", result["feedback"], flags=re.I)
    result["language"] = language
    return result


def assessment_text(review, language):
    if review.get("grading_mode") == "case_formative":
        from .learner_evidence import formative_text
        return formative_text(review, language)
    if review.get("grading_mode") == "question_rubric":
        from .rubric import assessment_text as rubric_text
        return rubric_text(review, language)
    adequate = review["ready_for_summary"] and not review["material_errors"]
    title = localized(language, "**本次回答反馈**", "**Feedback on your answer**")
    verdict = (localized(language, "已经覆盖当前问题的主要要求，可以进入总结。", "You have covered the main requirements of this question; we can summarize.") if adequate else
               localized(language, "下面是本次回答已经做到的内容和仍需完善的重点。", "Here is what your answer covers and what still needs attention."))
    parts = [title, verdict]
    for key, zh, en in [("strengths", "已经做到", "What you did well"), ("improvements", "这次的进步", "What improved"),
                        ("material_errors", "需要纠正", "Corrections"), ("gaps", "还可完善", "Remaining points")]:
        if review.get(key):
            parts.append("**" + localized(language, zh, en) + "**\n\n" + "\n".join("- " + s for s in review[key][:2]))
    return "\n\n".join(parts)


def repeated_focus(reviews, review):
    return bool(reviews and review.get("focus") not in {None, "none"} and
                reviews[-1].get("focus") == review["focus"])
