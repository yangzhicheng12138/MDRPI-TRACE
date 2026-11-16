"""Three progressive hints for subjective tasks, using published content only.

The caller owns hint counts, reference disclosure and assessment provenance.
These helpers do not assess answers, create a reference answer or change an
objective question's rules.
"""
import re
import unicodedata

from .hints import learning_scaffold, prepare_hints
from .language import localized


_ZH_STUCK = (
    r"(?:(?:这(?:道|个)?题|这个问题)(?:我)?|我)?"
    r"(?:还(?:是)?|仍(?:然)?|也|真的|完全|实在)?"
    r"(?:不会(?:(?:做|答|回答)(?:这(?:道|个)?题|这个问题|本题)?|"
    r"(?:整理|组织)(?:答案|回答|思路))?|答不上来|答不出来|回答不上来|回答不出来|"
    r"想不出答案|没有思路|没思路|不懂|"
    r"(?:不知道|不清楚)(?:答案|该(?:怎么|如何|咋)(?:答|回答|整理(?:答案|思路))|"
    r"(?:怎么|如何|咋)(?:(?:答|回答)(?:这(?:道|个)?题|这个问题|本题)?|"
    r"分析|整理(?:答案|思路)|组织(?:答案|回答)))?)"
    r"(?:了|啊|呢|呀)?"
)
_ZH_READ_HINTS = (
    r"(?:(?:我(?:把)?)?(?:所有|全部|这些|这(?:三|3|几)条|三条|3条)?提示"
    r"(?:已经|已|都|全都|全部)?(?:看(?:过|完|完了|了)|读(?:过|完|完了|了)|用完了?)|"
    r"我(?:都)?(?:已经|已)?(?:看(?:过|完|完了|了)|读(?:过|完|完了|了)|用完了?)"
    r"(?:所有|全部|这些|这(?:三|3|几)条|三条|3条)?提示)"
)
_EN_STUCK = (
    r"(?:no idea|(?:still\s+)?stuck|"
    r"i\s+(?:still\s+)?(?:do\s+not|don't|dont)\s+know"
    r"(?:\s+(?:the\s+answer|how\s+to\s+(?:answer(?:\s+(?:this|this\s+question))?|"
    r"(?:organize|organise|structure)\s+(?:my|the|an)\s+(?:answer|response))))?|"
    r"i(?:\s+am|'m)\s+(?:still\s+)?(?:not\s+sure|stuck)|"
    r"i\s+(?:still\s+)?(?:cannot|can't|cant)\s+answer(?:\s+(?:this|this\s+question))?|"
    r"i\s+have\s+no\s+idea)"
)
_EN_READ_HINTS = (
    r"(?:i(?:\s+have|'ve)?\s+(?:already\s+)?(?:read|seen|used)"
    r"(?:\s+(?:all|the|three|3)){0,3}\s+hints?|"
    r"(?:all\s+the|all|the|three|3)(?:\s+(?:three|3))?\s+hints?"
    r"(?:\s+have\s+been)?\s+(?:shown|read|used))"
)


def learning_help_intent(text):
    """Recognize whole help controls; substantive learner content stays an answer.

    A missing patient fact is not an admission of being unable to answer.
    Statements containing a clinical action plus difficulty are not stripped
    down to their difficulty clause.
    """
    if not isinstance(text, str):
        return None
    value = unicodedata.normalize("NFKC", text).replace("’", "'").strip()
    value = re.sub(r"\s+", " ", value).strip("。.!！?？ ").casefold()
    if not value:
        return None

    zh_reference = (
        r"(?:请(?:你)?|麻烦(?:你)?|能否|可以|你)?(?:直接)?"
        r"(?:告诉我|给我|提供|显示|展示|查看|看|让我看|我(?:想|要)(?:直接)?(?:看|知道))"
        r"(?:一下)?(?:这(?:道|个)?题(?:的)?|本题(?:的)?)?"
        r"(?:标准答案|参考答案|答案)(?:吧|好吗|可以吗|一下)?"
    )
    zh_give_up = (
        r"(?:我)?放弃(?:了)?(?:[,; ]\s*)?(?:我)?"
        r"(?:想|要)(?:直接)?(?:看|知道)(?:标准答案|参考答案|答案)"
    )
    en_reference = (
        r"(?:please\s+)?(?:(?:just|directly)\s+)?"
        r"(?:show\s+me|tell\s+me|give\s+me|let\s+me\s+see|view|see|"
        r"i\s+(?:want|would\s+like)\s+to\s+(?:see|view|know)|can\s+i\s+see)"
        r"\s+(?:the\s+|a\s+)?(?:standard\s+answer|reference\s+answer|answer)(?:\s+please)?"
    )
    en_give_up = (
        r"i\s+give\s+up(?:[,;]\s*|\s+(?:and\s+)?)"
        r"(?:i\s+)?(?:want\s+to\s+(?:see|know)|(?:please\s+)?(?:show|tell|give)\s+me)"
        r"\s+(?:the\s+)?(?:standard\s+answer|reference\s+answer|answer)"
    )
    zh_control_separator = r"(?:[,;.。]\s*|\s*)"
    en_control_separator = r"(?:[,;.。]\s*|\s+)"
    if any(re.fullmatch(pattern, value) for pattern in (
            zh_reference, zh_give_up, en_reference, en_give_up,
            _ZH_STUCK + zh_control_separator + zh_reference,
            _EN_STUCK + en_control_separator + en_reference)):
        return "reference"

    zh_request = r"(?:[,; ]\s*)?(?:请)?(?:给(?:我)?(?:一条|一点|一些|点)?提示|帮(?:帮)?我)"
    en_request = r"(?:[,;]\s*|\s+)(?:please\s+)?(?:give\s+me\s+(?:a\s+)?hint|help\s+me)"
    zh_sequence = _ZH_READ_HINTS + r"(?:[,;.。]\s*)?(?:但(?:是)?|可是|不过)?\s*" + _ZH_STUCK
    en_sequence = _EN_READ_HINTS + en_control_separator + r"(?:(?:but|however|yet|and)\s+)?" + _EN_STUCK
    if any(re.fullmatch(pattern, value) for pattern in (
            _ZH_STUCK, _ZH_STUCK + zh_request, zh_sequence, zh_sequence + zh_request,
            _EN_STUCK, _EN_STUCK + en_request, en_sequence, en_sequence + en_request)):
        return "stuck"
    return None


def subjective_hint_plan(item, raw, language, focus):
    """Return three hint strings and preparation reasons, preserving authored order.

    The third hint names the existing required criteria as an answer structure.
    It adds no patient finding, procedural parameter or new scoring requirement.
    A focus never reorders a procedure's published steps.
    """
    hints, reasons = prepare_hints(raw, language, focus)
    hints, reasons = list(hints[:2]), list(reasons)
    if len(hints) < 2:
        fallback, _ = prepare_hints({"hints": learning_scaffold(language, focus)}, language, focus)
        hints.extend(fallback[len(hints):2])
        reasons.append("incomplete_hint_sequence")

    rubric = item.get("rubric") if isinstance(item, dict) else None
    required = rubric.get("required", []) if isinstance(rubric, dict) else []
    labels = []
    for row in required if isinstance(required, list) else []:
        criterion = row.get("criterion") if isinstance(row, dict) else None
        label = criterion.get(language) if isinstance(criterion, dict) else criterion
        if isinstance(label, str) and label.strip():
            labels.append(label.strip().rstrip("。.!！"))
    if labels:
        structure = " → ".join(labels)
        third = localized(language,
            f"可以按“{structure}”的框架整理答案，每项一句。已经答对的内容保留，只补还缺的部分。",
            f"You can organize your answer as: {structure}. Use one sentence per point. Keep what you already answered correctly and add only the missing part.")
    else:
        third = localized(language,
            "可以按“题目已给信息 → 当前问题的要点 → 依据或需要核对的信息”的框架整理答案，每项一句。只写与当前问题有关的内容，未提供的信息保持未知。",
            "You can organize your answer as: supplied information → the point required by this task → its basis or information to check. Use one sentence per point, stay within this task, and keep unavailable information unknown.")
        reasons.append("missing_published_criterion_framework")
    hints.append(localized(language, "**组织答案：** ", "**Organize your answer:** ") + third)
    return hints, list(dict.fromkeys(reasons))


def guided_hint_text(body, level, language):
    """Wrap exactly one hint with encouragement and its three-hint budget."""
    if type(level) is not int or not 1 <= level <= 3:
        raise ValueError("Subjective hint level must be 1, 2 or 3.")
    encouragement = {
        1: ("没关系，我们一步一步来。", "It is okay to be unsure. Let's take one step at a time."),
        2: ("还没理顺也没关系，我们再看一个提示。", "It is okay if it is not clear yet. Let's look at one more hint."),
        3: ("可以慢慢来，我们再梳理一次。", "Take your time. Let's organize the answer once more."),
    }[level]
    parts = [localized(language, *encouragement),
             localized(language, f"**第 {level}/3 次提示 · 还剩 {3 - level} 次**",
                       f"**Hint {level}/3 · {3 - level} remaining**"), str(body).strip()]
    if level == 3:
        parts.append(localized(language,
            "如果仍答不上来，可以说“还是不会”或查看参考答案。",
            "If you are still stuck, say 'still stuck' or view the reference answer."))
    return "\n\n".join(parts)
