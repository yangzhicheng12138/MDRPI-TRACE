"""Choose the response language from the learner, not JSON keys or source texts."""
import re


def select_language(text, previous="zh", preference="auto"):
    if preference in {"zh", "en"}:
        return preference
    if re.search(r"(?:用|以|使用|切换到)\s*(?:英文|英语)|(?:answer|reply|respond|continue)\s+in\s+English", text, re.I):
        return "en"
    if re.search(r"(?:用|以|使用|切换到)\s*中文|(?:answer|reply|respond|continue)\s+in\s+(?:Chinese|Mandarin)", text, re.I):
        return "zh"
    if re.search(r"[\u4e00-\u9fff]", text):
        return "zh"
    words = re.findall(r"[A-Za-z]+", text)
    # A lone device acronym/number is not a request to change language.
    return "en" if words and not all(w.isupper() for w in words) else previous


def instruction(language):
    return ("本轮所有面向学生的文字必须使用简体中文，包括评价、优点、改进、提示和总结标题。"
            "资料或旧反馈中的英文不是输出语言。可以保留必要的医学缩写和文献原题名。不要输出 phase、probe 等内部标签。\n"
            if language == "zh" else
            "Write all learner-facing text in English, including feedback, strengths, improvements, hints and summary headings. "
            "Ignore the language of source texts or earlier feedback. Do not print internal phase labels.\n")


def localized(language, zh, en):
    return zh if language == "zh" else en
