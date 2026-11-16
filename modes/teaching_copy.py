"""Remove known developer review labels from learner-facing teaching copies.

This pure display helper has no project imports. It never rewrites stored
messages, authored answers, review metadata or model reference context.
"""
import re


_REPLACEMENTS = (
    (" · AI审核通过", ""),
    (" · AI-reviewed", ""),
    ("（AI审核通过）", ""),
    ("（AI审核）", ""),
    (" (AI-reviewed)", ""),
    ("审核：Codex。依据：", "依据："),
    ("Reviewed by Codex. Source basis:", "Source basis:"),
    ("AI-reviewed reference for the Chinese source case:", "Reference for the Chinese source case:"),
    ("AI-reviewed reference for the Chinese exercise:", "Reference for the Chinese exercise:"),
    ("AI-reviewed reference for the Chinese case:", "Reference for the Chinese case:"),
    ("Codex已按题面与所列来源核对参考答案和六步判定要求；AI审核通过，参考内容用于教学练习。原始题面与原文件未改写。", ""),
    ("AI生成·仅供练习·尚未逐题审核", "AI生成·仅供练习"),
    ("AI-generated · practice only · not individually reviewed", "AI-generated · practice only"),
)


def _clean_segment(text):
    for original, replacement in _REPLACEMENTS:
        text = text.replace(original, replacement)
    return text


def clean_teaching_copy(text, preserve=()):
    """Clean exact system labels while preserving quoted learner wording/code.

    Callers select teaching assistant/display or formal reference output. This
    function deliberately does not infer that arbitrary mentions of AI, review,
    generation or a named reviewer are removable.
    """
    if not isinstance(text, str) or not text:
        return text
    protected = [(match.start(), match.end()) for pattern in (
        r"(?m)^[ \t]*>[^\n]*(?:\n[ \t]*>[^\n]*)*",
        r"(?ms)^[ \t]*```[^\n]*\n.*?^[ \t]*```[^\n]*$",
        r"`[^`\n]+`",
    ) for match in re.finditer(pattern, text)]
    for value in preserve:
        if not isinstance(value, str) or not value:
            continue
        if not any(original in value for original, _ in _REPLACEMENTS):
            continue  # Short answers such as "A" must not split a system label.
        start = 0
        while True:
            index = text.find(value, start)
            if index < 0:
                break
            protected.append((index, index + len(value)))
            start = index + len(value)
    ranges = []
    for start, end in sorted(protected):
        if ranges and start <= ranges[-1][1]:
            ranges[-1] = (ranges[-1][0], max(end, ranges[-1][1]))
        else:
            ranges.append((start, end))
    parts, cursor = [], 0
    for start, end in ranges:
        parts.extend((_clean_segment(text[cursor:start]), text[start:end]))
        cursor = end
    parts.append(_clean_segment(text[cursor:]))
    return "".join(parts)
