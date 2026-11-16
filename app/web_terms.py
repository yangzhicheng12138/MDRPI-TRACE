"""Shared lexical terms for English and Chinese public page excerpts."""
import re


def web_terms(text):
    terms = set(re.findall(r"[a-z]{4,}", text.casefold()))
    terms.update(re.findall(r"\b(?:niv|icu|cpap)\b", text.casefold()))
    for run in re.findall(r"[\u4e00-\u9fff]+", text):
        terms.update(run[i:i + 2] for i in range(len(run) - 1))
    return terms - {"with", "from", "that", "this", "adult", "adults",
                    "如何", "什么", "哪些", "一下", "相关", "进行", "可以", "我们"}
