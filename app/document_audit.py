"""Document-level quality checks used by the clean retrieval build.

The first ingestion pass treated every PDF as one logical article.  In journal
downloads a PDF can occasionally contain an adjacent article (or a cover page
followed by a different article), so page-level chunk checks alone are not
enough.  This module records explainable boundary signals without changing the
source PDF or the original parsed/chunk files.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any


_TRANSLATION = str.maketrans(
    {
        "：": ":",
        "／": "/",
        "．": ".",
        "－": "-",
        "–": "-",
        "—": "-",
        "　": " ",
        "０": "0",
        "１": "1",
        "２": "2",
        "３": "3",
        "４": "4",
        "５": "5",
        "６": "6",
        "７": "7",
        "８": "8",
        "９": "9",
    }
)

_DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\]\)；;，,。]+", re.I)
_ABSTRACT_RE = re.compile(r"(?:【\s*摘要\s*】|［\s*摘要\s*］|\[\s*摘要\s*\])")
_EN_ABSTRACT_RE = re.compile(r"\babstract\b", re.I)
_GENRE_RE = re.compile(
    r"(?:指南|专家共识|Meta分析|系统评价|最佳证据|最佳实践|meta-analysis|systematic review|clinical practice guideline|original article)",
    re.I,
)


def normalise_bibliographic(text: str) -> str:
    """Normalise full-width punctuation and whitespace for comparisons."""

    value = unicodedata.normalize("NFKC", text or "").translate(_TRANSLATION)
    value = value.replace("\u200b", "").replace("\xad", "")
    return re.sub(r"\s+", " ", value).strip()


def extract_dois(text: str) -> list[str]:
    """Extract de-duplicated DOI-like identifiers from one page."""

    normalised = normalise_bibliographic(text)
    values: list[str] = []
    for match in _DOI_RE.findall(normalised):
        value = re.sub(r"[.,;:。；，)\]}]+$", "", match).casefold()
        if value and value not in values:
            values.append(value)
    return values


def _tokens(text: str) -> set[str]:
    value = normalise_bibliographic(text).casefold()
    cjk = re.findall(r"[\u4e00-\u9fff]", value)
    cjk_tokens = {"".join(cjk[index : index + 2]) for index in range(max(0, len(cjk) - 1))}
    words = set(re.findall(r"[a-z]{3,}", value))
    return {token for token in cjk_tokens | words if token}


def title_similarity(left: str, right: str) -> float:
    """Compute a language-agnostic token Jaccard similarity."""

    a, b = _tokens(left), _tokens(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def extract_title_candidate(text: str) -> str:
    """Find a likely article title near the beginning of a page.

    This intentionally returns an empty string when only a running header or a
    section label is present.  It is used as a boundary signal, not as a
    bibliographic authority.
    """

    lines = [normalise_bibliographic(line) for line in str(text).splitlines() if line.strip()]
    if not lines:
        return ""
    # Ignore common running headers and stop once the abstract body starts.
    candidates: list[str] = []
    for line in lines[:35]:
        if _ABSTRACT_RE.search(line) or re.match(r"^abstract\s*(?:objective|background|purpose|methods)?", line, re.I):
            break
        if len(line) < 10 or len(line) > 180:
            continue
        if re.search(r"(?:doi|作者单位|作者简介|通信作者|基金项目|received|accepted|corresponding|e-mail|万方数据|downloaded)", line, re.I):
            continue
        if re.fullmatch(r"(?:中华护理杂志|护理学杂志|journal[^\d]*|vol(?:ume)?\.?\s*\d+.*)", line, re.I):
            continue
        # A line made mostly of page/header metadata is not an article title.
        if len(re.findall(r"\d", line)) > max(6, len(line) // 3):
            continue
        candidates.append(line)
    if not candidates:
        return ""
    # Prefer a line that explicitly looks like an article title.  Otherwise
    # use the first substantive line (the usual location in Chinese journals).
    for candidate in candidates:
        if _GENRE_RE.search(candidate):
            return candidate
    return candidates[0]


def _page_boundary_signal(page: dict[str, Any], expected_title: str, primary_dois: set[str]) -> dict[str, Any] | None:
    page_number = int(page.get("page", 0) or 0)
    text = str(page.get("text", ""))
    top = text[:2400]
    title = extract_title_candidate(top)
    page_dois = set(extract_dois(top))
    cn_abstract = bool(_ABSTRACT_RE.search(top))
    en_abstract = bool(_EN_ABSTRACT_RE.search(top) and re.search(r"\b(?:objective|background|purpose|methods)\b", top, re.I))
    genre = bool(_GENRE_RE.search(title))
    similarity = title_similarity(title, expected_title) if title else 0.0
    differing_doi = bool(page_dois and primary_dois and not page_dois.issubset(primary_dois))

    reasons: list[str] = []
    score = 0
    # A DOI on a references page is expected and must not by itself imply a
    # second article.  Treat it as corroborating evidence only when an
    # article-start marker is present.
    if differing_doi and (cn_abstract or en_abstract) and title:
        reasons.append("different_doi_from_first_page")
        score += 1
    if cn_abstract and title and similarity < 0.45:
        reasons.append("unrelated_title_before_chinese_abstract")
        score += 4
    elif en_abstract and title and similarity < 0.20 and genre and differing_doi:
        reasons.append("unrelated_title_before_english_abstract")
        score += 4
    if title and genre and similarity < 0.25 and (cn_abstract or en_abstract):
        reasons.append("low_title_similarity")
        score += 1
    if score < 3:
        return None
    return {
        "page": page_number,
        "score": score,
        "title_candidate": title,
        "title_similarity": round(similarity, 4),
        "page_dois": sorted(page_dois),
        "reasons": reasons,
    }


def detect_boundary_pages(expected_title: str, pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return pages that likely begin a different article.

    A page is flagged only when there are multiple independent signals (for
    example a new article title/abstract and a DOI different from page 1).
    This avoids mistaking ordinary section headings, tables of contents, or
    references for a document boundary.
    """

    first_text = str(pages[0].get("text", "")) if pages else ""
    primary_dois = set(extract_dois(first_text))
    signals: list[dict[str, Any]] = []
    for page in pages[1:]:
        signal = _page_boundary_signal(page, expected_title, primary_dois)
        if signal:
            signals.append(signal)
    return signals


def is_placeholder(value: str) -> bool:
    """Whether a manifest title/author is a known placeholder."""

    return (value or "").strip().casefold() in {
        "标题",
        "作者",
        "title",
        "author",
        "unknown",
        "unk",
        "lippincott williams and wilkins",
        "thomson",
    }


def title_from_filename(filename: str) -> str:
    """Create a readable fallback title without inventing article metadata."""

    stem = Path(filename).stem
    stem = re.sub(r"^\d+_(?:19|20)\d{2}_", "", stem)
    stem = re.sub(r"\.{2,}", " ", stem)
    stem = stem.replace("_", " ")
    return re.sub(r"\s+", " ", stem).strip()
