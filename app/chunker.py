"""Structure-aware, page-traceable chunk construction."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from .schemas import ChunkRecord, DocumentRecord


_HEADING_KEYWORDS = (
    "recommendation",
    "recommendations",
    "implementation",
    "background",
    "introduction",
    "methods",
    "results",
    "discussion",
    "conclusion",
    "risk factors",
    "prevention",
    "assessment",
    "guideline",
    "建议",
    "推荐",
    "实施",
    "背景",
    "引言",
    "方法",
    "结果",
    "讨论",
    "结论",
    "危险因素",
    "预防",
    "评估",
)


@dataclass
class _Block:
    page: int
    section: str
    text: str


@dataclass
class _Piece:
    page: int
    section: str
    text: str


def _join_lines(lines: list[str]) -> str:
    output = ""
    for line in lines:
        if not output:
            output = line
            continue
        # Chinese text generally does not need an inserted space; Latin words do.
        if re.search(r"[\u4e00-\u9fff]$", output) or re.match(r"^[，。；：！？、)）】】]", line):
            output += line
        else:
            output += " " + line
    return output.strip()


def is_heading(line: str) -> bool:
    """Recognise common PDF headings without treating ordinary short text as one."""

    line = re.sub(r"\s+", " ", line).strip()
    line = re.sub(r"^(\d+(?:\.\d+)*)\s*\|\s*", r"\1 ", line)
    if not line or len(line) > 140:
        return False
    # A number alone, percentage, date, CI bound or bibliography entry is not
    # a section heading. Require a separator and a short heading-like label.
    numbered = re.match(r"^(?:\d+(?:\.\d+)*(?:[.)、]\s*|\s+)|[一二三四五六七八九十]+[、.]\s*)(.+)$", line)
    if numbered:
        label = numbered.group(1).strip()
        if (len(label) <= 65 and not re.search(r"[%％=。；;]|\b(?:CI|et\s+al)\b", label, re.I)
                and not re.match(r"[\d,.–—−-]", label)
                and (re.search(r"[\u4e00-\u9fff]", label) or label.istitle() or label.isupper()
                     or (label[:1].isupper() and len(label.split()) <= 8 and not re.search(r"[,.]", label))
                     or label.casefold() in _HEADING_KEYWORDS)):
            return True
    # Chinese clinical articles frequently use labels such as
    # “临床问题1：…”, “推荐意见：…” and “证据摘要：…”.  They do not start
    # with an Arabic/Chinese section number, so the generic heading rule used
    # to miss them and chunks consequently crossed several unrelated questions.
    if re.match(
        r"^(?:临床问题|研究问题|证据总结|最佳证据|关键点|主要结果|结论要点|护理要点|评估内容|预防措施|风险因素|高危部位)\s*[0-9一二三四五六七八九十]*\s*[:：]?",
        line,
    ):
        return True
    if re.match(r"^(?:第\s*[一二三四五六七八九十百]+\s*[章节部分]|附录|表\s*\d+|图\s*\d+)", line):
        return True
    if line.startswith(("【", "［")) and len(line) <= 120:
        return True
    if line.endswith((":", "：")) and len(line) <= 80:
        return True
    # English headings are often title-cased lines without a trailing colon.
    # For Chinese, keyword substring matches are unsafe: a normal sentence such
    # as "持续评估鼻梁..." must remain body text rather than become a heading.
    lower = line.lower()
    english_words = re.findall(r"[a-z]+", lower)
    if english_words and len(english_words) <= 8 and not re.search(r"[。！？；;]", line):
        title_like = line[:1].isupper() and (line.isupper() or line.istitle())
        if title_like and any(keyword in lower for keyword in _HEADING_KEYWORDS if re.search(r"[a-z]", keyword)):
            return True
    return False


def _page_blocks(page: dict, default_section: str = "未标注章节") -> list[_Block]:
    lines = page.get("text", "").splitlines()
    blocks: list[_Block] = []
    section = default_section
    paragraph: list[str] = []

    def flush() -> None:
        nonlocal paragraph
        text = _join_lines(paragraph)
        if text:
            blocks.append(_Block(page=int(page["page"]), section=section, text=text))
        paragraph = []

    for raw_line in lines:
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            flush()
            continue
        if is_heading(line):
            flush()
            section = line
            # Keep headings in the embedded text as well as metadata.  Query
            # terms often occur only in a clinical-question/section heading.
            paragraph.append(line)
        else:
            paragraph.append(line)
    flush()
    return blocks


def _split_long_text(text: str, max_chars: int) -> list[str]:
    # Whitespace and CJK character boundaries are safe; a decimal, signed
    # value, unit or Latin word must never be cut to satisfy a size target.
    boundaries = [match.end() for match in re.finditer(r"\s+|(?<=[\u4e00-\u9fff\u3002\uff0c\uff1b\uff1a])|(?=[\u4e00-\u9fff])", text)]
    output = []
    start = 0
    while len(text) - start > max_chars:
        candidates = [end for end in boundaries if start < end <= start + max_chars]
        end = max(candidates) if candidates else next((end for end in boundaries if end > start + max_chars), len(text))
        if text[start:end].strip():
            output.append(text[start:end].strip())
        start = end
    if text[start:].strip():
        output.append(text[start:].strip())
    return output


def _split_sentences(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    # Periods inside decimal values and DOI identifiers are not sentence ends.
    sentences = [item.strip() for item in re.split(r"(?<=[。！？!?；;])\s*|(?<=\.)\s+(?=[A-Z])", text) if item.strip()]
    if not sentences:
        return _split_long_text(text, max_chars)
    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > max_chars:
            pieces.append(current.strip())
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current.strip())
    # A single unbroken sentence can still exceed the target.
    output: list[str] = []
    for piece in pieces:
        if len(piece) <= max_chars:
            output.append(piece)
        else:
            output.extend(_split_long_text(piece, max_chars))
    return output


def _pieces(blocks: Iterable[_Block], target_chars: int) -> list[_Piece]:
    pieces: list[_Piece] = []
    for block in blocks:
        for text in _split_sentences(block.text, target_chars):
            pieces.append(_Piece(page=block.page, section=block.section, text=text))
    return pieces


def _tail(text: str, overlap_chars: int) -> str:
    if overlap_chars <= 0:
        return ""
    start = max(0, len(text) - overlap_chars)
    if start and re.match(r"[A-Za-z0-9.%+\-/\u2212\u2013\u2014]", text[start - 1]) and re.match(r"[A-Za-z0-9.%+\-/\u2212\u2013\u2014]", text[start]):
        boundary = re.search(r"\s+|(?=[\u4e00-\u9fff])", text[start:])
        if boundary is None:
            return ""
        start += boundary.end()
    return text[start:]


def _merge_short_assembled(
    assembled: list[tuple[str, set[int], list[str]]],
    target_chars: int,
    minimum_chars: int,
) -> list[tuple[str, set[int], list[str]]]:
    """Merge tiny layout/table fragments into a neighbouring evidence unit.

    PDF block extraction often yields one-character bullets, table cells, or
    wrapped English fragments.  Indexing each as an independent vector creates
    high-scoring noise.  We merge only within the same detected section and
    keep the original page set; fragments that cannot be safely merged remain
    traceable and can be excluded by the audit policy.
    """

    if minimum_chars <= 0 or not assembled:
        return assembled
    output: list[tuple[str, set[int], list[str]]] = []
    for text, pages, sections in assembled:
        if output and len(text) < minimum_chars and sections == output[-1][2] and min(pages) <= max(output[-1][1]) + 1:
            previous_text, previous_pages, previous_sections = output[-1]
            if len(previous_text) + len(text) + 1 <= target_chars + minimum_chars:
                output[-1] = (
                    f"{previous_text}\n{text}".strip(),
                    previous_pages | pages,
                    previous_sections,
                )
                continue
        output.append((text, set(pages), list(sections)))

    # A short first fragment may not have had a preceding compatible unit.  Try
    # one forward merge pass, again never crossing a section boundary.
    index = 0
    while index < len(output) - 1:
        text, pages, sections = output[index]
        next_text, next_pages, next_sections = output[index + 1]
        if len(text) < minimum_chars and sections == next_sections and min(next_pages) <= max(pages) + 1 and len(text) + len(next_text) + 1 <= target_chars + minimum_chars:
            output[index] = (f"{text}\n{next_text}".strip(), pages | next_pages, sections)
            del output[index + 1]
            continue
        index += 1
    return output


def build_chunks(
    document: DocumentRecord,
    pages: list[dict],
    target_chars: int = 700,
    overlap_chars: int = 100,
    merge_short_chars: int = 80,
) -> list[ChunkRecord]:
    """Build deterministic chunks while retaining section and page provenance."""

    if target_chars <= 0 or overlap_chars < 0 or overlap_chars >= target_chars:
        raise ValueError("Require target_chars > 0 and 0 <= overlap_chars < target_chars")
    if merge_short_chars < 0:
        raise ValueError("merge_short_chars must be non-negative")

    blocks: list[_Block] = []
    section = "未标注章节"
    previous_page: int | None = None
    for page in pages:
        if previous_page is not None and int(page["page"]) != previous_page + 1:
            section = "未标注章节"
        page_blocks = _page_blocks(page, section)
        if page_blocks:
            section = page_blocks[-1].section
            blocks.extend(page_blocks)
        previous_page = int(page["page"])
    pieces = _pieces(blocks, target_chars)
    if not pieces:
        return []

    assembled: list[tuple[str, set[int], list[str]]] = []
    current_text = ""
    current_pages: set[int] = set()
    current_sections: list[str] = []
    # One source page per character. Separators use None. This stays bounded
    # by chunk size and lets overlaps retain only the pages actually copied.
    current_origins: list[int | None] = []
    previous_piece_page: int | None = None
    for piece in pieces:
        # Do not carry overlap or body text across a genuine section/clinical
        # question boundary.  Cross-section overlap was a major source of
        # chunks containing two unrelated recommendations.
        gap = previous_piece_page is not None and piece.page > previous_piece_page + 1
        if current_text and current_sections and (piece.section != current_sections[-1] or gap):
            assembled.append((current_text.strip(), current_pages, current_sections))
            current_text = ""
            current_pages = set()
            current_sections = []
            current_origins = []
        candidate = piece.text if not current_text else f"{current_text}\n{piece.text}"
        if current_text and len(candidate) > target_chars:
            # If overlap copies the entire short prefix, emitting it separately
            # makes the later short-fragment merge duplicate that prefix.
            if len(current_text) > overlap_chars:
                assembled.append((current_text.strip(), current_pages, current_sections))
            tail = _tail(current_text, overlap_chars)
            origins = current_origins[-len(tail):] if tail else []
            raw = f"{tail}\n{piece.text}" if tail else piece.text
            raw_origins = origins + [None] + [piece.page] * len(piece.text) if tail else [piece.page] * len(piece.text)
            left = len(raw) - len(raw.lstrip())
            current_text = raw.strip()
            current_origins = raw_origins[left:left + len(current_text)]
            current_pages = {p for p in current_origins if p is not None}
            current_sections = [piece.section]
        else:
            current_origins += ([None] if current_text else []) + [piece.page] * len(piece.text)
            current_text = candidate
            current_pages.add(piece.page)
            if piece.section not in current_sections:
                current_sections.append(piece.section)
        previous_piece_page = piece.page
    if current_text:
        assembled.append((current_text.strip(), current_pages, current_sections))

    # Concatenating already-overlapping chunks would repeat the copied text
    # inside a single evidence unit and break source contiguity.
    if not overlap_chars:
        assembled = _merge_short_assembled(assembled, target_chars, merge_short_chars)

    chunks: list[ChunkRecord] = []
    for index, (text, pages_seen, sections) in enumerate(assembled, start=1):
        page_start = min(pages_seen) if pages_seen else None
        page_end = max(pages_seen) if pages_seen else None
        page_token = f"{page_start:04d}" if page_start is not None else "0000"
        chunk_id = f"{document.doc_id}_P{page_token}_C{index:03d}"
        chunks.append(
            ChunkRecord(
                chunk_id=chunk_id,
                doc_id=document.doc_id,
                text=text,
                title=document.title,
                authors=document.authors,
                year=document.year,
                organization=document.organization,
                source_type=document.source_type,
                evidence_tier=document.evidence_tier,
                population=document.population,
                device_type=document.device_type,
                topic=document.topic,
                section=" / ".join(sections) if sections else "未标注章节",
                page_start=page_start,
                page_end=page_end,
                page_verified=bool(pages_seen) and all(
                    page.get("page_verified", True) for page in pages if int(page["page"]) in pages_seen
                ),
                source_pages=sorted(pages_seen),
                source_pdf_sha256=document.sha256,
                preprocessing_warnings=sorted({warning for page in pages if int(page["page"]) in pages_seen
                                               for warning in page.get("preprocessing_warnings", [])}),
                kb_version=document.kb_version,
                language=document.language,
                include=document.include,
                review_status=document.review_status,
                filename=document.filename,
                char_count=len(text),
            )
        )
    return chunks
