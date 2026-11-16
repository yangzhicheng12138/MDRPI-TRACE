"""Page-preserving PDF text extraction using PyMuPDF."""

from __future__ import annotations

import re
import statistics
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import fitz

from .schemas import PageRecord


MIN_TEXT_CHARS_FOR_PAGE = 30
PARSER_VERSION = "pdf-parser-v6.2-body-numbers"


def normalise_page_text(text: str) -> str:
    """Remove extraction noise while preserving paragraph and line boundaries."""

    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def _compact(text: str) -> str:
    """Return a whitespace-free representation for layout heuristics."""

    return re.sub(r"\s+", "", text or "").strip()


def _text_blocks(page: Any) -> list[dict[str, Any]]:
    """Extract text blocks with coordinates from a PyMuPDF page.

    ``page.get_text("text", sort=True)`` sorts individual lines by their
    vertical coordinate.  On a two-column article this interleaves the left and
    right columns (line 1 left, line 1 right, line 2 left, ...), which is
    disastrous for retrieval.  Blocks retain enough geometry to reconstruct a
    reading order without introducing another PDF dependency.
    """

    blocks: list[dict[str, Any]] = []
    flags = fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_IMAGES
    for raw in page.get_text("dict", flags=flags, sort=False)["blocks"]:
        if raw.get("type") != 0:
            continue
        lines = raw.get("lines", [])
        middle = float(page.rect.width) / 2
        left = [line for line in lines if line["bbox"][2] <= middle]
        right = [line for line in lines if line["bbox"][0] >= middle]
        # Some PDFs put paragraphs from both columns into one text block.
        # Split only when every line lies wholly on one side of the gutter.
        groups = [left, right] if left and right and len(left) + len(right) == len(lines) else [lines]
        for group in groups:
            text = normalise_page_text("\n".join("".join(span["text"] for span in line["spans"]) for line in group))
            if not text:
                continue
            x0 = min(line["bbox"][0] for line in group)
            y0 = min(line["bbox"][1] for line in group)
            x1 = max(line["bbox"][2] for line in group)
            y1 = max(line["bbox"][3] for line in group)
            blocks.append({
                "x0": float(x0), "y0": float(y0), "x1": float(x1), "y1": float(y1),
                "width": float(x1 - x0), "height": float(y1 - y0), "text": text,
            })
            if len(group) == 1:
                spans = [span for span in group[0]["spans"] if span["text"].strip()]
                # The dominant span avoids using a raised citation or a tiny
                # spacer as the baseline of an ordinary text fragment.
                dominant = max(spans, key=lambda span: len(span["text"]))
                blocks[-1].update(line_baseline=float(dominant["origin"][1]),
                                  line_font_size=float(dominant["size"]))
    return blocks


def _order_column_fragments(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Read fragments sharing a native PDF baseline from left to right."""
    ordered = sorted(blocks, key=lambda item: (item["y0"], item["x0"]))
    output, row = [], []
    for block in ordered:
        same_line = (row and "line_baseline" in block and "line_baseline" in row[0]
                     and abs(block["line_baseline"] - row[0]["line_baseline"])
                     <= min(block["line_font_size"], row[0]["line_font_size"]) * 0.25)
        if row and not same_line:
            output.extend(sorted(row, key=lambda item: item["x0"]))
            row = []
        row.append(block)
    output.extend(sorted(row, key=lambda item: item["x0"]))
    return output


def _is_broad_block(block: dict[str, Any], page_width: float) -> bool:
    """Whether a block spans the page/gutter rather than one column."""

    width = float(block["width"])
    x0, x1 = float(block["x0"]), float(block["x1"])
    return width >= page_width * 0.72 or (x0 < page_width * 0.44 and x1 > page_width * 0.56)


def _column_groups(blocks: list[dict[str, Any]], page_width: float) -> list[list[dict[str, Any]]]:
    """Cluster narrow blocks by their left x-coordinate.

    Require sustained, wide text in both columns. Single words, indented
    keywords and figure labels do not establish additional columns.
    """

    narrow = [block for block in blocks if not _is_broad_block(block, page_width)
              and block["width"] >= page_width * 0.22]
    if not narrow:
        return []
    groups: list[list[dict[str, Any]]] = []
    for block in sorted(narrow, key=lambda item: (item["x0"], item["y0"])):
        if not groups:
            groups.append([block])
            continue
        median_x = statistics.median(item["x0"] for item in groups[-1])
        if float(block["x0"]) - median_x > page_width * 0.12:
            groups.append([block])
        else:
            groups[-1].append(block)
    groups = [group for group in groups if len(group) >= 2
              and sum(len(item["text"]) for item in group) >= 150]
    if len(groups) < 2:
        return []
    # Small table cells and rotated publisher notices must not introduce
    # extra columns or invalidate the two principal text columns.
    groups = sorted(sorted(groups, key=lambda group: sum(len(item["text"]) for item in group), reverse=True)[:2],
                    key=lambda group: statistics.median(item["x0"] for item in group))

    # A column model is only credible when the vertical ranges overlap.  A
    # page with a narrow title followed by one-column body text otherwise gets
    # misclassified as two columns.
    ranges = [
        (min(item["y0"] for item in group), max(item["y1"] for item in group))
        for group in groups
    ]
    overlap = max(0.0, min(item[1] for item in ranges) - max(item[0] for item in ranges))
    if overlap < 30.0:
        return []
    return groups


def order_page_blocks(page: Any) -> list[dict[str, Any]]:
    """Return page text blocks in a column-aware reading order.

    Full-width title/abstract blocks form a top region.  Once two sustained
    columns begin, blocks are emitted left column then right column, while any
    full-width table/figure block is inserted at its vertical position.  The
    original coordinates are retained for downstream cleaning/auditing.
    """

    return order_layout_blocks(_text_blocks(page), float(page.rect.width))


def order_layout_blocks(blocks: list[dict[str, Any]], page_width: float) -> list[dict[str, Any]]:
    """Apply the same reading order to native PDF blocks and OCR line boxes."""
    if not blocks:
        return []
    groups = _column_groups(blocks, page_width)
    if not groups:
        return _order_column_fragments(blocks)

    # The right column may start higher than the abstract in the left column.
    # Neither column's first paragraph belongs ahead of the other column.
    body_start = min(min(item["y0"] for item in group) for group in groups)
    top = [item for item in blocks if item["y0"] < body_start]
    rest = [item for item in blocks if item["y0"] >= body_start]
    columns: list[list[dict[str, Any]]] = [[] for _ in groups]
    broad: list[dict[str, Any]] = []
    medians = [statistics.median(item["x0"] for item in group) for group in groups]
    for item in rest:
        if _is_broad_block(item, page_width):
            broad.append(item)
            continue
        index = min(range(len(medians)), key=lambda value: abs(float(item["x0"]) - medians[value]))
        columns[index].append(item)
    columns = [_order_column_fragments(column) for column in columns]

    ordered = sorted(top, key=lambda item: (item["y0"], item["x0"]))
    cursors = [0] * len(columns)
    for wide in sorted(broad, key=lambda item: (item["y0"], item["x0"])):
        for index, column in enumerate(columns):
            while cursors[index] < len(column) and column[cursors[index]]["y0"] < wide["y0"]:
                ordered.append(column[cursors[index]])
                cursors[index] += 1
        ordered.append(wide)
    for index, column in enumerate(columns):
        ordered.extend(column[cursors[index] :])
    return ordered


_RETRIEVAL_BOILERPLATE = (
    re.compile(r"^doi\s*[:：]", re.I),
    re.compile(r"^(?:基金项目|作者单位|作者机构|作者简介|通信作者|作者贡献|利益冲突)"),
    re.compile(r"^(?:corresponding author|author contributions?|conflict[s]? of interest|funding|received|accepted|orcid)\b", re.I),
    re.compile(r"^(?:downloaded from|万方数据|知网|中国知网)", re.I),
    re.compile(r"^(?:收稿日期|修回日期|录用日期)"),
)


def _is_retrieval_boilerplate(text: str, page_height: float | None = None, y0: float | None = None, y1: float | None = None) -> bool:
    """Identify bibliographic/header blocks that should not enter retrieval."""

    compact = re.sub(r"\s+", " ", text or "").strip()
    if not compact:
        return True
    if any(pattern.search(compact) for pattern in _RETRIEVAL_BOILERPLATE):
        return True
    if re.search(r"\bDownloaded from https?://", compact, re.I) and re.search(r"terms and conditions|Wiley Online Library", compact, re.I):
        return True
    if re.match(r"^(?:Copyright\s*(?:\u00a9|\(c\))|\u00a9\s*\d{4})", compact, re.I):
        return True
    if re.fullmatch(r"(?:page\s*\d{1,4}(?:\s+of\s+\d{1,4})?|\d{1,4}\s+of\s+\d{1,4})", compact, re.I):
        return True
    # Bare numbers can be a wrapped range endpoint or a body measurement.
    # Treat them as page labels only when the containing block is in a margin.
    if (re.fullmatch(r"\d{1,4}", compact) and page_height is not None
            and ((y1 is not None and y1 <= 56) or (y0 is not None and y0 >= page_height - 40))):
        return True
    if re.search(r"(?:www\.|https?://).*", compact, re.I) and len(compact) < 260:
        return True
    # Typical running headers contain a journal name plus volume/issue/page;
    # do not remove substantive paragraphs that merely cite a journal.
    if re.search(r"\b(?:vol(?:ume)?\.?\s*\d+|\d{4}\s*,?\s*\d+\s*\(?\d*\)?\s*[:·]|第\s*\d+\s*卷)", compact, re.I) and len(compact) < 180:
        return True
    return False


def _margin_key(block: dict[str, Any], page_height: float) -> str:
    if float(block.get("y1", page_height)) > 56 and float(block.get("y0", 0)) < page_height - 40:
        return ""
    text = re.sub(r"\s+", " ", str(block.get("text", ""))).strip()
    if len(text) > 250 or re.search(r"[.!?\u3002\uff01\uff1f]$", text):
        return ""
    return re.sub(r"\d+", "#", text).casefold()


def normalise_retrieval_text(text: str, vocabulary: set[str] | None = None) -> str:
    """Normalise typography without changing numeric signs, powers or units."""
    ligatures = str.maketrans({"\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl", "\ufb05": "st", "\ufb06": "st"})
    text = text.translate(ligatures).replace("\u200b", "")
    # Some native PDFs encode the indent after a bullet as BEL. The source
    # renders this as spacing; other controls may encode data and stay flagged.
    text = re.sub(r"(•[ \t\n]*)\x07", r"\1", text)
    text = re.sub(r"\u00ad\s*\n\s*(?=[a-z])", "", text).replace("\u00ad", "")

    def join_wrap(match: re.Match) -> str:
        left, right = match.groups()
        # An ordinary hyphen can be meaningful. Remove it only when the
        # unhyphenated word is independently present in this source document.
        word = left + right
        return word if vocabulary and word.casefold() in vocabulary else left + "-" + right

    return re.sub(r"([A-Za-z]{2,})-\s*\n\s*([a-z]{2,})", join_wrap, text).strip()


def layout_review_reasons(blocks: list[dict[str, Any]]) -> list[str]:
    reasons = []
    small_words = sum(bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9,.;:'()/\-\u00ad]{0,34}", block["text"])) for block in blocks)
    if len(blocks) >= 40 and small_words >= len(blocks) * 0.45:
        reasons.append("fragmented_word_layout_needs_review")
    # Plain text has no cell/header association. Until a table has been
    # structurally reviewed, keep the page for review but not for retrieval.
    table_label = re.compile(r"^(?:(?i:ta\s*b\s*le)\s*\d+(?:[.:|\s]|$)|\u8868\s*\d+(?:\D|$))")
    for block in blocks:
        for line in block["text"].splitlines():
            line = re.sub(r"\s+", " ", line).strip()
            if table_label.match(line) and not re.search(r"\b(?:shows|presents|summarizes|summarises|provides|lists)\b", line):
                reasons.append("unstructured_table_needs_review")
                return reasons
    return reasons


@dataclass
class ReferenceState:
    active: bool = False


def is_reference_heading(text: str) -> bool:
    value = unicodedata.normalize("NFKC", text).strip()
    # Singular "Reference" commonly labels a table column, not bibliography.
    return bool(re.fullmatch(r"(?:\d+[.)]?\s+)?(?:参\s*考\s*文\s*献|references|bibliography)\s*[:：]?", value, re.I))


def bibliography_follows(lines: list[str]) -> bool:
    """Require entries after a heading; reject TOC and table column labels."""
    sample = unicodedata.normalize("NFKC", "\n".join(lines[:35])).replace("\u200b", "")
    first = lines[0] if lines else ""
    years = bool(re.search(r"\b(?:19|20)\d{2}\b", sample))
    numbered = bool(re.match(r"^\s*(?:\[\d+\]|\d+[.)])(?:\s|$)", first))
    author = bool(re.match(r"^[A-ZÀ-ž][A-Za-zÀ-ž’' -]+,\s*[A-Z]", first))
    corporate_author = bool(re.match(r"^[A-Z].{1,120}\.\s*\((?:19|20)\d{2}\)", first))
    doi = bool(re.search(r"10\.\d{4,9}/", sample))
    return years and (numbered or author or corporate_author or doi)


def looks_like_reference_page(text: str) -> bool:
    """Corroborate a reference continuation rather than swallowing appendices."""
    years = len(re.findall(r"\b(?:19|20)\d{2}\b", text))
    entries = len(re.findall(r"(?m)^\s*(?:\[\d+\]|\d+[.)])", text))
    text = text.replace("\u200b", "")
    citations = len(re.findall(r"(?:\b(?:19|20)\d{2}[;,]\s*\d|\[J\]|\b(?:Journal|Nurs|Wound)\b)", text, re.I))
    dois = len(re.findall(r"10\.\d{4,9}/", text))
    apa_authors = len(re.findall(r"(?m)^[A-ZÀ-ž][A-Za-zÀ-ž’' -]+,\s*[A-Z]", text))
    return years >= 2 and ((entries >= 2 and citations >= 2) or dois >= 2 or apa_authors >= 2)


def clean_retrieval_blocks(blocks: list[dict[str, Any]], page_height: float | None = None,
                           reference_state: ReferenceState | None = None,
                           running_margins: set[str] | None = None) -> tuple[str, list[str]]:
    """Create a retrieval-safe page view and report removed block text.

    This is intentionally conservative.  It removes bibliographic furniture
    and truncates a page after an explicit reference-list heading, while leaving
    substantive tables and recommendation text available for the chunk audit.
    """

    kept: list[str] = []
    removed: list[str] = []
    state = reference_state if reference_state is not None else ReferenceState()
    all_lines = [line.strip() for block in blocks for line in str(block.get("text", "")).splitlines() if line.strip()]
    line_index = 0
    for block in blocks:
        text = str(block.get("text", "")).strip()
        if not text:
            continue
        # A block can contain several lines (for example an affiliation and an
        # e-mail address).  Evaluate each line separately so that a useful
        # neighbouring sentence is not discarded with bibliographic furniture.
        if (page_height is not None and running_margins
                and _margin_key(block, page_height) in running_margins):
            removed.append(text)
            line_index += len([line for line in text.splitlines() if line.strip()])
            continue
        block_kept: list[str] = []
        block_removed: list[str] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            compact = re.sub(r"\s+", " ", line)
            line_index += 1
            following = all_lines[line_index:line_index + 4]
            if is_reference_heading(compact):
                # Web-to-PDF exports may have a REFERENCES navigation button
                # before the actual article (DOC072). It is not a section.
                navigation = any(re.match(r"^(?:CITE|SHARE|PDF|AUTHOR\(S\))\b", item, re.I) for item in following)
                if not navigation and bibliography_follows(all_lines[line_index:]):
                    state.active = True
                    block_removed.append(line)
                elif navigation:
                    block_removed.append(line)
                else:
                    # Keep ambiguous table labels and TOC entries; do not let
                    # an isolated word switch all subsequent pages to references.
                    block_kept.append(line)
                continue
            if state.active and re.match(r"^(?:appendix|appendices|supplementary (?:material|information)|chapter\s+\d+|附录|第.+章)\b", compact, re.I) and len(compact) < 100:
                state.active = False
            if state.active:
                block_removed.append(line)
                continue
            if _is_retrieval_boilerplate(compact, page_height, block.get("y0"), block.get("y1")):
                block_removed.append(line)
                continue
            block_kept.append(line)
        if block_kept:
            kept.append("\n".join(block_kept))
        if block_removed:
            removed.append("\n".join(block_removed))
    return "\n".join(kept).strip(), removed


def text_quality(text: str) -> dict[str, int | float | bool]:
    """Return explainable quality signals for extracted text.

    Broken embedded fonts commonly produce long runs of symbols and control
    characters while retaining only a small amount of meaningful Chinese,
    English, or numeric text.  The thresholds intentionally require a sizable
    sample so ordinary punctuation in a short page is not called corruption.
    """

    non_whitespace = sum(not character.isspace() for character in text)
    chinese = len(re.findall(r"[\u4e00-\u9fff]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    digits = len(re.findall(r"\d", text))
    semantic = chinese + latin + digits
    controls = sum(
        unicodedata.category(character).startswith("C") and not character.isspace()
        for character in text
    )
    symbols = sum(unicodedata.category(character).startswith(("P", "S")) for character in text)
    replacements = text.count("\ufffd")
    semantic_ratio = semantic / non_whitespace if non_whitespace else 0.0
    symbol_ratio = symbols / non_whitespace if non_whitespace else 0.0
    control_ratio = controls / non_whitespace if non_whitespace else 0.0
    replacement_ratio = replacements / non_whitespace if non_whitespace else 0.0
    is_garbled = bool(
        non_whitespace >= 400
        and (
            replacements >= max(20, non_whitespace * 0.05)
            or (symbol_ratio >= 0.40 and semantic_ratio < 0.65)
            or (control_ratio >= 0.05 and semantic_ratio < 0.65)
            or (semantic_ratio < 0.25)
        )
    )
    return {
        "non_whitespace_characters": non_whitespace,
        "semantic_characters": semantic,
        "chinese_characters": chinese,
        "latin_characters": latin,
        "digit_characters": digits,
        "control_characters": controls,
        "symbol_characters": symbols,
        "replacement_characters": replacements,
        "replacement_ratio": round(replacement_ratio, 6),
        "semantic_ratio": round(semantic_ratio, 6),
        "symbol_ratio": round(symbol_ratio, 6),
        "control_ratio": round(control_ratio, 6),
        "is_garbled": is_garbled,
    }


def inspect_pdf(path: str | Path) -> dict[str, Any]:
    """Read page count, PDF metadata, and a first-page sample for manifest inference."""

    pdf_path = Path(path)
    with fitz.open(pdf_path) as document:
        first_text = (
            "\n".join(block["text"] for block in order_page_blocks(document[0]))
            if document.page_count
            else ""
        )
        metadata = document.metadata or {}
        return {
            "page_count": document.page_count,
            "title": (metadata.get("title") or "").strip(),
            "author": (metadata.get("author") or "").strip(),
            "first_page_text": first_text[:12000],
        }


def parse_pdf(path: str | Path, doc_id: str) -> dict[str, Any]:
    """Extract every page and flag pages that likely require OCR.

    Page numbers are one-based PDF page indices. They are therefore traceable to
    the source file even when printed page labels are unavailable.
    """

    pdf_path = Path(path)
    pages: list[dict[str, Any]] = []
    empty_pages: list[int] = []
    ocr_pages: list[int] = []
    garbled_pages: list[int] = []
    with fitz.open(pdf_path) as document:
        ordered_pages = [order_page_blocks(page) for page in document]
        margin_counts: Counter[str] = Counter()
        for page, blocks in zip(document, ordered_pages):
            margin_counts.update({_margin_key(block, float(page.rect.height)) for block in blocks} - {""})
        running_margins = {key for key, count in margin_counts.items() if count >= 2}
        vocabulary = {word.casefold() for blocks in ordered_pages for block in blocks
                      for word in re.findall(r"\b[A-Za-z]{3,}\b", block["text"])}
        reference_state = ReferenceState()
        for index, (page, ordered_blocks) in enumerate(zip(document, ordered_pages), start=1):
            # Keep the complete, column-corrected text for audit/provenance.
            # ``retrieval_text`` is a derived view with running headers,
            # bibliographic furniture and reference lists removed.  Keeping
            # both views makes the cleaning reversible and auditable.
            text = "\n".join(str(block["text"]) for block in ordered_blocks).strip()
            reference_reset = reference_state.active and bool(text.strip()) and not looks_like_reference_page(text)
            if reference_reset:
                reference_state.active = False
            retrieval_text, removed_blocks = clean_retrieval_blocks(
                ordered_blocks, page_height=float(page.rect.height), reference_state=reference_state,
                running_margins=running_margins,
            )
            retrieval_text_before_normalisation = retrieval_text
            retrieval_text = normalise_retrieval_text(retrieval_text, vocabulary)
            count = len(text)
            needs_ocr = count < MIN_TEXT_CHARS_FOR_PAGE
            quality = text_quality(text)
            page_quality = "blank" if count == 0 else "needs_ocr" if needs_ocr else "ok"
            if bool(quality["is_garbled"]):
                page_quality = "garbled"
                garbled_pages.append(index)
            if count == 0:
                empty_pages.append(index)
            if needs_ocr:
                ocr_pages.append(index)
            page_record = asdict(
                PageRecord(
                    doc_id=doc_id,
                    page=index,
                    text=text,
                    parse_method="pymupdf",
                    needs_ocr=needs_ocr,
                    character_count=count,
                    page_verified=True,
                    quality_status=page_quality,
                )
            )
            page_record["retrieval_text"] = retrieval_text
            page_record["retrieval_text_before_normalisation"] = retrieval_text_before_normalisation
            page_record["preprocessing_warnings"] = layout_review_reasons(ordered_blocks)
            page_record["layout_block_count"] = len(ordered_blocks)
            page_record["layout_blocks"] = [dict(block, block_id=f"P{index:04d}_B{number:04d}") for number, block in enumerate(ordered_blocks)]
            page_record["page_width"] = float(page.rect.width)
            page_record["page_height"] = float(page.rect.height)
            page_record["reference_continuation_reset_needs_review"] = reference_reset
            page_record["reference_section_active_at_end"] = reference_state.active
            page_record["retrieval_removed_blocks"] = removed_blocks
            pages.append(page_record)
    total_chars = sum(int(item.get("character_count", 0)) for item in pages)
    all_text = "\n".join(str(item.get("text", "")) for item in pages)
    quality = text_quality(all_text)
    status = "ok"
    nonempty_pages = len(pages) - len(empty_pages)
    garbled_page_ratio = len(garbled_pages) / max(nonempty_pages, 1)
    if not pages or total_chars == 0 or len(ocr_pages) / max(len(pages), 1) >= 0.5:
        status = "needs_ocr"
    elif bool(quality["is_garbled"]) or garbled_page_ratio >= 0.5:
        status = "garbled"
    return {
        "doc_id": doc_id,
        "source_file": str(pdf_path),
        "parser_version": PARSER_VERSION,
        "normalisation": "ligatures_soft_hyphens_and_source_attested_line_wraps",
        "page_count": len(pages),
        "total_characters": total_chars,
        "empty_pages": empty_pages,
        "blank_pages": empty_pages,
        "ocr_pages": ocr_pages,
        "garbled_pages": garbled_pages,
        "text_quality": quality,
        "parse_status": status,
        "pages": pages,
    }
