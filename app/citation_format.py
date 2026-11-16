"""Reader-facing references; evidence identifiers remain in structured results."""

import re
import json
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote, unquote, urldefrag, urlsplit


REFERENCE_STANDARD = "GB/T 7714-2025"


@lru_cache(maxsize=1)
def _display_reviews():
    path = Path(__file__).resolve().parents[1] / "data/manifest/reference_display_overrides.json"
    return json.loads(path.read_text(encoding="utf-8"))["items"] if path.exists() else {}


def reference_metadata(source):
    """Flatten supplied metadata; never fill absent bibliographic facts."""
    result = {}
    for field in ("meta", "metadata"):
        if isinstance(source.get(field), dict):
            result.update({k: v for k, v in source[field].items() if v not in (None, "")})
    result.update({k: v for k, v in source.items() if k not in {"meta", "metadata"} and v not in (None, "")})
    for field, aliases in {"title": ("paper_title", "file_name"), "source_url": ("url", "publisher_url", "oa_url"),
                           "year": ("publication_year",)}.items():
        if not result.get(field):
            result[field] = next((result[k] for k in aliases if result.get(k)), "")
    # Earlier Europe PMC records stored the journal in organization. Restrict
    # compatibility to identifiable journal records, not arbitrary institutions.
    if (not result.get("journal") and result.get("source_type") == "journal_article"
            and (result.get("pmcid") or result.get("pmid"))):
        result["journal"] = result.get("organization", "")
    # Correct verified display facts without editing audited corpus payloads.
    identity = str(result.get("doc_id") or result.get("chunk_id") or result.get("evidence_id") or "")
    match = re.match(r"^(DOC\d+)(?:_|$)", identity)
    review = _display_reviews().get(match[1]) if match else None
    supplied_hash = result.get("source_pdf_sha256")
    if review:
        known_title = result.get("title") in review["original_titles"] + [review["fields"]["title"]]
        if (supplied_hash and supplied_hash != review["source_pdf_sha256"]) or (not supplied_hash and not known_title):
            review = None
    if not review:
        review = next((item for item in _display_reviews().values() if result.get("title") in item["original_titles"]
                       and (not supplied_hash or supplied_hash == item["source_pdf_sha256"])), None)
    if review:
        result.update(review["fields"])
        result["display_metadata_verification"] = review["verification"]
    return result


def _doi(source):
    source = reference_metadata(source)
    value = re.sub(r"^doi:\s*", "", str(source.get("doi") or "").strip(), flags=re.I)
    candidate = value or str(source.get("source_url") or source.get("url") or "")
    try:
        parts = urlsplit(candidate)
        if parts.scheme in {"http", "https"} and parts.hostname in {"doi.org", "dx.doi.org"}:
            return unquote(parts.path.lstrip("/"))
    except ValueError:
        pass
    return value


def _url(source):
    source = reference_metadata(source)
    for key in ("source_url", "url", "publisher_url", "oa_url"):
        url = str(source.get(key) or "").strip()
        try:
            parts = urlsplit(url)
            if parts.scheme in {"http", "https"} and parts.hostname and not parts.username and not parts.password and not re.search(r'[\s<>"\\]', url):
                return urldefrag(url)[0]
        except ValueError:
            pass
    return ""


def original_url(source):
    source = reference_metadata(source)
    if source.get("display_metadata_verification") and _url(source):
        return _url(source)
    doi = _doi(source)
    if re.fullmatch(r'10\.\d{4,9}/[^\s<>"\\]+', doi, re.I):
        return "https://doi.org/" + quote(doi, safe="/")
    url = _url(source)
    if url:
        return url
    source = reference_metadata(source)
    if re.fullmatch(r"\d+", str(source.get("pmid", ""))):
        return f"https://pubmed.ncbi.nlm.nih.gov/{source['pmid']}/"
    if re.fullmatch(r"PMC\d+", str(source.get("pmcid", "")), re.I):
        return f"https://pmc.ncbi.nlm.nih.gov/articles/{str(source['pmcid']).upper()}/"
    return ""


def _identities(source):
    source = reference_metadata(source)
    keys = []
    for field in ("doc_id", "pmid", "pmcid"):
        if source.get(field):
            keys.append((field, str(source[field]).strip().lower()))
    if _doi(source):
        keys.append(("doi", _doi(source).lower()))
    if _url(source):
        keys.append(("url", _url(source)))
    return keys or [("evidence_id", str(source.get("evidence_id") or source.get("chunk_id") or id(source)))]


def merge_reference_sources(sources):
    """Union shared identifiers, including excerpts that bridge two groups."""
    parents = list(range(len(sources)))
    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index
    identities = {}
    for index, raw in enumerate(sources):
        for key in _identities({"evidence_id": f"reference:{index}", **reference_metadata(raw)}):
            if key in identities:
                before, current = root(identities[key]), root(index)
                parents[max(before, current)] = min(before, current)
            identities[key] = index
    merged = {}
    for index, raw in enumerate(sources):
        group = root(index)
        row = merged.setdefault(group, {"source": {}, "positions": [], "locations": []})
        source = reference_metadata(deepcopy(raw))
        for key, value in source.items():
            if not row["source"].get(key) and value not in (None, ""):
                row["source"][key] = value
        row["positions"].append(index + 1)
        location = source.get("location") or (f"PDF第{source['page']}页" if source.get("page") else "")
        if location and location not in row["locations"]:
            row["locations"].append(location)
    return list(merged.values())



def numbered_references(result, step_order):
    """Deduplicate cited documents, numbering by first appearance in the answer."""
    sources = {s["evidence_id"]: s for s in result.get("citations", [])}
    entries = list(sources.items())
    merged = merge_reference_sources([source for _, source in entries])
    groups = [row["source"] for row in merged]
    by_evidence = {entries[position - 1][0]: group for group, row in enumerate(merged) for position in row["positions"]}
    numbered, references, numbers = {}, [], {}
    for step_id in step_order:
        step = result["six_steps"][step_id]
        for claim in step["claims"]:
            for eid in claim["evidence_ids"]:
                if eid not in by_evidence:
                    continue
                group = by_evidence[eid]
                if group not in numbered:
                    numbered[group] = len(references) + 1
                    references.append((numbered[group], groups[group]))
    for eid, group in by_evidence.items():
        if group in numbered:
            numbers[eid] = numbered[group]
            if sources[eid].get("doc_id"):
                numbers[sources[eid]["doc_id"]] = numbered[group]
    return numbers, references


def display_text(text, numbers):
    """Translate internal source labels that a model echoed in prose or caveats."""
    known = "|".join(re.escape(k) for k in sorted(numbers, key=len, reverse=True))
    internal = r"DOC\d+(?:_P\d+_C\d+(?:/C\d+)*)?|EXT_(?:WEB_[A-Za-z0-9]+|PMC\d+_[A-Za-z0-9]+)"
    pattern = re.compile(r"(?<![A-Za-z0-9_])(?:" + internal + ("|" + known if known else "") + r")(?![A-Za-z0-9_])")

    def label(match):
        token = match.group()
        number = numbers.get(token) or numbers.get(token.split("/")[0])
        return f"[{number}]" if number else "相关来源"

    text = re.sub(r"/v1/sources/(DOC\d+)(?:#page=\d+)?",
                  lambda m: f"[{numbers[m[1]]}]" if m[1] in numbers else "原文", text)
    # Remove only the outer brackets of an existing internal citation group.
    text = re.sub(r"\[([^\[\]\n]+)\]",
                  lambda m: pattern.sub(label, m[1]) if pattern.search(m[1]) else m[0], text)
    return pattern.sub(label, text)


def _clean(value):
    return re.sub(r"\s+", " ", str(value or "").replace("\ufeff", "")).strip().rstrip(".。")


def _authors(source):
    value = source.get("authors") or (source.get("organization") if reference_type(source) != "J" else "") or ""
    if isinstance(value, list):
        names = []
        for person in value:
            if isinstance(person, dict):
                family, given = _clean(person.get("family")), _clean(person.get("given"))
                if re.search(r"[\u4e00-\u9fff]", family + given):
                    names.append(family + given)
                else:
                    initials = " ".join("-".join(part[0].upper() for part in re.split(r"[-‐‑]", word) if part)
                                        for word in given.split() if word)
                    names.append((family + " " + initials).strip() or _clean(person.get("literal")))
            else:
                names.append(_clean(person))
    elif isinstance(value, str):
        names = re.split(r"\s*[;；]\s*", value) if re.search(r"[;；]", value) else [value]
        if len(names) == 1 and re.search(r",\s+[^,]+\s+[A-Z](?:[ A-Z.]*)$", value):
            names = re.split(r",\s*", value)
    else:
        names = []
    names = [_clean(name) for name in names if _clean(name) and
             _clean(name).casefold() not in {"admin", "administrator", "作者", "author", "unknown", "f-043"}]
    suffix = ""
    if len(names) > 3:
        suffix = ", " + ("等" if any(re.search(r"[\u4e00-\u9fff]", name) for name in names) else "et al")
        names = names[:3]
    return ", ".join(names) + suffix


def reference_type(source):
    supplied = source.get("document_type") or source.get("reference_type")
    if supplied in {"M", "J", "R", "S", "EB", "C", "D", "N", "DS", "PP", "Z"}:
        return supplied
    kind = source.get("source_type", "")
    if kind in {"journal_article", "original_research", "systematic_review", "meta_analysis", "scoping_review", "integrative_review"} or source.get("journal"):
        return "J"
    return {"web_page": "EB", "website": "EB", "book": "M", "standard": "S", "report": "R",
            "guideline": "R", "consensus": "R", "dissertation": "D", "conference_paper": "C",
            "dataset": "DS", "preprint": "PP"}.get(kind, "Z")


def reference_quality(source):
    """Describe supplied bibliographic gaps, not the clinical quality of a source.

    Author absence can be legitimate; only an explicit anonymous designation
    makes it intentional here. Volume and issue are conditional, never invented.
    """
    source = reference_metadata(source)
    kind = reference_type(source)
    missing = []
    if not _clean(source.get("title")):
        missing.append("题名")
    if kind == "S":
        if not source.get("standard_number"):
            missing.append("标准编号")
    else:
        if not _authors(source) and source.get("author_status") != "anonymous":
            missing.append("作者/责任机构")
        if not (source.get("year") or source.get("publication_date") or source.get("modified_date")):
            missing.append("年份/发布日期")
    if kind == "J":
        if not source.get("journal"):
            missing.append("期刊名")
        if not (source.get("pages") or source.get("article_number") or source.get("publication_status") == "online_first"):
            missing.append("页码或文章编号")
    if kind == "EB" and not (source.get("retrieved_at") or source.get("accessed_at")):
        missing.append("引用日期")
    warnings = []
    if kind == "Z" and source.get("document_type") != "Z":
        warnings.append("文献类型待核对")
    if "title_from_filename_verify" in str(source.get("notes", "")) and not source.get("display_metadata_verification"):
        warnings.append("题名来自文件名，尚未核对")
    return {"status": "incomplete" if missing or warnings else "available",
            "missing": missing, "warnings": warnings}


def reference_for_display(source):
    """Keep incomplete citations visibly qualified in answer text as in cards."""
    value = format_reference(source)
    quality = reference_quality(source)
    if quality["status"] == "incomplete":
        details = (["缺少" + "、".join(quality["missing"])] if quality["missing"] else []) + quality["warnings"]
        value += " （题录不完整：" + "；".join(details) + "）"
    return value


def format_reference(source):
    """GB/T 7714-2025 sequential-reference form for supplied bibliographic data.

    Unknown facts stay absent. Excerpt page numbers never become article pages.
    The standard text is referenced in docs/reference_format_gbt7714_2025.md.
    """
    source = reference_metadata(source)
    title = _clean(source.get("title")) or "题名待补充"
    kind = reference_type(source)
    # 2025 §8.9.2: standard number, title, type/carrier, access path.
    # Issuer and release/effective dates belong to the explanatory card.
    author = "" if kind == "S" else _authors(source)
    if kind == "S" and source.get("standard_number") and str(source["standard_number"]) not in title:
        title = _clean(source["standard_number"]) + " " + title
    if kind == "R" and source.get("report_number"):
        title += ": " + _clean(source["report_number"])
    url, doi = _url(source), _doi(source)
    valid_doi = bool(re.fullmatch(r'10\.\d{4,9}/[^\s<>"\\]+', doi, re.I))
    if not url and valid_doi:
        url = original_url(source)
    online = bool(url or valid_doi or source.get("pmid") or source.get("pmcid"))
    marker = kind + ("/OL" if online else "")
    text = (author + ". " if author else "") + title + "[" + marker + "]."
    year = _clean(source.get("year"))
    if kind == "M" and source.get("container_title"):
        editors = _authors({"authors": source.get("container_authors", [])})
        text = text.rstrip(".") + "//" + (editors + ". " if editors else "") + _clean(source["container_title"]) + "."
    if kind == "M" and source.get("edition"):
        text += " " + _clean(source["edition"]) + "."
    if kind == "J":
        journal = _clean(source.get("journal"))
        publication = journal
        if source.get("publication_status") == "online_first" and source.get("publication_date"):
            year = _clean(source["publication_date"])
        if year:
            publication += (", " if publication else "") + year
        volume, issue, pages = [_clean(source.get(k)) for k in ("volume", "issue", "pages")]
        pages = pages or _clean(source.get("article_number"))
        if volume:
            publication += (", " if publication else "") + volume
        if issue:
            publication += "(" + issue + ")"
        if pages:
            publication += (": " if publication else "") + pages.replace("–", "-").replace("—", "-")
        if publication:
            text += " " + publication + "."
    elif kind == "EB":
        created = _clean(source.get("publication_date") or source.get("modified_date"))
        if created:
            text += " (" + created[:10] + ")"
        elif year:
            text += " " + year
    elif kind != "S":
        place, publisher = [_clean(source.get(k)) for k in ("publication_place", "publisher")]
        publication = place + (": " if place and publisher else "") + publisher
        publication_date = _clean(source.get("publication_date")) if kind == "R" else ""
        if publication_date or year:
            publication += (", " if publication else "") + (publication_date or year)
        if source.get("pages"):
            publication += (": " if publication else "") + _clean(source["pages"]).replace("–", "-")
        if publication:
            text += " " + publication + "."
    accessed = _clean(source.get("retrieved_at") or source.get("accessed_at"))
    if online and accessed:
        text = (text if kind == "EB" else text.rstrip(".")) + "[" + accessed[:10] + "]."
    if url:
        text += " " + url + "."
    if valid_doi and doi.casefold() not in unquote(url).casefold():
        text += " DOI: " + doi + "."
    elif not valid_doi and (source.get("pmid") or source.get("pmcid")):
        locator = original_url(source)
        if locator and locator != url:
            text += " " + locator + "."
    elif doi and not valid_doi:
        # A malformed supplied identifier is visible as unconfirmed metadata,
        # never turned into an invented DOI hyperlink.
        text += " 待核对DOI: " + doi + "."
    return text.rstrip(".") + "."
