"""Original-request scope, date bounds and document identity shared by modes."""
from datetime import datetime, timezone
import hashlib
import re
from urllib.parse import unquote, urlsplit

from .query_policy import (ADULT, CHILD, NEONATAL, DEVICE_PATTERNS, DOMAIN,
                           inferred_population, recognition_text)

YEAR = r"(?:19|20)\d{2}"


def requested_years(question, *, current_year=None):
    """Parse explicit literature bounds; source years/standard numbers are not filters."""
    if re.search(r"不(?:限|限定|限制|设).{0,8}(?:发表|出版|年份|时间)|no date limit|without date restriction", question, re.I):
        return None
    basis = "first_publication" if re.search(r"首次|最早|首发|在线发表|first publi|online publi", question, re.I) else "publication_year"
    span = re.search(r"(?<![\d/])("+YEAR+r")\s*年?\s*(?:至|到|[-–—~～]|to\b|through\b)\s*("+YEAR+r")\s*年?", question, re.I)
    if span and re.search(r"发表|出版|文献|研究|综述|指南|资料|published|publication|studies|papers|articles|between", question, re.I):
        lo, hi = int(span[1]), int(span[2])
        return {"from_year": min(lo, hi), "to_year": max(lo, hi), "basis": basis}
    single = re.search(r"(?<![\d/-])("+YEAR+r")\s*年?\s*(?:发表|出版|published)|(?:发表于|出版于|published in)\s*("+YEAR+r")", question, re.I)
    if single:
        year = int(single[1] or single[2])
        return {"from_year": year, "to_year": year, "basis": basis}
    cutoff = re.search(r"(?<![\d/-])("+YEAR+r")\s*年\s*(及以前|及以后|以后|之后|以来|以前|之前|前|后)", question)
    if cutoff:
        year, word = int(cutoff[1]), cutoff[2]
        after = word in {"及以后", "以后", "之后", "以来", "后"}
        return {"from_year": year if after else 1900,
                "to_year": 2099 if after else year if word == "及以前" else year - 1, "basis": basis}
    recent = re.search(r"(?:近|最近)\s*([1-9]\d?)\s*年.{0,20}(?:发表|出版|文献|研究|综述|指南|资料)|last\s+([1-9]\d?)\s+years?", question, re.I)
    if recent:
        year = current_year or datetime.now(timezone.utc).year
        return {"from_year": year - int(recent[1] or recent[2]) + 1, "to_year": year, "basis": basis}
    return None


def publication_year(source, constraint=None):
    basis = (constraint or {}).get("basis")
    if basis == "first_publication":
        value = source.get("first_publication_date")
    else:
        value = source.get("bibliographic_year") or source.get("year") or source.get("publication_year")
        if not value and source.get("publication_date_source") in {"publisher_metadata", "registry_metadata"}:
            value = source.get("publication_date")
    match = re.match(r"^("+YEAR+r")", str(value or ""))
    return int(match[1]) if match else None


def year_decision(source, constraint):
    if not constraint:
        return {"eligible": True, "reason": "no_explicit_date_bound"}
    year = publication_year(source, constraint)
    return {"eligible": year is not None and constraint["from_year"] <= year <= constraint["to_year"],
            "year": year, "constraint": constraint,
            "reason": "publication_year_unverified" if year is None else "publication_year_in_range"
                      if constraint["from_year"] <= year <= constraint["to_year"] else "publication_year_out_of_range"}


def normalized_title(value):
    return re.sub(r"[\W_]", "", value).casefold().replace("的", "")


def requested_title(question):
    quoted = re.search(r"《([^》]{8,300})》", question)
    if quoted:
        return quoted[1]
    english = re.search(r"(?:请找|查找|寻找|find|locate)\s*([A-Za-z][A-Za-z0-9 :;,.'’()\-–]{18,280})", question, re.I)
    return english[1].strip(" ,.;") if english and re.search(r"edition|:",english[1],re.I) else ""


def request_constraints(question):
    text = recognition_text(question)
    devices = [d for d,p in DEVICE_PATTERNS.items() if re.search(p, text, re.I)]
    medical = bool(DOMAIN.search(text) or devices)
    return {"population": inferred_population(question), "devices": devices, "medical": medical,
            "years": requested_years(question), "title": requested_title(question),
            "original_requested": bool(re.search(r"原文|原始定义|原始研究|original (?:article|paper|definition)|full.?text", text, re.I)),
            "full_text_requested": bool(re.search(r"阅读全文|全文内容|full.?text (?:content|read)|read (?:the )?full", text, re.I))}


def source_identifiers(source):
    url = str(source.get("source_url") or source.get("url") or "")
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    text = str(source.get("text") or source.get("body_content") or "")
    head = re.split(r"\n\s*(?:#{1,4}\s*)?(?:References|参考文献)\b", text, maxsplit=1, flags=re.I)[0][:1800]
    doi = str(source.get("doi") or "").lower().removeprefix("https://doi.org/")
    if not doi:
        # DOI-bearing publisher URLs use /article/10..., /articles/10...
        # or /pdf/10... as well as /doi/10...; query parameters are not DOI.
        m=re.search(r"/(10\.\d{4,9}/[^\s<>\"?#]+)",unquote(parts.path),re.I)
        if m:doi=m[1].lower().rstrip('/.,;。)')
        if doi and re.search(r'/(?:pdf|epdf|pdfdirect)/10\.',parts.path,re.I):
            doi=re.sub(r'\.pdf$','',doi)
    if not doi:
        m = re.search(r"(?:doi\s*[:：]|doi\.org/|/doi/(?:abs/|full/|pdf/)?)\s*(10\.\d{4,9}/[^\s<>\"\[\]]+)", unquote(url)+"\n"+head, re.I)
        if m: doi = m[1].lower().rstrip(".,;。)")
    pmid = str(source.get("pmid") or "")
    if not pmid and host in {"pubmed.ncbi.nlm.nih.gov", "s2.ipubmed.cn", "www.ipubmed.cn"}:
        m = re.fullmatch(r"/(\d+)/?", parts.path)
        if m: pmid = m[1]
    pmcid = str(source.get("pmcid") or "").upper()
    if not pmcid:
        m = re.search(r"/(PMC\d+)(?:/|$)", parts.path, re.I)
        if m: pmcid = m[1].upper()
    standard = re.search(r"T\s*/?\s*CNAS\s*(\d+)\s*[-—─–]\s*(20\d{2})", str(source.get("title") or "")+"\n"+head, re.I)
    result = {k:v for k,v in {"doi":doi,"pmid":pmid,"pmcid":pmcid}.items() if v}
    if standard: result["standard_id"] = "T/CNAS " + standard[1] + "-" + standard[2]
    return result


def document_keys(source):
    ids = source_identifiers(source)
    keys = {k+":"+v.lower() for k,v in ids.items()}
    if source.get("doc_id"): keys.add("doc:"+source["doc_id"])
    title = normalized_title(str(source.get("article_title") or source.get("title") or ""))
    if len(title) >= 24 and not ids.get('standard_id'): keys.add("title:"+title)
    text = str(source.get("text") or source.get("body_content") or "")
    abstract = re.search(r"(?:^|\n)\s*(?:Objective|Objectives|目的)\s*[:：](.{250,})", text, re.S|re.I)
    if abstract:
        value = normalized_title(abstract[0])[:420]
        if len(value)>=250: keys.add("abstract:"+hashlib.sha256(value.encode()).hexdigest())
    if not keys:
        url = str(source.get("source_identity") or source.get("source_url") or source.get("url") or "")
        if url: keys.add("url:"+url.split("#",1)[0])
    return keys


def quoted_scope_matches(quotes, question):
    """Quoted examples about another device cannot establish direct support."""
    target=request_constraints(question)
    text='\n'.join(quotes)
    wanted=set(target['devices'])
    observed={d for d,p in DEVICE_PATTERNS.items() if re.search(p,text,re.I)}
    compatible=bool(wanted & observed or wanted <= {'cpap','niv_mask'} and observed <= {'cpap','niv_mask'})
    if wanted and observed and not compatible:return False
    people=inferred_population(text)
    if target['population'] in [['pediatric'],['neonatal']] and people==['adult']:return False
    if target['population']==['pediatric'] and people==['neonatal'] and not re.search(r'PICU|儿童|pediatric|paediatric|children',text,re.I):return False
    return True


def deduplicate_documents(sources, *, retain_local_passages=False):
    result, seen = [], set()
    for source in sources:
        if retain_local_passages and source.get("origin") != "external" and source.get("chunk_id"):
            keys = {"chunk:"+source["chunk_id"]}
        else:
            keys = document_keys(source)
        if not keys & seen:
            result.append(source)
        seen.update(keys)
    return result


def source_kind(source):
    parts = urlsplit(str(source.get("source_url") or source.get("url") or ""))
    host, path = (parts.hostname or "").lower(), parts.path.lower()
    if any(host == d or host.endswith("."+d) for d in ["renrendoc.com","book118.com","docin.com","wenku.baidu.com","doc88.com"]):
        return "unverified_upload"
    if (host == "pubmed.ncbi.nlm.nih.gov" and path == "/" and parts.query
            or re.search(r"showdownloadtoplist|teamstandards|/(?:search|searchresults)(?:/|$)", path, re.I)):
        return "navigation_list"
    if source.get("material_level") == "search_snippet": return "search_snippet"
    return source.get("document_kind") or "content_page"


def eligibility(source, question):
    """Hard exclusions complement model review; they never assert clinical support."""
    constraints = request_constraints(question)
    kind = source_kind(source)
    if constraints["medical"] or constraints["original_requested"]:
        if kind in {"unverified_upload","navigation_list","search_snippet"}:
            return {"eligible":False,"reason":kind}
        target = constraints["population"]
        title = str(source.get("article_title") or source.get("title") or "")
        text = title + "\n" + str(source.get("text") or source.get("body_content") or "")[:1600]
        actual = source.get("population") or inferred_population(title)
        if target == ["pediatric"] and (actual == ["neonatal"] or NEONATAL.search(title) and not re.search(r"PICU|儿童|pediatric|paediatric|children",title,re.I)):
            return {"eligible":False,"reason":"neonatal_evidence_for_pediatric_request"}
        if target in [["pediatric"],["neonatal"]] and actual == ["adult"]:
            return {"eligible":False,"reason":"adult_evidence_for_child_request"}
        if target == ['adult'] and actual in [['pediatric'],['neonatal']]:
            return {'eligible':False,'reason':'child_only_evidence_for_adult_request'}
        if target == ['neonatal'] and actual == ['pediatric'] and not NEONATAL.search(text):
            return {'eligible':False,'reason':'non_neonatal_evidence_for_neonatal_request'}
        actual_devices=source.get('device_type') or [d for d,p in DEVICE_PATTERNS.items() if re.search(p,title,re.I)]
        wanted=set(constraints['devices']); observed=set(actual_devices)
        # CPAP interface articles often use both nasal-prong and NIV labels.
        compatible=bool(wanted & observed or wanted <= {'cpap','niv_mask'} and observed <= {'cpap','niv_mask'})
        if wanted and observed and not compatible:
            return {'eligible':False,'reason':'device_mismatch'}
        if constraints["title"]:
            expected = normalized_title(constraints["title"])
            if expected and expected not in normalized_title(text[:1000]):
                return {"eligible":False,"reason":"requested_document_not_matched"}
    date = year_decision(source, constraints["years"])
    return date if not date["eligible"] else {"eligible":True,"reason":"candidate_requires_support_review","kind":kind}
