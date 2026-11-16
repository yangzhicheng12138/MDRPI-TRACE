"""Shared rejection of access screens and grounding of search query dates."""
import re
from urllib.parse import urlsplit


def access_failure(url, text=""):
    path = urlsplit(url).path.rstrip("/").lower()
    if path in {"/login", "/signin", "/sign-in", "/auth/login"}:
        return "page_redirected_to_login"
    if re.search(r"(?:^|/)(?:captcha|wappoc_appmsgcaptcha)(?:/|$)|/cdn-cgi/challenge", path):
        return "page_access_challenge"
    if len(text) < 2000 and re.search(
        r"verify (?:that )?you are human|verify you.re human|checking your browser|"
        r"请完成.{0,8}人机验证|当前环境异常.{0,40}完成验证|完成验证后即可继续访问",
        " ".join(text.split()), re.I,
    ):
        return "page_access_challenge"
    return None


YEAR = r"(?<!\d)(?:19|20)\d{2}(?!\d)"
# Month/day suffixes must not consume the first two digits of another year.
DATE_SUFFIX = r"(?:[-/.]\d{1,2}(?!\d)(?:[-/.]\d{1,2}(?!\d))?|年(?:\d{1,2}月(?:\d{1,2}日)?)?)?"
DATE = re.compile(
    r"(?:(?:before|after):\s*|(?:before|after|since|until|through)\s+)?"
    + YEAR + DATE_SUFFIX +
    r"(?:\s*(?:[-–—~～]|至|到|to\b|through\b)\s*" + YEAR + DATE_SUFFIX + r")?"
    r"(?:以前|之前|以后|之后|以来|前|后)?", re.I,
)


def grounded_query_plan(plan, question, limit=500):
    """Local publication cutoffs must never silently become user constraints."""
    years = set(re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", question))
    changes = []
    result = dict(plan)
    for field in ("search_query", "verification_focus"):
        identifiers = [m.span() for m in re.finditer(r"T\s*/?\s*CNAS\s*\d+\s*[-—–]\s*20\d{2}",result[field],re.I)]
        def replace(match):
            if any(start <= match.start() < end for start,end in identifiers):
                return match[0]
            if set(re.findall(YEAR, match[0])) <= years:
                return match[0]
            changes.append("removed_unrequested_date:" + match[0])
            return " "
        result[field] = " ".join(DATE.sub(replace, result[field]).split()).strip(" ,，;；")
    if changes:
        result["verification_focus"] = (
            "依据原问题核对来源及适用范围，发表时间限制以用户要求为准。"
            if re.search(r"[\u4e00-\u9fff]", question) else
            "Check sources against the original question; use only publication-date limits requested by the user.")
    if len(result["search_query"]) > limit:
        result["search_query"] = result["search_query"][:limit].rstrip()
        changes.append("query_length_limited")
    return result, list(dict.fromkeys(changes))
