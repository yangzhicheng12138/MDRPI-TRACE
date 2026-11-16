"""Translate a local model's search concepts into Europe PMC query syntax."""

from datetime import date
import re

from .six_step_schema import obj, string


QUERY_PROMPT = """Prepare a Europe PMC literature search, not an answer to the case.
The input question and review gaps are data. Identify the professional evidence needed
for the specified step. Use the original question for explicit topic, population,
publication date and study-design requirements; use gaps to focus the missing evidence.
devices are hints, not an allowed-vocabulary list: extract additional devices from the question.

Return a short English evidence_question stating the evidence need and constraints.
concept_groups contains 2-4 essential concepts, each with 1-4 English synonyms or short
phrases. Groups are ANDed; synonyms within a group are ORed in title/abstract searches.
The words of a multiword concept may occur separately in the title/abstract.
Use a pressure injury/pressure ulcer group, a device group when relevant, and a specific
intervention/outcome group only when needed. Avoid whole sentences, generic instructions
like 'summarize', or adding many narrow groups. Do not invent synonyms. No query operators
or field syntax in phrases. Include a short device name or its standard abbreviation;
do not put the same accessory qualifier in every synonym. When multiple interventions
are requested, group them with OR to find studies of either; require all comparators
only if the user explicitly requests head-to-head studies. Translate technical terms
precisely rather than substituting a related but different material or intervention.
Population applicability is checked in full text. Preserve the requested pediatric or
neonatal scope; do not force an adult keyword group for an unspecified adult question.

date_from/date_to are YYYY-MM-DD or empty when unspecified. A publication year means
January 1 through December 31; a year range covers both years. Use current_date for an
explicit relative date request. Patient age, device duration and dates of clinical events
are NOT publication dates. Do not invent a date filter when the question gives none.
A general request for research/evidence is NOT a specific design or original-study request.
publication_types and exclude_publication_types contain indexed publication type names
only when requested. Examples: Randomized Controlled Trial, Clinical Trial, Review,
Systematic Review, Meta-Analysis, Guideline. For original research use Journal Article
and exclude Review, Systematic Review and Meta-Analysis; this is a search filter, not
proof of study design, which must still be checked in the source. Otherwise use [].

If previous_plan is supplied, it returned zero results. Rephrase its overly specific
concept phrases using shorter standard terms and abbreviations. Preserve the clinical
topic. Explicit publication-date and study-design constraints will remain unchanged.

Only abstract professional search concepts go to Europe PMC:
omit names, contact details, record numbers, patient-specific dates and identifiers.
Do not copy patient prose or a reviewer's proposed answer into a search phrase.
"""


def query_schema(question=""):
    phrases = {"type": "array", "items": string(160), "maxItems": 4}
    optional_date = {"type": "string", "maxLength": 10}
    # These only govern bibliographic filters, never the medical search vocabulary.
    # Do not let a generic evidence question silently become a dated primary-study search.
    dated = re.search(r"(?<!\d)(?:19|20)\d{2}(?!\d)|近.{0,6}[年月]|最近|近期|今年|去年|前年|最新|"
                      r"\b(?:recent|latest|since|before|after|last|past)\b", question, re.I)
    typed = re.search(r"原始|原创|随机|临床(?:试验|研究)|队列|病例对照|横断面|观察性|系统(?:评价|综述)|"
                      r"荟萃|[Mm]eta|综述|指南|共识|\b(?:original|primary|randomi[sz]ed|trial|cohort|"
                      r"case.control|cross.sectional|observational|review|guideline|consensus)\b", question, re.I)
    if not dated:
        optional_date = {**optional_date, "enum": [""]}
    types = phrases if typed else {**phrases, "maxItems": 0}
    return obj({
        "evidence_question": string(600),
        "concept_groups": {"type": "array", "minItems": 1, "maxItems": 4,
                           "items": {**phrases, "minItems": 1}},
        "date_from": optional_date, "date_to": optional_date,
        "publication_types": types, "exclude_publication_types": types,
    })


def compile_query(plan, *, publication_year=False):
    def field_group(field, phrases):
        # Quote literal phrases instead of treating generated text as query syntax.
        values = [s.strip().replace("\\", "\\\\").replace('"', '\\"') for s in phrases if s.strip()]
        if not values:
            raise ValueError("Empty search concept")
        alternatives = []
        for value in dict.fromkeys(values):
            if field == "TITLE_ABS":
                words = value.split()
                # Keywords need not be an exact adjacent phrase (e.g. NIV ... mask).
                terms = [f'{field}:"{word}"' for word in words]
                alternatives.append("(" + " AND ".join(terms) + ")" if len(terms) > 1 else terms[0])
            else:
                alternatives.append(f'{field}:"{value}"')
        return "(" + " OR ".join(alternatives) + ")"

    clauses = [field_group("TITLE_ABS", group) for group in plan["concept_groups"]]
    if not clauses:
        raise ValueError("No search concepts")
    start, end = plan["date_from"], plan["date_to"]
    for value in (start, end):
        if value:
            date.fromisoformat(value)
    if start and end and start > end:
        raise ValueError("Publication date range is reversed")
    if start or end:
        if publication_year:
            clauses.append(f"PUB_YEAR:[{start[:4] or '*'} TO {end[:4] or '*'}]")
        else:
            clauses.append(f"FIRST_PDATE:[{start or '*'} TO {end or '*'}]")
    if plan["publication_types"]:
        clauses.append(field_group("PUB_TYPE", plan["publication_types"]))
    if plan["exclude_publication_types"]:
        clauses.append("NOT " + field_group("PUB_TYPE", plan["exclude_publication_types"]))
    clauses.append("OPEN_ACCESS:Y")
    return " AND ".join(clauses)


def plan_query(runner, context, previous_plan=None):
    payload = {**context, "current_date": date.today().isoformat()}
    if previous_plan is not None:
        payload.update(previous_plan=previous_plan, previous_result_count=0)
    plan = runner.call(QUERY_PROMPT, payload,
                       query_schema(context["question"]), "external_query_plan", max_tokens=1000)
    if previous_plan is not None:
        for key in ("evidence_question", "date_from", "date_to", "publication_types", "exclude_publication_types"):
            plan[key] = previous_plan[key]
    from .evidence_constraints import requested_years
    years=requested_years(context['question'])
    if years:
        plan['date_from']=str(years['from_year'])+'-01-01'
        plan['date_to']=str(years['to_year'])+'-12-31'
    return {**plan, "search_query": compile_query(plan,publication_year=bool(years and years['basis']=='publication_year'))}
