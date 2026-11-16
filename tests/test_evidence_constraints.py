"""Observed retrieval failures: negation, dates, article identity and unsafe sources."""
from dataclasses import replace
from pathlib import Path

import pytest

from app.config import AppConfig
from app.evidence_constraints import (deduplicate_documents, eligibility, requested_years,
    request_constraints, source_identifiers, year_decision)
from app.evidence_router import local_batch
from app.evidence_snapshot import RunBudget, RunLog
from app.query_policy import query_scope, recognition_text
from test_rag import evidence, pipeline, config


@pytest.mark.parametrize('question,people,devices',[
 ('早产儿鼻CPAP鼻部损伤，不要直接套用成人面罩证据。',['neonatal'],{'cpap'}),
 ('儿童气管插管损伤，不能只给成人无创通气面罩资料。',['pediatric'],{'endotracheal_tube'}),
 ('成人NIV压力性损伤，排除儿童研究。',['adult'],{'niv_mask'}),
 ('Find neonatal CPAP pressure injury evidence, do not use adult NIV studies.',['neonatal'],{'cpap'}),
 ('儿童压力性损伤，而非成人气管切开问题。',['pediatric'],set()),
 ('不要成人面罩证据而是查找新生儿CPAP压力性损伤研究。',['neonatal'],{'cpap'}),
 ('Do not use adult NIV studies but find pediatric ETT pressure injury studies.',['pediatric'],{'endotracheal_tube'}),
])
def test_negative_evidence_scope_does_not_become_target(question,people,devices):
    cfg=AppConfig(Path('.'),{'filters':{'validated_populations':['adult','pediatric','neonatal'],
                                      'strict_population_isolation':True}})
    before=question
    scope=query_scope(question,cfg)
    assert not scope.reason and scope.population==people and set(scope.devices)==devices
    assert question==before


def test_positive_age_and_inclusive_comparisons_are_preserved():
    cfg=AppConfig(Path('.'),{'filters':{'validated_populations':['adult','pediatric','neonatal'],
                                      'strict_population_isolation':True}})
    assert query_scope('对比成人和儿童压力性损伤',cfg).reason=='population_scope_ambiguous'
    assert query_scope('不仅成人还包括儿童压力性损伤',cfg).reason=='population_scope_ambiguous'
    assert '成人' in recognition_text('不要忽略成人证据')
    assert '无创通气面罩' in recognition_text('患者68岁，鼻梁发红，不能摘掉无创通气面罩。')
    assert query_scope('患者，68岁，鼻梁红斑按压不褪色，无创通气4小时。',cfg).population==['adult']
    assert query_scope('患儿10岁气管插管，不能仅给成人资料。',cfg,population=['adult']).reason=='query_filter_mismatch'


@pytest.mark.parametrize('question,years',[
 ('请找2024至2026年发表的新生儿CPAP系统综述',(2024,2026)),
 ('Find published studies between 2024-2026',(2024,2026)),
 ('只要2025年发表的指南',(2025,2025)),
 ('查找2020年以前发表的压力性损伤研究',(1900,2019)),
 ('近3年发表的相关资料',(2024,2026)),
 ('T/CNAS 34-2023标准原文',None),
 ('核对SECURE第二版年份，文件名2024',None),
 ('指南检索截止2022年6月，不限定发表年份',None),
 ('患者68岁，使用面罩4小时',None),
])
def test_explicit_dates_are_not_identifiers_or_source_cutoffs(question,years):
    value=requested_years(question,current_year=2026)
    assert ((value['from_year'],value['to_year']) if value else None) == years


def test_date_basis_and_unknown_date_are_not_assumed():
    q='2024至2026年发表的系统综述'
    years=requested_years(q)
    assert not year_decision({'year':2022},years)['eligible']
    assert not year_decision({'publication_date':'2025-01-01','publication_date_source':'search_result'},years)['eligible']
    s={'bibliographic_year':2024,'first_publication_date':'2023-12-15'}
    assert year_decision(s,years)['eligible']
    assert not year_decision(s,requested_years('2024至2026年首次发表的文献'))['eligible']


def test_local_date_filter_runs_before_ranking_and_keeps_source_text():
    current=replace(evidence(),year=2025)
    old=replace(evidence('DOC002_P0001_C001'),year=2022)
    engine=pipeline(items=[old,current])
    r=engine.retrieve('2024至2026年发表的NIV面罩压力性损伤皮肤评估研究')
    assert [s['year'] for s in r['evidence']]==[2025]
    assert r['evidence'][0]['text']==current.text and engine.last_retrieval_candidates[0].year==2025
    assert r['date_constraints']['rejected_candidates'][0]['year']==2022


def test_performance_followup_retains_original_date_constraint(tmp_path):
    engine=pipeline(items=[replace(evidence(),year=2022),replace(evidence('DOC002_P0001_C001'),year=2025)])
    found=local_batch(engine,['adult NIV prevention'],['niv_mask'],['adult'],RunBudget(),RunLog(tmp_path),
                      original_question='只要2024至2026年发表的NIV研究')
    assert {s['year'] for s in found.values()}=={2025}


def test_mirror_identity_uses_doi_pmid_not_url_and_preserves_local_passages():
    a={'source_url':'https://pubmed.ncbi.nlm.nih.gov/40426095/','doi':'10.1186/s12887-025-05700-x','title':'Study A'}
    b={**a,'source_url':'https://example.org/mirror'}
    c={'source_url':'https://other.org/paper','doi':'10.1007/other','title':'Study B'}
    assert len(deduplicate_documents([a,b,c]))==2
    assert source_identifiers({'url':'https://s2.ipubmed.cn/40426095/'})['pmid']=='40426095'
    passages=[{'doc_id':'DOC001','chunk_id':'a','origin':'local'}, {'doc_id':'DOC001','chunk_id':'b','origin':'local'}]
    assert len(deduplicate_documents(passages,retain_local_passages=True))==2


def test_medical_upload_lists_wrong_population_are_not_eligible_evidence():
    question='儿童PICU气管插管损伤研究，不能只给成人无创通气资料。'
    for s in [{'source_url':'https://www.renrendoc.com/paper/123','title':'儿童指南解读'},
              {'source_url':'https://pubmed.ncbi.nlm.nih.gov/?term=author','title':'Search results'},
              {'source_url':'https://journal.org/a','title':'新生儿NICU器械压力性损伤研究'}]:
        assert not eligibility(s,question)['eligible']
    assert eligibility({'source_url':'https://www.renrendoc.com/paper/123','title':'Cooking lesson'},'找烹饪课件')['eligible']
    assert request_constraints('请找Device-related pressure ulcers: SECURE prevention. Second edition的原文')['title'].endswith('edition')
