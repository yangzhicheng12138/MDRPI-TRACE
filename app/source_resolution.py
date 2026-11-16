"""Resolve discovered identifiers through public registry records, within budget."""
import html
import json
import re
from urllib.parse import urlencode, urlsplit

from .evidence_constraints import (document_keys, eligibility, normalized_title,
                                   request_constraints, source_identifiers, source_kind)
from .external_evidence import BASE, BudgetExceeded, ExternalFailure, validate_public_url


def _literal(value):
    return value.replace('\\', '\\\\').replace('"', '\\"')


class SourceResolver:
    def __init__(self, web):
        self.web, self.cache, self.calls = web, {}, 0

    def _records(self, query, *, discovery=False):
        if query in self.cache:
            return self.cache[query]
        if self.calls >= 3:
            return []
        kind = "search_requests" if discovery else "fetch_requests"
        if not self.web.budget.available(kind):
            return []
        self.calls += 1
        url = BASE + 'search?' + urlencode({'query':query,'format':'json','resultType':'core','pageSize':5})
        self.web._check_cancelled()
        self.web.log.write('source_registry_request', {'query':query,'url':url,'budget_kind':kind})
        try:
            data, _ = self.web.fetcher(url,self.web.budget,kind,validator=validate_public_url,
                                     timeout_seconds=min(8,self.web.budget.remaining()))
            records = json.loads(data).get('resultList',{}).get('result',[])
            if not isinstance(records,list): raise ValueError('Invalid registry records')
            records = [r for r in records if isinstance(r,dict) and isinstance(r.get('title'),str)]
            self.web._check_cancelled()
        except BudgetExceeded:
            raise
        except Exception as exc:
            records=[]
            self.web.log.write('source_registry_failure',{'error_type':type(exc).__name__})
        self.cache[query]=records
        self.web.log.write('source_registry_result', {'query':query,'records':records})
        return records

    def _row(self, record, original, query):
        title = record['title']
        pmid = str(record.get('pmid') or (record.get('id') if record.get('source')=='MED' else '') or '')
        doi = str(record.get('doi') or '')
        pmcid = str(record.get('pmcid') or '')
        if not re.fullmatch(r'\d+',pmid): return None
        url = 'https://pubmed.ncbi.nlm.nih.gov/'+pmid+'/'
        year = str(record.get('pubYear') or '')
        abstract = html.unescape(re.sub(r'<[^>]+>',' ',str(record.get('abstractText') or '')))
        journal = record.get('journalInfo') or {}
        journal_name=(journal.get('journal') or {}).get('title','') if isinstance(journal,dict) else ''
        header = (title+'\n'+str(record.get('authorString') or '')+'\n'+journal_name+'\n'
                  +'Journal publication year: '+year+'\nFirst publication date: '+str(record.get('firstPublicationDate') or '')
                  +'\nPMID: '+pmid+'\nDOI: '+doi)
        body = header+'\n\nAbstract:\n'+abstract if abstract else header
        level = 'article_abstract' if abstract else 'article_record'
        # Original publisher/repository bodies retain their own provenance. A
        # mirror/list can only become a registry URL with the registry's text.
        original_url = str(original.get('url') or '')
        original_host=(urlsplit(original_url).hostname or '').lower()
        mirror=original_host in {'s2.ipubmed.cn','www.ipubmed.cn','library.nih.go.kr','ouci.dntb.gov.ua'}
        url_ids=source_identifiers({'url':original_url})
        original_title=normalized_title(str(original.get('article_title') or original.get('title') or ''))
        registry_title=normalized_title(title)
        title_agrees=bool(registry_title and original_title and (registry_title in original_title or original_title in registry_title))
        own_url=bool(url_ids.get('pmid')==pmid or doi and url_ids.get('doi','').lower()==doi.lower())
        if original.get('body_content') and source_kind(original)!='navigation_list' and not mirror and (own_url or title_agrees):
            url,body=original_url,original['body_content']
            level=original.get('resolved_material_level','web_page_excerpt')
        elif pmcid and record.get('isOpenAccess')=='Y' and self.web.budget.available('fetch_requests'):
            try:
                from .external_evidence import parse_fulltext
                data,full_url=self.web.fetcher(BASE+pmcid+'/fullTextXML',self.web.budget,'fetch_requests',
                    validator=validate_public_url,timeout_seconds=min(8,self.web.budget.remaining()))
                excerpts,event=parse_fulltext(data,record,full_url,query,
                    population=request_constraints(getattr(self.web,'request_question',query))['population'] or None)
                if excerpts:
                    body='\n\n'.join(s['text'] for s in excerpts)[:16000]
                    url=excerpts[0]['source_url'];level='full_text_excerpt'
            except BudgetExceeded: raise
            except Exception as exc:
                self.web.log.write('source_fulltext_unavailable',{'pmid':pmid,'error_type':type(exc).__name__})
        return {**original,'title':title,'article_title':title,'url':url,'body_content':body,
            'doi':doi,'pmid':pmid,'pmcid':pmcid,'bibliographic_year':int(year) if re.fullmatch(r'\d{4}',year) else None,
            'year':int(year) if re.fullmatch(r'\d{4}',year) else None,'source_type':'journal_article',
            'first_publication_date':record.get('firstPublicationDate'),
            'publication_date':record.get('firstPublicationDate'),'publication_date_source':'registry_metadata',
            'authors':record.get('authorString'),'journal':journal_name,
            'volume':journal.get('volume'),'issue':journal.get('issue'),'pages':record.get('pageInfo'),
            'publication_types':(record.get('pubTypeList') or {}).get('pubType',[]),
            'resolved_material_level':level,'document_kind':'journal_article','registry_verified':True,
            'resolved_from':original_url,'search_backend':original.get('search_backend','deepseek')}

    def expand(self, discoveries, query, question):
        constraints=request_constraints(question)
        if not any(constraints[k] for k in ('medical','original_requested','years','title')):
            return discoveries
        rows=[]
        for row in discoveries:
            # A discovery snippet labels what has been read so far. It must
            # not prevent fetching that result's actual page body.
            decision=eligibility({**row,'material_level':row.get('resolved_material_level','discovered_page')},question)
            if not decision['eligible'] and decision['reason'] not in {'publication_year_unverified','requested_document_not_matched','navigation_list'}:
                self.web.log.write('source_candidate_rejected',{'url':row.get('url'),**decision})
                continue
            rows.append(row)
        # One batch resolves PMID/DOI aliases. An exact requested title can
        # recover a specific article from author/navigation pages.
        ids=[source_identifiers(r) for r in rows]
        terms=[]
        for ident in ids:
            if ident.get('pmid'): terms.append('EXT_ID:'+ident['pmid'])
            elif ident.get('doi'): terms.append('DOI:"'+_literal(ident['doi'])+'"')
        wanted=constraints['title']
        if wanted and re.search(r'[A-Za-z]{5}',wanted): terms.append('TITLE:"'+_literal(wanted)+'"')
        if not terms and constraints['years']:
            for row in rows:
                title=row.get('article_title') or row.get('title','')
                if len(title)>35 and len(re.findall(r'[A-Za-z]',title))>=25:
                    terms.append('TITLE:"'+_literal(title.split(' - ')[0])+'"')
        records=self._records(' OR '.join(dict.fromkeys(terms))) if terms else []
        used=set();resolved=[]
        for row,ident in zip(rows,ids):
            matches=[r for r in records if (ident.get('pmid') and ident['pmid']==str(r.get('pmid') or r.get('id'))
                or ident.get('doi') and ident['doi'].lower()==str(r.get('doi','')).lower()
                or normalized_title(r['title']) in normalized_title(str(row.get('title','')))
                or wanted and normalized_title(wanted)==normalized_title(r['title'])
                    and normalized_title(r['title']) in normalized_title(str(row.get('body_content',''))[:1200]))]
            if matches:
                record=matches[0];value=self._row(record,row,query)
                if value: resolved.append(value);used.add(str(record.get('id')))
            elif source_kind(row)!='navigation_list': resolved.append(row)
        for record in records:
            if wanted and normalized_title(wanted)==normalized_title(record['title']) and str(record.get('id')) not in used:
                row=self._row(record,{'search_backend':'deepseek'},query)
                if row:resolved.insert(0,row)
        # Supplement an unsuccessful biomedical discovery using the SAME
        # model-generated query, not a second invented patient narrative.
        usable=[r for r in resolved if eligibility({**r,'material_level':r.get('resolved_material_level','discovered_page')},question)['eligible']]
        if not usable and constraints['medical'] and re.search(r'[A-Za-z]{3}',query) and self.web.budget.available('search_requests'):
            for record in self._records(query,discovery=True):
                value=self._row(record,{'search_backend':'europepmc'},query)
                if value and eligibility(value,question)['eligible']: resolved.append(value)
                if len(resolved)>=5:break
        # Keep richer original or registry text first, deduplicate before the
        # two-body admission limit so mirrors cannot crowd out a new study.
        result,seen=[],set()
        for row in sorted(resolved,key=lambda r:not r.get('registry_verified')):
            keys=document_keys(row)
            if not keys & seen:result.append(row)
            seen.update(keys)
        return result
