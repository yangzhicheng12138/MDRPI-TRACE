# Data and teaching materials

## Included examples

- `openrn.json`: 30 adaptations of publicly available Open RN textbook exercises, with per-item attribution, original links and licenses.
- `mdrpi_specialty.json`: eight project-authored MDRPI exercises, including equivalent answers, common misconceptions and progressive hints.
- `modes/teaching.py`: three explicitly fictional practice scenarios.
- Rubrics, content-review records and catalog classifications cover these included examples only. AI review records document content development, not independent clinical expert validation.

Original Open RN content retains CC BY 4.0. Supplied teaching adaptations retain CC BY-SA 4.0. These terms apply to the teaching materials, not the full text of linked documents. External PDFs, source pages and model weights are not redistributed.

## Data prepared separately

Formal research cases, generated experimental answers, research scores, user uploads, conversations, learning records, model weights and knowledge-source documents are excluded. Runtime data defaults to `data/interactive_modes/`, which is excluded by `.gitignore`.

## Build a knowledge base

Place PDFs you are entitled to use in `data/source/PDFs/` and adjust `config.yaml`. Index preparation retains provenance checks and review admission; unreviewed chunks should not be treated as reviewed evidence.

```bash
python scripts/01_scan_documents.py --help
python scripts/02_parse_documents.py --help
python scripts/03_build_chunks.py --help
python scripts/04_audit_chunks.py --help
python scripts/04_review_manifest.py --help
python scripts/05_build_vector_db.py --help
```

Complete inventory, parsing, chunking, content checks, source review and indexing in that order. Each command documents its inputs and outputs through `--help`. Build the searchable release index after review. Do not commit full texts or indexes without redistribution rights.

## Register authoring materials

The authoring registry is `data/question_generation/development_materials.json`; its loading contract is implemented in `modes/question_generation_materials.py`. The clean repository omits this file, so Question Studio initially shows no materials.

Registration requires independent development cases, teaching evidence, source citations, population/device scope, and file/item fingerprints tied to reviewed content. The current compatibility format uses reviewed development content in `data/teaching_bank/expansion_250.json` and legally held local PDFs. The filename is an existing contract name; exactly 250 items are not required.

The code checks registration, content review and source fingerprints before authoring. Missing or mismatched materials stop that generation request. Keep development materials and formal research cases separate.

## Add teaching questions

Follow the public examples for stems, answers, criteria, acceptable expressions, misconceptions and hints. When a stem or answer changes, update its criteria and review fingerprints too. Do not reuse a stale review conclusion for changed content.

English and Chinese interface labels do not translate source content. Provide bilingual question fields and reference points explicitly when adding bilingual teaching materials.
