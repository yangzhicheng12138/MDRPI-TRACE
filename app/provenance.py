"""Content identities for reproducible local research runs."""

from pathlib import Path

from .corpus import json_digest, sha256_file


RUNTIME_FILES = (
    "api.py", "config.py", "corpus.py", "embedding.py", "local_llm.py", "metadata.py",
    "provenance.py", "query_policy.py", "rag.py", "reranker.py", "retriever.py", "vector_store.py",
    "lexical.py", "applicability.py", "population_isolation.py",
    "answer_layer.py", "structured_answer.py", "case_facts.py", "six_step_schema.py",
    "six_step_answer.py", "reflection_checks.py", "reflection_workflow.py", "revision_scope_review.py", "evidence_guard.py", "answer_guards.py",
    "evidence_router.py", "external_evidence.py", "external_query.py", "evidence_snapshot.py",
    "web_search.py", "parallel_search.py", "web_terms.py", "web_reader.py",
    "deepseek_search.py", "web_content.py",
    "search_planner.py", "planned_search.py", "evidence_constraints.py", "source_resolution.py", "reflection_backend.py", "reflection_inputs.py",
)


def runtime_identity():
    root = Path(__file__).resolve().parent
    files = {"app/" + name: sha256_file(root / name) for name in RUNTIME_FILES}
    files["scripts/17_serve_local_model.py"] = sha256_file(root.parent / "scripts/17_serve_local_model.py")
    return {"files": files, "sha256": json_digest(files)}


def code_identity(root):
    root = Path(root)
    files = {str(p.relative_to(root)): sha256_file(p)
             for directory in ("app", "scripts", "tests") for p in sorted((root / directory).glob("*.py"))}
    files["requirements.txt"] = sha256_file(root / "requirements.txt")
    return {"files": files, "sha256": json_digest(files)}
