from modes.backend import render_references, fit_messages
import pytest


def test_references_use_supplied_metadata_and_one_bibliography():
    sources = [{"title": "Example article", "doi": "10.1/example", "year": 2024},
               {"title": "Example page", "source_type": "web_page", "source_url": "https://example.org/page"}]
    text = render_references("Claim [1]; another claim [2]. Unsupported number [99].", sources)
    assert text.count("### 参考资料") == 1
    assert "10.1/example" in text and "https://example.org/page" in text
    assert "[99]" not in text


def test_no_fabricated_bibliography_without_sources():
    assert "参考资料" not in render_references("General explanation.", [])
    assert render_references("The user's own identifier is DOC001.", []) == "The user's own identifier is DOC001."


def test_context_trimming_retains_latest_request_and_full_turns():
    class Config:
        def get(self, *args, **kwargs):
            return 180
    class Counter:
        config = Config()
        def count_tokens(self, messages):
            return len(messages) * 10, "test counter"
    messages = [{"role": r, "content": str(i)} for i, r in enumerate(["system", "user", "assistant", "user", "assistant", "user"])]
    fitted, trimmed = fit_messages(Counter(), messages, 20)
    assert fitted == [messages[0], messages[-1]]
    assert trimmed == 4
    assert len(messages) == 6


def test_multiple_excerpts_of_one_article_have_one_reference():
    sources = [{"doc_id": "DOC1", "origin": "external", "title": "Same article", "text": "First excerpt"},
               {"doc_id": "DOC1", "origin": "external", "title": "Same article", "text": "Second excerpt"}]
    text = render_references("First claim [1]. Second claim [2].", sources)
    assert text.count("Same article") == 1
    assert "First claim [1]. Second claim [1]." in text


def test_grouped_citations_are_numbered_and_listed():
    sources = [{"doc_id": f"DOC{i}", "origin": "external", "title": f"Article {i}"} for i in range(1, 4)]
    text = render_references("Claim [2, 3]; another [2，1].", sources)
    assert "Claim [1, 2]; another [1, 3]." in text
    assert text.count("Article 2") == 1
    assert "[3] Article 1[Z]." in text


def test_web_result_is_not_discarded_when_local_results_fill_context(tmp_path, monkeypatch):
    import httpx
    import app.web_search
    from modes.backend import Backend, TurnLog
    local = [{"chunk_id": f"DOC{i}_C1", "title": f"Local {i}", "text": f"local {i}"} for i in range(8)]
    web = {"evidence_id": "EXT_WEB_example", "source_identity": "https://example.org/article", "title": "Web result", "text": "New external evidence"}
    real_client = httpx.Client
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"status": "evidence_found", "evidence": local}))
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: real_client(**{**kwargs, "transport": transport}))
    # Exercise the real public search() return contract; replace network work only.
    monkeypatch.setattr(app.web_search.DDGSSearch, "_live", lambda *args, **kwargs:
        {"query": "development", "status": "evidence_available", "evidence": [web], "events": []})
    config = tmp_path / "config.yaml"
    config.write_text("llm:\n  base_url: http://127.0.0.1:8001/v1\nweb_search:\n  provider: ddgs\n")
    sources, events = Backend(config).sources("A development question", "web", TurnLog(tmp_path / "log"), lambda *_: None)
    assert web in sources
    assert any(s in local for s in sources)
    assert events[-1]["status"] == "evidence_available"


@pytest.mark.parametrize("preparation,expected_notice", [
    ({"shortened": True, "strategy": "token_head_and_tail", "input_tokens": 9430, "query_tokens": 8000}, True),
    ({"shortened": False}, False),
    (None, False),
])
def test_server_token_shortening_is_visible_and_logged(tmp_path, monkeypatch, preparation, expected_notice):
    import json
    import httpx
    from modes.backend import Backend, TurnLog
    from modes.chat_tools import source_notes
    real_client = httpx.Client
    body = {"status": "evidence_found", "evidence": [], "query_preparation": preparation}
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=body))
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: real_client(**{**kwargs, "transport": transport}))
    config = tmp_path / "config.yaml"
    config.write_text("llm:\n  base_url: http://127.0.0.1:8001/v1\n")
    _, events = Backend(config).sources("病例背景。末尾当前问题？", "local", TurnLog(tmp_path / "log"), lambda *_: None)
    assert any(e.get("status") == "query_shortened" for e in events) is expected_notice
    assert any("末尾当前问题" in note for note in source_notes(events)) is expected_notice
    log = json.loads(next((tmp_path / "log").glob("*retrieval.json")).read_text())
    assert log["query_shortened"] is expected_notice
    assert log["server_query_preparation"] == preparation


@pytest.mark.parametrize("detail", ["末尾当前问题超过资料检索的实际 token 预算，请缩短当前问题后重试。",
                                   [{"msg": "too long", "input": "PRIVATE_CASE_BODY"}]])
def test_rejected_retrieval_input_has_visible_reason_without_echoing_case(tmp_path, monkeypatch, detail):
    import httpx
    from modes.backend import Backend, TurnLog
    from modes.chat_tools import source_notes
    real_client = httpx.Client
    transport = httpx.MockTransport(lambda request: httpx.Response(422, json={"detail": detail}))
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: real_client(**{**kwargs, "transport": transport}))
    config = tmp_path / "config.yaml"
    config.write_text("llm:\n  base_url: http://127.0.0.1:8001/v1\n")
    sources, events = Backend(config).sources("当前问题：需拆分的病例内容", "local", TurnLog(tmp_path / "log"), lambda *_: None)
    assert sources == []
    notes = source_notes(events)
    assert any("本次知识库检索未完成" in note for note in notes)
    assert not any("PRIVATE_CASE_BODY" in note for note in notes)
    if isinstance(detail, str):
        assert any(detail in note for note in notes)
