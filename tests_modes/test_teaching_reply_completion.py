import copy
import json

import pytest

from modes.backend import Backend, TurnLog
from modes.teaching_router import incomplete_teaching_reply


@pytest.mark.parametrize("text", [
    "根据现有资料，你可以从以下几个方面考虑：",
    "可以从以下几个方面考虑。", "Consider the following:",
    "先整理资料，然后，", "判断依据\n1. ",
])
def test_dangling_explanations_need_completion(text):
    assert incomplete_teaching_reply(text)


@pytest.mark.parametrize("text", [
    "先区分已知资料和需要确认的信息。", "为什么这个因素会影响愈合？",
    "请思考：病史与伤口愈合有什么关系？", "Consider the patient's context.",
    "例如：\n- 解释线索之间的关系。", "时间写作 12:30", "符号为“：”",
])
def test_complete_short_explanations_are_not_retried(text):
    assert not incomplete_teaching_reply(text)


class Replies:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = []
        self.closed = False

    def structured(self, messages, schema, name, **kwargs):
        self.calls.append((name, json.loads(messages[-1]["content"]), kwargs))
        return {"reply": next(self.replies)}

    def close(self):
        self.closed = True


def setup_backend(tmp_path, monkeypatch, replies):
    config = tmp_path / "config.yaml"
    config.write_text("llm:\n  base_url: http://127.0.0.1:8001/v1\n")
    backend = Backend(config)
    llm = Replies(replies)
    monkeypatch.setattr(backend, "_llm", lambda: llm)
    state = {"mode": "teaching", "language": "zh", "stage": "answer", "case": "测试病例",
             "messages": [], "hint_events": [], "reviews": []}
    return backend, llm, state, TurnLog(tmp_path / "log")


def test_one_repair_preserves_locked_basis_and_shared_deadline(tmp_path, monkeypatch):
    completed = "先找出题目已经提供的信息，再说明这些信息与题目所问问题的关系；不能把缺失资料当成已知事实。"
    backend, llm, state, log = setup_backend(tmp_path, monkeypatch, ["可以从以下几方面考虑：", completed])
    original = copy.deepcopy(state)
    assert backend.teaching_dialogue(state, "解释这道题", "explain", log) == completed
    assert len(llm.calls) == 2
    first, repaired = llm.calls
    assert not first[1]["reference_access"]["allowed"]
    assert repaired[1]["saved_reference_basis"] is None
    assert repaired[1]["published_current_task_points"] == []
    assert {k: v for k, v in repaired[1].items() if k != "unfinished_draft"} == first[1]
    assert repaired[2]["deadline"] == first[2]["deadline"]
    assert state == original  # No hint use, grade, answer exposure or step advance.
    assert llm.closed


def test_a_second_incomplete_reply_is_not_accepted(tmp_path, monkeypatch):
    backend, llm, state, log = setup_backend(tmp_path, monkeypatch, ["分析如下：", "可以考虑："])
    with pytest.raises(ValueError, match="未写完整"):
        backend.teaching_dialogue(state, "解释这道题", "explain", log)
    assert len(llm.calls) == 2
    assert llm.closed


def test_complete_reply_needs_no_extra_model_call(tmp_path, monkeypatch):
    backend, llm, state, log = setup_backend(tmp_path, monkeypatch, ["请解释病例线索之间的关系。"])
    assert backend.teaching_dialogue(state, "解释这道题", "explain", log) == "请解释病例线索之间的关系。"
    assert len(llm.calls) == 1
    assert llm.closed
