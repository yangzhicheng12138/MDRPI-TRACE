"""Bounded format recovery for ordinary chat, without changing clinical pipelines."""
from copy import deepcopy
import json
import re

from jsonschema import Draft202012Validator

from app.local_llm import InvalidModelOutput
from app.search_planner import planning_runner
from .tasks import check_cancelled


CHAT_CALLS = {"chat_tool_plan", "chat_tool_continue", "chat_preference_replan",
              "web_query_plan", "search_sufficiency"}


def _redact(value):
    return re.sub(r"\bsk-[A-Za-z0-9_-]{12,}\b", "[redacted API key]", value)


class _ResponseTrace:
    """Capture response data only; never persist request headers or credentials."""
    def __init__(self, client, owner):
        self.client, self.owner = client, owner

    def post(self, *args, **kwargs):
        response = self.client.post(*args, **kwargs)
        if response.status_code == 200:
            try:
                body = response.json()
                self.owner.last_response = {k: body[k] for k in ("id", "model", "choices", "usage") if k in body}
            except (ValueError, TypeError):
                self.owner.last_response = {"body": response.text[:40000]}
        return response

    def close(self):
        self.client.close()


class ChatPlanningRunner:
    def __init__(self, runner):
        self.runner = runner
        self.last_response = None
        cloud = getattr(runner, "cloud", None)
        if cloud is not None:
            # Keep cancellation and configured provider/fallback behavior intact.
            previous = cloud.llm.wrap_client
            cloud.llm.wrap_client = lambda client: _ResponseTrace(previous(client) if previous else client, self)

    def __getattr__(self, name):
        return getattr(self.runner, name)

    def call(self, prompt, payload, schema, name, max_tokens=900):
        if name not in CHAT_CALLS:
            return self.runner.call(prompt, payload, schema, name, max_tokens=max_tokens)
        final_review = (name == "search_sufficiency" and isinstance(payload.get("round"), int)
                        and isinstance(payload.get("round_limit"), int)
                        and payload["round"] >= payload["round_limit"])
        actual_schema = schema
        if final_review and "next_query" in schema.get("properties", {}):
            # No next query will run after the last round. Its length must not
            # discard otherwise valid evidence assessments. All other fields,
            # including source IDs and quotes, keep their original constraints.
            actual_schema = deepcopy(schema)
            actual_schema["properties"]["next_query"]["maxLength"] = 4096
            prompt += '\nThis is the final search round. Set next_query=""; assess the actual evidence normally.'
        for attempt in range(2):
            check_cancelled()
            self.budget.remaining()
            self.last_response = None
            try:
                value = self.runner.call(prompt, payload, actual_schema, name, max_tokens=max_tokens)
                if final_review:
                    value = {**value, "next_query": ""}
                    Draft202012Validator(schema).validate(value)
                if attempt:
                    self.log.write("chat_plan_recovered", {"name": name, "attempts": attempt + 1})
                return value
            except InvalidModelOutput as exc:
                # Context overflow is not a generation-format problem; another
                # request with more feedback would only make it worse.
                if "context budget" in str(exc).lower():
                    raise
                raw = getattr(exc, "raw_response", None) or self.last_response
                self.log.write("chat_plan_format_error", {"name": name, "attempt": attempt + 1,
                    "error_type": type(exc).__name__, "message": _redact(str(exc))[:600],
                    "raw_response": json.loads(_redact(json.dumps(raw, ensure_ascii=False)))})
                if attempt:
                    raise
                check_cancelled()
                self.budget.remaining()
                # Feedback goes at the end of the variable payload, keeping the
                # system prefix stable. No failed tool is executed or fabricated.
                feedback = {"error": _redact(str(exc))[:600], "attempts_remaining": 1,
                    "instruction": "Return one complete JSON object matching the supplied schema. "
                        "Keep reasons and queries short. Choose only necessary available tools with valid args. "
                        "Do not invent source IDs, quotes, file IDs, or execution results."}
                try:
                    content = raw["choices"][0]["message"]["content"]
                    if isinstance(content, str):
                        feedback["previous_output_excerpt"] = _redact(content)[:3000]
                except (TypeError, KeyError, IndexError):
                    pass
                payload = {**payload, "format_repair": feedback}
                max_tokens = min(1800, max_tokens + 300)


def recover_chat_runner(config, runner):
    if config is not None and not config.get("interactive", "chat_planner_recovery", default=True):
        return runner
    return runner if isinstance(runner, ChatPlanningRunner) else ChatPlanningRunner(runner)


def chat_planning_runner(config, local, **kwargs):
    return recover_chat_runner(config, planning_runner(config, local, **kwargs))
