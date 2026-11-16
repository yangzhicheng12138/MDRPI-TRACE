"""Loopback-only chat client; source text never follows a remote redirect."""

from __future__ import annotations

import json
import os
import time
from copy import deepcopy
from urllib.parse import urljoin, urlparse

import httpx


SELECTION_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "assessment": {"type": "string", "minLength": 1, "maxLength": 600},
        "decision": {"type": "string", "enum": ["supported", "insufficient", "conflict"]},
        "evidence_ids": {"type": "array", "items": {"type": "string"}},
        "conflict_pairs": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "properties": {"left_id": {"type": "string"}, "right_id": {"type": "string"}},
            "required": ["left_id", "right_id"],
        }},
    },
    "required": ["assessment", "decision", "evidence_ids", "conflict_pairs"],
}


class ModelUnavailable(RuntimeError):
    pass


class InvalidModelOutput(ValueError):
    pass


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object key")
        result[key] = value
    return result


class LocalLLM:
    def __init__(self, config, transport=None):
        self.config = config
        self.base_url = os.environ.get("MDRPI_LLM_BASE_URL", str(config.get("llm", "base_url", default="http://127.0.0.1:8001/v1"))).rstrip("/")
        parsed = urlparse(self.base_url)
        if (parsed.scheme not in {"http", "https"} or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
                or parsed.username or parsed.password or parsed.query or parsed.fragment):
            raise ValueError("Local model URL must use loopback HTTP(S) without credentials or query parameters")
        self.model = os.environ.get("MDRPI_LLM_MODEL", str(config.get("llm", "model", default="ministral-3-14b")))
        headers = {}
        if os.environ.get("MDRPI_LLM_API_KEY"):
            headers["Authorization"] = "Bearer " + os.environ["MDRPI_LLM_API_KEY"]
        self.client = httpx.Client(
            timeout=float(config.get("llm", "timeout_seconds", default=180)),
            trust_env=False, follow_redirects=False, headers=headers, transport=transport,
        )

    def close(self):
        self.client.close()

    def available(self) -> bool:
        try:
            response = self.client.get(self.base_url + "/models", timeout=5)
            response.raise_for_status()
            return self.model in {item["id"] for item in response.json()["data"]}
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            return False

    def count_tokens(self, messages, *, deadline=None) -> tuple[int, str]:
        try:
            kwargs = {"timeout": self._remaining_timeout(deadline, 30)} if deadline is not None else {}
            response = self.client.post(urljoin(self.base_url + "/", "../tokenize"),
                                        **kwargs,
                                        json={"model": self.model, "messages": messages, "add_generation_prompt": True})
            if response.status_code in {404, 405}:
                # UTF-8 byte count plus framing is a conservative fallback for
                # byte-based tokenizers. It reduces context, never cuts a chunk.
                return sum(len(item["content"].encode("utf-8")) for item in messages) + 256, "utf8_upper_bound"
            response.raise_for_status()
            count = response.json()["count"]
            if not isinstance(count, int) or isinstance(count, bool) or count < 1:
                raise ValueError("Invalid token count")
            return count, "server_tokenizer"
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            if isinstance(exc, httpx.TimeoutException):
                self._remaining_timeout(deadline, 30)
            raise ModelUnavailable("Local model tokenization is unavailable") from exc

    def select_evidence(self, messages, *, allowed_ids=None, trace=None) -> dict:
        schema = deepcopy(SELECTION_SCHEMA)
        if allowed_ids is not None:
            if not isinstance(allowed_ids, list) or not all(isinstance(item, str) and item for item in allowed_ids):
                raise ValueError("Allowed evidence IDs must be a list of nonempty strings")
            allowed_ids = list(dict.fromkeys(allowed_ids))
            properties = schema["properties"]
            limit = min(len(allowed_ids), int(self.config.get("rag", "max_selected_evidence", default=4)))
            properties["evidence_ids"]["maxItems"] = limit
            properties["conflict_pairs"]["maxItems"] = limit if len(allowed_ids) >= 2 else 0
            if allowed_ids:
                properties["evidence_ids"]["items"]["enum"] = allowed_ids
                for side in ("left_id", "right_id"):
                    properties["conflict_pairs"]["items"]["properties"][side]["enum"] = allowed_ids
            else:
                properties["decision"]["enum"] = ["insufficient"]
            branches = []
            for decision, min_ids, min_pairs in (("supported", 1, 0), ("insufficient", 0, 0), ("conflict", 2, 1)):
                if limit < min_ids:
                    continue
                branch = deepcopy(schema)
                fields = branch["properties"]
                fields["decision"]["enum"] = [decision]
                fields["evidence_ids"]["minItems"] = min_ids
                fields["conflict_pairs"]["minItems"] = min_pairs
                if decision == "insufficient":
                    fields["evidence_ids"]["maxItems"] = 0
                if decision != "conflict":
                    fields["conflict_pairs"]["maxItems"] = 0
                branches.append(branch)
            schema = {"anyOf": branches}
        request = {
            "model": self.model, "messages": messages,
            "temperature": float(self.config.get("llm", "temperature", default=0)),
            "max_tokens": int(self.config.get("llm", "max_tokens", default=1200)),
            "stream": False,
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "evidence_selection", "strict": True, "schema": schema,
            }},
        }
        try:
            response = self.client.post(self.base_url + "/chat/completions", json=request)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ModelUnavailable("Local model generation is unavailable") from exc
        try:
            choice = response.json()["choices"][0]
            if choice.get("finish_reason") != "stop":
                raise InvalidModelOutput("Incomplete model output")
            content = choice["message"]["content"]
            if not isinstance(content, str) or len(content) > 20000:
                raise InvalidModelOutput("Invalid model content")
            value = json.loads(content, object_pairs_hook=_unique_object)
            if not isinstance(value, dict) or set(value) != set(SELECTION_SCHEMA["required"]):
                raise InvalidModelOutput("Unexpected model response fields")
            assessment = value.pop("assessment")
            if not isinstance(assessment, str) or not assessment.strip() or len(assessment) > 600:
                raise InvalidModelOutput("Invalid applicability assessment")
            # Keep the model's unverified note out of the public selection and
            # quoted answer. A caller-owned trace avoids shared request state.
            if trace is not None:
                trace["applicability_assessment"] = assessment
            return value
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise InvalidModelOutput("Model did not return a complete evidence selection") from exc

    @staticmethod
    def _remaining_timeout(deadline, limit):
        if deadline is None:
            return limit
        from .evidence_snapshot import BudgetExceeded
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise BudgetExceeded("Shared model deadline exceeded")
        return min(limit, remaining)

    def structured(self, messages, schema, name, max_tokens=2400, *, deadline=None):
        """Bounded local structured generation for the development answer path."""
        self.last_generation_usage = None
        count, _ = self.count_tokens(messages, deadline=deadline) if deadline is not None else self.count_tokens(messages)
        if count + max_tokens + 128 > int(self.config.get('llm', 'context_window', default=8192)):
            raise InvalidModelOutput('Structured context budget exceeded')
        # vLLM's grammar backend lacks uniqueItems; enforce it on the response.
        def grammar(value):
            if isinstance(value, dict):
                return {k: grammar(v) for k, v in value.items() if k != 'uniqueItems'}
            if isinstance(value, list):
                return [grammar(v) for v in value]
            return value

        try:
            response = self.client.post(self.base_url + '/chat/completions', json={
                'model': self.model, 'messages': messages, 'temperature': 0,
                'max_tokens': max_tokens, 'stream': False,
                'response_format': {'type': 'json_schema', 'json_schema': {
                    'name': name, 'strict': True, 'schema': grammar(schema)}}},
                timeout=self._remaining_timeout(deadline, float(self.config.get('llm', 'structured_timeout_seconds', default=600))))
            response.raise_for_status()
        except httpx.HTTPError as exc:
            if isinstance(exc, httpx.TimeoutException):
                self._remaining_timeout(deadline, 600)
            raise ModelUnavailable('Structured generation unavailable') from exc
        try:
            body = response.json()
            self.last_generation_usage = body.get('usage')
            choice = body['choices'][0]
            if choice.get('finish_reason') != 'stop':
                raise ValueError('Truncated output')
            content = choice['message']['content']
            if not isinstance(content, str) or len(content) > 40000:
                raise ValueError('Invalid content')
            value = json.loads(content, object_pairs_hook=_unique_object)
            from jsonschema import validate, ValidationError
            try:
                validate(value, schema)
            except ValidationError as exc:
                raise ValueError('Schema mismatch') from exc
            return value
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            error = InvalidModelOutput(f'Invalid structured output: {exc}')
            try:
                error.raw_response = response.json()
            except ValueError:
                error.raw_response = {"body": response.text[:40000]}
            raise error from exc
