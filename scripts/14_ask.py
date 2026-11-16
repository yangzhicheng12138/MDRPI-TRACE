#!/usr/bin/env python3
"""Ask the local RAG service, or use the same pipeline without a server."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlparse

from common import add_project_to_import_path, project_config_path

add_project_to_import_path()

from app.config import AppConfig  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=project_config_path())
    parser.add_argument("--query", required=True)
    parser.add_argument("--device-type", default=None)
    parser.add_argument("--population", default=None)
    parser.add_argument("--api", default=None, help="Loopback RAG API URL (defaults to configured service port)")
    parser.add_argument("--local", action="store_true", help="Direct mode; stop the RAG API first to release Qdrant's local lock")
    parser.add_argument("--retrieve-only", action="store_true")
    parser.add_argument("--legacy", action="store_true", help="Historical verified source extracts")
    parser.add_argument("--web-mode", choices=["off", "live", "replay"], default="off")
    parser.add_argument("--web-strategy", choices=["on_demand", "verify"], default="on_demand")
    parser.add_argument("--replay-run-id")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output", type=Path, default=None, help="New output file; existing files are never overwritten")
    args = parser.parse_args()
    config = AppConfig.from_file(args.config)
    values = lambda value: [item.strip() for item in value.split(",") if item.strip()] if value is not None else None
    request = {"query": args.query, "device_type": values(args.device_type), "population": values(args.population)}
    try:
        if args.local:
            from app.rag import RAGPipeline
            pipeline = RAGPipeline(config)
            try:
                if args.retrieve_only or args.legacy:
                    result = (pipeline.retrieve if args.retrieve_only else pipeline.answer)(**request)
                else:
                    from app.reflection_workflow import reflective_answer
                    result = reflective_answer(pipeline, **request, web_mode=args.web_mode,
                                               replay_run_id=args.replay_run_id, web_strategy=args.web_strategy)
            finally:
                pipeline.close()
        else:
            import httpx
            url = (args.api or f"http://127.0.0.1:{config.get('service', 'port', default=8020)}").rstrip("/")
            parsed = urlparse(url)
            if parsed.scheme not in {"http", "https"} or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
                raise ValueError("RAG API must be a loopback URL")
            with httpx.Client(trust_env=False, follow_redirects=False, timeout=1860) as client:
                request.update(web_mode=args.web_mode, replay_run_id=args.replay_run_id, web_strategy=args.web_strategy)
                endpoint = "/v1/retrieve" if args.retrieve_only else "/v1/legacy/answer" if args.legacy else "/v1/answer"
                response = client.post(url + endpoint, json=request)
                result = response.json()
                if "status" not in result:
                    response.raise_for_status()
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as handle:
                json.dump(result, handle, ensure_ascii=False, indent=2)
        if args.json or args.retrieve_only:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(f"status={result['status']}")
            print(result["answer"])
        return 1 if result["status"] in {"index_unavailable", "model_unavailable", "invalid_model_output", "technical_failure"} else 0
    except Exception as exc:
        print(f"RAG request failed: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
