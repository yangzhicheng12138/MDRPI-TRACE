#!/usr/bin/env python3
"""Launch existing Ministral weights with compiled CUDA graphs or optional eager execution."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

from common import add_project_to_import_path, project_config_path

add_project_to_import_path()

from app.config import AppConfig  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=project_config_path())
    parser.add_argument("--gpu", required=True, type=int, help="An available physical GPU; no other processes are stopped")
    parser.add_argument("--memory-utilization", type=float, default=0.45)
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--enforce-eager", action="store_true",
                        help="Restore eager execution; default uses the validated single-request CUDA Graph/compile configuration")
    parser.add_argument("--log", type=Path, required=True, help="New log file, never overwritten")
    args = parser.parse_args()
    if not 0.1 <= args.memory_utilization <= 0.95 or args.gpu < 0:
        raise ValueError("Invalid GPU resource limits")
    config = AppConfig.from_file(args.config)
    python = config.path("local_model", "python")
    weights = config.path("local_model", "weights")
    if not python.is_file() or not (weights / "config.json").is_file():
        raise ValueError("Configured local model environment or weights are missing")
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.update(CUDA_VISIBLE_DEVICES=str(args.gpu), TZ="Asia/Shanghai", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
               TOKENIZERS_PARALLELISM="false", VLLM_NO_USAGE_STATS="1", DO_NOT_TRACK="1")
    command = [str(python), "-m", "vllm.entrypoints.openai.api_server", "--model", str(weights),
               "--served-model-name", str(config.get("llm", "model")), "--host", "127.0.0.1",
               "--port", str(args.port), "--dtype", "bfloat16", "--quantization", "bitsandbytes",
               "--load-format", "bitsandbytes", "--gpu-memory-utilization", str(args.memory_utilization),
               "--max-model-len", str(config.get("llm", "context_window", default=8192)),
               "--max-num-seqs", "1", "--max-num-batched-tokens", "1024",
               "--structured-outputs-config", json.dumps(config.get("reflection", "structured_outputs", default={
                   "backend": "xgrammar", "disable_any_whitespace": True})),
               "--limit-mm-per-prompt", '{"image":0}', "--disable-log-requests", "--generation-config", "vllm"]
    if args.enforce_eager:
        command.append("--enforce-eager")
    else:
        command += ["--compilation-config", json.dumps({
            "level": 3, "cudagraph_mode": "FULL_AND_PIECEWISE", "cudagraph_capture_sizes": [1]})]
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open("x", encoding="utf-8") as log:
        print("Starting loopback model service; log:", args.log, flush=True)
        return subprocess.call(command, env=env, stdout=log, stderr=subprocess.STDOUT)


if __name__ == "__main__":
    raise SystemExit(main())
