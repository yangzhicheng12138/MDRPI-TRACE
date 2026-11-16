#!/usr/bin/env python3
"""Serve the local audited RAG pipeline with serialized model access."""

import argparse
import os
from pathlib import Path
import sys
import time

from common import add_project_to_import_path, project_config_path

add_project_to_import_path()

from app.config import AppConfig  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=project_config_path())
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--device", default=None, help="Override both retrieval models, e.g. cpu or cuda:0")
    parser.add_argument("--embedding-device", default=None,
                        help="Override only the embedding model device")
    parser.add_argument("--reranker-device", default=None,
                        help="Override only the reranker model device")
    parser.add_argument("--log", type=Path, default=None, help="New service log file")
    args = parser.parse_args()
    config = AppConfig.from_file(args.config)
    if all(config.get(section, "local_files_only", default=False) for section in ("embedding", "reranker")):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if args.device:
        config.values.setdefault("embedding", {})["device"] = args.device
        config.values.setdefault("reranker", {})["device"] = args.device
    if args.embedding_device:
        config.values.setdefault("embedding", {})["device"] = args.embedding_device
    if args.reranker_device:
        config.values.setdefault("reranker", {})["device"] = args.reranker_device
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        output = args.log.open("x", encoding="utf-8", buffering=1)
        sys.stdout = output
        sys.stderr = output
    os.environ["TZ"] = "Asia/Shanghai"
    time.tzset()
    threads = str(config.get("service", "cpu_threads", default=4))
    os.environ["OMP_NUM_THREADS"] = threads
    os.environ["MKL_NUM_THREADS"] = threads
    started = time.monotonic()

    def progress(stage):
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S %z')}] "
              f"startup +{time.monotonic() - started:.1f}s: {stage}", flush=True)

    progress("importing torch")
    import torch
    torch.set_num_threads(int(threads))
    progress("importing API and retrieval dependencies")
    from app.api import create_app
    from app.rag import RAGPipeline
    import uvicorn
    progress("creating RAG pipeline")
    engine = RAGPipeline(config)
    try:
        progress("checking index")
        engine.retriever.check_index()
        progress(f"loading embedding model; device={config.get('embedding', 'device')}")
        engine.retriever.embedder._load()
        progress(f"loading reranker; device={config.get('reranker', 'device')}")
        engine.reranker._load()
        progress("models loaded; starting HTTP server")
        uvicorn.run(create_app(config, engine), host="127.0.0.1", port=args.port or int(config.get("service", "port", default=8020)), workers=1)
    finally:
        engine.close()


if __name__ == "__main__":
    main()
