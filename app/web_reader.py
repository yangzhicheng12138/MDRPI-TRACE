"""Optional Crawl4AI browser worker, isolated from model dependencies."""

import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile

from .external_evidence import ExternalFailure, validate_public_url


class Crawl4AIReader:
    def __init__(self, python, *, timeout=30):
        self.python = str(python)
        self.timeout = timeout

    def __call__(self, url, budget):
        validate_public_url(url)
        budget.consume("fetch_requests")
        timeout = min(self.timeout, budget.remaining())
        worker = Path(__file__).resolve().parents[1] / "scripts/read_web_page.py"
        env = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}}
        env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        env.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(Path(self.python).parent.parent / "browsers"))
        with tempfile.TemporaryDirectory(prefix="mdrpi-browser-") as directory:
            output = Path(directory) / "result.json"
            with (Path(directory) / "worker.log").open("wb") as stream:
                try:
                    process = subprocess.Popen([self.python, str(worker), "--url", url,
                        "--timeout", str(timeout), "--output", str(output)],
                        stdout=stream, stderr=subprocess.STDOUT, env=env, start_new_session=True)
                except OSError as exc:
                    raise ExternalFailure("browser_worker_unavailable:" + type(exc).__name__) from exc
                try:
                    process.wait(timeout=timeout)
                except subprocess.TimeoutExpired as exc:
                    # The group contains only this worker and its browser children.
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    raise ExternalFailure("browser_request_timeout") from exc
            if not output.exists():
                raise ExternalFailure("browser_worker_failed:" + str(process.returncode))
            result = json.loads(output.read_text())
        if result.get("error"):
            raise ExternalFailure(result["error"])
        if result.get("status_code") != 200:
            raise ExternalFailure("browser_http_" + str(result.get("status_code")))
        final_url = result.get("url") or url
        validate_public_url(final_url)
        data = result.get("html", "").encode("utf-8")
        if len(data) > 4_000_000:
            raise ExternalFailure("external_response_too_large")
        return data, final_url
