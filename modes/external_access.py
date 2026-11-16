"""Password-protected, isolated entry point for a single external tester."""
from __future__ import annotations

import base64
import binascii
import os
import secrets
from pathlib import Path
from urllib.parse import urlsplit

from fastapi.responses import FileResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import Headers, MutableHeaders
from starlette.middleware.gzip import GZipMiddleware

from modes.runtime import ROOT
from modes.server import create_app as create_workspace_app


class TestAccessGate:
    """Protect every HTTP path without buffering streaming responses."""

    def __init__(self, app, *, username, password, origins):
        self.app = app
        self.username = username.encode("utf-8")
        self.password = password.encode("utf-8")
        self.origins = frozenset(origins)
        if not username or len(self.password) < 16 or not self.origins:
            raise ValueError("测试入口必须设置用户名、至少16字节的口令和允许的来源。")

    def authenticated(self, authorization):
        try:
            scheme, value = authorization.split(" ", 1)
            if scheme.lower() != "basic" or len(value) > 4096:
                return False
            username, password = base64.b64decode(value, validate=True).split(b":", 1)
            user_ok = secrets.compare_digest(username, self.username)
            password_ok = secrets.compare_digest(password, self.password)
            return user_ok and password_ok
        except (ValueError, binascii.Error):
            return False

    async def __call__(self, scope, receive, send):
        if scope["type"] == "websocket":
            # The shared legacy Chainlit application is intentionally not mounted.
            await send({"type": "websocket.close", "code": 1008})
            return
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        if not self.authenticated(headers.get("authorization", "")):
            response = PlainTextResponse(
                "请输入测试账号和访问口令。", status_code=401,
                headers={"WWW-Authenticate": 'Basic realm="MDRPI test", charset="UTF-8"',
                         "Cache-Control": "no-store"},
            )
            await response(scope, receive, send)
            return
        origin = headers.get("origin")
        if scope["method"] not in {"GET", "HEAD", "OPTIONS"} and origin is not None and origin not in self.origins:
            await PlainTextResponse("此来源不能修改测试会话。", status_code=403)(scope, receive, send)
            return

        async def private_response(message):
            if message["type"] == "http.response.start":
                response_headers = MutableHeaders(scope=message)
                response_headers["Cache-Control"] = "private, no-store"
                response_headers["X-Content-Type-Options"] = "nosniff"
            await send(message)

        await self.app(scope, receive, private_response)


def protected_workspace(service=None, *, username, password, origins, with_ui=True):
    # Only the current workspace is exposed. The legacy UI does not share its
    # global service or websocket transport with this isolated test instance.
    app = create_workspace_app(service, with_ui=False)
    if with_ui:
        frontend = ROOT / "modes/frontend"

        @app.get("/ui/")
        def workspace():
            return FileResponse(frontend / "index.html")

        @app.get("/ui")
        def workspace_redirect():
            return RedirectResponse("/ui/")

        app.mount("/ui/assets", GZipMiddleware(StaticFiles(directory=frontend), minimum_size=500, compresslevel=6))
    app.add_middleware(TestAccessGate, username=username, password=password, origins=origins)
    return app


def create_app():
    """Fail closed if the launcher omitted credentials or a separate data root."""
    data_setting = os.environ.get("MDRPI_MODES_DATA", "")
    password_setting = os.environ.get("MDRPI_TEST_PASSWORD_FILE", "")
    public_url = os.environ.get("MDRPI_TEST_PUBLIC_URL", "").rstrip("/")
    username = os.environ.get("MDRPI_TEST_USERNAME", "")
    port = int(os.environ.get("MDRPI_MODES_PORT", "8022"))
    if not data_setting or not password_setting or not username:
        raise RuntimeError("测试入口缺少独立数据目录或访问口令，拒绝启动。")
    data_root = Path(data_setting).resolve()
    regular_data = (ROOT / "data/interactive_modes").resolve()
    if data_root == regular_data or regular_data in data_root.parents or data_root in regular_data.parents:
        raise RuntimeError("测试入口不能使用或包含现有会话目录。")
    public = urlsplit(public_url)
    if public.scheme != "https" or not public.netloc or public.username or public.password or public.path or public.query or public.fragment:
        raise RuntimeError("请设置测试入口的 HTTPS 来源网址。")
    secret_file = Path(password_setting)
    if secret_file.stat().st_mode & 0o077:
        raise RuntimeError("访问口令文件只允许文件所有者读写，请先设置 chmod 600。")
    password = secret_file.read_text(encoding="utf-8").strip()
    if len(password.encode("utf-8")) < 16:
        raise RuntimeError("访问口令至少需要16字节。")
    return protected_workspace(
        username=username, password=password,
        origins={public_url, f"http://127.0.0.1:{port}", f"http://localhost:{port}"},
    )
