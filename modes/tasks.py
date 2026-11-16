"""Per-turn cancellation for interactive modes; never changes research execution."""
import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
import threading

import httpx


class TurnCancelled(BaseException):
    """Control flow, like asyncio.CancelledError; not a model/tool failure."""


CURRENT_TURN = ContextVar("interactive_turn", default=None)


class TurnControl:
    def __init__(self):
        self.cancelled = threading.Event()
        self.lock = threading.Lock()
        self.callbacks = set()
        self.partial_text = ""

    def check(self):
        if self.cancelled.is_set():
            raise TurnCancelled()

    def cancel(self):
        with self.lock:
            self.cancelled.set()
            callbacks = list(self.callbacks)
        for callback in callbacks:
            callback()

    def observe(self, kind, value):
        self.check()
        if kind == "token":
            self.partial_text += value
        elif kind == "answer_text":
            self.partial_text = value

    @contextmanager
    def interrupt_with(self, callback):
        with self.lock:
            self.check()
            self.callbacks.add(callback)
        try:
            yield
        finally:
            with self.lock:
                self.callbacks.discard(callback)


def check_cancelled():
    control = CURRENT_TURN.get()
    if control:
        control.check()


class CancellableClient:
    """Sync facade over HTTPX async I/O, scoped to one interactive LLM client.

    Reuses LocalLLM's request construction and response parsing. Closing the
    request on cancellation disconnects this generation, not the model server.
    """
    def __init__(self, original, *, transport=None):
        self.original = original
        self.control = CURRENT_TURN.get()
        self.loop = asyncio.new_event_loop()
        self.client = httpx.AsyncClient(timeout=original.timeout, headers=original.headers,
                                       trust_env=False, follow_redirects=False, transport=transport)

    def _run(self, coroutine, interruptible=True):
        task = self.loop.create_task(coroutine)

        def cancel_task():
            try:
                self.loop.call_soon_threadsafe(task.cancel)
            except RuntimeError:
                pass  # Request finished and its loop closed concurrently.

        try:
            if interruptible and self.control:
                with self.control.interrupt_with(cancel_task):
                    return self.loop.run_until_complete(task)
            return self.loop.run_until_complete(task)
        except (asyncio.CancelledError, TurnCancelled):
            task.cancel()
            self.loop.run_until_complete(asyncio.gather(task, return_exceptions=True))
            raise TurnCancelled() from None

    def request(self, method, url, **kwargs):
        request = self.original.build_request(method, url, **kwargs)
        return self._run(self.client.send(request))

    def post(self, url, **kwargs):
        return self.request("POST", url, **kwargs)

    def get(self, url, **kwargs):
        return self.request("GET", url, **kwargs)

    @contextmanager
    def stream(self, method, url, **kwargs):
        request = self.original.build_request(method, url, **kwargs)
        response = self._run(self.client.send(request, stream=True))
        try:
            yield StreamingResponse(self, response)
        finally:
            self._run(response.aclose(), interruptible=False)

    def close(self):
        try:
            self._run(self.client.aclose(), interruptible=False)
        finally:
            self.loop.close()
            self.original.close()


class StreamingResponse:
    def __init__(self, client, response):
        self.client, self.response = client, response

    def raise_for_status(self):
        self.response.raise_for_status()

    def iter_lines(self):
        lines = self.response.aiter_lines()
        while True:
            try:
                yield self.client._run(lines.__anext__())
            except StopAsyncIteration:
                return
