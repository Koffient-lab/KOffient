"""A local llama-server (host build of the same llama.cpp version) and one chat request to it."""

from __future__ import annotations

import json
import queue
import re
import socket
import subprocess
import time
from pathlib import Path

import requests

from ..model import sha256_file

SERVER_BINARY = "llama-server.exe"


class ServerError(RuntimeError):
    pass


def describe_host_engine(engine_dir: Path) -> dict:
    """Passport of the host engine: version and commit as the binary reports them, checksum of the binary."""
    binary = engine_dir / SERVER_BINARY
    if not binary.is_file():
        raise ServerError(f"{SERVER_BINARY} not found in {engine_dir}")
    proc = subprocess.run([str(binary), "--version"], capture_output=True, text=True, timeout=60)
    text = proc.stdout + proc.stderr
    version = re.search(r"build (\d+), commit ([0-9a-f]+)", text)
    if not version:
        raise ServerError(f"cannot read the version of {binary}: {text.strip()[:200]}")
    built = re.search(r"built with (.+)", text)
    return {
        "name": "llama.cpp",
        "build_id": engine_dir.name,
        "tag": f"b{version.group(1)}",
        "commit": version.group(2)[:7],
        "built_with": built.group(1).strip() if built else None,
        "backend": "cpu",
        "binaries_sha256": {SERVER_BINARY: sha256_file(binary)},
    }


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class LlamaServer:
    """Context manager: starts llama-server on a free local port, stops it on exit.

    The server has a single slot on purpose. With several slots the requests are decoded in one batch, and the
    answer to a request then depends on what else happened to be in the batch.
    """

    def __init__(self, engine_dir: Path, model: Path, log_path: Path, *, ctx: int = 4096, threads: int = 2,
                 start_timeout_s: int = 300):
        self.command = [
            str(engine_dir / SERVER_BINARY), "-m", str(model), "--host", "127.0.0.1",
            "-c", str(ctx), "-np", "1", "-t", str(threads), "--jinja", "--no-webui",
        ]
        self.log_path = log_path
        self.start_timeout_s = start_timeout_s
        self.url = ""
        self._proc: subprocess.Popen | None = None
        self._log = None

    def __enter__(self) -> "LlamaServer":
        port = _free_port()
        self.url = f"http://127.0.0.1:{port}"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(self.log_path, "wb")
        self._proc = subprocess.Popen(self.command + ["--port", str(port)], stdout=self._log,
                                      stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        deadline = time.monotonic() + self.start_timeout_s
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                self.__exit__(None, None, None)
                raise ServerError(f"llama-server exited with code {self._proc.returncode}, see {self.log_path}")
            try:
                if requests.get(f"{self.url}/health", timeout=5).status_code == 200:
                    return self
            except requests.RequestException:
                pass
            time.sleep(0.5)
        self.__exit__(None, None, None)
        raise ServerError(f"llama-server did not become ready in {self.start_timeout_s} s, see {self.log_path}")

    def __exit__(self, *exc) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._log:
            self._log.close()
            self._log = None

    def chat(self, messages: list[dict], max_tokens: int, tools: list[dict] | None = None,
             grammar: str | None = None, timeout_s: int = 600) -> dict:
        """One greedy chat completion. Returns text, parsed tool calls, finish reason and token counts."""
        payload = {"messages": messages, "max_tokens": max_tokens, "temperature": 0, "seed": 1,
                   "cache_prompt": False, "stream": False}
        if tools:
            payload["tools"] = tools
        if grammar:
            payload["grammar"] = grammar
        response = requests.post(f"{self.url}/v1/chat/completions", json=payload, timeout=timeout_s)
        if response.status_code != 200:
            raise ServerError(f"HTTP {response.status_code}: {response.text[:300]}")
        body = response.json()
        choice = body["choices"][0]
        message = choice["message"]
        calls = []
        for call in message.get("tool_calls") or []:
            arguments = call["function"].get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except ValueError:
                    pass
            calls.append({"name": call["function"]["name"], "arguments": arguments})
        usage = body.get("usage") or {}
        return {
            "content": message.get("content") or "",
            "tool_calls": calls,
            "finish_reason": choice.get("finish_reason"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
        }


class ServerPool:
    """Several independent single-slot servers for the same model; chat() uses whichever one is free."""

    def __init__(self, engine_dir: Path, model: Path, log_dir: Path, *, servers: int = 8, ctx: int = 4096,
                 threads: int = 2):
        self.size = servers
        self._servers = [LlamaServer(engine_dir, model, log_dir / f"server-{i}.log", ctx=ctx, threads=threads)
                         for i in range(servers)]
        self._free: queue.Queue[LlamaServer] = queue.Queue()

    def __enter__(self) -> "ServerPool":
        try:
            for server in self._servers:
                self._free.put(server.__enter__())
        except ServerError:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *exc) -> None:
        for server in self._servers:
            server.__exit__(None, None, None)

    def chat(self, *args, **kwargs) -> dict:
        server = self._free.get()
        try:
            return server.chat(*args, **kwargs)
        finally:
            self._free.put(server)
