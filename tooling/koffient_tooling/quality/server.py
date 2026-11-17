"""A local llama-server (host build of the same llama.cpp version) and one chat request to it."""

from __future__ import annotations

import json
import queue
import re
import socket
import subprocess
import threading
import time
from pathlib import Path

import requests

from ..model import sha256_file

SERVER_BINARY = "llama-server.exe"
# Models with a thinking mode (Qwen3) answer in their non-thinking mode, the same footing as models without one.
# Templates that do not use the variable ignore it, so the prompt of other models does not change.
CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}
# Greedy decoding with every penalty switched off, sent with each request. The converter writes the model's own
# generation_config (e.g. Qwen2.5: repetition_penalty 1.1) into the GGUF and llama-server uses it as the default,
# so without this files of the same model could be decoded differently depending on who converted them.
SAMPLING = {"temperature": 0, "seed": 1, "repeat_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0,
            "dry_multiplier": 0.0}


class ServerError(RuntimeError):
    pass


class UnparsedOutput(ServerError):
    """The model answered, but llama-server could not parse the answer (e.g. a malformed tool call).

    This is the model's mistake, not a broken run: it is scored as a wrong answer.
    """


# llama-server's message when the chat parser rejects the model's own output (HTTP 500).
_UNPARSED_MARKERS = ("does not match the expected",)


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
        "backend": "cuda" if (engine_dir / "ggml-cuda.dll").is_file() else "cpu",
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
                 gpu_layers: str | None = None, start_timeout_s: int = 300):
        self.command = [
            str(engine_dir / SERVER_BINARY), "-m", str(model), "--host", "127.0.0.1",
            "-c", str(ctx), "-np", "1", "-t", str(threads), "--jinja", "--no-webui",
            # Every request is computed from scratch (cache_prompt is off), so the host prompt cache (up to 8 GB per
            # server by default) and context checkpoints (state copies, large for hybrid models) only cost memory.
            "--cache-ram", "0", "--ctx-checkpoints", "0",
        ]
        if gpu_layers is not None:
            self.command += ["-ngl", gpu_layers]
        self.log_path = log_path
        self.start_timeout_s = start_timeout_s
        self.url = ""
        self._proc: subprocess.Popen | None = None
        self._log = None

    def __enter__(self) -> "LlamaServer":
        port = _free_port()
        self.url = f"http://127.0.0.1:{port}"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(self.log_path, "ab")  # append: a restarted server keeps the log of the crash above
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

    def is_dead(self, wait_s: float = 15.0) -> bool:
        """True when the process has exited. A crash resets the connection before the process is gone, so wait."""
        if self._proc is None:
            return False
        try:
            self._proc.wait(timeout=wait_s)
        except subprocess.TimeoutExpired:
            return False
        return True

    def restart(self) -> None:
        self.__exit__(None, None, None)
        with open(self.log_path, "ab") as log:
            log.write(b"\n==== koffient: server died, restarted ====\n")
        self.__enter__()

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
        payload = {"messages": messages, "max_tokens": max_tokens, **SAMPLING,
                   "cache_prompt": False, "stream": False, "chat_template_kwargs": CHAT_TEMPLATE_KWARGS}
        if tools:
            payload["tools"] = tools
        if grammar:
            payload["grammar"] = grammar
        response = requests.post(f"{self.url}/v1/chat/completions", json=payload, timeout=timeout_s)
        if response.status_code != 200:
            if response.status_code == 500 and any(m in response.text for m in _UNPARSED_MARKERS):
                raise UnparsedOutput(response.text[:300])
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

    # Servers on the CUDA build sometimes die with "out of memory" while the card has GBs free (journal P37-P40).
    MAX_RESTARTS = 3

    def __init__(self, engine_dir: Path, model: Path, log_dir: Path, *, servers: int = 8, ctx: int = 4096,
                 threads: int = 2, gpu_layers: str | None = None):
        self.size = servers
        self._servers = [LlamaServer(engine_dir, model, log_dir / f"server-{i}.log", ctx=ctx, threads=threads,
                                     gpu_layers=gpu_layers)
                         for i in range(servers)]
        self._free: queue.Queue[LlamaServer] = queue.Queue()
        self._lock = threading.Lock()
        self.restarts = 0

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
        """Ask a free server. If the server process died on this request, restart it and ask again.

        A server with one slot answers deterministically (journal P31), so the answer from the restarted server is
        the one the dead server would have given. Restarts are counted and land in the task record.
        """
        server = self._free.get()
        try:
            for attempt in range(self.MAX_RESTARTS + 1):
                try:
                    return server.chat(*args, **kwargs)
                except requests.ConnectionError:
                    # Only a dropped connection can mean a crash; an HTTP error from a live server (e.g. output the
                    # parser rejected) is an answer and must not wait for the process to exit.
                    if not server.is_dead() or attempt == self.MAX_RESTARTS:
                        raise
                    with self._lock:
                        self.restarts += 1
                    server.restart()
        finally:
            self._free.put(server)
