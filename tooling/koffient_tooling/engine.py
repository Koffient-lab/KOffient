"""Engine passport: what a llama.cpp build directory was configured with."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from .model import sha256_file

BINARIES = ("llama-bench", "llama-completion")

# Cache entries that change which CPU kernels end up in the binary, or how it is linked.
RECORDED_FLAGS = (
    "CMAKE_BUILD_TYPE", "ANDROID_ABI", "ANDROID_PLATFORM", "BUILD_SHARED_LIBS",
    "GGML_NATIVE", "GGML_CPU_ARM_ARCH", "GGML_CPU_ALL_VARIANTS", "GGML_BACKEND_DL",
    "GGML_SSE42", "GGML_AVX", "GGML_AVX2", "GGML_AVX512", "GGML_AVX_VNNI",
    "GGML_FMA", "GGML_F16C", "GGML_BMI2",
    "GGML_OPENMP", "GGML_LLAMAFILE", "GGML_CPU_KLEIDIAI",
)


class EngineError(RuntimeError):
    """The build directory is missing, incomplete, or not a CMake build."""


def parse_cmake_cache(text: str) -> dict[str, str]:
    cache = {}
    for line in text.splitlines():
        match = re.match(r"^([A-Za-z0-9_.\-]+):[A-Z]+=(.*)$", line)
        if match:
            cache[match.group(1)] = match.group(2)
    return cache


def _git(source_dir: Path, *args: str) -> str | None:
    try:
        proc = subprocess.run(["git", "-C", str(source_dir), *args], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def describe_build(build_dir: Path) -> dict:
    """Read the `engine` block of a record from a CMake build directory."""
    cache_file = build_dir / "CMakeCache.txt"
    if not cache_file.is_file():
        raise EngineError(f"{build_dir} has no CMakeCache.txt")
    cache = parse_cmake_cache(cache_file.read_text(encoding="utf-8", errors="replace"))

    binaries = {}
    for name in BINARIES:
        path = build_dir / "bin" / name
        if not path.is_file():
            raise EngineError(f"{path} is missing: build the '{name}' target first")
        binaries[name] = sha256_file(path)

    source_dir = Path(cache.get("CMAKE_HOME_DIRECTORY", ""))
    ndk = re.search(r"/ndk/([\d.]+)/", cache.get("CMAKE_TOOLCHAIN_FILE", "").replace("\\", "/"))
    return {
        "name": "llama.cpp",
        "build_id": build_dir.name,
        "abi": cache.get("ANDROID_ABI"),
        # Source checkout state now; cross-checked against the commit llama-bench reports at run time.
        "tag": _git(source_dir, "describe", "--tags", "--always") if source_dir.is_dir() else None,
        "commit": _git(source_dir, "rev-parse", "--short", "HEAD") if source_dir.is_dir() else None,
        "ndk": ndk.group(1) if ndk else None,
        "cmake_flags": {key: cache[key] for key in RECORDED_FLAGS if key in cache},
        "binaries_sha256": binaries,
    }
