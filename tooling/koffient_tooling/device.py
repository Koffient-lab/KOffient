"""Thin adb wrapper: push files, run commands, collect device facts."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

# CPU features that decide which ggml kernels can run (x86 flags / arm64 Features).
INTERESTING_CPU_FEATURES = (
    "sse4_2", "avx", "avx2", "fma", "f16c", "bmi2", "avx_vnni", "avx512f", "avx512_vnni",
    "asimd", "asimddp", "i8mm", "sve", "sve2", "sme",
)

_FACTS_SCRIPT = r"""
echo abi=$(getprop ro.product.cpu.abi)
echo os_release=$(getprop ro.build.version.release)
echo sdk=$(getprop ro.build.version.sdk)
echo model=$(getprop ro.product.model)
echo qemu=$(getprop ro.kernel.qemu)$(getprop ro.boot.qemu)
echo soc=$(getprop ro.soc.model)
echo ncpu=$(nproc)
echo mem_kb=$(grep MemTotal /proc/meminfo | tr -s ' ' | cut -d' ' -f2)
echo cpu_name=$(grep -m1 -E '^(model name|Hardware)' /proc/cpuinfo | cut -d: -f2)
echo cpu_flags=$(grep -m1 -E '^(flags|Features)' /proc/cpuinfo | cut -d: -f2)
"""


class DeviceError(RuntimeError):
    """adb is missing, the device is unreachable, or a device command failed."""


@dataclass
class ShellResult:
    returncode: int
    stdout: str
    stderr: str


def find_adb() -> str:
    exe = "adb.exe" if os.name == "nt" else "adb"
    for var in ("ANDROID_HOME", "ANDROID_SDK_ROOT"):
        root = os.environ.get(var)
        if root:
            candidate = Path(root) / "platform-tools" / exe
            if candidate.is_file():
                return str(candidate)
    found = shutil.which("adb")
    if found:
        return found
    raise DeviceError("adb not found: set ANDROID_HOME or put platform-tools on PATH")


def parse_facts(text: str, serial: str) -> dict:
    """Turn the key=value output of _FACTS_SCRIPT into the `device` block of a record."""
    raw = {}
    for line in text.splitlines():
        key, sep, value = line.partition("=")
        if sep:
            raw[key.strip()] = value.strip()

    abi = raw.get("abi", "")
    is_emulator = serial.startswith("emulator-") or "1" in raw.get("qemu", "")
    flags = set(raw.get("cpu_flags", "").split())
    mem_kb = int(raw["mem_kb"]) if raw.get("mem_kb", "").isdigit() else None
    return {
        "kind": "emulator" if is_emulator else "phone",
        "serial": serial,
        "abi": abi,
        "os": f"Android {raw.get('os_release', '?')}",
        "sdk": int(raw["sdk"]) if raw.get("sdk", "").isdigit() else None,
        "model": raw.get("model") or None,
        "cpu": raw.get("cpu_name") or raw.get("soc") or None,
        "cpu_features": [f for f in INTERESTING_CPU_FEATURES if f in flags],
        "cores": int(raw["ncpu"]) if raw.get("ncpu", "").isdigit() else None,
        "ram_mb": round(mem_kb * 1024 / 1e6) if mem_kb else None,
        # Speed numbers are only comparable to phones when measured on real arm hardware.
        "speed_is_phone_grade": (not is_emulator) and abi.startswith("arm"),
    }


class Device:
    def __init__(self, serial: str | None = None, adb: str | None = None):
        self.adb = adb or find_adb()
        self.serial = serial or self._only_device()

    def _adb(self, *args: str, timeout: float | None = None, binary: bool = False):
        cmd = [self.adb]
        if getattr(self, "serial", None):
            cmd += ["-s", self.serial]
        cmd += list(args)
        kwargs = {} if binary else {"encoding": "utf-8", "errors": "replace"}
        try:
            return subprocess.run(cmd, capture_output=True, timeout=timeout, **kwargs)
        except subprocess.TimeoutExpired as exc:
            raise DeviceError(f"adb timed out after {timeout:.0f}s: {' '.join(args)[:200]}") from exc

    def _only_device(self) -> str:
        out = self._adb("devices").stdout
        serials = [line.split("\t")[0] for line in out.splitlines() if line.endswith("\tdevice")]
        if len(serials) != 1:
            raise DeviceError(f"expected exactly one connected device, found {serials or 'none'}; pass --device")
        return serials[0]

    def shell(self, script: str, timeout: float | None = None, check: bool = False) -> ShellResult:
        proc = self._adb("shell", script, timeout=timeout)
        result = ShellResult(proc.returncode, proc.stdout, proc.stderr)
        if check and result.returncode != 0:
            tail = (result.stderr or result.stdout).strip()[-400:]
            raise DeviceError(f"device command failed (rc={result.returncode}): {tail}")
        return result

    def read_file(self, remote: str) -> bytes:
        proc = self._adb("exec-out", f"cat {shlex.quote(remote)}", binary=True)
        if proc.returncode != 0:
            raise DeviceError(f"cannot read {remote} from device")
        return proc.stdout

    def sha256(self, remote: str) -> str | None:
        result = self.shell(f"sha256sum {shlex.quote(remote)} 2>/dev/null")
        if result.returncode != 0 or not result.stdout.strip():
            return None
        return result.stdout.split()[0]

    def push_if_changed(self, local: Path, remote: str, local_sha256: str, executable: bool = False) -> bool:
        """Push `local` unless the device already holds a file with the same sha256. Returns True if pushed."""
        if self.sha256(remote) == local_sha256:
            return False
        parent = remote.rsplit("/", 1)[0]
        self.shell(f"mkdir -p {shlex.quote(parent)}", check=True)
        proc = self._adb("push", str(local), remote, timeout=1800)
        if proc.returncode != 0:
            raise DeviceError(f"adb push failed for {local.name}: {(proc.stderr or proc.stdout).strip()[-300:]}")
        if self.sha256(remote) != local_sha256:
            raise DeviceError(f"checksum mismatch after pushing {local.name}")
        if executable:
            self.shell(f"chmod 755 {shlex.quote(remote)}", check=True)
        return True

    def facts(self) -> dict:
        facts = parse_facts(self.shell(_FACTS_SCRIPT, check=True).stdout, self.serial)
        facts["rooted"] = self.has_root()
        return facts

    def has_root(self) -> bool:
        result = self.shell("su 0 id -u 2>/dev/null")
        return result.returncode == 0 and result.stdout.strip() == "0"

    def drop_caches(self) -> None:
        """Evict the page cache so the next model load reads from storage. Needs root."""
        self.shell('su 0 sh -c "sync; echo 3 > /proc/sys/vm/drop_caches"', check=True)
