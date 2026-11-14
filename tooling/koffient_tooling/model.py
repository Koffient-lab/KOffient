"""Model passport: identity and the actual contents of a GGUF file."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path

K_QUANT_BLOCK = 256  # K-quants need tensor rows that are a multiple of this, else llama.cpp falls back


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_tensors(tensors: list[tuple[str, str, int, int]]) -> dict:
    """Aggregate (name, type, n_bytes, n_params) rows into the size-related passport fields."""
    by_type: dict[str, int] = defaultdict(int)
    total_bytes = total_params = 0
    for _name, type_name, n_bytes, n_params in tensors:
        by_type[type_name] += n_bytes
        total_bytes += n_bytes
        total_params += n_params
    names = {row[0] for row in tensors}
    return {
        "params": total_params,
        "bits_per_param": round(total_bytes * 8 / total_params, 3) if total_params else None,
        "tensor_types_mb": {
            name: round(size / 1e6, 1) for name, size in sorted(by_type.items(), key=lambda kv: -kv[1])
        },
        # The vocabulary matrix is stored twice when the file carries its own output head.
        "separate_output_head": "output.weight" in names,
    }


def describe_model(path: Path, source: str | None = None) -> dict:
    """Read the `model` block of a record from a GGUF file."""
    import gguf  # heavy import (numpy), keep it out of module load

    reader = gguf.GGUFReader(str(path))

    def field(key: str):
        entry = reader.fields.get(key)
        return entry.contents() if entry is not None else None

    arch = field("general.architecture")
    file_type = field("general.file_type")
    try:
        quant_label = gguf.LlamaFileType(file_type).name.removeprefix("MOSTLY_").lower()
    except (ValueError, TypeError):
        quant_label = None
    width = field(f"{arch}.embedding_length") if arch else None

    passport = {
        "name": field("general.name") or path.stem,
        "file": path.name,
        "source": source,
        "sha256": sha256_file(path),
        "file_bytes": path.stat().st_size,
        "architecture": arch,
        "quant_label": quant_label,
        "embedding_length": width,
        "block_count": field(f"{arch}.block_count") if arch else None,
        "k_quant_compatible_width": (width % K_QUANT_BLOCK == 0) if isinstance(width, int) else None,
    }
    passport.update(
        summarize_tensors(
            [(t.name, t.tensor_type.name, int(t.n_bytes), int(t.n_elements)) for t in reader.tensors]
        )
    )
    return passport
