#!/usr/bin/env python3
"""Export a real ONNX model to a topology-aware .koff v0.2 artifact.

Requires: pip install onnx numpy
Rust CLI: cargo (default) or --koff-bin path/to/koff

This tool does NOT improve inference latency by itself. It rewrites ONNX initializers
into external_data offsets, physically places their bytes according to a Rust
layout plan, and packages the graph/weights/plan into a single .koff file.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def get_cli(bin_path: str | None) -> list[str]:
    if bin_path:
        return [bin_path]
    if shutil.which("koff"):
        return ["koff"]
    native_root = Path(__file__).resolve().parents[1]
    if not shutil.which("cargo"):
        raise RuntimeError("Rust Cargo was not found; install Rust or provide --koff-bin")
    return ["cargo", "run", "--quiet", "--manifest-path", str(native_root / "Cargo.toml"),
            "--package", "koffient-cli", "--"]


def run_cli(cli: list[str], *args: str) -> None:
    subprocess.run([*cli, *args], check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Topology-aware ONNX to KOFF exporter")
    parser.add_argument("input", type=Path, help="Original ONNX model (embedded/external initializers)")
    parser.add_argument("output", type=Path, help="Result .koff file")
    parser.add_argument("--policy", choices=["topology", "original"], default="topology")
    parser.add_argument("--alignment", type=int, default=64)
    parser.add_argument("--koff-bin", help="Path to a compiled `koff` CLI executable")
    parser.add_argument("--keep-stage", type=Path, help="Keep intermediate graph, weights, TSV")
    args = parser.parse_args()
    if args.alignment < 16 or args.alignment > 1048576 or args.alignment & (args.alignment - 1):
        parser.error("--alignment must be a power of two from 16 through 1048576")

    try:
        import onnx
        from onnx import numpy_helper
        from onnx.external_data_helper import set_external_data
    except ImportError as exc:
        raise SystemExit("This script requires `pip install onnx numpy`") from exc

    model = onnx.load(str(args.input), load_external_data=True)
    if not model.graph.initializer:
        raise ValueError("Model has no dense initializers. Empty/sparse-only models are not supported.")
    for node in model.graph.node:
        for attr in node.attribute:
            if attr.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS):
                raise ValueError("Control-flow subgraphs (If/Loop/Scan) are not supported in this MVP exporter")
    if len({t.name for t in model.graph.initializer}) != len(model.graph.initializer):
        raise ValueError("Duplicate initializer names")
    if len(model.graph.sparse_initializer):
        raise ValueError("Sparse initializers are not supported in this MVP exporter")

    cli = get_cli(args.koff_bin)
    tensors: dict[int, bytes] = {}
    tensor_names: dict[str, int] = {}
    for index, tensor in enumerate(model.graph.initializer, start=1):
        if tensor.data_type == onnx.TensorProto.STRING:
            raise ValueError("String initializer is not supported")
        raw = numpy_helper.to_array(tensor).tobytes(order="C")
        if not raw:
            raise ValueError(f"Empty initializer: {tensor.name}")
        tensors[index] = raw
        tensor_names[tensor.name] = index

    producers: dict[str, int] = {}
    graph_lines = [f"weight {i} {len(data)}" for i, data in tensors.items()]
    for index, node in enumerate(model.graph.node, start=1):
        parents = sorted({producers[x] for x in node.input if x in producers})
        weights = sorted({tensor_names[x] for x in node.input if x in tensor_names})
        graph_lines.append("node {} {} {}".format(
            index,
            ",".join(map(str, parents)) if parents else "-",
            ",".join(map(str, weights)) if weights else "-",
        ))
        for output in node.output:
            if output:
                if output in producers:
                    raise ValueError(f"Duplicate node output name: {output}")
                producers[output] = index

    def produce(stage: Path) -> None:
        stage.mkdir(parents=True, exist_ok=True)
        graph_path = stage / "graph.txt"
        layout_path = stage / "layout.tsv"
        graph_path.write_text("\n".join(graph_lines) + "\n", encoding="utf-8")
        run_cli(cli, "topo-export", str(graph_path), str(layout_path), str(args.alignment), args.policy)
        placement: dict[int, tuple[int, int]] = {}
        for row in layout_path.read_text(encoding="utf-8").splitlines():
            if not row or row.startswith("#"):
                continue
            tensor_id, _step, offset, size = map(int, row.split("\t"))
            placement[tensor_id] = (offset, size)
        if len(placement) != len(tensors):
            raise ValueError("Rust layout is missing model weights")
        for i, data in tensors.items():
            if i not in placement or placement[i][1] != len(data):
                raise ValueError(f"Tensor size mismatch in layout, tensor {i}")

        weights_path = stage / "weights.bin"
        blob_size = max(offset + length for offset, length in placement.values())
        with weights_path.open("wb") as fp:
            fp.truncate(blob_size)
            for tensor_id, raw in tensors.items():
                fp.seek(placement[tensor_id][0])
                fp.write(raw)

        # Rewire ONNX graph to a single external weights file. Offset is relative to
        # the start of weights.bin (NOT to the absolute .koff file offset).
        for tensor in model.graph.initializer:
            tensor_id = tensor_names[tensor.name]
            offset, length = placement[tensor_id]
            # set_external_data requires a raw_data field, even if the source
            # initializer used typed protobuf arrays instead.
            tensor.raw_data = tensors[tensor_id]
            set_external_data(tensor, location="weights.bin", offset=offset, length=length)
            tensor.ClearField("raw_data")
            for field in ("float_data", "int32_data", "string_data", "int64_data", "double_data", "uint64_data"):
                tensor.ClearField(field)

        graph_output = stage / "model.onnx"
        # All tensor bytes are now in weights.bin, not embedded in the protobuf.
        onnx.save_model(model, str(graph_output))

        # Validate that a standard ONNX reader reconstructs every initializer.
        loaded = onnx.load(str(graph_output), load_external_data=True)
        onnx.checker.check_model(loaded)
        for tensor in loaded.graph.initializer:
            digest = hashlib.sha256(numpy_helper.to_array(tensor).tobytes(order="C")).digest()
            original = hashlib.sha256(tensors[tensor_names[tensor.name]]).digest()
            if digest != original:
                raise ValueError(f"External weight round-trip failed: {tensor.name}")

        args.output.parent.mkdir(parents=True, exist_ok=True)
        run_cli(cli, "pack-external", str(graph_output), str(weights_path),
                str(layout_path), str(args.output), args.input.stem, str(args.alignment))
        run_cli(cli, "validate", str(args.output))
        print(f"KOFF created: {args.output} | weights={len(tensors)} | policy={args.policy}")

    if args.keep_stage:
        produce(args.keep_stage)
    else:
        with tempfile.TemporaryDirectory(prefix="koffient-onnx-") as temp_dir:
            produce(Path(temp_dir))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, subprocess.CalledProcessError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
