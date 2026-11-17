#!/usr/bin/env python3
"""End-to-end test (requires onnx + numpy and compiled koff CLI).

Creates an ONNX model with initializers deliberately stored in the opposite
order to their first use, then tests original/topological .koff artifacts.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


def run(*args: str) -> None:
    subprocess.run(list(args), check=True)


def main() -> None:
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    from onnx.reference import ReferenceEvaluator

    ap = argparse.ArgumentParser()
    ap.add_argument("--koff-bin", required=True, help="Compiled koff binary")
    args = ap.parse_args()
    script = Path(__file__).resolve().parents[1] / "tooling" / "pack_onnx_topology.py"
    with tempfile.TemporaryDirectory(prefix="koffient-it-") as temp:
        p = Path(temp)
        late = numpy_helper.from_array(np.array([[0.5, 1], [2, 3]], dtype=np.float32), name="W_late")
        early = numpy_helper.from_array(np.array([[1, 0], [0, 2]], dtype=np.float32), name="W_early")
        inputs = [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2])]
        outputs = [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2])]
        graph = helper.make_graph([
            helper.make_node("MatMul", ["X", "W_early"], ["H"]),
            helper.make_node("MatMul", ["H", "W_late"], ["Y"]),
        ], "topology-test", inputs, outputs, initializer=[late, early])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        source = p / "source.onnx"
        onnx.save_model(model, str(source))
        x = np.array([[2, 3]], dtype=np.float32)
        expected = ReferenceEvaluator(onnx.load(str(source))).run(None, {"X": x})[0]

        for policy in ["original", "topology"]:
            artifact = p / f"{policy}.koff"
            run(sys.executable, str(script), str(source), str(artifact),
                "--koff-bin", args.koff_bin, "--policy", policy)
            bundle = p / policy
            run(args.koff_bin, "extract-bundle", str(artifact), str(bundle))
            restored = onnx.load(str(bundle / "model.onnx"), load_external_data=True)
            onnx.checker.check_model(restored)
            result = ReferenceEvaluator(restored).run(None, {"X": x})[0]
            np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)
            run(args.koff_bin, "validate", str(artifact))

        original = (p / "original" / "weights.bin").read_bytes()
        optimized = (p / "topology" / "weights.bin").read_bytes()
        assert original != optimized, "Topology layout did not change physical order"
        assert original[:late.raw_data.__len__()] == late.raw_data
        assert optimized[:early.raw_data.__len__()] == early.raw_data
        print("PASS: two valid KOFF files, distinct physical layouts, identical ONNX inference")


if __name__ == "__main__":
    main()
