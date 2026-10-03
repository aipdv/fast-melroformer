"""Latency of single ANE blocks: which part of the stack is slow? (1 layer, time or freq, q_tiles)

bench/guard.sh .venv/bin/python bench/ane_part.py time 3
"""
import sys, time
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import OUT, gpu_lock, load_reference  # noqa: E402
from fastmbr.ane import AneStack  # noqa: E402

kind, tiles = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 3
frames = int(sys.argv[3]) if len(sys.argv) > 3 else 1101
with gpu_lock():
    stack = AneStack(load_reference(torch.float32, device="cpu"), frames, q_tiles=tiles, layers=[0], kinds=(kind,)).eval()
    x = torch.randn(1, 384, 60, frames)
    ml = ct.convert(torch.jit.trace(stack, x), inputs=[ct.TensorType(name="x", shape=x.shape, dtype=np.float16)],
                    convert_to="mlprogram", compute_precision=ct.precision.FLOAT16,
                    minimum_deployment_target=ct.target.macOS15, skip_model_load=True)
    pkg = OUT / "part.mlpackage"
    ml.save(str(pkg))
    mm = ct.models.MLModel(str(pkg), compute_units=ct.ComputeUnit.CPU_AND_NE)
    xi = {"x": x.numpy().astype(np.float16)}
    mm.predict(xi)
    ts = []
    for _ in range(5):
        t = time.perf_counter(); mm.predict(xi); ts.append(time.perf_counter() - t)
    print(f"{kind} block, q_tiles={tiles}, frames={frames}: {sorted(ts)[2]*1000:.0f} ms")
