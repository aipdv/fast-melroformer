"""How fast is the Neural Engine on this model's shapes? Toy FF stack as 1x1 convs, (1, C, bands, frames).

uv run bench/ane_probe.py [frames]
"""
import sys, time
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from torch import nn

from common import gpu_lock

F_ = int(sys.argv[1]) if len(sys.argv) > 1 else 1101
BANDS, DIM, REPEAT = 60, 384, 4
OUT = Path(__file__).resolve().parents[1] / "out"


class FF(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Conv2d(DIM, DIM * 4, 1)
        self.b = nn.Conv2d(DIM * 4, DIM, 1)

    def forward(self, x):
        return x + self.b(torch.nn.functional.gelu(self.a(x)))


class Stack(nn.Module):
    def __init__(self):
        super().__init__()
        self.ffs = nn.ModuleList([FF() for _ in range(REPEAT)])

    def forward(self, x):
        for f in self.ffs:
            x = f(x)
        return x


if __name__ == "__main__":
    m = Stack().eval()
    for p in m.parameters():
        nn.init.normal_(p, std=0.02)
    x = torch.randn(1, DIM, BANDS, F_)
    traced = torch.jit.trace(m, x)
    flops = REPEAT * 2 * 2 * DIM * DIM * 4 * BANDS * F_
    with gpu_lock():
        ml = ct.convert(traced, inputs=[ct.TensorType(name="x", shape=x.shape)], convert_to="mlprogram",
                        compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.macOS15)
        pkg = OUT / "ane_probe.mlpackage"
        OUT.mkdir(exist_ok=True)
        ml.save(str(pkg))
        xi = {"x": x.numpy().astype(np.float32)}
        for cu in (ct.ComputeUnit.CPU_AND_NE, ct.ComputeUnit.CPU_AND_GPU, ct.ComputeUnit.ALL):
            mm = ct.models.MLModel(str(pkg), compute_units=cu)
            mm.predict(xi)
            ts = []
            for _ in range(5):
                t = time.perf_counter(); mm.predict(xi); ts.append(time.perf_counter() - t)
            t = sorted(ts)[2]
            print(f"{cu.name:12s} frames={F_} {t*1000:7.1f} ms  {flops/t/1e12:5.2f} TFLOPS")
