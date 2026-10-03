"""Can the Neural Engine run this model's attention? Toy attention block (qkv conv, attention, out conv).

uv run bench/ane_attn_probe.py {time|freq} {sdpa|einsum} [frames]
time: 60 band-sequences of `frames` tokens.  freq: `frames` sequences of 60 band-tokens.
Prints per-op device placement from MLComputePlan, then latency per compute unit.
"""
import collections, sys, time
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from common import gpu_lock

AXIS, KIND = sys.argv[1], sys.argv[2]
FR = int(sys.argv[3]) if len(sys.argv) > 3 else 1101
BANDS, DIM, H, DH = 60, 384, 8, 64
OUT = Path(__file__).resolve().parents[1] / "out"


class Attn(nn.Module):
    """x: (1, C, bands, frames). Attention runs along frames (time) or bands (freq)."""

    def __init__(self):
        super().__init__()
        self.qkv = nn.Conv2d(DIM, 3 * H * DH, 1, bias=False)
        self.out = nn.Conv2d(H * DH, DIM, 1, bias=False)

    def forward(self, x):
        if AXIS == "freq":
            x = x.transpose(2, 3)  # (1, C, frames, bands): sequences are frames, tokens are bands
        S, N = (FR, BANDS) if AXIS == "freq" else (BANDS, FR)  # fixed ints: ANE needs static shapes
        q, k, v = self.qkv(x).chunk(3, dim=1)  # each (1, H*DH, S, N): S sequences, N tokens
        if KIND == "sdpa":
            def heads(t):  # -> (S, H, N, DH)
                return t.reshape(H, DH, S, N).permute(2, 0, 3, 1)
            o = F.scaled_dot_product_attention(heads(q), heads(k), heads(v))  # (S, H, N, DH)
            o = o.permute(1, 3, 0, 2).reshape(1, H * DH, S, N)
        else:  # Apple ml-ane-transformers style: per-head einsum, sequences ride in dim 2
            outs = []
            for qh, kh, vh in zip(q.split(DH, 1), k.split(DH, 1), v.split(DH, 1)):
                w = torch.einsum("bcsq,bcsk->bsqk", qh, kh) * DH**-0.5
                outs.append(torch.einsum("bsqk,bcsk->bcsq", w.softmax(dim=-1), vh))
            o = torch.cat(outs, dim=1)
        o = self.out(o)
        return o.transpose(2, 3) if AXIS == "freq" else o


if __name__ == "__main__":
    m = Attn().eval()
    for p in m.parameters():
        nn.init.normal_(p, std=0.02)
    x = torch.randn(1, DIM, BANDS, FR)
    flops = 2 * BANDS * FR * DIM * 4 * H * DH + (4 * BANDS * FR * FR * H * DH if AXIS == "time" else 4 * FR * BANDS * BANDS * H * DH)
    with gpu_lock():
        ml = ct.convert(torch.jit.trace(m, x), inputs=[ct.TensorType(name="x", shape=x.shape)], convert_to="mlprogram",
                        compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.macOS15)
        pkg = OUT / f"attn_{AXIS}_{KIND}_{FR}.mlpackage"
        ml.save(str(pkg))
        compiled = ct.models.MLModel(str(pkg), compute_units=ct.ComputeUnit.CPU_AND_NE).get_compiled_model_path()
        plan = ct.models.compute_plan.MLComputePlan.load_from_path(compiled, compute_units=ct.ComputeUnit.CPU_AND_NE)
        fn = plan.model_structure.program.functions["main"]
        place = collections.Counter()
        for op in fn.block.operations:
            u = plan.get_compute_device_usage_for_mlprogram_operation(op)
            if u is not None:
                place[(op.operator_name, type(u.preferred_compute_device).__name__)] += 1
        print("placement:", dict(place))
        xi = {"x": x.numpy()}
        for cu in (ct.ComputeUnit.CPU_AND_NE, ct.ComputeUnit.CPU_AND_GPU):
            mm = ct.models.MLModel(str(pkg), compute_units=cu)
            mm.predict(xi)
            ts = []
            for _ in range(5):
                t = time.perf_counter(); mm.predict(xi); ts.append(time.perf_counter() - t)
            t = sorted(ts)[2]
            print(f"{AXIS} {KIND} frames={FR} {cu.name:12s} {t*1000:7.1f} ms  {flops/t/1e12:5.2f} TFLOPS")
