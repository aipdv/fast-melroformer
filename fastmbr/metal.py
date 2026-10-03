"""fast16 + custom Metal kernels for the attention glue (torch.mps.compile_shader).

    from fastmbr.metal import build
    model = build()  # fp16 MelBandRoformer on MPS

Kernels:
- qkv_rope: reads the fused qkv+gates GEMM output (b*n, 1544) once and writes q, k (RoPE
  applied, fp32 math) and v straight into the contiguous (3, b, h, n, d) layout SDPA wants.
- gate_merge: sigmoid(gate) * o and the (b, h, n, d) -> (b, n, h*d) head merge in one pass.
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from fastmbr.patch import FastAttention, accelerate

SRC = """
#include <metal_stdlib>
using namespace metal;

// grid (D/2, H, b*n). y row = [q | k | v | gates], q/k heads in split-half order.
kernel void qkv_rope(device const half* y [[buffer(0)]], device const float* cs [[buffer(1)]],
                     device const float* sn [[buffer(2)]], device half* qkv [[buffer(3)]],
                     constant uint& N [[buffer(4)]], constant uint& C [[buffer(5)]],
                     uint3 g [[thread_position_in_grid]], uint3 G [[threads_per_grid]]) {
    const uint j = g.x, h = g.y, m = g.z, half_d = G.x, D = 2 * half_d, H = G.y, M = G.z;
    const uint t = m % N, bb = m / N, inner = H * D;
    device const half* row = y + m * C + h * D + j;
    const float c = cs[t * half_d + j], s = sn[t * half_d + j];
    const uint o = ((bb * H + h) * N + t) * D + j, plane = M * inner;
    float a = row[0], b = row[half_d];
    qkv[o] = half(a * c - b * s); qkv[o + half_d] = half(b * c + a * s);
    a = row[inner]; b = row[inner + half_d];
    qkv[plane + o] = half(a * c - b * s); qkv[plane + o + half_d] = half(b * c + a * s);
    qkv[2 * plane + o] = row[2 * inner]; qkv[2 * plane + o + half_d] = row[2 * inner + half_d];
}

// grid (D, H, b*n). out[m, h*D + d] = o[bb, h, t, d] * sigmoid(y[m, gate0 + h])
kernel void gate_merge(device const half* o [[buffer(0)]], device const half* y [[buffer(1)]],
                       device half* out [[buffer(2)]], constant uint& N [[buffer(3)]],
                       constant uint& C [[buffer(4)]], constant uint& gate0 [[buffer(5)]],
                       uint3 g [[thread_position_in_grid]], uint3 G [[threads_per_grid]]) {
    const uint d = g.x, h = g.y, m = g.z, D = G.x, H = G.y;
    const uint t = m % N, bb = m / N;
    const float gate = 1.0f / (1.0f + exp(-float(y[m * C + gate0 + h])));
    out[(m * H + h) * D + d] = half(float(o[((bb * H + h) * N + t) * D + d]) * gate);
}
"""

_lib = None


def lib():
    global _lib
    if _lib is None:
        _lib = torch.mps.compile_shader(SRC)
    return _lib


class MetalAttention(FastAttention):
    def rope32(self, n, device):
        key = (n, "f32", device)
        if key not in self._rope:
            ang = torch.arange(n, device=device, dtype=torch.float32)[:, None] * self.freqs.to(device)
            self._rope[key] = (ang.cos().contiguous(), ang.sin().contiguous())
        return self._rope[key]

    def forward(self, x):
        b, n, _ = x.shape
        h, d, inner = self.h, self.d, self.inner
        y = F.linear(F.rms_norm(x, (x.shape[-1],), self.gamma, eps=1e-12), self.w, self.b)
        C = y.shape[-1]
        cos, sin = self.rope32(n, x.device)
        qkv = torch.empty(3, b, h, n, d, device=x.device, dtype=x.dtype)
        L = lib()
        L.qkv_rope(y, cos, sin, qkv, n, C, threads=(d // 2, h, b * n), group_size=(d // 2, h, 1))
        o = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
        m = torch.empty(b, n, inner, device=x.device, dtype=x.dtype)
        L.gate_merge(o, y, m, n, C, 3 * inner, threads=(d, h, b * n), group_size=(d, 4, 1))
        return self.out(m)


def build():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
    from common import load_reference
    from audio_separator.separator.uvr_lib_v5.roformer import mel_band_roformer as mbr

    model = accelerate(load_reference(torch.float16))
    for mod in list(model.modules()):
        if isinstance(mod, mbr.Transformer):
            for pair in mod.layers:
                pair[0].__class__ = MetalAttention  # same buffers, new forward
    return model
