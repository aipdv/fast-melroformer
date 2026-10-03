"""One attention block, time and freq shapes: fast16 vs metal speed + error vs fp32 reference.

.venv/bin/python bench/metal_micro.py
Random weights (timing does not depend on them). Holds the GPU lock.
"""
import statistics, sys, time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import gpu_lock, sync  # noqa: E402

D = "mps"


def tm(fn, reps=10):
    fn(); sync(D)
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); sync(D); ts.append(time.perf_counter() - t)
    return statistics.median(ts) * 1000


def main():
    from audio_separator.separator.uvr_lib_v5.roformer import mel_band_roformer as mbr
    from rotary_embedding_torch import RotaryEmbedding
    from fastmbr.patch import FastAttention
    from fastmbr import metal

    torch.manual_seed(0)
    ref = mbr.Attention(384, heads=8, dim_head=64, rotary_embed=RotaryEmbedding(64)).to(D).eval()
    variants = {}
    variants["fast16"] = FastAttention(ref).half()
    mt = FastAttention(ref).half(); mt.__class__ = metal.MetalAttention
    variants["metal"] = mt
    with torch.inference_mode():
        for name, shape in [("time", (60, 1101, 384)), ("freq", (1101, 60, 384))]:
            x = torch.randn(*shape, device=D)
            y32 = ref(x)
            xh = x.half()
            for v, m in variants.items():
                err = (m(xh).float() - y32).norm() / y32.norm()
                print(f"{name} {v:8s} {tm(lambda: m(xh)):7.1f} ms  rel err {err:.2e}")


if __name__ == "__main__" and len(sys.argv) == 1:
    with gpu_lock():
        main()


def ops():
    """Op-level split of the metal attention block (sync around each op)."""
    import torch.nn.functional as F
    from audio_separator.separator.uvr_lib_v5.roformer import mel_band_roformer as mbr
    from rotary_embedding_torch import RotaryEmbedding
    from fastmbr.patch import FastAttention
    from fastmbr import metal

    ref = mbr.Attention(384, heads=8, dim_head=64, rotary_embed=RotaryEmbedding(64)).to(D).eval()
    a = FastAttention(ref).half()
    L = metal.lib()
    with torch.inference_mode():
        for name, (b, n) in [("time", (60, 1101)), ("freq", (1101, 60))]:
            x = torch.randn(b, n, 384, device=D, dtype=torch.float16)
            xn = F.rms_norm(x, (384,), a.gamma, eps=1e-12)
            y = F.linear(xn, a.w, a.b)
            cos, sin = metal.MetalAttention.rope32(a, n, D)
            qkv = torch.empty(3, b, 8, n, 64, device=D, dtype=x.dtype)
            L.qkv_rope(y, cos, sin, qkv, n, 1544, threads=(32, 8, b * n), group_size=(32, 8, 1))
            o = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
            mg = torch.empty(b, n, 512, device=D, dtype=x.dtype)
            r = {
                "rms_norm": lambda: F.rms_norm(x, (384,), a.gamma, eps=1e-12),
                "linear+b": lambda: F.linear(xn, a.w, a.b),
                "linear": lambda: F.linear(xn, a.w),
                "qkv_rope": lambda: L.qkv_rope(y, cos, sin, qkv, n, 1544, threads=(32, 8, b * n), group_size=(32, 8, 1)),
                "sdpa": lambda: F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2]),
                "gate_merge": lambda: L.gate_merge(o, y, mg, n, 1544, 1536, threads=(64, 8, b * n), group_size=(64, 4, 1)),
                "out": lambda: a.out(mg),
                "out+res": lambda: torch.addmm(x.view(-1, 384), mg.view(-1, 512), a.out.weight.t()),
                "residual": lambda: x + x,
                "empty": lambda: torch.empty(3, b, 8, n, 64, device=D, dtype=x.dtype),
            }
            for k, f in r.items():
                print(f"{name} {k:10s} {tm(f):6.1f} ms")


if __name__ == "__main__" and "ops" in sys.argv:
    with gpu_lock():
        ops()


def ff_ops():
    """FF block variants and 2D-vs-3D GEMM input."""
    import torch.nn.functional as F
    from audio_separator.separator.uvr_lib_v5.roformer import mel_band_roformer as mbr

    ff = mbr.FeedForward(384).to(D).half().eval()
    l1, l2 = ff.net[1], ff.net[4]
    with torch.inference_mode():
        for name, (b, n) in [("time", (60, 1101)), ("freq", (1101, 60))]:
            x = torch.randn(b, n, 384, device=D, dtype=torch.float16)
            x2 = x.view(-1, 384)
            h = torch.randn(b * n, 1536, device=D, dtype=torch.float16)
            r = {
                "ff ref 3D": lambda: ff(x) + x,
                "ff 2D": lambda: ff(x2) + x2,
                "l1 3D +b": lambda: l1(x),
                "l1 2D +b": lambda: l1(x2),
                "l1 2D nob": lambda: F.linear(x2, l1.weight),
                "gelu": lambda: F.gelu(h),
                "l2 2D +b": lambda: l2(h),
                "l2 2D nob": lambda: F.linear(h, l2.weight),
                "l2 addmm res": lambda: torch.addmm(x2, h, l2.weight.t()),
            }
            for k, f in r.items():
                print(f"{name} {k:13s} {tm(f):6.1f} ms")


if __name__ == "__main__" and "ff" in sys.argv:
    with gpu_lock():
        ff_ops()
