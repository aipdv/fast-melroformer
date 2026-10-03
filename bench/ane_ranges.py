"""Activation ranges per block of the ANE-layout stack (torch fp32). Explains fp16 precision loss on the ANE."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ane_stack import capture  # noqa: E402
from common import gpu_lock, load_reference  # noqa: E402
from fastmbr.ane import AneStack  # noqa: E402

with gpu_lock():
    x, _ = capture()
    st = AneStack(load_reference(torch.float32, device="cpu")).eval()
    h = torch.from_numpy(x)
    with torch.inference_mode():
        for i, b in enumerate(st.blocks):
            a = b.attn(h); r1 = a + h
            hid = b.ff.a(b.ff.norm(r1)); f = b.ff.b(torch.nn.functional.gelu(hid)); r2 = f + r1
            out = b.norm(r2)
            print(f"b{i:2d} in {h.abs().max():7.1f} attn {a.abs().max():7.1f} ffhid {hid.abs().max():7.1f} "
                  f"ffout {f.abs().max():7.1f} resid {r2.abs().max():7.1f} out {out.abs().max():7.1f} "
                  f"|gamma|max {b.norm.g.abs().max():6.2f}")
            h = out
