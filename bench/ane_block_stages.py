"""Stage-by-stage fidelity of one ANE block (default: last freq block, 11) fed the exact fp32 reference input."""
import sys
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ane_stack import capture  # noqa: E402
from common import OUT, gpu_lock, load_reference, sdr  # noqa: E402
from fastmbr.ane import AneStack  # noqa: E402
import fastmbr.ane  # noqa: E402,F401

IDX = int(sys.argv[1]) if len(sys.argv) > 1 else 11


class Stages(nn.Module):
    def __init__(self, b, act):
        super().__init__()
        self.b, self.act = b, act

    def forward(self, x):
        b = self.b
        a = b.attn(x); r1 = a + x
        n = b.ff.norm(r1); hid = b.ff.a(n); g = self.act(hid); f = b.ff.b(g); r2 = f + r1
        return a, r1, n, hid, g, f, r2, b.norm(r2)


NAMES = ["attn", "resid1", "ffnorm", "ffhid", "gelu", "ffout", "resid2", "out"]
with gpu_lock():
    x, _ = capture()
    st = AneStack(load_reference(torch.float32, device="cpu")).eval()
    h = torch.from_numpy(x)
    with torch.inference_mode():
        for b in st.blocks[:IDX]:
            h = b(h)
        from fastmbr.ane import gelu
        ref = [t.numpy() for t in Stages(st.blocks[IDX], torch.nn.functional.gelu)(h)]  # exact erf GELU, fp32
        m = Stages(st.blocks[IDX], gelu).eval()  # what the ANE model uses (ANE_GELU)
        traced = torch.jit.trace(m, h)
    ml = ct.convert(traced, inputs=[ct.TensorType(name="x", shape=h.shape, dtype=np.float16)],
                    outputs=[ct.TensorType(name=n, dtype=np.float16) for n in NAMES], convert_to="mlprogram",
                    compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.macOS15, skip_model_load=True)
    pkg = OUT / "ane_stages.mlpackage"
    ml.save(str(pkg))
    out = ct.models.MLModel(str(pkg), compute_units=ct.ComputeUnit.CPU_AND_NE).predict({"x": h.numpy().astype(np.float16)})
    for i, n in enumerate(NAMES):
        print(f"block {IDX} {n:7s} {sdr(ref[i], out[n].astype(np.float32)):5.1f} dB")
