"""Kim Mel-RoFormer with the 12 transformer blocks on the Neural Engine (Core ML) and the rest on the GPU.

    from fastmbr.hybrid import build
    model = build()  # call like the reference: model(audio (1, 2, 485100) on mps) -> vocals

The Core ML package comes from `bench/ane_stack.py convert` (out/ane_stack_1101.mlpackage).
"""
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "out" / "ane_stack_1101.mlpackage"


class AneCall(nn.Module):
    """Stands in for the first time-transformer: gets (bands, frames, d), returns the whole stack's output."""

    def __init__(self, pkg=PKG):
        super().__init__()
        import shutil
        import coremltools as ct
        compiled = pkg.with_suffix(".mlmodelc")  # keep the compiled model: compiling the big graph takes minutes
        if not compiled.exists():
            m = ct.models.MLModel(str(pkg), compute_units=ct.ComputeUnit.CPU_AND_NE)
            shutil.copytree(m.get_compiled_model_path(), compiled)
            del m
        self.ml = ct.models.CompiledMLModel(str(compiled), compute_units=ct.ComputeUnit.CPU_AND_NE)

    def forward(self, x):  # (f, t, d) -> (1, d, f, t) -> ANE -> back
        dev, dt = x.device, x.dtype
        a = x.permute(2, 0, 1)[None].to(torch.float16).cpu().numpy()
        y = self.ml.predict({"x": a})["y"]
        return torch.from_numpy(np.ascontiguousarray(y[0].transpose(1, 2, 0))).to(dev, dt)


def build():
    sys.path.insert(0, str(ROOT / "bench"))
    from common import load_reference

    model = load_reference(torch.float16)
    model.layers = nn.ModuleList([nn.ModuleList([AneCall(), nn.Identity()])])  # forward loop runs once
    return model
