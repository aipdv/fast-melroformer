"""Per-block fidelity of the ANE stack vs the same stack in torch fp32 (finds where fp16 on the ANE loses precision).

ANE_ALL_OUTPUTS=1 bench/guard.sh .venv/bin/python bench/ane_blocks.py
"""
import os, sys
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
assert os.environ.get("ANE_ALL_OUTPUTS") == "1"
from ane_stack import capture  # noqa: E402
from common import OUT, gpu_lock, load_reference, sdr  # noqa: E402
from fastmbr.ane import AneStack  # noqa: E402

with gpu_lock():
    x, _ = capture()
    stack = AneStack(load_reference(torch.float32, device="cpu")).eval()
    xt = torch.from_numpy(x)
    with torch.inference_mode():
        ref = [r.numpy() for r in stack(xt)]
        traced = torch.jit.trace(stack, xt)
    names = [f"b{i}" for i in range(len(ref))]
    ml = ct.convert(traced, inputs=[ct.TensorType(name="x", shape=xt.shape, dtype=np.float16)],
                    outputs=[ct.TensorType(name=n, dtype=np.float16) for n in names], convert_to="mlprogram",
                    compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.macOS15, skip_model_load=True)
    pkg = OUT / "ane_blocks.mlpackage"
    ml.save(str(pkg))
    del ml, traced
    out = ct.models.MLModel(str(pkg), compute_units=ct.ComputeUnit.CPU_AND_NE).predict({"x": x.astype(np.float16)})
    for i, n in enumerate(names):
        kind = "time" if i % 2 == 0 else "freq"
        print(f"block {i:2d} ({kind}): {sdr(ref[i], out[n].astype(np.float32)):5.1f} dB")
