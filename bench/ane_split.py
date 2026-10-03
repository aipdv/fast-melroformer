"""Where does an ane-engine chunk spend time: GPU/CPU steps, input prep, Core ML predict, output copy back."""
import sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import chunks, gpu_lock  # noqa: E402

with gpu_lock():
    from fastmbr import hybrid
    m = hybrid.build()
    ane = m.layers[0][0]
    T = {"prep": 0.0, "predict": 0.0, "back_copy": 0.0}
    ml = ane.ml

    def fwd(x):
        dev, dt = x.device, x.dtype
        t = time.perf_counter(); a = x.permute(2, 0, 1)[None].to(torch.float16).contiguous().cpu().numpy(); T["prep"] += time.perf_counter() - t
        t = time.perf_counter(); y = ml.predict({"x": a})["y"]; T["predict"] += time.perf_counter() - t
        t = time.perf_counter(); r = torch.from_numpy(np.ascontiguousarray(y[0].transpose(1, 2, 0))).to(dev, dt); T["back_copy"] += time.perf_counter() - t
        return r

    ane.forward = fwd
    xs = torch.from_numpy(chunks(3)).to("mps")
    with torch.inference_mode():
        m(xs[:1]); torch.mps.synchronize()
        for k in T: T[k] = 0.0
        t0 = time.perf_counter()
        for i in range(3):
            m(xs[i:i+1]).cpu()
        tot = (time.perf_counter() - t0) / 3
    T = {k: v / 3 for k, v in T.items()}
    print(f"per chunk {tot:.2f} s: " + ", ".join(f"{k} {v:.2f}" for k, v in T.items()) + f", rest (GPU/CPU model steps) {tot - sum(T.values()):.2f}")
