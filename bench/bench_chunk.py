"""Per-chunk speed + fidelity of model variants, interleaved A/B to cancel heat drift.

uv run bench/bench_chunk.py ref16 fast16 [--reps 3]
Fidelity = SDR of the vocals output vs the fp32 reference on 3 real TV chunks.
"""
import argparse, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CHUNK, chunks, gpu_lock, load_reference, reference_outputs, sdr, timeit  # noqa: E402


def build(name):
    if name == "ref32":
        return load_reference(torch.float32)
    if name == "ref16":
        return load_reference(torch.float16)
    if name == "fast16":
        from fastmbr.patch import accelerate
        return accelerate(load_reference(torch.float16))
    if ":" in name:  # "fastmbr.metal:build" -> any module exposing build() -> model on mps
        import importlib
        mod, fn = name.split(":")
        return getattr(importlib.import_module(mod), fn)()
    raise SystemExit(f"unknown variant {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variants", nargs="+")
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    x = chunks(3)
    with gpu_lock():
        ref = reference_outputs(x)
        models = {v: build(v) for v in a.variants}
        dtype = {v: next(m.parameters()).dtype for v, m in models.items()}
        times = {v: [] for v in models}
        fid = {}
        with torch.inference_mode():
            for v, m in models.items():  # fidelity on all 3 chunks
                y = np.stack([m(torch.from_numpy(c[None]).to("mps")).float().cpu().numpy()[0] for c in x])
                fid[v] = sdr(ref, y)
            xc = torch.from_numpy(x[1:2])
            for _ in range(a.reps):  # interleave A B A B
                for v, m in models.items():
                    xi = xc.to("mps")
                    times[v].append(timeit(lambda: m(xi), warmup=0, reps=1))
    secs = CHUNK / 44100
    for v in models:
        t = float(np.median(times[v]))
        print(f"{v:10s} {t:6.2f} s/chunk  {secs/t:5.2f}x real time  fidelity {fid[v]:5.1f} dB")


if __name__ == "__main__":
    main()
