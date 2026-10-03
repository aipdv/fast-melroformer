"""Do the GPU and the ANE overlap? Time each alone, then together (ANE in a thread, GPU in the main thread)."""
import sys, threading, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import chunks, gpu_lock  # noqa: E402

with gpu_lock():
    from fastmbr.hybrid import AneCall
    from fastmbr.metal import build
    gpu = build()
    ane = AneCall()
    x = torch.from_numpy(chunks(1)).to("mps")
    feats = torch.randn(60, 1101, 384, dtype=torch.float16)

    def g():
        with torch.inference_mode():
            gpu(x); torch.mps.synchronize()

    def a():
        ane(feats)

    for f in (g, a):
        f()
    t = time.perf_counter(); g(); tg = time.perf_counter() - t
    t = time.perf_counter(); a(); ta = time.perf_counter() - t
    # GIL check: does a Python loop progress while predict runs?
    ticks = [0]; done = threading.Event()
    th = threading.Thread(target=lambda: (a(), done.set())); th.start()
    while not done.is_set():
        ticks[0] += 1
    th.join()
    print(f"alone: GPU chunk {tg:.2f} s, ANE stack {ta:.2f} s; python ticks during ANE predict: {ticks[0]:,} (0 = GIL held)")
    res = {}
    def ra():
        t0 = time.perf_counter(); a(); a(); res["ane"] = time.perf_counter() - t0
    th = threading.Thread(target=ra)
    t0 = time.perf_counter(); th.start(); g(); res["gpu"] = time.perf_counter() - t0; th.join()
    wall = time.perf_counter() - t0
    print(f"together: GPU chunk {res['gpu']:.2f} s, 2x ANE stack {res['ane']:.2f} s, wall {wall:.2f} s "
          f"(serial would be {tg + 2*ta:.2f} s)")
