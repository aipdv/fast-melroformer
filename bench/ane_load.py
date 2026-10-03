"""How long does loading the ANE stack take (first load compiles; later loads should hit the cache)?"""
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import gpu_lock  # noqa: E402

with gpu_lock():
    from fastmbr.hybrid import AneCall
    t = time.perf_counter(); a = AneCall(); print(f"load {time.perf_counter() - t:.1f} s")
