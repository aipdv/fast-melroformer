"""CPU seconds burned by one Core ML ANE predict (busy-waiting or host-side work would show here)."""
import resource, sys, time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import gpu_lock  # noqa: E402

with gpu_lock():
    from fastmbr.hybrid import AneCall
    a = AneCall()
    x = torch.randn(60, 1101, 384, dtype=torch.float16)
    a(x)
    r0 = resource.getrusage(resource.RUSAGE_SELF); t = time.perf_counter()
    a(x)
    r1 = resource.getrusage(resource.RUSAGE_SELF); w = time.perf_counter() - t
    print(f"predict wall {w:.2f} s, user CPU {r1.ru_utime - r0.ru_utime:.2f} s, sys CPU {r1.ru_stime - r0.ru_stime:.2f} s")
