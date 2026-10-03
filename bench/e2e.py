"""End to end on the 3 min TV clip: speed of each engine and fidelity vs the fp32 reference (same chunking).

bench/guard.sh .venv/bin/python bench/e2e.py gpu ane dual
The fp32 reference vocals are cached in out/e2e_ref32.npy (first run computes them, ~4 min).
"""
import gc, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CLIP, OUT, gpu_lock, sdr  # noqa: E402
from fastmbr.separate import load_audio, separate  # noqa: E402

with gpu_lock():
    audio = load_audio(CLIP)
    ref_p = OUT / "e2e_ref32.npy"
    if not ref_p.exists():
        ref, _ = separate(audio, "ref32")
        np.save(ref_p, ref.astype(np.float32))
        gc.collect(); torch.mps.empty_cache()
    ref = np.load(ref_p)
    for engine in sys.argv[1:]:
        v, secs = separate(audio, engine)
        print(f"  -> fidelity vs fp32 reference: {sdr(ref, v):.1f} dB")
        del v
        gc.collect(); torch.mps.empty_cache()
