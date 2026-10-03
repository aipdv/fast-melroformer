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
    import json
    res_p = Path(__file__).resolve().parents[1] / "docs" / "e2e_results.json"
    res = json.loads(res_p.read_text()) if res_p.exists() else {}
    for engine in sys.argv[1:]:
        v, secs = separate(audio, engine)
        f = sdr(ref, v)
        print(f"  -> fidelity vs fp32 reference: {f:.1f} dB")
        res[engine] = {"seconds": round(secs, 1), "x_real_time": round(audio.shape[1] / 44100 / secs, 2), "fidelity_db": round(f, 1)}
        res_p.parent.mkdir(exist_ok=True)
        res_p.write_text(json.dumps(res, indent=1))
        del v
        gc.collect(); torch.mps.empty_cache()
