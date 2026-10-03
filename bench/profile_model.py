"""Module-level time split of one 11 s chunk (syncs around each module, so totals run a bit high).

uv run bench/profile_model.py fast16
"""
import collections, sys, time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench_chunk import build  # noqa: E402
from common import chunks, gpu_lock, sync  # noqa: E402


def wrap(T, name, mod):
    f = mod.forward

    def w(*a, **k):
        sync("mps"); t = time.perf_counter(); r = f(*a, **k); sync("mps"); T[name] += time.perf_counter() - t
        return r

    mod.forward = w


if __name__ == "__main__":
    with gpu_lock():
        m = build(sys.argv[1])
        T = collections.Counter()
        wrap(T, "band_split", m.band_split)
        for tt, ft in m.layers:
            for kind, blk in (("time", tt), ("freq", ft)):
                for attn, ff in blk.layers:
                    wrap(T, f"{kind}_attn", attn)
                    wrap(T, f"{kind}_ff", ff)
        for me in m.mask_estimators:
            wrap(T, "mask_est", me)
        x = torch.from_numpy(chunks(1)).to("mps")
        with torch.inference_mode():
            m(x); T.clear()
            sync("mps"); t0 = time.perf_counter(); m(x); sync("mps"); tot = time.perf_counter() - t0
        print(f"total {tot:.2f} s per chunk")
        for k, v in T.most_common():
            print(f"  {k:11s} {v:5.2f} s {100*v/tot:4.0f}%")
        print(f"  other (STFT, gather, scatter, iSTFT, reshapes) {tot - sum(T.values()):.2f} s")
