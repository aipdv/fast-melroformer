"""Op-level time of one attention block (time and freq shapes) on MPS fp16, reference code path.

uv run bench/profile_attn.py
"""
import collections, time

import torch

from common import gpu_lock, load_reference, sync

D = "mps"


def profile(attn, x, reps=5):
    from audio_separator.separator.uvr_lib_v5.roformer.rotary import rotate_queries_or_keys
    from einops import rearrange

    T = collections.Counter()

    def step(name, f):
        sync(D); t = time.perf_counter(); r = f(); sync(D); T[name] += time.perf_counter() - t
        return r

    with torch.inference_mode():
        for i in range(reps + 1):
            if i == 1:
                T.clear()
            h = step("norm", lambda: attn.norm(x))
            qkv = step("qkv_linear", lambda: attn.to_qkv(h))
            q, k, v = step("split_heads", lambda: rearrange(qkv, "b n (qkv h d) -> qkv b h n d", qkv=3, h=attn.heads).contiguous())
            q = step("rope_q", lambda: rotate_queries_or_keys(attn.rotary_embed, q))
            k = step("rope_k", lambda: rotate_queries_or_keys(attn.rotary_embed, k))
            o = step("sdpa", lambda: attn.attend(q, k, v))
            g = step("gates", lambda: o * rearrange(attn.to_gates(h), "b n h -> b h n 1").sigmoid())
            o = step("merge_heads", lambda: rearrange(g, "b h n d -> b n (h d)").contiguous())
            step("out_linear", lambda: attn.to_out(o))
    tot = sum(T.values())
    return tot / reps, {k: v / tot for k, v in T.items()}


if __name__ == "__main__":
    with gpu_lock():
        m = load_reference(torch.float16)
        time_attn = m.layers[0][0].layers[0][0]
        freq_attn = m.layers[0][1].layers[0][0]
        for name, attn, shape in [("time", time_attn, (60, 1101, 384)), ("freq", freq_attn, (1101, 60, 384))]:
            x = torch.randn(*shape, device=D, dtype=torch.float16)
            tot, parts = profile(attn, x)
            print(f"{name} attention block: {tot*1000:.0f} ms")
            for k, v in sorted(parts.items(), key=lambda kv: -kv[1]):
                print(f"  {k:12s} {v*100:5.1f}%  {v*tot*1000:6.1f} ms")
