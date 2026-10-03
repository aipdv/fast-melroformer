"""Drop-in speedups for audio-separator's MelBandRoformer on Apple GPUs. Same weights, same math.

    from fastmbr.patch import accelerate
    accelerate(model)  # in place, after load_state_dict and .half().to("mps")

What it changes (profile: bench/profile_attn.py):
- RMSNorm: F.rms_norm in the model dtype instead of an fp32 F.normalize round trip.
- RoPE: cos/sin tables built once per sequence length. The reference rotates
  interleaved pairs (0,1),(2,3)... in fp32. We permute q/k weight rows per head to the
  split-half layout, so the rotation is two slices and one cat in fp16. q.k is unchanged
  because q and k get the same permutation.
- Output gates folded into the qkv matmul (one GEMM, 1536 + 8 outputs).
- Gate multiply fused with the head merge.
"""
import types

import torch
import torch.nn.functional as F
from torch import nn


def _rmsnorm_forward(self, x):
    # F.normalize(x) * sqrt(d) * gamma == x / sqrt(mean(x^2)) * gamma
    return F.rms_norm(x, (x.shape[-1],), self.gamma, eps=1e-12)


def _split_half_perm(dim_head):
    # new position j reads old position perm[j]: evens first, then odds
    return torch.cat([torch.arange(0, dim_head, 2), torch.arange(1, dim_head, 2)])


class FastAttention(nn.Module):
    def __init__(self, ref):
        super().__init__()
        h, inner = ref.heads, ref.to_qkv.weight.shape[0] // 3
        d = inner // h
        self.h, self.d, self.inner = h, d, inner
        w = ref.to_qkv.weight.detach().view(3, h, d, -1).clone()
        p = _split_half_perm(d).to(w.device)
        w[0], w[1] = w[0][:, p], w[1][:, p]  # q, k only; v keeps its order
        gw, gb = ref.to_gates.weight.detach(), ref.to_gates.bias.detach()
        self.register_buffer("w", torch.cat([w.reshape(3 * inner, -1), gw]))
        self.register_buffer("b", torch.cat([torch.zeros(3 * inner, dtype=gb.dtype, device=gb.device), gb]))
        self.register_buffer("gamma", ref.norm.gamma.detach())
        self.register_buffer("freqs", ref.rotary_embed.freqs.detach().float())
        self.out = ref.to_out[0]
        self._rope = {}

    def rope(self, n, dtype, device):
        key = (n, dtype, device)
        if key not in self._rope:
            ang = torch.arange(n, device=device, dtype=torch.float32)[:, None] * self.freqs.to(device)
            self._rope[key] = (ang.cos().to(dtype), ang.sin().to(dtype))
        return self._rope[key]

    def forward(self, x):
        b, n, _ = x.shape
        y = F.linear(F.rms_norm(x, (x.shape[-1],), self.gamma, eps=1e-12), self.w, self.b)
        qkv = y[..., : 3 * self.inner].view(b, n, 3, self.h, self.d).permute(2, 0, 3, 1, 4)
        cos, sin = self.rope(n, x.dtype, x.device)
        half = self.d // 2
        qk = qkv[:2]
        x1, x2 = qk[..., :half], qk[..., half:]
        qk = torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
        o = F.scaled_dot_product_attention(qk[0], qk[1], qkv[2])
        g = y[..., 3 * self.inner :].sigmoid().unsqueeze(-1)  # (b, n, h, 1)
        return self.out((o.transpose(1, 2) * g).reshape(b, n, self.inner))


def accelerate(model):
    from audio_separator.separator.uvr_lib_v5.roformer import mel_band_roformer as mbr

    for mod in list(model.modules()):
        if isinstance(mod, mbr.RMSNorm):
            mod.forward = types.MethodType(_rmsnorm_forward, mod)
        if isinstance(mod, mbr.Transformer):
            for pair in mod.layers:
                pair[0] = FastAttention(pair[0])
    return model
