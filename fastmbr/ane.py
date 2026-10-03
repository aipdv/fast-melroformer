"""The 12 transformer blocks of Kim Mel-RoFormer rewritten for the Apple Neural Engine.

Layout: x is (1, C=384, F=60 bands, T=frames), the (B, C, H, W) form the ANE likes.
- Linear -> 1x1 Conv2d. Gates folded into the qkv conv (as in fastmbr.patch).
- Time attention: sequences = bands (dim 2), tokens = frames (dim 3).
  Freq attention: sequences = frames (dim 3), tokens = bands (dim 2).
- Per-head einsum (Apple SPLIT_EINSUM). Time attention tiles queries so one head's
  score block stays small.
- RMSNorm is scale invariant, so inputs are divided by a constant first: x^2 cannot overflow fp16.
- RoPE uses the split-half head layout (q/k rows permuted at load), constant cos/sin tables.

AneStack(ref_model, frames) holds weights copied from audio-separator's MelBandRoformer.
Input/output match `x` between band_split and mask_estimators, permuted (b t f d) -> (b d f t).
"""
import math
import os

import torch
import torch.nn.functional as F
from torch import nn

from fastmbr.patch import _split_half_perm

ATTN_LAYOUT = os.environ.get("ANE_ATTN_LAYOUT", "batched")  # "apple" = keys-on-channels einsum
ALL_OUTPUTS = os.environ.get("ANE_ALL_OUTPUTS") == "1"  # profiling: return every block output
GELU = os.environ.get("ANE_GELU", "native")  # "tanh": +1.5 dB end to end but +25% ANE time (NOTES.md)
BAND_GROUPS = int(os.environ.get("ANE_BAND_GROUPS", "15"))  # time block 319 -> 258 ms (1 vs 15 groups); 60 groups 245 ms but a huge graph
NORM_PRESCALE = 8.0  # measured norm-input RMS 0.25..3.6 (bench/ane_stack.py check): (x/8)^2 stays in fp16 normal range, sums < 65504


class Norm(nn.Module):
    """RMSNorm over channels (dim 1): x / sqrt(mean(x^2)) * gamma."""

    def __init__(self, gamma):
        super().__init__()
        self.register_buffer("g", gamma.detach().float().view(1, -1, 1, 1))

    def forward(self, x):
        x = x * (1.0 / NORM_PRESCALE)
        return x * torch.rsqrt((x * x).mean(dim=1, keepdim=True) + 1e-12) * self.g


def conv(w, b=None):
    c = nn.Conv2d(w.shape[1], w.shape[0], 1, bias=b is not None)
    c.weight.data = w.detach().float()[:, :, None, None].clone()
    if b is not None:
        c.bias.data = b.detach().float().clone()
    return c


class Attn(nn.Module):
    def __init__(self, ref, axis, n_tokens, q_tiles=1):
        super().__init__()
        h, inner = ref.heads, ref.to_qkv.weight.shape[0] // 3
        d = inner // h
        self.h, self.d, self.inner, self.axis, self.q_tiles = h, d, inner, axis, q_tiles
        w = ref.to_qkv.weight.detach().float().view(3, h, d, -1).clone()
        p = _split_half_perm(d)
        w[0], w[1] = w[0][:, p], w[1][:, p]
        w[0] *= d**-0.5  # fold the softmax scale into q
        self.norm = Norm(ref.norm.gamma)
        self.qkv = conv(torch.cat([w.reshape(3 * inner, -1), ref.to_gates.weight.detach().float()]),
                        torch.cat([torch.zeros(3 * inner), ref.to_gates.bias.detach().float()]))
        self.out = conv(ref.to_out[0].weight)
        ang = torch.arange(n_tokens, dtype=torch.float32)[:, None] * ref.rotary_embed.freqs.detach().float()  # (n, d/2)
        shape = (1, d // 2, 1, n_tokens) if axis == "time" else (1, d // 2, n_tokens, 1)
        self.register_buffer("cos", ang.cos().T.reshape(shape))
        self.register_buffer("sin", ang.sin().T.reshape(shape))

    def rope(self, t):
        half = self.d // 2
        a, b = t[:, :half], t[:, half:]
        return torch.cat([a * self.cos - b * self.sin, b * self.cos + a * self.sin], dim=1)

    def forward(self, x):
        y = self.qkv(self.norm(x))
        i, d = self.inner, self.d
        outs = []
        for h in range(self.h):
            q = self.rope(y[:, h * d:(h + 1) * d])
            k = self.rope(y[:, i + h * d:i + (h + 1) * d])
            v = y[:, 2 * i + h * d:2 * i + (h + 1) * d]
            if ATTN_LAYOUT == "skip":  # profiling only: drop the attention core, keep projections
                o = v
            elif self.axis == "time" and ATTN_LAYOUT == "apple":
                # Apple SPLIT_EINSUM: keys on the channel axis, softmax over channels. (1, Tk, F, Tq)
                kt = k.permute(0, 3, 2, 1)  # (1, Tk, F, d)
                o = torch.cat([torch.einsum("bkhq,bchk->bchq", torch.einsum("bchq,bkhc->bkhq", qt, kt).softmax(1), v)
                               for qt in q.chunk(self.q_tiles, dim=3)], dim=3)
            elif self.axis == "time" and ATTN_LAYOUT == "nosoftmax":  # profiling only
                o = torch.cat([torch.einsum("bsqk,bcsk->bcsq", torch.einsum("bcsq,bcsk->bsqk", qt, k), v)
                               for qt in q.chunk(self.q_tiles, dim=3)], dim=3)
            elif self.axis == "time" and ATTN_LAYOUT == "manual":  # softmax as max/sub/exp/sum/mul
                def sm(w):
                    e = torch.exp(w - w.amax(-1, keepdim=True))
                    return e * (1.0 / e.sum(-1, keepdim=True))
                o = torch.cat([torch.einsum("bsqk,bcsk->bcsq", sm(torch.einsum("bcsq,bcsk->bsqk", qt, k)), v)
                               for qt in q.chunk(self.q_tiles, dim=3)], dim=3)
            elif self.axis == "time":  # (1, d, F, T): attend over T per band; tile bands x queries so ops stay small
                rows = []
                for qb, kb, vb in zip(q.chunk(BAND_GROUPS, 2), k.chunk(BAND_GROUPS, 2), v.chunk(BAND_GROUPS, 2)):
                    rows.append(torch.cat([torch.einsum("bsqk,bcsk->bcsq", torch.einsum("bcsq,bcsk->bsqk", qt, kb).softmax(-1), vb)
                                           for qt in qb.chunk(self.q_tiles, dim=3)], dim=3))
                o = torch.cat(rows, dim=2)
            else:  # attend over F per frame
                o = torch.einsum("btqk,bckt->bcqt", torch.einsum("bcqt,bckt->btqk", q, k).softmax(-1), v)
            outs.append(o * torch.sigmoid(y[:, 3 * i + h:3 * i + h + 1]))
        return self.out(torch.cat(outs, dim=1))


class FF(nn.Module):
    def __init__(self, ref):
        super().__init__()
        net = ref.net
        self.norm, self.a, self.b = Norm(net[0].gamma), conv(net[1].weight, net[1].bias), conv(net[4].weight, net[4].bias)

    def forward(self, x):
        return self.b(gelu(self.a(self.norm(x))))


def gelu(x):
    if GELU == "tanh":  # tanh form built from basic ops; the ANE's native gelu loses ~15 dB (bench/ane_block_stages.py)
        return 0.5 * x * (1.0 + torch.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))
    if GELU == "split":
        # gelu(x) = relu(x) - |x| * Phi(-|x|). relu is exact on the ANE; the low-precision transcendental
        # only touches the small correction term (<= 0.17). Phi(z) ~ sigmoid(1.5976 z + 0.07056 z^3), max err 1.4e-4.
        a = torch.abs(x)
        return torch.relu(x) - a * torch.sigmoid(-(1.5976 * a + 0.07056 * a * a * a))
    if GELU == "sigmoid":
        return x * torch.sigmoid(1.702 * x)
    return F.gelu(x)


class Block(nn.Module):
    """One audio-separator Transformer (depth 1 + output norm) along one axis."""

    def __init__(self, ref, axis, n_tokens, q_tiles):
        super().__init__()
        (attn, ff), = ref.layers
        self.attn, self.ff, self.norm = Attn(attn, axis, n_tokens, q_tiles), FF(ff), Norm(ref.norm.gamma)

    def forward(self, x):
        x = self.attn(x) + x
        x = self.ff(x) + x
        return self.norm(x)


class AneStack(nn.Module):
    def __init__(self, ref_model, frames=1101, bands=60, q_tiles=3, layers=None, kinds=("time", "freq")):
        super().__init__()
        pairs = ref_model.layers if layers is None else [ref_model.layers[i] for i in layers]
        self.blocks = nn.ModuleList()
        for tt, ft in pairs:
            if "time" in kinds:
                self.blocks.append(Block(tt, "time", frames, q_tiles))
            if "freq" in kinds:
                self.blocks.append(Block(ft, "freq", bands, 1))

    def forward(self, x):
        outs = []
        for b in self.blocks:
            x = b(x)
            outs.append(x)
        return tuple(outs) if ALL_OUTPUTS else x
