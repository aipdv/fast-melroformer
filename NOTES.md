# Lab notebook

Machine: MacBook Air M3, 8 GB, 8-core GPU, 16-core Neural Engine, fanless. macOS 15 (Darwin 24.6).
Model: Kim Mel-Band RoFormer vocals (`vocals_mel_band_roformer.ckpt`, sha256 87201f4d…), 228M params, fp32 913 MB.
Test audio: 3 min of a TV episode (not in git). Chunk = 1101 STFT frames = 11 s.
Rule: one timed job at a time (`bench/common.py: gpu_lock`). RAM pressure and heat change timings a lot on this machine.

## 2026-10-03 Baselines (audio-separator 0.47, torch 2.14.1)

End to end, 3 min clip, RAM free:

| Setup | Time | Speed vs real time |
|---|---|---|
| fp32 (old run, RAM busy) | 3:52 | 0.78x |
| `--use_native_fp16` | 2:25 | 1.24x |
| mlx-audio-separator 0.1.19, `--mdxc_overlap 11` | 3:01 | 0.99x |

30 s clip: fp32 28 s, autocast 34 s (RAM busy), fp16 23 s, torch.compile no gain, fp16+compile 51 s, MLX 18 s.
fp16 output vs fp32: 70.6 dB SDR (identical to the ear). MLX vs PyTorch fp32: 39.3 dB.

mlx-audio-separator notes:
- `--mdxc_overlap` is the hop in seconds, not a window count. Default 8 => 8 s hop on 11 s chunks (1.4x the work).
- `--experimental_roformer_fast_norm` gives wrong audio (4.2 dB vs reference): `mx.fast.rms_norm(...) * self.scale` scales by sqrt(dim) twice. Dropping `* self.scale` gives 39.3 dB. No speed gain on M3.

## Where the time goes (MPS fp16, one 11 s chunk, ~7.4 s)

Module level: time attention 36%, freq attention 28%, time FF 12%, freq FF 13%, mask estimators 4%, STFT/glue 7%.
Params: mask estimators 201.5M (88%), transformer 23.7M, band split 3.1M.
=> weight quantization / ternary weights cut RAM, not time. Time is in attention glue.

Op level inside one attention block (`bench/profile_attn.py`):

| op | time block (60 x 1101) | freq block (1101 x 60) |
|---|---|---|
| total | 442 ms | 366 ms |
| RoPE q+k | 131 ms (30%) | 170 ms (46%) |
| SDPA | 148 ms (34%) | 11 ms (3%) |
| qkv linear | 62 ms | 71 ms |
| RMSNorm | 33 ms | 41 ms |
| gates | 30 ms | 31 ms |
| out linear | 27 ms | 32 ms |

RoPE is applied in fp32 with angles rebuilt per call (rotary_embedding_torch + audio-separator's fp32 guard).
RMSNorm upcasts to fp32 too. These are memory-bound elementwise passes over 66k x 512 tensors.
Hypothesis H1: precomputed fp16 cos/sin + fused kernels cut attention time ~45% => ~1.6x end to end.

## H1 result: fastmbr/patch.py (fp16 RoPE tables, fused gates into qkv GEMM, F.rms_norm)

`bench/bench_chunk.py ref16 fast16`, 3 interleaved reps, one 11 s chunk:

| variant | s/chunk | x real time | fidelity vs fp32 |
|---|---|---|---|
| ref16 | 8.24 | 1.34x | 55.6 dB |
| fast16 | 5.14 | 2.14x | 56.3 dB |

1.6x from glue removal alone. Output identical to the ear (both > 55 dB).
