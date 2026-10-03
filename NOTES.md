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

## 2026-10-03 Crash and safety rules

The Mac froze (out of RAM) while an ANE conversion probe and a GPU kernel job ran at the same time.
Now: `bench/guard.sh` runs one job at a time, kills it below 8% free RAM, caps the MPS pool
(watermark ratios). The GPU lock reclaims itself when its owner dies. Checkpoint loads are mmap'd and cast to fp16 first.

## H2 result: fastmbr/metal.py (custom Metal kernels via torch.mps.compile_shader)

- `qkv_rope`: one pass from the fused qkv+gates GEMM output to contiguous q, k (RoPE, fp32 math), v in SDPA layout.
- `gate_merge`: sigmoid gate x attention output + head merge in one pass.

| variant | s/chunk | x real time | fidelity vs fp32 |
|---|---|---|---|
| fast16 | 4.84 | 2.27x | 56.3 dB |
| metal | 4.01 | 2.74x | 54.7 dB |

GPU floor estimate: GEMM fp16 peaks at 1.42 TFLOPS here; the model is ~4.1-4.5 TFLOP per chunk => ~3 s/chunk best case.

## ANE probe

Toy FF stack (4 x 384->1536->384 as 1x1 convs on (1, 384, 60, 1101)), fp16:
CPU_AND_NE 126 ms = 4.9 TFLOPS; ALL 110 ms = 5.7 TFLOPS; CPU_AND_GPU 474 ms = 1.3 TFLOPS.
=> the ANE has ~4x the GPU's matmul throughput on this shape.

## Research summary (attention methods, ANE practice) — see the agent report digest

- Chinese-lab sparse/linear attention (NSA, DSA, MoBA, MLA, Lightning, Kimi Linear): little to no gain at seq 1101/60, non-causal.
  DSA keeps every key at <=2048 tokens; MLA only saves KV cache. Best training-free: SpargeAttn-style dynamic block sparsity (time axis only, <=22% of FLOPs).
- Time-attention matmuls are ~22% of FLOPs; FF + projections dominate. So the hardware (ANE) matters more than attention tricks.
- ANE rules: fixed shapes, (B,C,1,S)-style layout with 1x1 convs, per-head einsum (Apple SPLIT_EINSUM), tile queries so one
  head's scores stay small, keep STFT/complex outside, rank <= 4 for batched matmul, watch fp16 overflow in norms.
- Smule windowed checkpoint (mbr-win10-sink8): 11.17 vs 12.12 dB, plain dense ops possible.

## H3: transformer stack on the Neural Engine (fastmbr/ane.py, bench/ane_stack.py, fastmbr/hybrid.py)

ANE-layout rewrite matches the reference in torch fp32: 116.5 dB. Norm inputs RMS 0.25..3.6 => prescale 8 before squaring.
Core ML (fp16, macOS15, CPU_AND_NE): all 4764 ops placed on the ANE. Package 47 MB.

| piece (1101 frames) | ANE ms |
|---|---|
| full stack (12 blocks) | 2470 |
| time block | 319 (q_tiles 3), 315 (6); Apple keys-on-channel einsum 348 |
| time block without attention core | 98 |
| time block without softmax | 182 => softmax ~137 ms/block (581M elements, ~4.2 G elem/s) |
| time block, explicit max/exp/sum softmax | 316 (no gain) |
| freq block | 130; without attention core 105 |
| 801 frames: time 193, freq 98 | only ~11% better per second of audio |

=> On the ANE the time-attention softmax is the wall (6 x 137 ms). Everything else is fast.

End to end (bench_chunk, one chunk, ANE stack + GPU front/back, serial):

| variant | s/chunk | x real time | fidelity |
|---|---|---|---|
| metal (GPU only) | 4.03 | 2.73x | 54.7 dB |
| hybrid (ANE stack) | 3.00 | 3.67x | 40.2 dB |

Next: (1) find where ANE fp16 loses precision (target >= 45 dB), (2) run GPU and ANE at the same time on different chunks.

## H4: ANE precision and tiling

Per-block fidelity of the ANE stack vs torch fp32 (bench/ane_blocks.py): 53.7 dB after block 0, ~1.5 dB lost per block, then
45.7 -> 35.9 dB at the last block. Activations are small (max |x| < 56), so no overflow.
Stage probe of block 11 (bench/ane_block_stages.py): the ANE's native GELU drops 58.7 -> 42.8 dB.
tanh-form GELU: 48.8 dB at that stage, but end to end only 40.2 -> 41.7 dB and the stack gets 25% slower. Kept native (option ANE_GELU=tanh).

Time attention tiled over 15 band groups x 3 query tiles (each op ~3 MB of scores, near the ANE's on-chip size):
time block 319 -> 258 ms; full stack 2.47 -> 1.67 s per 11 s chunk; 30,492 ops, all on the ANE.
First load compiles for ~4.5 min; the compiled .mlmodelc is cached (out/), later loads take 2.2 s.

Concurrency, first try (whole stack on ANE without band groups, GPU running a full chunk at the same time):
GPU chunk 5.2 -> 11.2 s, wall 11.2 s vs 12.0 s serial. Little overlap. Likely shared memory bandwidth (the ungrouped
ANE softmax streams ~1 GB of scores per block through DRAM). Python's GIL is released during Core ML predict.

MLX bug reported upstream: https://github.com/ssmall256/mlx-audio-separator/issues/6
