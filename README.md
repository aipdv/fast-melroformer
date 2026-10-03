# fast-melroformer

Kim's Mel-Band RoFormer (vocal / dialogue separation) running **4.2x faster than real time** on an 8 GB M3 MacBook Air.
Same weights, same output (56.5 dB match to fp32, inaudible). Start was 0.78x.

![optimization history](docs/optimization.png)

## What

- `fastmbr/patch.py`: GPU patch for audio-separator's model: fp16 RoPE tables, gates folded into the qkv matmul, fused RMSNorm.
- `fastmbr/metal.py`: custom Metal kernels (`torch.mps.compile_shader`) for RoPE + head split and gate + head merge.
- `fastmbr/ane.py`: the 12 transformer blocks rewritten for the Apple Neural Engine
  ((1, C, bands, frames) layout, 1x1 convs, per-head einsum, attention tiled over 15 band groups x 3 query tiles).
- `fastmbr/hybrid.py`: Neural Engine runs the transformer stack (Core ML); the GPU runs STFT, band split, mask estimators, iSTFT.
- `fastmbr/separate.py`: CLI. File in (any format ffmpeg reads), `_vocals.wav` and `_other.wav` out.
- `bench/`: every measurement in `NOTES.md` is reproducible from here. `bench/guard.sh` runs one job at a time with a RAM watchdog.

## How

```bash
uv sync
# weights + yaml from audio-separator (vocals_mel_band_roformer.*) in ~/models/melroformer/
bench/guard.sh .venv/bin/python bench/ane_stack.py convert   # once: builds out/ane_stack_1101_g15.mlpackage
.venv/bin/python -m fastmbr.separate episode.mp4 -o out/      # first run compiles for the ANE (~5 min), then cached
```

Engines: `--engine ane` (default, fastest), `gpu` (Metal kernels only), `ref16` / `ref32` (unmodified model).

## Where the time went

See `NOTES.md` (lab notebook, every number and dead end). Short version:
- RoPE and RMSNorm ran in fp32 with tables rebuilt per call: 30-46% of attention time on MPS. Fixing that: 1.6x.
- GPU fp16 matmul tops out at 1.4 TFLOPS here; the Neural Engine does 4.9-5.7 TFLOPS on the same shapes.
- On the Neural Engine, the time-attention softmax is the wall. Tiling it so each op's scores stay small: stack 2.47 -> 1.67 s per 11 s chunk.
- Weight quantization / ternary weights would not help speed: 88% of the weights sit in the mask estimators, which take 4% of the time.
