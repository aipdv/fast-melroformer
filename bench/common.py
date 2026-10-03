"""Shared bench helpers: GPU lock, model + audio loading, timing, fidelity."""
import contextlib, os, statistics, time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml

MODELS = Path(os.environ.get("MBR_DIR", "~/models/melroformer")).expanduser()
CKPT = MODELS / "vocals_mel_band_roformer.ckpt"
CFG = MODELS / "vocals_mel_band_roformer.yaml"
CLIP = MODELS / "clip180.wav"  # 3 min of TV audio, 44.1 kHz stereo (not in git)
CHUNK = 441 * (1101 - 1)  # 485100 samples = 1101 STFT frames, what audio-separator feeds per chunk
OUT = Path(__file__).resolve().parents[1] / "out"
LOCK = Path("/tmp/fast-melroformer.gpu.lock")


@contextlib.contextmanager
def gpu_lock(poll=2.0):
    """One timed GPU/ANE job at a time on this 8 GB machine. mkdir is atomic."""
    while True:
        try:
            LOCK.mkdir()
            break
        except FileExistsError:
            time.sleep(poll)
    try:
        yield
    finally:
        LOCK.rmdir()


def load_reference(dtype=torch.float32, device="mps"):
    """Kim Mel-RoFormer exactly as audio-separator builds it."""
    from audio_separator.separator.uvr_lib_v5.roformer.mel_band_roformer import MelBandRoformer

    cfg = yaml.load(CFG.read_text(), Loader=yaml.FullLoader)["model"]
    m = MelBandRoformer(**cfg)
    sd = torch.load(CKPT, map_location="cpu")
    m.load_state_dict(sd.get("state_dict", sd))
    return m.to(dtype).to(device).eval()


def chunks(n=3, start_s=20.0):
    """n consecutive model-sized chunks of real TV audio, shape (n, 2, CHUNK) float32."""
    audio, sr = sf.read(CLIP, dtype="float32", always_2d=True)
    assert sr == 44100
    s = int(start_s * sr)
    return np.stack([audio[s + i * CHUNK : s + (i + 1) * CHUNK].T for i in range(n)])


def sync(device):
    if str(device).startswith("mps"):
        torch.mps.synchronize()


def timeit(fn, warmup=1, reps=3, device="mps"):
    """Median wall seconds of fn() after warmup."""
    for _ in range(warmup):
        fn()
        sync(device)
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        sync(device)
        ts.append(time.perf_counter() - t)
    return statistics.median(ts)


def sdr(ref, est):
    """dB. Fidelity to the fp32 reference output (higher = closer; >40 is inaudible)."""
    ref, est = np.asarray(ref, np.float64), np.asarray(est, np.float64)
    return 10 * np.log10((ref**2).sum() / max(((ref - est) ** 2).sum(), 1e-20))


def reference_outputs(x):
    """fp32 reference vocals for the bench chunks, cached in out/."""
    OUT.mkdir(exist_ok=True)
    p = OUT / f"ref_{x.shape[0]}x.npy"
    if p.exists():
        return np.load(p)
    m = load_reference(torch.float32)
    with torch.inference_mode():
        y = np.stack([m(torch.from_numpy(c[None]).to("mps")).float().cpu().numpy()[0] for c in x])
    np.save(p, y)
    del m
    torch.mps.empty_cache()
    return y
