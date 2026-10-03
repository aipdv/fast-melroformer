"""Separate voice from everything else with Kim Mel-RoFormer, fast on Apple silicon.

    .venv/bin/python -m fastmbr.separate input.(wav|mp3|mp4|mkv) [-o outdir] [--engine ane2|ane|gpu|dual|ref16|ref32]

Writes <outdir>/<name>_vocals.wav and <name>_other.wav (44.1 kHz stereo, float32).
Chunks are 485100 samples (11 s, 1101 frames) with no overlap, like audio-separator's default for this model.
Engines:
  gpu   fastmbr.metal (custom Metal kernels)
  ane   transformer stack on the Neural Engine, rest on the GPU (fastmbr.hybrid)
  ane2  two ane workers (default): one chunk's GPU/CPU steps overlap the other's ANE stack
  dual  gpu and ane workers pull chunks from one queue at the same time
  ref16 / ref32  audio-separator's model as is (for comparison)
"""
import argparse, queue, subprocess, sys, threading, time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
SR, CHUNK = 44100, 441 * 1100


def load_audio(path):
    """(2, n) float32 at 44.1 kHz. ffmpeg decodes anything (video, mp3, 5.1)."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0", "-ac", "2", "-ar", str(SR),
                          "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, 2).T.copy()


def models(engine):
    from common import load_reference
    if engine == "ref32":
        return [load_reference(torch.float32)]
    if engine == "ref16":
        return [load_reference(torch.float16)]
    from fastmbr import hybrid, metal
    return {"gpu": lambda: [metal.build()], "ane": lambda: [hybrid.build()],
            "ane2": lambda: [hybrid.build(), hybrid.build()],  # 2 workers: one's GPU/CPU steps overlap the other's ANE stack
            "dual": lambda: [metal.build(), hybrid.build()]}[engine]()


def separate(audio, engine="dual", log=print):
    n = audio.shape[1]
    nchunks = -(-n // CHUNK)
    padded = np.zeros((2, nchunks * CHUNK), np.float32)
    padded[:, :n] = audio
    vocals = np.zeros_like(padded)
    work = queue.Queue()
    for i in range(nchunks):
        work.put(i)
    done, errors = {}, []

    def worker(m, name):
        try:
            run(m, name)
        except Exception as e:  # surface worker failures instead of returning silent zeros
            errors.append(e)

    def run(m, name):
        with torch.inference_mode():
            while True:
                try:
                    i = work.get_nowait()
                except queue.Empty:
                    return
                with lock:
                    x = torch.from_numpy(padded[None, :, i * CHUNK:(i + 1) * CHUNK]).to("mps")
                    vocals[:, i * CHUNK:(i + 1) * CHUNK] = m(x).float().cpu().numpy()[0]
                done[i] = name

    ms = models(engine)
    import contextlib
    from fastmbr.hybrid import MPS_LOCK
    lock = MPS_LOCK if engine == "ane2" else contextlib.nullcontext()
    names = {"dual": ["gpu", "ane"], "ane2": ["ane-a", "ane-b"]}.get(engine, [engine])
    t0 = time.perf_counter()
    threads = [threading.Thread(target=worker, args=(m, nm)) for m, nm in zip(ms, names)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    secs = time.perf_counter() - t0
    if errors:
        raise errors[0]
    assert len(done) == nchunks, f"only {len(done)}/{nchunks} chunks done"
    split = {nm: sum(1 for v in done.values() if v == nm) for nm in names}
    log(f"{engine}: {n / SR:.0f} s of audio in {secs:.1f} s = {n / SR / secs:.2f}x real time; chunks per engine {split}")
    return vocals[:, :n], secs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("-o", "--outdir", default=".")
    ap.add_argument("--engine", default="ane2", choices=["gpu", "ane", "ane2", "dual", "ref16", "ref32"])
    a = ap.parse_args()
    audio = load_audio(a.input)
    vocals, _ = separate(audio, a.engine)
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(a.input).stem
    sf.write(out / f"{stem}_vocals.wav", vocals.T, SR, subtype="FLOAT")
    sf.write(out / f"{stem}_other.wav", (audio - vocals).T, SR, subtype="FLOAT")
    print("wrote", out / f"{stem}_vocals.wav", out / f"{stem}_other.wav")


if __name__ == "__main__":
    main()
