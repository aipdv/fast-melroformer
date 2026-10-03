"""Build, check, convert and time the ANE transformer stack (fastmbr/ane.py).

uv run bench/guard.sh .venv/bin/python bench/ane_stack.py check     # torch fp32 equivalence + activation ranges
uv run bench/guard.sh .venv/bin/python bench/ane_stack.py convert   # -> out/ane_stack_<frames>.mlpackage
uv run bench/guard.sh .venv/bin/python bench/ane_stack.py run       # placement, ANE latency, fidelity of stack output
"""
import collections, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import OUT, chunks, gpu_lock, load_reference, sdr  # noqa: E402
from fastmbr.ane import AneStack  # noqa: E402

FRAMES = 1101
from fastmbr.ane import BAND_GROUPS  # noqa: E402
PKG = OUT / f"ane_stack_{FRAMES}_g{BAND_GROUPS}.mlpackage"
IO = OUT / "stack_io.npz"


def capture():
    """Reference fp32: input to the transformer stack and its output, for one real chunk. (1, 384, 60, T)."""
    if IO.exists():
        z = np.load(IO)
        return z["x"], z["y"]
    m = load_reference(torch.float32)
    got = {}
    m.band_split.register_forward_hook(lambda mod, i, o: got.__setitem__("x", o))
    m.mask_estimators[0].register_forward_pre_hook(lambda mod, i: got.__setitem__("y", i[0]))
    with torch.inference_mode():
        m(torch.from_numpy(chunks(1)).to("mps"))
    x, y = (got[k].float().cpu().permute(0, 3, 2, 1).contiguous().numpy() for k in ("x", "y"))  # (b t f d) -> (b d f t)
    np.savez(IO, x=x, y=y)
    return x, y


def check():
    x, y = capture()
    ref = load_reference(torch.float32, device="cpu")
    stack = AneStack(ref, FRAMES).eval()
    rms = collections.defaultdict(list)
    from fastmbr.ane import Norm
    for name, mod in stack.named_modules():
        if isinstance(mod, Norm):
            mod.register_forward_pre_hook(lambda m, i, n=name: rms[n].append(i[0].pow(2).mean(1).sqrt()))
    with torch.inference_mode():
        out = stack(torch.from_numpy(x)).numpy()
    print(f"torch ANE-layout stack vs reference: {sdr(y, out):.1f} dB")
    lo = min(float(v[0].min()) for v in rms.values()); hi = max(float(v[0].max()) for v in rms.values())
    amax = max(float(v[0].max()) for v in rms.values())
    print(f"norm input RMS over tokens: min {lo:.3g}, max {hi:.3g} (fp16 safe if min/64 > ~3e-4 and max/64*sqrt(384) < 256)")


def convert():
    import coremltools as ct
    x, _ = capture()
    ref = load_reference(torch.float32, device="cpu")
    stack = AneStack(ref, FRAMES).eval()
    del ref
    xt = torch.from_numpy(x)
    with torch.inference_mode():
        traced = torch.jit.trace(stack, xt)
    ml = ct.convert(traced, inputs=[ct.TensorType(name="x", shape=xt.shape, dtype=np.float16)],
                    outputs=[ct.TensorType(name="y", dtype=np.float16)], convert_to="mlprogram",
                    compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.macOS15,
                    skip_model_load=True)
    ml.save(str(PKG))
    print("saved", PKG)


def run():
    import coremltools as ct
    x, y = capture()
    mm = ct.models.MLModel(str(PKG), compute_units=ct.ComputeUnit.CPU_AND_NE)
    plan = ct.models.compute_plan.MLComputePlan.load_from_path(mm.get_compiled_model_path(), compute_units=ct.ComputeUnit.CPU_AND_NE)
    place = collections.Counter()
    for op in plan.model_structure.program.functions["main"].block.operations:
        u = plan.get_compute_device_usage_for_mlprogram_operation(op)
        if u is not None:
            place[type(u.preferred_compute_device).__name__] += 1
    print("op placement:", dict(place))
    xi = {"x": x.astype(np.float16)}
    out = mm.predict(xi)["y"]
    ts = []
    for _ in range(3):
        t = time.perf_counter(); mm.predict(xi); ts.append(time.perf_counter() - t)
    print(f"ANE stack: {sorted(ts)[1]:.2f} s per {FRAMES}-frame chunk; fidelity vs reference stack {sdr(y, out.astype(np.float32)):.1f} dB")


if __name__ == "__main__":
    with gpu_lock():
        {"check": check, "convert": convert, "run": run}[sys.argv[1]]()
