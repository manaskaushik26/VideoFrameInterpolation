"""
verify_export.py -- prove the exported integers do what training measured. numpy only.

Runs model/vfi_ref.forward (the bit-exact model of the RTL) on the held-out frame pair
train_vfi.py exported, then checks three things:

  1. vfi_ref's flow agrees with the quantised torch model's, to within a few LSB.
     They differ only in rounding mode (torch rounds half-to-even, the MAC half-up),
     so a large difference means the export layout or a shift is wrong.
  2. No hidden layer saturates over a large fraction of pixels -- that would mean the
     shifts are wrong for real data.
  3. The frame synthesised from vfi_ref's flow beats the no-model 50/50 blend.

It also writes per-layer golden dumps (golden/<layer>.bin, raw int16, beat order) --
the files to diff the RTL and the board against, replacing the old golden_*.hex
which came from a float model with biases and a tanh that the hardware does not have.

    python verify_export.py run_mnist/export
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model"))
import vfi_ref  # noqa: E402


def warp(frame, fx, fy):
    """Bilinear backward warp with border clamping -- grid_sample(align_corners=True,
    padding_mode='border'), which is what training used. frame (C, H, W)."""
    C, H, W = frame.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    x = np.clip(xx + fx, 0, W - 1)
    y = np.clip(yy + fy, 0, H - 1)
    x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int)
    x1 = np.minimum(x0 + 1, W - 1); y1 = np.minimum(y0 + 1, H - 1)
    wx, wy = x - x0, y - y0
    return ((1 - wy) * ((1 - wx) * frame[:, y0, x0] + wx * frame[:, y0, x1])
            + wy * ((1 - wx) * frame[:, y1, x0] + wx * frame[:, y1, x1]))


def psnr(a, b):
    return 10 * np.log10(1.0 / max(np.mean((a - b) ** 2), 1e-10))


def main(d):
    meta = json.load(open(os.path.join(d, "vfi_model.json")))
    n = meta["img"]
    weights = {l["name"]: vfi_ref.load_weights(os.path.join(d, "%s_weight_q15.bin" % l["name"]),
                                               l["c_out"], l["c_in"]) for l in meta["layers"]}
    shifts = meta["shifts"]
    bias = {"flow": np.fromfile(os.path.join(d, "flow_bias_q15.bin"), dtype="<i2")}
    x = vfi_ref.load_dump(os.path.join(d, "golden_input.bin"), n, n, 6)

    gold = os.path.join(d, "golden")
    os.makedirs(gold, exist_ok=True)
    vfi_ref.dump(x, os.path.join(gold, "input.bin"))
    print("[*] running vfi_ref (bit-exact RTL model) at %dx%d -- takes a few minutes" % (n, n))
    flow, infos = vfi_ref.forward(x, weights, shifts, bias=bias, dump_dir=gold, verbose=True)

    ok = True
    ref = vfi_ref.load_dump(os.path.join(d, "golden_flow_torch.bin"), n, n, 2).astype(np.int64)
    diff = np.abs(flow.astype(np.int64) - ref)
    print("\n[1] flow vs quantised torch model: max %d LSB, mean %.3f LSB, %.3f%% differ"
          % (diff.max(), diff.mean(), 100.0 * (diff > 0).mean()))
    if diff.max() > 64:
        print("    FAIL -- far beyond rounding; check layer order, transpose, shifts")
        ok = False

    worst = max(infos[:-1], key=lambda i: i["sat_frac"])
    print("[2] worst hidden-layer saturation: %s %.2f%%" % (worst["layer"], 100 * worst["sat_frac"]))
    if worst["sat_frac"] > 0.05:
        print("    WARN -- more than 5% of outputs clip; the shift for that layer is too small")

    f0, f1, f2 = np.load(os.path.join(d, "golden_frames.npy")).astype(np.float64)
    px = vfi_ref.from_q15(flow) * meta["flow_px_per_unit"]          # (H, W, 2), pixels
    fx, fy = px[..., 0], px[..., 1]
    mid = 0.5 * (warp(f0, 0.5 * fx, 0.5 * fy) + warp(f2, -0.5 * fx, -0.5 * fy))
    p_board, p_blend = psnr(mid, f1), psnr(0.5 * (f0 + f2), f1)
    print("[3] this frame: board path %.2f dB vs 50/50 blend %.2f dB  (flow |max| %.1f px)"
          % (p_board, p_blend, np.abs(px).max()))
    if p_board <= p_blend:
        print("    FAIL -- the deployed integer model does no better than no model at all")
        ok = False

    print("\nshifts for host/vfi_driver.VfiLayer:", shifts)
    print("flow bias (host-side add):", bias["flow"].tolist())
    print("golden per-layer dumps:", gold)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "run/export"))
