"""
Checks that demo.py's pre- and post-processing is the training convention, using
the export's own held-out triplet -- the one verify_export.py scored.

  - preprocess() turns the held-out frames into exactly golden_input.bin, the
    tensor the golden flow was computed from (so step 1 matches training);
  - midpoint() (cv2.remap) agrees with train/verify_export.py's numpy warp,
    which is grid_sample(align_corners=True, border) as training used, and both
    beat the 50/50 blend on that triplet;
  - a flow that is negative in x moves content the right way: the sign
    convention of the warp, which the PSNR check above would also catch, but
    this names it;
  - the whole --sim demo runs end to end on a tiny synthetic video.

Run: python host/test_demo.py
"""

import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "model"))
sys.path.insert(0, os.path.join(REPO, "train"))

import cv2            # noqa: E402
import demo           # noqa: E402
import vfi_ref as R   # noqa: E402
from verify_export import warp as train_warp  # noqa: E402

EXPORT = R.DEFAULT_EXPORT
FAILS = []


def check(name, cond):
    print("  %-4s %s" % ("PASS" if cond else "FAIL", name))
    if not cond:
        FAILS.append(name)


def main():
    weights, shifts, bias, meta = R.load_export(EXPORT)
    n = meta["img"]
    f0, f1, f2 = np.load(os.path.join(EXPORT, "golden_frames.npy"))     # (3, n, n) RGB [0,1]
    to_bgr = lambda f: cv2.cvtColor(np.rint(f.transpose(1, 2, 0) * 255).astype(np.uint8),
                                    cv2.COLOR_RGB2BGR)
    a, truth, b = to_bgr(f0), to_bgr(f1), to_bgr(f2)

    print("--- preprocessing matches training ---")
    x = demo.preprocess(a, b, n)
    gold_x = R.load_dump(os.path.join(EXPORT, "golden_input.bin"), n, n, 6)
    check("preprocess(frame0, frame2) == golden_input.bin, value for value",
          np.array_equal(x, gold_x))

    print("--- warp and blend match training ---")
    flow = R.load_dump(os.path.join(EXPORT, "golden", "flow.bin"), n, n, 2)   # after bias
    fx, fy = demo.flow_to_pixels(flow, meta["flow_px_per_unit"], n, n)
    mid = demo.midpoint(a, b, fx, fy)
    px = R.from_q15(flow) * meta["flow_px_per_unit"]
    ref = 0.5 * (train_warp(f0.astype(np.float64), 0.5 * px[..., 0], 0.5 * px[..., 1])
                 + train_warp(f2.astype(np.float64), -0.5 * px[..., 0], -0.5 * px[..., 1]))
    ref_bgr = to_bgr(ref.astype(np.float64))
    diff = np.abs(mid.astype(int) - ref_bgr.astype(int))
    check("cv2.remap midpoint == training's grid_sample warp to within 1 grey level "
          "(max %d)" % diff.max(), diff.max() <= 1)
    blend = np.clip(0.5 * (a.astype(np.float32) + b.astype(np.float32)) + 0.5, 0, 255).astype(np.uint8)
    p_model, p_blend = demo.psnr(mid, truth), demo.psnr(blend, truth)
    print("       held-out triplet: model %.2f dB, blend %.2f dB" % (p_model, p_blend))
    check("model beats the 50/50 blend on the held-out triplet", p_model > p_blend + 3)

    print("--- warp sign convention ---")
    img = np.zeros((20, 20, 3), np.uint8)
    img[:, 10] = 255
    out = demo.warp(img, np.full((20, 20), -3, np.float32), np.zeros((20, 20), np.float32))
    check("flow x = -3 samples 3 px to the left, so the bar moves right to column 13",
          out[5, 13, 0] == 255 and out[5, 10, 0] == 0)

    print("--- --sim end to end on a synthetic clip ---")
    d = tempfile.mkdtemp()
    src = os.path.join(d, "clip.mp4")
    vw = cv2.VideoWriter(src, cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 120))
    for t in range(4):
        fr = np.zeros((120, 160, 3), np.uint8)
        cv2.circle(fr, (40 + 8 * t, 60), 12, (255, 255, 255), -1)
        vw.write(fr)
    vw.release()
    out = os.path.join(d, "clip_2x.mp4")
    png = os.path.join(d, "side.png")
    rc = demo.main([src, "--sim", "--out", out, "--png", png])
    cap = cv2.VideoCapture(out)
    n_out = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    check("exit code 0", rc == 0)
    check("4 frames in -> 7 frames out (got %d)" % n_out, n_out == 7)
    check("fps doubled (got %.1f)" % fps, abs(fps - 20) < 0.1)
    check("side-by-side PNG written", os.path.exists(png) and cv2.imread(png) is not None)

    print("")
    print("FAIL  %d checks" % len(FAILS) if FAILS else "PASS  all demo checks")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
