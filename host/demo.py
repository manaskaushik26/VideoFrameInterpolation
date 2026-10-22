#!/usr/bin/env python3
"""
demo.py -- frame-rate doubling: video in, video at 2x fps out, flow from the PL.

    python3 demo.py sample.mp4                       # on the PYNQ-Z2, flow from the PL
    python demo.py sample.mp4 --sim                  # anywhere, flow from vfi_ref.py

For every consecutive pair (a, b) of input frames it writes a, then a frame
synthesised at the midpoint, so the output has twice the frames at twice the fps.

The pipeline, each step exactly as train/train_vfi.py defined it:

  1. preprocess: cv2.resize to 112x112 (INTER_AREA), BGR -> RGB, /255, then
     Q1.15 = round(x * 32768) clamped to 32767; stack a RGB ++ b RGB as
     (row, col, channel) int16 -- the layout the PL's stream takes.
  2. flow: run_network on the PL (conv1..conv5 ReLU, flow with CFG[14] linear),
     or vfi_ref.forward with --sim. Both are bit-exact with each other.
  3. + flow bias, on the PS: the RTL has none.
  4. flow to pixels: value / 32768 * flow_px_per_unit (32); ch0 = x, ch1 = y.
  5. midpoint: 0.5 * (warp(a, +flow/2) + warp(b, -flow/2)), a bilinear BACKWARD
     warp with border clamping -- grid_sample(align_corners=True,
     padding_mode='border') in training, cv2.remap(BORDER_REPLICATE) here.

The network sees 112x112 only. The flow is upsampled to the video's own size
(vectors scaled by W/112 and H/112, since step 1 squashes the aspect ratio the
same way), and the full-resolution frames are warped -- flow fields are smooth,
images are not.

--png writes one side-by-side comparison from a triplet (t, t+1, t+2), where the
real frame t+1 is the ground truth the synthesised one is scored against:

    frame t  |  50/50 blend  |  model  |  true frame t+1

Files, in one folder on the board: vfi.bit vfi.hwh vfi_driver.py vfi_ref.py
demo.py, and the trained export (vfi_model.json, *_weight_q15.bin,
flow_bias_q15.bin) in --model, default ./export_gpu.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "model"))

try:
    import cv2
except ImportError:
    sys.exit("demo.py needs OpenCV (cv2): it is on the PYNQ image; elsewhere pip install opencv-python")

Q = 32768


def default_model_dir():
    for d in (os.path.join(HERE, "export_gpu"), os.path.join(REPO, "train", "export_gpu"), HERE):
        if os.path.exists(os.path.join(d, "vfi_model.json")):
            return d
    return os.path.join(HERE, "export_gpu")


# ---------------------------------------------------------------- pre / post

def preprocess(a_bgr, b_bgr, size):
    """Two BGR frames -> (size, size, 6) int16 Q1.15, a RGB ++ b RGB."""
    def one(f):
        f = cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(f, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
    x = np.concatenate([one(a_bgr), one(b_bgr)], axis=2)
    return np.minimum(np.rint(x * Q), Q - 1).astype(np.int16)


def flow_to_pixels(flow_q15, px_per_unit, out_w, out_h):
    """(n, n, 2) Q1.15 after bias -> (out_h, out_w) x and y displacement in pixels."""
    n = flow_q15.shape[0]
    px = flow_q15.astype(np.float32) / Q * px_per_unit
    fx = cv2.resize(px[..., 0], (out_w, out_h), interpolation=cv2.INTER_LINEAR) * (out_w / n)
    fy = cv2.resize(px[..., 1], (out_w, out_h), interpolation=cv2.INTER_LINEAR) * (out_h / n)
    return fx, fy


def warp(frame, fx, fy):
    """Backward warp: out(p) = frame(p + f), bilinear, border clamped."""
    h, w = frame.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(frame, gx + fx, gy + fy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def midpoint(a, b, fx, fy):
    """0.5 * (warp(a, +f/2) + warp(b, -f/2)), in float, returned as uint8."""
    wa = warp(a.astype(np.float32), 0.5 * fx, 0.5 * fy)
    wb = warp(b.astype(np.float32), -0.5 * fx, -0.5 * fy)
    return np.clip(0.5 * (wa + wb) + 0.5, 0, 255).astype(np.uint8)


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 10 * np.log10(255.0 ** 2 / max(mse, 1e-10))


# ---------------------------------------------------------------- flow sources

class SimFlow:
    """vfi_ref.forward: the bit-exact integer model of the PL, numpy only."""

    name = "vfi_ref (sim)"

    def __init__(self, model_dir):
        import vfi_ref
        self.R = vfi_ref
        self.weights, self.shifts, self.bias, self.meta = vfi_ref.load_export(model_dir)

    def __call__(self, x):
        flow, _ = self.R.forward(x, self.weights, self.shifts, bias=self.bias)
        return flow


class BoardFlow:
    """run_network on the PL, then the flow bias on the PS."""

    name = "PL (PYNQ-Z2)"

    def __init__(self, model_dir, bit):
        from pynq import Overlay
        import vfi_driver as D
        self.D = D
        self.weights, self.layers, self.bias = D.load_model(model_dir)
        with open(os.path.join(model_dir, "vfi_model.json")) as f:
            self.meta = json.load(f)
        n = self.meta["img"]
        ol = Overlay(bit)
        self.vfi = D.Vfi(ol.vfi, ol.dma_fm, ol.dma_wt, img_w=n, img_h=n, n_out=24, max_ch=128)

    def __call__(self, x):
        flow, report = self.vfi.run_network(x, self.weights, self.layers, warn=False)
        return self.D.add_flow_bias(flow, self.bias)


# ---------------------------------------------------------------- main

def read_frames(path, start, count):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit("cannot open %s" % path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    while count <= 0 or len(frames) < count:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames, fps


def side_by_side(frames, flow_src, k, size, px_per_unit, path):
    """Triplet (k, k+1, k+2): synthesise k+1 from k and k+2, score against the real one."""
    a, truth, b = frames[k], frames[k + 1], frames[k + 2]
    h, w = a.shape[:2]
    flow = flow_src(preprocess(a, b, size))
    fx, fy = flow_to_pixels(flow, px_per_unit, w, h)
    model = midpoint(a, b, fx, fy)
    blend = np.clip(0.5 * (a.astype(np.float32) + b.astype(np.float32)) + 0.5, 0, 255).astype(np.uint8)
    p_model, p_blend = psnr(model, truth), psnr(blend, truth)

    tiles = [(a, "frame t"), (blend, "50/50 blend  %.2f dB" % p_blend),
             (model, "model  %.2f dB" % p_model), (truth, "true frame t+1")]
    th = 360
    tw = int(round(w * th / h))
    row = []
    for img, label in tiles:
        t = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
        cv2.rectangle(t, (0, 0), (tw, 30), (0, 0, 0), -1)
        cv2.putText(t, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        row.append(t)
    cv2.imwrite(path, np.concatenate(row, axis=1))
    mag = np.hypot(fx, fy)
    return p_model, p_blend, float(mag.max()), float(mag.mean())


def main(argv=None):
    ap = argparse.ArgumentParser(description="VFI demo: double a video's frame rate")
    ap.add_argument("video")
    ap.add_argument("--out", default=None, help="output video (default <video>_2x.mp4)")
    ap.add_argument("--sim", action="store_true", help="flow from vfi_ref.py instead of the PL")
    ap.add_argument("--bit", default=os.path.join(HERE, "vfi.bit"))
    ap.add_argument("--model", default=default_model_dir(), help="trained export folder")
    ap.add_argument("--start", type=int, default=0, help="first input frame")
    ap.add_argument("--frames", type=int, default=0, help="input frames to use (0 = all)")
    ap.add_argument("--png", default=None, help="write a side-by-side comparison PNG")
    ap.add_argument("--png-frame", type=int, default=None,
                    help="triplet start for --png, relative to --start (default: middle)")
    a = ap.parse_args(argv)

    flow_src = SimFlow(a.model) if a.sim else BoardFlow(a.model, a.bit)
    meta = flow_src.meta
    size, px_per_unit = meta["img"], meta["flow_px_per_unit"]
    print("flow from %s, model %s (shifts %s, flow bias %s)"
          % (flow_src.name, a.model, meta["shifts"], meta["flow_bias_q15"]))

    frames, fps = read_frames(a.video, a.start, a.frames)
    if len(frames) < 2:
        sys.exit("need at least two frames")
    h, w = frames[0].shape[:2]
    out = a.out or os.path.splitext(a.video)[0] + "_2x.mp4"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 2 * fps, (w, h))
    if not vw.isOpened():
        sys.exit("cannot open %s for writing" % out)
    print("%s: %d frames %dx%d at %.2f fps -> %s: %d frames at %.2f fps"
          % (a.video, len(frames), w, h, fps, out, 2 * len(frames) - 1, 2 * fps))

    t_flow, t0 = 0.0, time.time()
    for i in range(len(frames) - 1):
        fa, fb = frames[i], frames[i + 1]
        t = time.time()
        flow = flow_src(preprocess(fa, fb, size))
        t_flow += time.time() - t
        fx, fy = flow_to_pixels(flow, px_per_unit, w, h)
        vw.write(fa)
        vw.write(midpoint(fa, fb, fx, fy))
        if (i + 1) % 10 == 0 or i == len(frames) - 2:
            print("  %d/%d pairs, flow %.2f s/pair, total %.2f s/pair"
                  % (i + 1, len(frames) - 1, t_flow / (i + 1), (time.time() - t0) / (i + 1)))
    vw.write(frames[-1])
    vw.release()
    print("wrote %s" % out)

    if a.png:
        if len(frames) < 3:
            sys.exit("--png needs at least three frames")
        k = a.png_frame if a.png_frame is not None else (len(frames) - 3) // 2
        k = max(0, min(k, len(frames) - 3))
        pm, pb, mx, mean = side_by_side(frames, flow_src, k, size, px_per_unit, a.png)
        print("side-by-side, input frames %d and %d, synthesised %d: model %.2f dB, blend %.2f dB "
              "(flow |max| %.1f px, mean %.2f px at %dx%d) -> %s"
              % (a.start + k, a.start + k + 2, a.start + k + 1, pm, pb, mx, mean, w, h, a.png))
    return 0


if __name__ == "__main__":
    sys.exit(main())
