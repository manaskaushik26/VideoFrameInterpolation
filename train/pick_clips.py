"""
pick_clips.py -- choose the real-video clips train_vfi.py --videos should learn from.

    python pick_clips.py UCF-101/ --out clips/            # then: train_vfi.py --videos clips/train
    python pick_clips.py my_footage/ --out clips/ --max-frames 60000

Every clip is measured the way the network will see it -- resized to 112x112 with
INTER_AREA, consecutive triplets (t, t+1, t+2) -- and kept only if it can teach
something the hardware network can learn:

  motion   the network warps by the flow between t and t+2, and six 3x3 layers
           see only +-6 px at 112x112. Motion too fast for optical flow to track at
           112x112 is caught separately (see flow_check), since there the measured
           number is an underestimate. Per clip, the 90th-percentile Farneback flow
           magnitude between t and t+2 (the moving part of the frame) must lie in
           [--min-px, --max-px], default [0.75, 6]. Below that a 50/50 blend is
           already right and the clip teaches nothing -- sample.mp4 is one: copying
           the previous frame scores 59 dB on it. Above it the motion is outside
           the receptive field.
  cuts     a triplet that spans a scene cut has no correct middle frame. A clip
           whose largest neighbour-to-neighbour histogram distance is above
           --cut-hist and --cut-ratio times its own median is dropped.
  size     at least --min-frames frames.

Diversity and an honest validation split. UCF101 names clips
v_<Class>_g<GG>_c<CC>: clips in one group share the actor and the room, so if
two of them land on opposite sides of train_vfi.py's by-file split, validation
is scored on near-duplicates of training data. So at most --per-group clips
(default 1) are taken per (class, group), classes are drawn round-robin until
--max-frames is reached (train_vfi.py decodes every frame into RAM: about
37 KB per frame at 112x112, so 100k frames is ~3.7 GB), and the clips are laid
out as:

  <out>/train/   for train_vfi.py --videos; it holds out 10% of these itself
  <out>/demo/    DEMO candidates from groups never used in train/ -- the model
                 has not seen the actor or the room -- with clear, in-range
                 motion. Use one for host/demo.py.
  <out>/picks.csv  every clip measured, kept or not, and why

Files are hard-linked when possible (no extra disk), else copied. Needs numpy
and opencv only.
"""

import argparse
import csv
import os
import re
import shutil
import sys
from collections import defaultdict

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("pick_clips.py needs OpenCV: pip install opencv-python")

SIZE = 112
EXTS = (".mp4", ".avi", ".mov", ".mkv")
UCF = re.compile(r"v_(?P<cls>.+)_g(?P<g>\d+)_c(?P<c>\d+)", re.I)


def read_small(path, max_frames):
    """Frames (up to max_frames) at 112x112, resized exactly as training does -> (grey
    frames, one hue/value histogram per frame for cut detection, fps, w, h)."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out, hists = [], []
    while len(out) < max_frames:
        ok, f = cap.read()
        if not ok:
            break
        f = cv2.resize(f, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        out.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
        hists.append(cv2.calcHist([cv2.cvtColor(f, cv2.COLOR_BGR2HSV)], [0, 2], None,
                                  [16, 16], [0, 180, 0, 256]))
    cap.release()
    return out, hists, fps, w, h


_GX, _GY = np.meshgrid(np.arange(SIZE, dtype=np.float32), np.arange(SIZE, dtype=np.float32))


def flow_check(a, b):
    """-> (|flow| a->b in px at 112x112, residual ratio).

    Farneback (5 levels, 21 px window) is exact to ~10.5 px at 112x112 and then
    collapses -- 14 px of true motion reads as ~5, 24 px as ~2.6 -- and so does every
    other estimator tried (DIS, larger windows, half resolution): it is a limit of
    tiny frames. So the number alone would wave a too-fast clip through. The check
    is whether the flow explains the motion: warp b back by it and compare with a.
    Calibrated on known shifts, residual / raw difference is <= 0.14 when the flow
    is right (0.5..10.5 px) and >= 0.62 once it has collapsed (12 px and up).
    """
    f = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 5, 21, 5, 7, 1.5, 0)
    w = cv2.remap(b, _GX + f[..., 0], _GY + f[..., 1], cv2.INTER_LINEAR,
                  borderMode=cv2.BORDER_REPLICATE)
    c = (slice(8, -8), slice(8, -8))
    raw = float(np.mean(np.abs(a.astype(np.float32) - b)[c]))
    res = float(np.mean(np.abs(a.astype(np.float32) - w)[c]))
    return np.hypot(f[..., 0], f[..., 1]), (res / raw if raw > 3.0 else 0.0)


def measure(path, samples, scan):
    """-> dict: frames, fps, size, motion p90/median (px, t->t+2 at 112), max frame diff."""
    frames, hists, fps, w, h = read_small(path, scan)
    n = len(frames)
    m = {"frames": n, "fps": round(fps, 2), "size": "%dx%d" % (w, h),
         "p90_px": 0.0, "median_px": 0.0, "resid_ratio": 0.0, "cut_hist": 0.0}
    if n < 3:
        return m
    # Scene cuts: hue/value histogram distance between neighbouring frames. Motion moves
    # content around but keeps its histogram; a cut replaces it. Raw frame difference
    # does not work here: on a moving textured clip, normal motion already differs by
    # 14 grey levels per frame and a synthetic cut by only 23. Calibrated: continuous
    # clips <= 0.025, cuts 0.11-0.36 (sample.mp4 has one at 0.93).
    d = np.array([cv2.compareHist(hists[i], hists[i + 1], cv2.HISTCMP_BHATTACHARYYA)
                  for i in range(n - 1)])
    m["cut_hist"] = round(float(d.max()), 3)
    m["cut_x_median"] = round(float(d.max() / max(np.median(d), 0.005)), 1)
    p90s, meds, ratios = [], [], []
    for t in np.linspace(0, n - 3, min(samples, n - 2)).astype(int):
        mag, ratio = flow_check(frames[t], frames[t + 2])
        p90s.append(float(np.percentile(mag, 90)))
        meds.append(float(np.median(mag)))
        ratios.append(ratio)
    m["resid_ratio"] = round(float(np.median(ratios)), 2)
    m["p90_px"] = round(float(np.median(p90s)), 2)
    m["median_px"] = round(float(np.median(meds)), 2)
    return m


def link_or_copy(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src", help="folder of videos, searched recursively (e.g. UCF-101/)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-frames", type=int, default=100_000,
                    help="training frame budget; ~37 KB of RAM each in train_vfi.py")
    ap.add_argument("--per-group", type=int, default=1, help="max clips per (class, group)")
    ap.add_argument("--min-px", type=float, default=0.75)
    ap.add_argument("--max-px", type=float, default=6.0)
    ap.add_argument("--cut-hist", type=float, default=0.08,
                    help="a scene cut is a neighbour histogram distance above this ...")
    ap.add_argument("--cut-ratio", type=float, default=6.0,
                    help="... AND this many times the clip's median distance")
    ap.add_argument("--max-resid", type=float, default=0.5,
                    help="max warp residual / raw frame difference; above it the motion "
                         "is too fast to track (the measured px is then an underestimate)")
    ap.add_argument("--min-frames", type=int, default=30)
    ap.add_argument("--samples", type=int, default=8, help="triplets measured per clip")
    ap.add_argument("--scan", type=int, default=400, help="frames read per clip for measuring")
    ap.add_argument("--demo", type=int, default=5, help="demo candidates to set aside")
    a = ap.parse_args(argv)

    files = sorted(os.path.join(dp, f) for dp, _, fs in os.walk(a.src) for f in fs
                   if f.lower().endswith(EXTS))
    if not files:
        sys.exit("no %s files under %s" % ("/".join(EXTS), a.src))
    print("measuring %d clips at %dx%d ..." % (len(files), SIZE, SIZE))

    rows = []
    for i, p in enumerate(files):
        m = measure(p, a.samples, a.scan)
        name = os.path.splitext(os.path.basename(p))[0]
        u = UCF.match(name)
        m.update(path=p, name=name,
                 cls=u.group("cls") if u else os.path.basename(os.path.dirname(p)),
                 group=("g" + u.group("g")) if u else name)
        if m["frames"] < a.min_frames:
            m["verdict"] = "too short"
        elif m["cut_hist"] > a.cut_hist and m["cut_x_median"] > a.cut_ratio:
            m["verdict"] = "scene cut"
        elif m["p90_px"] < a.min_px:
            m["verdict"] = "static"
        elif m["p90_px"] > a.max_px:
            m["verdict"] = "too fast"
        elif m["p90_px"] > 1.5 and m["resid_ratio"] > a.max_resid:
            m["verdict"] = "too fast"          # untrackable: the flow does not explain it
        else:
            m["verdict"] = "ok"
        rows.append(m)
        if (i + 1) % 200 == 0 or i + 1 == len(files):
            print("  %d/%d" % (i + 1, len(files)))

    ok = [r for r in rows if r["verdict"] == "ok"]

    # Demo candidates first, from whole groups set aside: mid-range motion, longest first.
    by_group = defaultdict(list)
    for r in ok:
        by_group[(r["cls"], r["group"])].append(r)
    demo_groups, demo = set(), []
    for r in sorted((r for r in ok if 2.0 <= r["p90_px"] <= 5.0), key=lambda r: -r["frames"]):
        if len(demo) >= a.demo:
            break
        key = (r["cls"], r["group"])
        if key in demo_groups or any(d["cls"] == r["cls"] for d in demo):
            continue                                   # one demo per class, for variety
        demo_groups.add(key)
        demo.append(r)
        r["verdict"] = "demo"

    # Training: round-robin over classes, at most per_group clips per group, within budget.
    per_cls = defaultdict(list)
    for (cls, g), rs in sorted(by_group.items()):
        if (cls, g) in demo_groups:
            continue
        per_cls[cls].extend(sorted(rs, key=lambda r: r["name"])[:a.per_group])
    train, budget = [], a.max_frames
    queues = {c: list(v) for c, v in per_cls.items()}
    while budget > 0 and any(queues.values()):
        for c in sorted(queues):
            if queues[c] and budget > 0:
                r = queues[c].pop(0)
                if r["frames"] <= budget or not train:
                    train.append(r)
                    r["verdict"] = "train"
                    budget -= r["frames"]
    for r in rows:
        if r["verdict"] == "ok":
            r["verdict"] = "ok, not needed (budget or per-group cap)"

    for sub, chosen in (("train", train), ("demo", demo)):
        d = os.path.join(a.out, sub)
        os.makedirs(d, exist_ok=True)
        for r in chosen:
            link_or_copy(r["path"], os.path.join(d, os.path.basename(r["path"])))
    with open(os.path.join(a.out, "picks.csv"), "w", newline="") as f:
        cols = ["verdict", "name", "cls", "group", "frames", "fps", "size",
                "p90_px", "median_px", "resid_ratio", "cut_hist", "cut_x_median", "path"]
        w = csv.DictWriter(f, cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["verdict"], r["name"])))

    tally = defaultdict(int)
    for r in rows:
        tally[r["verdict"].split(",")[0]] += 1
    tr_frames = sum(r["frames"] for r in train)
    p90 = np.array([r["p90_px"] for r in train]) if train else np.zeros(1)
    print("\nverdicts: %s" % ", ".join("%s %d" % kv for kv in sorted(tally.items())))
    print("train: %d clips, %d classes, %d frames (~%.1f GB in train_vfi.py), "
          "motion p90 median %.2f px [%.2f..%.2f]"
          % (len(train), len({r['cls'] for r in train}), tr_frames, tr_frames * SIZE * SIZE * 3 / 1e9,
             np.median(p90), p90.min(), p90.max()))
    for r in demo:
        print("demo:  %-40s %4d frames  motion %.2f px  (%s, group not in train)"
              % (r["name"], r["frames"], r["p90_px"], r["cls"]))
    print("\nnext:  python train_vfi.py --videos %s --out run_video"
          % os.path.join(a.out, "train"))
    if len(train) < 2:
        print("WARNING: train_vfi.py needs at least 2 videos")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
