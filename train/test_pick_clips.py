"""
Known-answer test for pick_clips.py: synthetic clips whose motion and cuts are set
exactly, named like UCF101, each of which must get the right verdict.

A clip translates a multi-scale noise texture so that the motion between frames t
and t+2, after the 112x112 resize, is exactly the stated number of pixels.

Run: python train/test_pick_clips.py      (numpy + opencv, about a minute)
"""

import csv
import os
import shutil
import sys
import tempfile

import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pick_clips  # noqa: E402

FAILS = []


def check(name, cond):
    print("  %-4s %s" % ("PASS" if cond else "FAIL", name))
    if not cond:
        FAILS.append(name)


def make_clips(d):
    rng = np.random.default_rng(0)
    tex = sum(cv2.GaussianBlur(rng.normal(0, 1, (900, 3000)), (0, 0), s) * s for s in (1, 3, 8))
    tex = cv2.cvtColor(cv2.normalize(tex, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
                       cv2.COLOR_GRAY2BGR)

    def clip(name, px112, n=80, cut=False):
        dx = px112 / 2 * 320 / 112                  # per frame at 320 wide
        vw = cv2.VideoWriter(os.path.join(d, name + ".avi"),
                             cv2.VideoWriter_fourcc(*"MJPG"), 25, (320, 240))
        for t in range(n):
            y0 = 500 if (cut and t >= n // 2) else 100
            f = cv2.warpAffine(tex, np.float32([[1, 0, -dx * t], [0, 1, 0]]),
                               (3000, 900))[y0:y0 + 240, 50:370]
            if cut and t >= n // 2:
                f = cv2.convertScaleAbs(f, alpha=0.6, beta=60)
            vw.write(np.ascontiguousarray(f))
        vw.release()

    expect = {}
    for name, px, want in [
        ("v_Walk_g01_c01", 2.5, "train"), ("v_Walk_g01_c02", 2.5, "ok, not needed"),
        ("v_Walk_g02_c01", 2.0, "train"), ("v_Run_g01_c01", 4.0, None),
        ("v_Run_g02_c01", 3.5, None), ("v_Jump_g01_c01", 3.0, None),
        ("v_Jump_g02_c01", 4.5, None), ("v_Sit_g01_c01", 0.0, "static"),
        ("v_Trot_g01_c01", 7.0, "too fast"),        # measurable, above the +-6 px field
        ("v_Sprint_g01_c01", 14.0, "too fast"),     # flow collapses; caught by residual
    ]:
        clip(name, px)
        expect[name] = (px, want)
    clip("v_Pan_g01_c01", 3.0, cut=True)
    expect["v_Pan_g01_c01"] = (3.0, "scene cut")
    return expect


def main():
    tmp = tempfile.mkdtemp()
    src, out = os.path.join(tmp, "src"), os.path.join(tmp, "out")
    os.makedirs(src)
    expect = make_clips(src)
    rc = pick_clips.main([src, "--out", out, "--demo", "2"])
    rows = {r["name"]: r for r in csv.DictReader(open(os.path.join(out, "picks.csv")))}

    print("--- verdicts ---")
    check("exit code 0", rc == 0)
    for name, (px, want) in sorted(expect.items()):
        got = rows[name]["verdict"]
        if want is not None:
            check("%-18s %4.1f px -> %-15s (got %s)" % (name, px, want, got), got.startswith(want))
    for cls in ("Run", "Jump"):
        v = sorted(rows["v_%s_g0%d_c01" % (cls, g)]["verdict"] for g in (1, 2))
        check("%s: one group to demo, the other to train (got %s)" % (cls, v), v == ["demo", "train"])

    print("--- measurement ---")
    for name, (px, _) in expect.items():
        if 0 < px <= 6 and "Pan" not in name:
            got = float(rows[name]["p90_px"])
            check("%-18s motion measured %.2f px, true %.1f" % (name, got, px), abs(got - px) < 0.25)

    print("--- layout ---")
    train = sorted(os.listdir(os.path.join(out, "train")))
    demo = sorted(os.listdir(os.path.join(out, "demo")))
    check("train/ holds exactly the train verdicts",
          train == sorted(n + ".avi" for n, r in rows.items() if r["verdict"] == "train"))
    check("demo/ shares no (class, group) with train/",
          not {d.rsplit("_c", 1)[0] for d in demo} & {t.rsplit("_c", 1)[0] for t in train})

    shutil.rmtree(tmp, ignore_errors=True)
    print("")
    print("FAIL  %d checks" % len(FAILS) if FAILS else "PASS  all clip-picker checks")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
