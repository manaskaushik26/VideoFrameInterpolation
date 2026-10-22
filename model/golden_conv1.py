"""
golden_conv1 -- integer golden model for ONE layer, conv1 (6 -> 32), with the
trained weights, plus the stimulus/expected files the xsim testbench reads.

    python model/golden_conv1.py                 # writes build/conv1/
    python model/golden_conv1.py --check-sim     # after xsim: compare its dump

The hardware has N_OUT = 24 MACs, so conv1's 32 output channels run as two
passes, exactly as host/vfi_driver.py:run_layer does it:

    pass 0: kernels  0..23
    pass 1: kernels 24..31, then 16 all-zero kernels (outputs discarded)

Every file here is one 16-bit value per line in hex, in AXI-Stream beat order,
so the testbench $readmemh's them with no reordering:

    input.hex        112*112*6   beats, (row, col, channel) channel-minor
    w_pass{p}.hex    24*6*9      beats, (out, in, kh, kw)
    exp_pass{p}.hex  112*112*24  beats, (row, col, channel) channel-minor
    conv1_golden.bin 112*112*32  raw <i2, the whole layer stitched

The arithmetic is vfi_ref.conv_stage, which mirrors conv_mac_3x3 + act_unit and
is itself tested in test_vfi_ref.py. It is not trusted on its own terms here
either: a few hundred outputs are recomputed below by a scalar loop written from
the definition, and the whole layer is compared against the golden conv1.bin
that verify_export.py wrote when the model was exported.
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import vfi_ref  # noqa: E402

H = W = 112
C_IN, C_OUT, N_OUT = 6, 32, vfi_ref.N_OUT_MAX


def write_hex(path, values):
    v = np.asarray(values, dtype=np.int16).reshape(-1).view(np.uint16)
    with open(path, "w", newline="\n") as f:
        f.write("\n".join("%04x" % x for x in v))
        f.write("\n")


def read_hex(path):
    with open(path) as f:
        v = [int(t, 16) for t in f.read().split()]
    return np.array(v, dtype=np.uint16).view(np.int16)


def scalar_ref(img, kern, shift, y, x, o, relu=True):
    """One output value, from the definition, in plain Python ints."""
    acc = 0
    for c in range(img.shape[2]):
        for kr in range(3):
            for kc in range(3):
                iy, ix = y + kr - 1, x + kc - 1
                if 0 <= iy < H and 0 <= ix < W:
                    acc += int(img[iy, ix, c]) * int(kern[o, c, kr, kc])
    r = (acc + (1 << (shift - 1))) >> shift
    r = max(-32768, min(32767, r))
    return max(0, r) if relu else r


def passes(kern):
    """(n_out, C_IN, 3, 3) kernel sets per pass, zero-padded like vfi_driver."""
    out = []
    for lo, hi in vfi_ref.output_passes(C_OUT, N_OUT):
        k = np.zeros((N_OUT, C_IN, 3, 3), dtype=np.int16)
        k[: hi - lo] = kern[lo:hi]
        out.append((lo, hi, k))
    return out


def generate(export, out_dir, act="relu", stim="mnist"):
    cfg = json.load(open(os.path.join(export, "vfi_model.json")))
    shift = cfg["shifts"]["conv1"]
    kern = vfi_ref.load_weights(os.path.join(export, "conv1_weight_q15.bin"), C_OUT, C_IN)
    mode = vfi_ref.ACT_RELU if act == "relu" else vfi_ref.ACT_LINEAR
    if stim == "mnist":
        img = vfi_ref.load_dump(os.path.join(export, "golden", "input.bin"), H, W, C_IN)
    else:
        # Dense: every pixel lit, uniform over the [0, 1) range real pixels use.
        # The held-out Moving-MNIST pair is ~90% black background, so most of its
        # windows are all-zero and test nothing.
        img = np.random.default_rng(20261013).integers(0, 32768, size=(H, W, C_IN)).astype(np.int16)

    out, info = vfi_ref.conv_stage(img, kern, shift, mode)
    acc = vfi_ref.conv3x3_acc(img, kern)
    rounded = (acc + (1 << (shift - 1))) >> shift
    print("conv1 [%s, %s input]: shift %d, %dx%dx%d -> %s"
          % (act, stim, shift, H, W, C_IN, out.shape))
    print("  output range [%d, %d]: %d negative, %d zero, %d positive"
          % (out.min(), out.max(), int(np.sum(out < 0)), int(np.sum(out == 0)),
             int(np.sum(out > 0))))
    print("  pre-activation: %d negative; saturated high %d, low %d (%.3f%%)"
          % (int(np.sum(rounded < 0)), int(np.sum(rounded > 32767)),
             int(np.sum(rounded < -32768)), 100.0 * info["sat_frac"]))

    # Check 1: a scalar loop from the definition, on corners, edges and randoms.
    rng = np.random.default_rng(1)
    pts = [(0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1), (0, 57), (57, 0), (H - 1, 57)]
    pts += [tuple(p) for p in rng.integers(0, H, size=(60, 2))]
    bad = 0
    for y, x in pts:
        for o in range(C_OUT):
            if scalar_ref(img, kern, shift, y, x, o, mode == vfi_ref.ACT_RELU) != out[y, x, o]:
                bad += 1
    print("  scalar cross-check: %d values, %d mismatches" % (len(pts) * C_OUT, bad))
    if bad:
        sys.exit("FAIL: vfi_ref disagrees with the scalar definition")

    # Check 2: the golden dump written at export time (that run is ReLU, MNIST).
    ref_path = os.path.join(export, "golden", "conv1.bin")
    if os.path.exists(ref_path) and act == "relu" and stim == "mnist":
        ref = vfi_ref.load_dump(ref_path, H, W, C_OUT)
        n = int(np.sum(ref != out))
        print("  vs export golden/conv1.bin: %d mismatches of %d" % (n, out.size))
        if n:
            sys.exit("FAIL: does not reproduce the exported golden conv1")

    os.makedirs(out_dir, exist_ok=True)
    write_hex(os.path.join(out_dir, "input.hex"), img)
    vfi_ref.dump(out, os.path.join(out_dir, "conv1_golden.bin"))
    sat = []
    for p, (lo, hi, k) in enumerate(passes(kern)):
        part, pinfo = vfi_ref.conv_stage(img, k, shift, mode)
        assert np.array_equal(part[:, :, : hi - lo], out[:, :, lo:hi])
        write_hex(os.path.join(out_dir, "w_pass%d.hex" % p), k)
        write_hex(os.path.join(out_dir, "exp_pass%d.hex" % p), part)
        sat.append(int(pinfo["sat_sticky"]))
        print("  pass %d: channels %2d..%2d -> w_pass%d.hex, exp_pass%d.hex  (sat_sticky %d)"
              % (p, lo, hi - 1, p, p, sat[-1]))
    # shift, then the expected sat_sticky per pass -- read by the testbench
    with open(os.path.join(out_dir, "cfg.hex"), "w", newline="\n") as f:
        f.write("%02x\n" % shift + "".join("%02x\n" % s for s in sat))
    print("wrote %s" % out_dir)


def check_sim(out_dir):
    """Stitch the testbench's per-pass dumps and diff the whole layer."""
    gold = vfi_ref.load_dump(os.path.join(out_dir, "conv1_golden.bin"), H, W, C_OUT)
    sim = np.zeros_like(gold)
    for p, (lo, hi) in enumerate(vfi_ref.output_passes(C_OUT, N_OUT)):
        path = os.path.join(out_dir, "sim_pass%d.hex" % p)
        v = read_hex(path)
        if v.size != H * W * N_OUT:
            sys.exit("FAIL: %s has %d values, expected %d" % (path, v.size, H * W * N_OUT))
        part = v.reshape(H, W, N_OUT)
        sim[:, :, lo:hi] = part[:, :, : hi - lo]
        if np.any(part[:, :, hi - lo:]):
            sys.exit("FAIL: pass %d's zero-kernel channels are not zero" % p)

    diff = np.argwhere(sim != gold)
    print("RTL (xsim) vs Python golden, conv1, %d values" % gold.size)
    if len(diff):
        for y, x, c in diff[:10]:
            print("  (%d,%d,ch%d): sim %d  golden %d" % (y, x, c, sim[y, x, c], gold[y, x, c]))
        sys.exit("FAIL: %d mismatches" % len(diff))
    vfi_ref.dump(sim, os.path.join(out_dir, "conv1_sim.bin"))
    print("PASS: bit-exact, 0 mismatches")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--export", default=os.path.join(REPO, "train", "export_gpu"))
    ap.add_argument("--out", default=os.path.join(REPO, "build", "conv1"))
    ap.add_argument("--act", choices=["relu", "linear"], default="relu")
    ap.add_argument("--input", choices=["mnist", "random"], default="mnist")
    ap.add_argument("--check-sim", action="store_true")
    a = ap.parse_args()
    check_sim(a.out) if a.check_sim else generate(a.export, a.out, a.act, a.input)
