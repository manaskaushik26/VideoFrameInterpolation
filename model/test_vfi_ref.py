"""
Tests for vfi_ref, the bit-exact reference model.

A reference model that is itself wrong is worse than none: it would make the RTL look
broken when it is right, or right when it is broken. So nothing here trusts the fast
path. The central test recomputes every result with a naive triple-loop written
straight from the definition, the same way sim/tb_conv_layer_mc.sv writes its
reference, and requires exact agreement.

Coverage, and why each case exists:

  - vectorised conv3x3_acc vs a naive from-the-definition loop. The matmul formulation
    is the only clever code in the model; an axis mix-up there produces a full-size,
    plausible-looking output with the kernel silently transposed.
  - padding checked positionally with a delta kernel, so an off-by-one at an edge
    cannot hide in an aggregate.
  - rounding is half-up including on negatives, where truncation and rounding differ
    in sign as well as magnitude.
  - both saturation arms are asserted to actually fire. tb_conv_mac_3x3 counts arm
    hits and fails if any never fires, because the superseded design's clamp was
    unreachable code that no test noticed.
  - cross-check against host/test_vfi_driver.py's independently written ref_conv.
    Two implementations written separately agreeing is worth more than either alone.

Run: python model/test_vfi_ref.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vfi_ref as R


FAILS = []


def check(name, cond):
    print("  %-4s %s" % ("PASS" if cond else "FAIL", name))
    if not cond:
        FAILS.append(name)


# ---------------------------------------------------------------- naive reference

def naive_acc(img, kern):
    """3x3 cross-correlation with `same` zero padding, written from the definition.

    Deliberately the slowest possible formulation: no numpy broadcasting, no tricks,
    nothing that could share a bug with the implementation under test.
    """
    H, W, c_in = img.shape
    c_out = kern.shape[0]
    out = np.zeros((H, W, c_out), dtype=np.int64)
    for oy in range(H):
        for ox in range(W):
            for o in range(c_out):
                acc = 0
                for c in range(c_in):
                    for kr in range(3):
                        for kc in range(3):
                            iy, ix = oy + kr - 1, ox + kc - 1
                            if 0 <= iy < H and 0 <= ix < W:
                                acc += int(img[iy, ix, c]) * int(kern[o, c, kr, kc])
                out[oy, ox, o] = acc
    return out


def rand_img(rng, h, w, c):
    return rng.integers(R.OUT_MIN, R.OUT_MAX + 1, size=(h, w, c), dtype=np.int64).astype(np.int16)


def rand_kern(rng, c_out, c_in):
    return rng.integers(R.OUT_MIN, R.OUT_MAX + 1, size=(c_out, c_in, 3, 3), dtype=np.int64).astype(np.int16)


# ---------------------------------------------------------------- tests

def test_conv_vs_naive():
    print("--- vectorised convolution vs naive definition ---")
    rng = np.random.default_rng(20260831)
    shapes = [(1, 1, 1, 1), (1, 5, 1, 3), (5, 1, 3, 1), (4, 4, 1, 1),
              (7, 5, 6, 4), (3, 3, 8, 2), (6, 6, 3, 24)]
    for h, w, c_in, c_out in shapes:
        img, kern = rand_img(rng, h, w, c_in), rand_kern(rng, c_out, c_in)
        check("%dx%dx%d -> %d channels matches the definition" % (h, w, c_in, c_out),
              np.array_equal(R.conv3x3_acc(img, kern), naive_acc(img, kern)))


def test_padding_is_positional():
    print("--- `same` zero padding, tap by tap ---")
    # A kernel that is 1 at exactly one tap turns the convolution into a shift, so a
    # wrong padding edge or a transposed tap index shows up as a displaced image
    # rather than as a slightly-off number.
    H = W = 4
    img = np.arange(1, H * W + 1, dtype=np.int16).reshape(H, W, 1)
    ok = True
    for kr in range(3):
        for kc in range(3):
            kern = np.zeros((1, 1, 3, 3), dtype=np.int16)
            kern[0, 0, kr, kc] = 1
            got = R.conv3x3_acc(img, kern)[:, :, 0]
            exp = np.zeros((H, W), dtype=np.int64)
            for oy in range(H):
                for ox in range(W):
                    iy, ix = oy + kr - 1, ox + kc - 1
                    if 0 <= iy < H and 0 <= ix < W:
                        exp[oy, ox] = img[iy, ix, 0]
            if not np.array_equal(got, exp):
                ok = False
    check("all 9 taps read (y+kr-1, x+kc-1) with zeros outside", ok)

    # The centre tap must be identity: this is the one that catches a whole-window
    # transpose, which every other tap test would still pass in mirrored form.
    kern = np.zeros((1, 1, 3, 3), dtype=np.int16)
    kern[0, 0, 1, 1] = 1
    check("centre tap [1][1] is the pixel itself",
          np.array_equal(R.conv3x3_acc(img, kern)[:, :, 0], img[:, :, 0].astype(np.int64)))


def test_rounding_half_up():
    print("--- round half up, at accumulator width ---")
    # shift=1: exact halves round up (toward +inf), matching (acc + 1) >> 1.
    acc = np.array([[[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]]], dtype=np.int64)
    out, _ = R.requantise(acc, 1)
    check("halves round toward +inf, negatives included",
          list(out[0, 0]) == [-2, -2, -1, -1, 0, 0, 1, 1, 2, 2, 3])

    check("shift of 0 adds no rounding bias",
          np.array_equal(R.requantise(np.array([[[7, -7]]], dtype=np.int64), 0)[0][0, 0],
                         np.array([7, -7], dtype=np.int16)))

    # Truncation would differ here; if this ever passes with a >> that lacks the bias,
    # the rounding stage has been dropped.
    acc = np.array([[[3]]], dtype=np.int64)
    check("rounding differs from truncation where it should",
          int(R.requantise(acc, 1)[0][0, 0, 0]) == 2)


def test_saturation_arms_fire():
    print("--- saturation, both arms ---")
    big = np.array([[[1 << 40, -(1 << 40), 0]]], dtype=np.int64)
    out, sat = R.requantise(big, 15)
    check("positive arm clamps to OUT_MAX", int(out[0, 0, 0]) == R.OUT_MAX and bool(sat[0, 0, 0]))
    check("negative arm clamps to OUT_MIN", int(out[0, 0, 1]) == R.OUT_MIN and bool(sat[0, 0, 1]))
    check("in-range value is not flagged", int(out[0, 0, 2]) == 0 and not bool(sat[0, 0, 2]))

    # Boundary: exactly OUT_MAX must NOT be flagged. The RTL compares `rounded >
    # OUT_MAX`, strictly greater. host/test_vfi_driver.py's mock uses >=, which
    # over-reports by one value; this model follows the RTL.
    edge = np.array([[[R.OUT_MAX, R.OUT_MIN, R.OUT_MAX + 1]]], dtype=np.int64)
    out, sat = R.requantise(edge, 0)
    check("exactly OUT_MAX is not saturation (RTL uses strict >)", not bool(sat[0, 0, 0]))
    check("exactly OUT_MIN is not saturation", not bool(sat[0, 0, 1]))
    check("one past OUT_MAX is saturation", bool(sat[0, 0, 2]))

    check("ACC_W overflow raises rather than wrapping silently",
          _raises(OverflowError, R.requantise, np.array([[[R.ACC_MAX]]], dtype=np.int64), 4))


def test_activation():
    print("--- act_unit ---")
    x = np.array([[[-32768, -1, 0, 1, 32767]]], dtype=np.int16)
    check("MODE 1 ReLU zeroes negatives, preserves the rest",
          list(R.activate(x, R.ACT_RELU)[0, 0]) == [0, 0, 0, 1, 32767])
    check("MODE 0 linear passes everything", np.array_equal(R.activate(x, R.ACT_LINEAR), x))
    check("MODE 2 (LIF) is rejected, not silently treated as ReLU",
          _raises(ValueError, R.activate, x, 2))


def test_conv_stage_composition():
    print("--- conv_stage: conv -> requantise -> activate ---")
    rng = np.random.default_rng(7)
    img, kern = rand_img(rng, 5, 5, 4), rand_kern(rng, 6, 4)
    out, info = R.conv_stage(img, kern, 20, R.ACT_RELU)
    exp, sat = R.requantise(R.conv3x3_acc(img, kern), 20)
    check("stage equals its three steps applied in order",
          np.array_equal(out, np.where(exp < 0, np.int16(0), exp)))
    check("activation is applied after the clamp, not before", (out >= 0).all())
    check("sat_count matches the requantise mask", info["sat_count"] == int(sat.sum()))

    # A shift too small must clip and say so -- this is the sat_sticky diagnostic the
    # hardware exposes for tuning OUT_SHIFT, so it has to be trustworthy.
    _, lo = R.conv_stage(img, kern, 4, R.ACT_RELU)
    _, hi = R.conv_stage(img, kern, 30, R.ACT_RELU)
    check("a too-small shift saturates and sets sat_sticky", lo["sat_sticky"])
    check("a large shift does not saturate", not hi["sat_sticky"])


def test_pass_splitting():
    print("--- multi-pass output channel split ---")
    check("128 output channels take 6 passes at N_OUT=24", len(R.output_passes(128)) == 6)
    check("24 fits in one pass", R.output_passes(24) == [(0, 24)])
    ok = True
    for c_out in range(1, 130):
        p = R.output_passes(c_out)
        covered = [i for a, b in p for i in range(a, b)]
        if covered != list(range(c_out)):
            ok = False
    check("every channel is covered exactly once, in order, for 1..129", ok)
    check("a final short pass is not padded into the range", R.output_passes(25)[-1] == (24, 25))


def test_weight_and_dump_io(tmp):
    print("--- weight and dump I/O ---")
    rng = np.random.default_rng(11)
    kern = rand_kern(rng, 5, 3)
    p = os.path.join(tmp, "w.bin")
    kern.astype("<i2").tofile(p)
    check("weight file reshapes (out, in, kh, kw) with no transpose",
          np.array_equal(R.load_weights(p, 5, 3), kern))
    check("a wrong-sized weight file is rejected",
          _raises(ValueError, R.load_weights, p, 5, 4))

    fmap = rand_img(rng, 4, 6, 3)
    d = os.path.join(tmp, "f.bin")
    R.dump(fmap, d)
    check("dump round-trips in (row, col, channel) order",
          np.array_equal(R.load_dump(d, 4, 6, 3), fmap))
    check("a wrong-shaped dump is rejected", _raises(ValueError, R.load_dump, d, 4, 6, 4))


def test_bias():
    print("--- host-side bias ---")
    x = np.array([[[32760, -32760]]], dtype=np.int16)
    check("bias saturates rather than wrapping",
          list(R.add_bias(x, np.array([100, -100]))[0, 0]) == [R.OUT_MAX, R.OUT_MIN])
    check("bias length must match the channel count",
          _raises(ValueError, R.add_bias, x, np.array([1, 2, 3])))


def test_cross_check_host_driver():
    print("--- cross-check against host/test_vfi_driver.py's ref_conv ---")
    here = os.path.dirname(os.path.abspath(__file__))
    host = os.path.join(os.path.dirname(here), "host")
    if not os.path.exists(os.path.join(host, "test_vfi_driver.py")):
        check("host ref_conv available", False)
        return
    sys.path.insert(0, host)
    from test_vfi_driver import ref_conv

    rng = np.random.default_rng(99)
    ok = True
    for h, w, c_in, c_out, shift in [(5, 5, 3, 4, 20), (4, 7, 8, 6, 18), (3, 3, 1, 2, 15)]:
        img, kern = rand_img(rng, h, w, c_in), rand_kern(rng, c_out, c_in)
        mine, _ = R.conv_stage(img, kern, shift, R.ACT_RELU)
        theirs, _ = ref_conv(img, kern, shift)
        if not np.array_equal(mine, theirs):
            ok = False
    check("two independently written implementations agree exactly", ok)


def _raises(exc, fn, *a):
    try:
        fn(*a)
    except exc:
        return True
    except Exception:
        return False
    return False


def main():
    import tempfile
    print("vfi_ref -- reference model checks\n")
    test_conv_vs_naive()
    test_padding_is_positional()
    test_rounding_half_up()
    test_saturation_arms_fire()
    test_activation()
    test_conv_stage_composition()
    test_pass_splitting()
    with tempfile.TemporaryDirectory() as tmp:
        test_weight_and_dump_io(tmp)
    test_bias()
    test_cross_check_host_driver()

    print()
    if FAILS:
        print("FAIL  %d check(s) failed:" % len(FAILS))
        for f in FAILS:
            print("  - %s" % f)
        return 1
    print("PASS  all reference model checks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
