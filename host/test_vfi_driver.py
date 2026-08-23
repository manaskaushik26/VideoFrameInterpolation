"""
Offline tests for vfi_driver, with the hardware replaced by a software model.

No board required. What this checks is the part of the driver most likely to be
wrong and least likely to be noticed: tensor layout.

  - multi-pass stitching. A layer with more output channels than the MAC array
    runs in ceil(ch_out / n_out) passes, and the partial results are reassembled
    here. An off-by-one in that assembly produces a full-size, plausible-looking
    output with channels silently misplaced -- nothing crashes, and on real
    weights it would just look like a bad model.
  - the zero-padded final pass when ch_out is not a multiple of n_out, and the
    discarding of the surplus channels the hardware always emits.
  - weight file reshape: (out, in, kh, kw), no transpose.
  - Q1.15 conversion round-trip and saturation.

The mock computes the same `same`-padded cross-correlation, requantisation and
ReLU the RTL does, so a whole-layer result assembled from passes is compared
against a single direct convolution. Those must agree exactly.

Run:  python host/test_vfi_driver.py
"""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vfi_driver import (  # noqa: E402
    Vfi, VfiLayer, VfiError, load_weight_file, quantise_q15, dequantise_q15,
    REG_ID, REG_CFG, REG_CTRL, REG_STATUS, REG_FRAMES, ID_MAGIC,
    STATUS_WEIGHTS_LOADED, STATUS_SAT_STICKY,
)

IMG_W, IMG_H, N_OUT = 8, 6, 4


def ref_conv(fmap, kernels, shift):
    """Same-padded cross-correlation + requantise + ReLU, in 64-bit."""
    h, w, c_in = fmap.shape
    n_out = kernels.shape[0]
    x = fmap.astype(np.int64)
    k = kernels.astype(np.int64)
    acc = np.zeros((h, w, n_out), dtype=np.int64)
    pad = np.zeros((h + 2, w + 2, c_in), dtype=np.int64)
    pad[1:-1, 1:-1] = x
    for n in range(n_out):
        for kr in range(3):
            for kc in range(3):
                acc[:, :, n] += (pad[kr:kr + h, kc:kc + w, :] * k[n, :, kr, kc]).sum(axis=2)
    if shift == 0:
        r = acc
    else:
        r = (acc + (1 << (shift - 1))) >> shift
    r = np.clip(r, -32768, 32767)
    return np.maximum(r, 0).astype(np.int16), bool((r >= 32767).any() or (r <= -32768).any())


class MockIP:
    def __init__(self):
        self.regs = {REG_ID: ID_MAGIC, REG_CFG: 0, REG_STATUS: 0, REG_FRAMES: 0}

    def read(self, a):
        return self.regs.get(a, 0)

    def write(self, a, v):
        if a == REG_CTRL:
            if v & 1:
                self.regs[REG_STATUS] &= ~STATUS_SAT_STICKY
            return
        self.regs[a] = v


class _SendChannel:
    """Consumes a transfer and computes what the PL would compute."""

    def __init__(self, ip, hw):
        self.ip, self.hw = ip, hw

    def transfer(self, buf):
        arr = np.array(buf, dtype=np.int16)
        cfg = self.ip.read(REG_CFG)
        c_in, shift = cfg & 0xFF, (cfg >> 8) & 0x3F

        if arr.size == self.hw["n_out"] * c_in * 9:          # kernels
            self.hw["kernels"] = arr.reshape(self.hw["n_out"], c_in, 3, 3)
            self.ip.regs[REG_STATUS] |= STATUS_WEIGHTS_LOADED
        elif arr.size == IMG_H * IMG_W * c_in:               # feature map
            fmap = arr.reshape(IMG_H, IMG_W, c_in)
            out, sat = ref_conv(fmap, self.hw["kernels"], shift)
            self.hw["out"] = out
            if sat:
                self.ip.regs[REG_STATUS] |= STATUS_SAT_STICKY
            self.ip.regs[REG_FRAMES] += 1
        else:
            raise AssertionError(f"unexpected transfer of {arr.size} values")

    def wait(self):
        pass


class _RecvChannel:
    """Hands back whatever the last feature-map transfer produced."""

    def __init__(self, hw):
        self.hw = hw
        self._buf = None

    def transfer(self, buf):
        self._buf = buf

    def wait(self):
        if self._buf is None:
            raise AssertionError("recv wait() with no outstanding transfer")
        self._buf[:] = self.hw["out"].reshape(-1)
        self._buf = None


class MockDma:
    # Send and receive are distinct objects, so a transfer is never ambiguous.
    # Marking buffers instead (an earlier attempt) mistook the feature-map send
    # for the result buffer, because every allocation looked the same.
    def __init__(self, ip, hw):
        self.sendchannel = _SendChannel(ip, hw)
        self.recvchannel = _RecvChannel(hw)


def _patch_allocate():
    """pynq.allocate stand-in: a plain ndarray with freebuffer()."""
    import types
    mod = types.ModuleType("pynq")

    class Buf(np.ndarray):
        def freebuffer(self):
            pass

    def allocate(shape, dtype):
        return np.zeros(shape, dtype=dtype).view(Buf)

    mod.allocate = allocate
    sys.modules["pynq"] = mod


def make_vfi(n_out=N_OUT):
    hw = {"n_out": n_out, "kernels": None, "out": None}
    ip = MockIP()
    fm = MockDma(ip, hw)
    wt = MockDma(ip, hw)
    return Vfi(ip, fm, wt, img_w=IMG_W, img_h=IMG_H, n_out=n_out, max_ch=128), hw


def check(name, cond, fails):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        fails.append(name)


def main():
    _patch_allocate()
    rng = np.random.default_rng(7)
    fails = []

    print("--- multi-pass stitching vs one direct convolution ---")
    # ch_out both a multiple of n_out and not, to cover the padded final pass.
    for ch_in, ch_out in [(3, 4), (3, 8), (6, 10), (2, 1), (5, 4), (8, 13)]:
        vfi, _ = make_vfi()
        shift = 20
        fmap = rng.integers(-2000, 2000, (IMG_H, IMG_W, ch_in)).astype(np.int16)
        kern = rng.integers(-3000, 3000, (ch_out, ch_in, 3, 3)).astype(np.int16)

        got, st = vfi.run_layer(fmap, kern, VfiLayer(ch_in, ch_out, shift))
        exp, _ = ref_conv(fmap, kern, shift)

        n_passes = -(-ch_out // N_OUT)
        check(f"{ch_in}->{ch_out}: {n_passes} pass(es), result matches",
              got.shape == exp.shape and np.array_equal(got, exp), fails)

    print("--- surplus channels from a padded final pass are discarded ---")
    vfi, _ = make_vfi()
    fmap = rng.integers(-2000, 2000, (IMG_H, IMG_W, 3)).astype(np.int16)
    kern = rng.integers(-3000, 3000, (5, 3, 3, 3)).astype(np.int16)   # 5 = 2 passes
    got, st = vfi.run_layer(fmap, kern, VfiLayer(3, 5, 20))
    exp, _ = ref_conv(fmap, kern, 20)
    check("ch_out=5 with n_out=4 gives exactly 5 channels", got.shape[2] == 5, fails)
    check("ch_out=5 values match", np.array_equal(got, exp), fails)

    print("--- saturation is reported ---")
    vfi, _ = make_vfi()
    fmap = np.full((IMG_H, IMG_W, 2), 30000, dtype=np.int16)
    kern = np.full((4, 2, 3, 3), 30000, dtype=np.int16)
    _, st = vfi.run_layer(fmap, kern, VfiLayer(2, 4, 0))    # shift 0 must clip
    check("shift=0 with full-scale operands reports saturation",
          any(s["saturated"] for s in st), fails)

    vfi, _ = make_vfi()
    _, st = vfi.run_layer(np.zeros((IMG_H, IMG_W, 2), np.int16),
                          np.zeros((4, 2, 3, 3), np.int16), VfiLayer(2, 4, 20))
    check("zeros do not report saturation",
          not any(s["saturated"] for s in st), fails)

    print("--- frame counter ---")
    vfi, _ = make_vfi()
    fmap = rng.integers(-100, 100, (IMG_H, IMG_W, 3)).astype(np.int16)
    kern = rng.integers(-100, 100, (9, 3, 3, 3)).astype(np.int16)  # 3 passes
    vfi.run_layer(fmap, kern, VfiLayer(3, 9, 20))
    check("3 passes increment FRAMES by 3", vfi.frames() == 3, fails)

    print("--- weight file reshape ---")
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "w.bin")
        ref = rng.integers(-32768, 32767, (7, 5, 3, 3)).astype(np.int16)
        ref.tofile(p)
        back = load_weight_file(p, 7, 5)
        check("(out,in,kh,kw) round-trips with no transpose", np.array_equal(back, ref), fails)
        try:
            load_weight_file(p, 7, 4)
            check("wrong shape is rejected", False, fails)
        except VfiError:
            check("wrong shape is rejected", True, fails)

    print("--- Q1.15 ---")
    f = np.array([0.0, 0.5, -0.5, 0.999969482421875, -1.0])
    q = quantise_q15(f)
    check("quantise round-trips", np.allclose(dequantise_q15(q), f, atol=1 / 32768), fails)
    check("saturates at +1.0", quantise_q15(np.array([2.0]))[0] == 32767, fails)
    check("saturates at -1.0", quantise_q15(np.array([-2.0]))[0] == -32768, fails)

    print("--- guards ---")
    vfi, _ = make_vfi()
    try:
        vfi.run_layer(np.zeros((IMG_H + 1, IMG_W, 3), np.int16),
                      np.zeros((4, 3, 3, 3), np.int16), VfiLayer(3, 4, 20))
        check("wrong image size is rejected", False, fails)
    except VfiError:
        check("wrong image size is rejected", True, fails)

    ip = MockIP()
    ip.regs[REG_ID] = 0xDEADBEEF
    try:
        Vfi(ip, MockDma(ip, {}), MockDma(ip, {}), IMG_W, IMG_H, N_OUT)
        check("wrong bitstream ID is rejected", False, fails)
    except VfiError:
        check("wrong bitstream ID is rejected", True, fails)

    print()
    if fails:
        print(f"FAIL  {len(fails)} check(s): {', '.join(fails)}")
        return 1
    print("PASS  all driver layout checks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
