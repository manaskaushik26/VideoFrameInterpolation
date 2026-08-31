"""
Tests for bringup.py, the staged first-run script.

bringup.py runs on borrowed board time, and a script that crashes on its own
first line wastes exactly the resource it exists to protect. So it is exercised
here against the mock hardware in test_vfi_driver.py, retargeted from that
module's small test geometry to the real 112x112x24 shape the bitstream is built
for. No board required.

The happy path is the easy half. The half that matters is the failure paths: a
bring-up script whose diagnosis is wrong is worse than none, because it sends you
hunting in the wrong place with a board on loan. Each negative case below injects
one specific fault and requires the script to stop at the right stage.

The all-zeros case is the one to keep. That is precisely how the superseded
design failed -- a datapath that synthesised cleanly, passed its unit tests and
computed a constant zero -- and if it ever recurs this script must name it rather
than report a successful transfer of nothing.

Run: python host/test_bringup.py
"""

import io
import os
import sys
import tempfile
import types
from contextlib import redirect_stdout

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "model"))

import test_vfi_driver as T          # noqa: E402  (mock hardware lives here)
import bringup                        # noqa: E402

FAILS = []


def check(name, cond):
    print("  %-4s %s" % ("PASS" if cond else "FAIL", name))
    if not cond:
        FAILS.append(name)


class _Corrupt:
    """Wraps a receive channel so the returned data can be damaged on purpose."""

    def __init__(self, inner, mode):
        self.inner, self.mode = inner, mode

    def transfer(self, buf):
        self._buf = buf
        self.inner.transfer(buf)

    def wait(self):
        self.inner.wait()
        if self.mode == "zeros":
            self._buf[:] = 0
        elif self.mode == "one_value":
            self._buf[0] = self._buf[0] + 1 if self._buf[0] < 32767 else 0


def build_env(bad_id=False, corrupt=None):
    """A fake `pynq` backed by the mock PL, at the real bitstream geometry."""
    T.IMG_W, T.IMG_H = bringup.IMG_W, bringup.IMG_H

    hw = {"n_out": bringup.N_OUT, "kernels": None, "out": None}
    ip = T.MockIP()
    if bad_id:
        ip.regs[T.REG_ID] = 0x00000000      # unprogrammed PL looks like this
    fm, wt = T.MockDma(ip, hw), T.MockDma(ip, hw)
    if corrupt:
        fm.recvchannel = _Corrupt(fm.recvchannel, corrupt)

    class FakeOverlay:
        def __init__(self, path):
            self.ip_dict = {"vfi": {}, "dma_fm": {}, "dma_wt": {}}
            self.vfi, self.dma_fm, self.dma_wt = ip, fm, wt

    class Buf(np.ndarray):
        def freebuffer(self):
            pass

    mod = types.ModuleType("pynq")
    mod.Overlay = FakeOverlay
    mod.allocate = lambda shape, dtype: np.zeros(shape, dtype=dtype).view(Buf)
    sys.modules["pynq"] = mod

    bit = os.path.join(tempfile.mkdtemp(), "vfi.bit")
    with open(bit, "wb") as fh:
        fh.write(b"\x00")
    return bit


def run(**kw):
    """Run bringup end to end, returning (exit_code, captured_output)."""
    bit = build_env(**kw)
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = bringup.main(["bringup.py", bit])
    return rc, buf.getvalue()


# ---------------------------------------------------------------- scenarios

def test_happy_path():
    print("--- all stages pass against working mock hardware ---")
    rc, out = run()
    check("exit code is 0", rc == 0)
    check("all six stages ran", out.count("PASS") >= 6)
    check("stage 5 confirms an exact match", "match exactly" in out)
    # The sizes that would have failed under the old 14-bit length register.
    check("moves a full-size feature map (150,528 in / 602,112 out)",
          "150528" in out and "602112" in out)


def test_bad_id_stops_at_stage_1():
    print("--- unprogrammed PL / wrong bitstream ---")
    rc, out = run(bad_id=True)
    check("exit code is non-zero", rc != 0)
    check("stops at stage 1", "Stopped at stage 1" in out)
    check("reports the ID it actually read", "0x00000000" in out)
    check("explains that this means the PL is not programmed",
          "not programmed" in out)
    check("does not run later stages", "[2/6]" not in out)


def test_constant_zero_is_named():
    print("--- the historic failure: datapath returns a constant zero ---")
    rc, out = run(corrupt="zeros")
    check("exit code is non-zero", rc != 0)
    check("stops at stage 4, not later", "Stopped at stage 4" in out)
    check("says every value is zero", "every returned value is zero" in out)
    check("names it as the old design's failure mode",
          "computes nothing" in out)
    check("does not reach the comparison stage", "[5/6]" not in out)


def test_single_wrong_value_is_caught():
    print("--- one incorrect value out of 301,056 ---")
    rc, out = run(corrupt="one_value")
    check("exit code is non-zero", rc != 0)
    check("stops at stage 5", "Stopped at stage 5" in out)
    check("counts the mismatches", "1 of 301056 values differ" in out)
    check("prints the offending value with its coordinates", "hw " in out)
    check("suggests where to look", "kernel layout" in out)


def main():
    print("bringup.py -- staged script checks\n")
    test_happy_path()
    test_bad_id_stops_at_stage_1()
    test_constant_zero_is_named()
    test_single_wrong_value_is_caught()
    print()
    if FAILS:
        print("FAIL  %d check(s) failed:" % len(FAILS))
        for f in FAILS:
            print("  - %s" % f)
        return 1
    print("PASS  all bring-up script checks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
