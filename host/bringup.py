#!/usr/bin/env python3
"""
bringup.py -- staged first-run test for the VFI overlay on a PYNQ-Z2.

Run this on the board, before anything else. Nothing in this project has ever
executed on real silicon; everything known about it comes from simulation and
from Vivado reports, and this project's defining lesson is that Vivado will
report success on a design that computes nothing.

The stages go from smallest to largest, each one adding exactly one thing, so a
failure names its own cause instead of leaving you to bisect a whole system:

    1  overlay loads, ID register reads 0x56464931       AXI-Lite read
    2  CFG writes and reads back, incl. CFG[14] linear    AXI-Lite write
    3  kernels DMA in, weights_loaded asserts             small transfer (2.5 KB)
    4  feature map DMA round trip returns the right size  large transfer (600 KB)
    5  the returned data matches model/vfi_ref.py exactly the arithmetic
    6  FRAMES counter advanced, saturation reported       housekeeping

Stage 3 vs stage 4 is the important split. Stage 3 moves 2,592 bytes and stage 4
moves 602,112. Until 2026-08-31 both DMAs ran with c_sg_length_width at its
default of 14, capping a single transfer at 16,383 bytes -- so stage 3 would have
passed and stage 4 failed. If you see exactly that, you are running an old
bitstream: check SG_LENGTH_WIDTH in the .hwh, which should read 26.

Usage
-----
    python3 bringup.py [path/to/vfi.bit]

Copy these next to it: vfi.bit, the matching .hwh (named to match the .bit, e.g.
vfi.hwh), vfi_driver.py, and model/vfi_ref.py. The reference model is numpy-only,
so it runs on the board unchanged.

Exit code is 0 only if every stage passed.
"""

import os
import sys
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "model"))

import vfi_driver as D

try:
    import vfi_ref as R
except ImportError:
    R = None

# Must match the bitstream. IMG_W/IMG_H and N_OUT are synthesis-time parameters,
# not runtime configuration -- only the channel count and shift are runtime.
IMG_W = IMG_H = 112
N_OUT = 24
MAX_CH = 128

# Stage 4/5 stimulus. CH_IN=6 is conv1's shape (two RGB frames stacked).
CH_IN = 6
SHIFT = 20      # not a trained value -- see the note in vfi_driver's docstring
SEED = 20260831

results = []


def stage(n, title):
    def deco(fn):
        fn._stage = (n, title)
        return fn
    return deco


def _fail(msg, *hints):
    """Raise with a diagnosis attached, so the runner can print what to check."""
    e = AssertionError(msg)
    e.hints = list(hints)
    raise e


# ------------------------------------------------------------------ stages

@stage(1, "Overlay loads and the ID register reads back")
def stage1(ctx, bit_path):
    from pynq import Overlay

    if not os.path.exists(bit_path):
        _fail("bitstream not found: %s" % bit_path,
              "Copy vfi.bit from %LOCALAPPDATA%/vfi_build/ on the build machine.",
              "The matching .hwh must sit beside it with the same basename.")

    ol = Overlay(bit_path)
    ctx["overlay"] = ol

    names = sorted(ol.ip_dict.keys())
    print("      IP blocks in the overlay: %s" % ", ".join(names))

    # Instance names come from the BD cell names in syn/build.tcl and were
    # verified against the generated vfi_bd.hwh: vfi, dma_fm, dma_wt. The
    # fallback exists only to give a better message than AttributeError.
    ip = getattr(ol, "vfi", None) or getattr(ol, "vfi_0", None)
    if ip is None:
        _fail("no 'vfi' IP in this overlay (found: %s)" % ", ".join(names),
              "The .hwh does not match the .bit, or this is the wrong bitstream.",
              "Expected a cell named 'vfi' -- see syn/build.tcl.")

    got = ip.read(D.REG_ID)
    if got != D.ID_MAGIC:
        _fail("ID register reads 0x%08X, expected 0x%08X" % (got, D.ID_MAGIC),
              "0x00000000 usually means the PL is not programmed, or the "
              "address map in the .hwh does not match this .bit.",
              "A garbage value means you are reading a different IP block.",
              "This check exists because the old host code silently addressed "
              "the wrong instance and produced plausible nonsense.")

    ctx["ip"] = ip
    return "ID = 0x%08X (\"VFI1\")" % got


@stage(2, "CFG register writes and reads back")
def stage2(ctx, _):
    ip = ctx["ip"]
    for channels, shift, linear in [(6, 20, False), (128, 15, True), (32, 0, False)]:
        ip.write(D.REG_CFG, ((shift & 0x3F) << 8) | (channels & 0xFF)
                 | (D.CFG_LINEAR if linear else 0))
        back = ip.read(D.REG_CFG)
        if (back & 0xFF) != channels or ((back >> 8) & 0x3F) != shift:
            _fail("CFG readback 0x%04X != channels=%d shift=%d"
                  % (back, channels, shift),
                  "Reads work (stage 1 passed) but writes do not land.",
                  "Check the AXI-Lite write path in rtl/axis/axil_regs.sv.")
        if bool(back & D.CFG_LINEAR) != linear:
            _fail("CFG[14] (linear) reads back %d, wrote %d"
                  % (bool(back & D.CFG_LINEAR), linear),
                  "This is a bitstream from before the runtime activation bit "
                  "(2026-10-23). It applies ReLU to every layer, so the flow "
                  "layer would lose every negative value.",
                  "Copy the kit's vfi.bit AND vfi.hwh again; check SHA256SUMS.txt.")
    return "3 configurations written and read back, linear bit included"


@stage(3, "Kernel DMA -- small transfer, weights_loaded asserts")
def stage3(ctx, _):
    ol = ctx["overlay"]
    vfi = D.Vfi(ctx["ip"], ol.dma_fm, ol.dma_wt,
                img_w=IMG_W, img_h=IMG_H, n_out=N_OUT, max_ch=MAX_CH,
                check_id=False)          # already checked, with a better message
    ctx["vfi"] = vfi

    rng = np.random.default_rng(SEED)
    kern = rng.integers(-4000, 4000, size=(N_OUT, CH_IN, 3, 3)).astype(np.int16)
    ctx["kern"] = kern

    vfi._configure(CH_IN, SHIFT)
    vfi.clear_saturation()
    vfi._send(vfi.dma_wt.sendchannel, kern)

    st = vfi.status()
    if not st["weights_loaded"]:
        _fail("weights_loaded did not assert after %d bytes" % kern.nbytes,
              "Expected %d beats (N_OUT * CH_IN * 9)." % (N_OUT * CH_IN * 9),
              "If this hangs rather than failing, the weight DMA never got "
              "TREADY -- check the s_wt stream and the PL reset.")
    return "%d kernels (%d bytes) loaded" % (N_OUT * CH_IN, kern.nbytes)


@stage(4, "Feature map DMA -- large transfer, round trip")
def stage4(ctx, _):
    vfi = ctx["vfi"]
    rng = np.random.default_rng(SEED + 1)
    fmap = rng.integers(-20000, 20000,
                        size=(IMG_H, IMG_W, CH_IN)).astype(np.int16)
    ctx["fmap"] = fmap

    in_bytes = fmap.nbytes
    out_bytes = IMG_H * IMG_W * N_OUT * 2
    print("      sending %d bytes, expecting %d back" % (in_bytes, out_bytes))

    out = vfi._run_pass(fmap, ctx["kern"], SHIFT)
    ctx["out"] = out

    if out.shape != (IMG_H, IMG_W, N_OUT):
        _fail("returned %r, expected %r" % (out.shape, (IMG_H, IMG_W, N_OUT)))

    # The old design's whole failure was a datapath that computed a constant
    # zero while synthesising cleanly. Call that out by name if it recurs.
    if not out.any():
        _fail("every returned value is zero",
              "This is the old design's exact failure mode: a datapath that "
              "builds clean and computes nothing.",
              "Check that weights_loaded is still set and that cfg_channels "
              "matches the map you sent.")
    return "round trip OK, %d bytes in / %d out, %.1f%% non-zero" % (
        in_bytes, out_bytes, 100.0 * np.count_nonzero(out) / out.size)


@stage(5, "Output matches the bit-exact reference model")
def stage5(ctx, _):
    if R is None:
        _fail("model/vfi_ref.py not importable",
              "Copy it next to this script. It is numpy-only and runs on the "
              "board unchanged.")

    exp, info = R.conv_stage(ctx["fmap"], ctx["kern"], SHIFT, R.ACT_RELU)
    got = ctx["out"]

    if got.shape != exp.shape:
        _fail("shape %r vs reference %r" % (got.shape, exp.shape))

    diff = (got != exp)
    n_bad = int(diff.sum())
    if n_bad:
        idx = np.argwhere(diff)[:5]
        lines = ["  (%d,%d,ch%d): hw %d, ref %d"
                 % (y, x, c, got[y, x, c], exp[y, x, c]) for y, x, c in idx]
        _fail("%d of %d values differ (%.3f%%)"
              % (n_bad, got.size, 100.0 * n_bad / got.size),
              "First mismatches:\n" + "\n".join(lines),
              "All values wrong: check the kernel layout (out,in,kh,kw, no "
              "transpose) and that cfg_shift took effect.",
              "Only edges wrong: padding. Only some channels: the pass "
              "stitching in run_layer.")

    return "all %d values match exactly (reference saturated %.2f%%)" % (
        got.size, 100.0 * info["sat_frac"])


@stage(6, "Frame counter and saturation status")
def stage6(ctx, _):
    vfi = ctx["vfi"]
    frames = vfi.frames()
    st = vfi.status()
    if frames < 1:
        _fail("FRAMES reads %d after a completed pass" % frames,
              "Data came back, so the datapath ran -- this is the counter or "
              "its register, not the pipeline.")
    note = ""
    if st["saturated"]:
        note = ("  NOTE: output saturated at shift=%d. Expected with random "
                "stimulus; it is the signal you tune a real shift with." % SHIFT)
    return "FRAMES = %d, saturated = %s%s" % (frames, st["saturated"],
                                              ("\n" + note) if note else "")


STAGES = [stage1, stage2, stage3, stage4, stage5, stage6]


# ------------------------------------------------------------------ runner

def main(argv):
    bit = argv[1] if len(argv) > 1 else os.path.join(HERE, "vfi.bit")

    print("VFI bring-up")
    print("  bitstream : %s" % bit)
    print("  geometry  : %dx%d, CH_IN=%d, N_OUT=%d, shift=%d"
          % (IMG_W, IMG_H, CH_IN, N_OUT, SHIFT))
    print("  reference : %s" % ("model/vfi_ref.py" if R else "NOT FOUND"))
    print()

    ctx = {}
    for fn in STAGES:
        n, title = fn._stage
        print("[%d/%d] %s" % (n, len(STAGES), title))
        try:
            detail = fn(ctx, bit)
        except Exception as e:                       # noqa: BLE001
            print("      FAIL  %s" % e)
            for h in getattr(e, "hints", []):
                print("            - %s" % h)
            if not isinstance(e, AssertionError):
                print()
                traceback.print_exc()
            print("\nStopped at stage %d. Later stages did not run, because "
                  "they depend on this one." % n)
            return 1
        print("      PASS  %s" % detail)
        results.append((n, title))
    print()
    print("PASS  all %d stages. The conv engine works on hardware, and its "
          "output is bit-exact against the reference model." % len(STAGES))
    print("Next: cache coherency under load, then a full six-layer network "
          "once the model is trained.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
