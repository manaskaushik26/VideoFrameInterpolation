#!/usr/bin/env python3
"""
conv1_board.py -- run conv1 (6 -> 32) on the PL with the TRAINED weights and
require a bit-exact match against the golden model.

Run it on the board after bringup.py passes. bringup.py proves the arithmetic
with random kernels; this runs the layer the network actually uses, with the
same input frame pair, weights and shift as the xsim run (sim/tb_conv1_golden.sv)
that already matched model/golden_conv1.py exactly. So the three agree or the
board is the odd one out:

    Python golden  ==  xsim (RTL)  ==  PL on the board

conv1 has 32 output channels and the MAC array is 24 wide, so this is two
hardware passes (kernels 0..23, then 24..31 + 16 zero kernels), stitched by
vfi_driver.run_layer -- the first multi-pass layer to run on silicon.

Files: vfi.bit vfi.hwh vfi_driver.py vfi_ref.py next to this script, and in
--data (default ./export_gpu, the trained export as the kit ships it):

    vfi_model.json  conv1_weight_q15.bin  golden/input.bin  golden/conv1.bin

A flat folder holding input.bin and conv1_golden.bin instead also works.

Usage:  python3 conv1_board.py [--bit vfi.bit] [--data DIR]
Exit code is 0 only on an exact match.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "model"))

import vfi_driver as D  # noqa: E402
import vfi_ref as R     # noqa: E402

IMG_W = IMG_H = 112
N_OUT = 24
MAX_CH = 128
C_IN, C_OUT = 6, 32


def main(argv=None):
    ap = argparse.ArgumentParser(description="conv1 on the PL vs the golden model")
    ap.add_argument("--bit", default=os.path.join(HERE, "vfi.bit"))
    kit_export = os.path.join(HERE, "export_gpu")
    ap.add_argument("--data", default=kit_export if os.path.isdir(kit_export) else HERE)
    a = ap.parse_args(argv)

    def data(name, alt=None):
        for p in (os.path.join(a.data, name), alt and os.path.join(a.data, alt)):
            if p and os.path.exists(p):
                return p
        print("FAIL  missing %s in %s" % (name, a.data))
        sys.exit(2)

    shift = json.load(open(data("vfi_model.json")))["shifts"]["conv1"]
    img = R.load_dump(data("input.bin", os.path.join("golden", "input.bin")), IMG_H, IMG_W, C_IN)
    kern = R.load_weights(data("conv1_weight_q15.bin"), C_OUT, C_IN)
    gold = R.load_dump(data("conv1_golden.bin", os.path.join("golden", "conv1.bin")),
                       IMG_H, IMG_W, C_OUT)

    # Recompute the golden on the board too: catches a golden file that does not
    # belong to these weights, independently of the checksum.
    ref, info = R.conv_stage(img, kern, shift, R.ACT_RELU)
    if not np.array_equal(ref, gold):
        print("FAIL  conv1_golden.bin does not match vfi_ref on these weights -- "
              "the data files are from different exports")
        return 2
    print("golden: conv1, shift %d, %d values, %d nonzero, %d saturated"
          % (shift, gold.size, int(np.count_nonzero(gold)), info["sat_count"]))

    from pynq import Overlay
    ol = Overlay(a.bit)
    vfi = D.Vfi(ol.vfi, ol.dma_fm, ol.dma_wt, img_w=IMG_W, img_h=IMG_H,
                n_out=N_OUT, max_ch=MAX_CH)          # raises unless ID = "VFI1"

    rc = run_and_compare(vfi, "held-out MNIST pair", img, kern, shift, gold,
                         os.path.join(a.data, "conv1_board.bin"))

    # ReLU hides every negative pre-activation, and the MNIST pair is mostly black
    # background, so only ~12k of its 401k outputs are nonzero. A dense input with
    # the same weights lights up ~206k. Its golden is computed here, on the board.
    dense = np.random.default_rng(20261013).integers(
        0, 32768, size=(IMG_H, IMG_W, C_IN)).astype(np.int16)
    dense_gold, _ = R.conv_stage(dense, kern, shift, R.ACT_RELU)
    rc |= run_and_compare(vfi, "dense random input", dense, kern, shift, dense_gold,
                          os.path.join(a.data, "conv1_board_dense.bin"))

    print("")
    if rc == 0:
        print("PASS  bit-exact on both inputs: the board computes conv1 exactly as "
              "the golden model and the RTL simulation do.")
    return rc


def run_and_compare(vfi, label, img, kern, shift, gold, out_path):
    print("")
    print("--- %s: %d nonzero golden values of %d ---"
          % (label, int(np.count_nonzero(gold)), gold.size))
    frames0 = vfi.frames()
    t0 = time.perf_counter()
    out, statuses = vfi.run_layer(img, kern, D.VfiLayer(C_IN, C_OUT, shift, "conv1"))
    dt = time.perf_counter() - t0

    for s in statuses:
        print("  pass %d: channels %2d..%2d  weights_loaded=%d  sat_sticky=%d"
              % (s["pass"], s["channels"][0], s["channels"][1] - 1,
                 s["weights_loaded"], s["saturated"]))
    print("  %d passes in %.1f ms (PL + DMA + PS stitching); FRAMES %d -> %d"
          % (len(statuses), dt * 1e3, frames0, vfi.frames()))
    R.dump(out, out_path)

    bad = np.argwhere(out != gold)
    if len(bad) == 0:
        print("  bit-exact: 0 mismatches")
        return 0

    for y, x, c in bad[:10]:
        print("  (row %d, col %d, ch %d): board %d  golden %d"
              % (y, x, c, out[y, x, c], gold[y, x, c]))
    chans = sorted(set(int(c) for c in bad[:, 2]))
    edge = np.mean((bad[:, 0] == 0) | (bad[:, 0] == IMG_H - 1) |
                   (bad[:, 1] == 0) | (bad[:, 1] == IMG_W - 1))
    print("FAIL  %d mismatches, in channels %s" % (len(bad), chans))
    if not np.any(out):
        print("      every returned value is zero: the datapath ran and computed nothing "
              "(the superseded design's failure). Check weights_loaded and CFG.")
    elif all(c >= N_OUT for c in chans) or all(c < N_OUT for c in chans):
        print("      confined to one pass: suspect pass stitching / zero-padding in "
              "vfi_driver.run_layer before the PL.")
    elif edge > 0.9:
        print("      almost all on the image border: line-buffer `same` padding.")
    else:
        print("      spread across the frame: kernel layout or cfg_shift. bringup.py "
              "stage 5 should have caught this -- re-run it.")
    return 1

if __name__ == "__main__":
    sys.exit(main())
