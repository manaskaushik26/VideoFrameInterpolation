#!/usr/bin/env python3
"""
perf.py -- measure one conv pass on hardware. Run this only AFTER bringup.py
passes all six stages; it assumes the arithmetic is already proven correct and
measures nothing but time.

What the number means. One pass streams IMG_W*IMG_H*CH_IN windows through the
MAC array, one per cycle, so the PL's own floor at 75 MHz is

    112 * 112 * 6 = 75,264 cycles = 1.004 ms

The floors are computed at the clock the board actually runs, read from
pynq's Clocks.fclk0_mhz after the overlay loads (it can be below 75, e.g. 62.5
MHz on a 1000 MHz IO PLL). If it can't be read, 75 MHz is assumed and the
output says so.

Anything above that is PS overhead: two DMA descriptors, the kernel reload, and
pynq's buffer allocation. The gap is the interesting part -- it tells you
whether a six-layer network will be bound by the MAC array or by the host loop,
which is the decision the layer sequencer rests on.

The kernel transfer is deliberately included, because run_layer() reloads
weights on every pass and that is what a real network pays.

Then, if the trained export is next to it (./export_gpu), one full network run
per iteration with a per-layer breakdown -- 17 passes, the number that sets the
demo's frame rate -- against the PL floor of sum(112*112*max(c_in, 24)) cycles.

Usage:  python3 perf.py [path/to/vfi.bit] [iterations]
"""

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import vfi_driver as D

IMG_W = IMG_H = 112
N_OUT = 24
MAX_CH = 128
CH_IN = 6
SHIFT = 20


def pl_clock():
    """(MHz, description) -- the measured FCLK0, or the design value, labelled."""
    mhz = D.read_fclk0_mhz()
    if mhz is None:
        return D.FCLK_DESIGN_MHZ, "ASSUMED, Clocks.fclk0_mhz unreadable"
    return mhz, "measured, Clocks.fclk0_mhz"


def main(argv):
    bit = argv[1] if len(argv) > 1 else os.path.join(HERE, "vfi.bit")
    iters = int(argv[2]) if len(argv) > 2 else 10

    from pynq import Overlay
    ol = Overlay(bit)
    fclk_mhz, fclk_src = pl_clock()
    vfi = D.Vfi(ol.vfi, ol.dma_fm, ol.dma_wt,
                img_w=IMG_W, img_h=IMG_H, n_out=N_OUT, max_ch=MAX_CH)

    rng = np.random.default_rng(7)
    kern = rng.integers(-4000, 4000, size=(N_OUT, CH_IN, 3, 3)).astype(np.int16)
    fmap = rng.integers(-20000, 20000, size=(IMG_H, IMG_W, CH_IN)).astype(np.int16)

    vfi._run_pass(fmap, kern, SHIFT)          # warm up: first call pays page faults

    t0 = time.perf_counter()
    for _ in range(iters):
        vfi._run_pass(fmap, kern, SHIFT)
    ms = 1000.0 * (time.perf_counter() - t0) / iters

    macs = IMG_W * IMG_H * CH_IN * N_OUT * 9
    floor_ms = 1000.0 * (IMG_W * IMG_H * CH_IN) / (fclk_mhz * 1e6)

    print("PL clock        : %.2f MHz (%s)" % (fclk_mhz, fclk_src))
    print("passes          : %d (+1 warm-up)" % iters)
    print("shape           : %dx%d, CH_IN=%d, N_OUT=%d" % (IMG_W, IMG_H, CH_IN, N_OUT))
    print("per pass        : %.2f ms" % ms)
    print("PL floor @%.1f MHz: %.2f ms  (one window-channel per cycle)" % (fclk_mhz, floor_ms))
    print("overhead        : %.2f ms (%.0f%% of wall time is PS/DMA)"
          % (ms - floor_ms, 100.0 * (ms - floor_ms) / ms))
    print("throughput      : %.2f GMAC/s over %d MACs" % (macs / (ms * 1e6), macs))
    print("frames counted  : %d" % vfi.frames())
    print()
    model = next((d for d in (os.path.join(HERE, "export_gpu"),
                              os.path.join(os.path.dirname(HERE), "train", "export_gpu"))
                  if os.path.exists(os.path.join(d, "vfi_model.json"))), None)
    if model is None:
        print("no ./export_gpu -- skipping the full-network timing")
        return 0

    weights, layers, bias = D.load_model(model)
    x0 = np.fromfile(os.path.join(model, "golden", "input.bin"), dtype="<i2").reshape(IMG_H, IMG_W, 6)
    vfi.run_network(x0, weights, layers, warn=False)                 # warm up
    per_layer = {l.name: 0.0 for l in layers}
    n_net = max(1, iters // 5)
    t0 = time.perf_counter()
    for _ in range(n_net):
        x = x0
        for wgt, layer in zip(weights, layers):
            t = time.perf_counter()
            x, _ = vfi.run_layer(x, wgt, layer)
            per_layer[layer.name] += time.perf_counter() - t
        D.add_flow_bias(x, bias)
    net_ms = 1000.0 * (time.perf_counter() - t0) / n_net

    print()
    print("full network    : %.1f ms per frame pair (%d run(s)) -> %.2f interpolated frames/s"
          % (net_ms, n_net, 1000.0 / net_ms))
    floor_total = 0.0
    print("  layer   passes  ms      PL floor ms")
    for layer in layers:
        passes = -(-layer.ch_out // N_OUT)
        floor = 1000.0 * passes * IMG_W * IMG_H * max(layer.ch_in, N_OUT) / (fclk_mhz * 1e6)
        floor_total += floor
        print("  %-6s  %d       %6.1f  %6.1f" % (layer.name, passes,
                                                  1000.0 * per_layer[layer.name] / n_net, floor))
    print("  PL floor total %.1f ms at %.2f MHz; the rest is PS/DMA/stitching"
          % (floor_total, fclk_mhz))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
