#!/usr/bin/env python3
"""
network_board.py -- the whole trained network on the PL, layer by layer, each
layer required to match the golden dump bit-exactly; then the flow end to end.

Run on the board after bringup.py and conv1_board.py pass.

    conv1   6 -> 32   2 passes       conv4 128 -> 64   3 passes
    conv2  32 -> 64   3 passes       conv5  64 -> 32   2 passes
    conv3  64 -> 128  6 passes       flow   32 -> 2    1 pass, ReLU off (CFG[14])

It CHAINS, as run_network does and as the demo will: each layer's input is the
board's own output of the layer before, so an error anywhere reaches the flow.
After every layer the result is compared against the export's golden/<layer>.bin,
so a failure names the layer it started in. The flow is compared after the
host-side bias, against golden/flow.bin -- the same file the chained xsim run
(sim/tb_network_golden.sv) matched.

Files: vfi.bit vfi.hwh vfi_driver.py next to this script, and the trained
export in --model (default ./export_gpu) with its golden/ folder.

Usage:  python3 network_board.py [--bit vfi.bit] [--model DIR]
Exit code is 0 only if every layer is bit-exact.
"""

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import vfi_driver as D  # noqa: E402

N_OUT, MAX_CH = 24, 128


def load_dump(path, h, w, c):
    v = np.fromfile(path, dtype="<i2")
    if v.size != h * w * c:
        raise SystemExit("FAIL  %s: %d values, expected %d" % (path, v.size, h * w * c))
    return v.reshape(h, w, c)


def main(argv=None):
    ap = argparse.ArgumentParser(description="whole network on the PL vs golden")
    ap.add_argument("--bit", default=os.path.join(HERE, "vfi.bit"))
    ap.add_argument("--model", default=os.path.join(HERE, "export_gpu"))
    ap.add_argument("--out-dir", default=HERE, help="where flow_board.bin is written")
    a = ap.parse_args(argv)

    weights, layers, bias = D.load_model(a.model)
    gold = os.path.join(a.model, "golden")
    import json
    n = json.load(open(os.path.join(a.model, "vfi_model.json")))["img"]
    x = load_dump(os.path.join(gold, "input.bin"), n, n, 6)

    from pynq import Overlay
    ol = Overlay(a.bit)
    vfi = D.Vfi(ol.vfi, ol.dma_fm, ol.dma_wt, img_w=n, img_h=n, n_out=N_OUT, max_ch=MAX_CH)

    print("whole network on the PL, %dx%d, chained, vs %s" % (n, n, gold))
    print("layer   in->out  shift act     passes  ms       mismatches  saturated")
    total, rc = 0.0, 0
    for wgt, layer in zip(weights, layers):
        t0 = time.perf_counter()
        try:
            x, st = vfi.run_layer(x, wgt, layer)
        except D.VfiError as e:
            print("FAIL  %s: %s" % (layer.name, e))
            return 1
        dt = time.perf_counter() - t0
        total += dt
        check = D.add_flow_bias(x, bias) if layer.name == "flow" else x
        ref = load_dump(os.path.join(gold, "%s.bin" % layer.name), n, n, layer.ch_out)
        bad = np.argwhere(check != ref)
        print("%-6s  %3d->%-3d  %3d   %-6s  %d       %7.1f  %10d  %s"
              % (layer.name, layer.ch_in, layer.ch_out, layer.shift,
                 "linear" if layer.linear else "relu", len(st), dt * 1e3, len(bad),
                 any(s["saturated"] for s in st)))
        if len(bad):
            for y, xx, c in bad[:5]:
                print("    (row %d, col %d, ch %d): board %d  golden %d"
                      % (y, xx, c, check[y, xx, c], ref[y, xx, c]))
            chans = sorted(set(int(c) for c in bad[:, 2]))
            print("FAIL  %s: %d mismatches in channels %s" % (layer.name, len(bad), chans[:20]))
            if layer.name == "flow" and np.all(x >= 0):
                print("      no negative value came back: the flow layer ran with ReLU on. "
                      "Wrong (old) bitstream? bringup.py stage 2 checks CFG[14].")
            rc = 1
            break

    if rc == 0:
        flow = check
        np.asarray(flow, dtype="<i2").tofile(os.path.join(a.out_dir, "flow_board.bin"))
        print("")
        print("network: %.0f ms per frame pair on the PL path (%.2f fps), "
              "%d negative flow values returned" % (total * 1e3, 1.0 / total, int(np.sum(x < 0))))
        print("PASS  every layer bit-exact; the board's flow == golden/flow.bin")
    return rc


if __name__ == "__main__":
    sys.exit(main())
