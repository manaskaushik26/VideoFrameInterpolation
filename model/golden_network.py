"""
golden_network -- stimulus and expected results for the WHOLE network in xsim,
and the check of what xsim produced.

    python model/golden_network.py               # writes build/net/
    (run sim/tb_network_golden.sv)
    python model/golden_network.py --check-sim   # diff xsim's layers vs golden

sim/tb_network_golden.sv runs all six layers on one vfi_top at the bitstream's
exact configuration (112x112, MAX_CH=128, N_OUT=24, ACT_MODE=1), each layer as
ceil(c_out/24) passes driven over AXI-Lite and AXI-Stream the way
host/vfi_driver.run_layer does it. It CHAINS: each layer's input is the RTL's own
stitched output of the previous layer, not a golden file, so an error anywhere
propagates to the flow exactly as it would on the board.

That covers the multi-pass layers specifically -- conv2 (64 out, 3 passes), conv3
(128 out, 6 passes), conv4 (64 out, 3 passes) -- and the flow layer with ReLU off
via CFG[14].

Files written to build/net/, one 16-bit hex value per line, AXI-Stream beat order:

    cfg.txt            one line per layer: name c_in c_out shift linear n_pass
    sat.txt            expected sat_sticky per pass, in run order
    input.hex          112*112*6, (row, col, channel)
    w_<layer>.hex      n_pass*24 kernels (out, in, kh, kw), the last pass's
                       surplus kernels zero -- exactly what run_layer DMAs
    exp_<layer>.hex    the layer's full output as the PL returns it. For the flow
                       layer that is BEFORE the host-side bias.

The expected values are vfi_ref's, and are first checked against the export's
own golden/<layer>.bin -- the flow after adding the bias -- so the chain is
anchored to the dumps the trained model was verified with, not to this file.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import vfi_ref as R  # noqa: E402

N_OUT = R.N_OUT_MAX


def write_hex(path, values):
    v = np.asarray(values, dtype=np.int16).reshape(-1).view(np.uint16)
    with open(path, "w", newline="\n") as f:
        f.write("\n".join("%04x" % x for x in v))
        f.write("\n")


def read_hex(path):
    with open(path) as f:
        return np.array([int(t, 16) for t in f.read().split()], dtype=np.uint16).view(np.int16)


def generate(export, out_dir):
    weights, shifts, bias, meta = R.load_export(export)
    n = meta["img"]
    gold = os.path.join(export, "golden")
    os.makedirs(out_dir, exist_ok=True)

    x = R.load_dump(os.path.join(gold, "input.bin"), n, n, 6)
    write_hex(os.path.join(out_dir, "input.hex"), x)

    cfg_lines, sat_lines = [], []
    for name, c_in, c_out, act in R.NETWORK:
        linear = int(act == R.ACT_LINEAR)
        kern = weights[name]
        passes = R.output_passes(c_out, N_OUT)
        padded = np.zeros((len(passes) * N_OUT, c_in, 3, 3), dtype=np.int16)
        padded[:c_out] = kern
        write_hex(os.path.join(out_dir, "w_%s.hex" % name), padded)

        # Per pass, exactly as the hardware sees it, so sat_sticky is per pass too.
        out = np.zeros((n, n, c_out), dtype=np.int16)
        for p, (lo, hi) in enumerate(passes):
            part, info = R.conv_stage(x, padded[p * N_OUT:(p + 1) * N_OUT], shifts[name], act)
            if np.any(part[:, :, hi - lo:]):
                sys.exit("FAIL: zero kernels produced nonzero output in %s pass %d" % (name, p))
            out[:, :, lo:hi] = part[:, :, :hi - lo]
            sat_lines.append("%d" % int(info["sat_sticky"]))

        whole, _ = R.conv_stage(x, kern, shifts[name], act)
        assert np.array_equal(whole, out), "pass stitching disagrees with the whole layer"

        ref = R.load_dump(os.path.join(gold, "%s.bin" % name), n, n, c_out)
        check = R.add_bias(out, bias[name]) if name in bias else out
        bad = int(np.sum(check != ref))
        print("%-6s %3d -> %-3d  shift %2d  %-6s  %d pass(es)  vs golden/%s.bin: %d mismatches%s"
              % (name, c_in, c_out, shifts[name], "linear" if linear else "relu",
                 len(passes), name, bad, "  (after +bias %s)" % bias[name].tolist()
                 if name in bias else ""))
        if bad:
            sys.exit("FAIL: vfi_ref does not reproduce the export's golden %s" % name)

        write_hex(os.path.join(out_dir, "exp_%s.hex" % name), out)
        cfg_lines.append("%s %d %d %d %d %d" % (name, c_in, c_out, shifts[name], linear, len(passes)))
        x = out

    with open(os.path.join(out_dir, "cfg.txt"), "w", newline="\n") as f:
        f.write("\n".join(cfg_lines) + "\n")
    with open(os.path.join(out_dir, "sat.txt"), "w", newline="\n") as f:
        f.write("\n".join(sat_lines) + "\n")
    print("flow before bias: %d of %d values negative" % (int(np.sum(x < 0)), x.size))
    print("wrote %s  (%d passes in total)" % (out_dir, len(sat_lines)))


def check_sim(export, out_dir):
    """Diff every layer xsim dumped against golden, and the flow end to end."""
    weights, shifts, bias, meta = R.load_export(export)
    n = meta["img"]
    gold = os.path.join(export, "golden")
    ok = True
    print("RTL (xsim, chained) vs golden, %dx%d" % (n, n))
    for name, c_in, c_out, act in R.NETWORK:
        path = os.path.join(out_dir, "sim_%s.hex" % name)
        if not os.path.exists(path):
            print("  %-6s  MISSING %s" % (name, path))
            ok = False
            continue
        v = read_hex(path)
        if v.size != n * n * c_out:
            print("  %-6s  %d values, expected %d" % (name, v.size, n * n * c_out))
            ok = False
            continue
        sim = v.reshape(n, n, c_out)
        if name in bias:
            sim = R.add_bias(sim, bias[name])        # the host-side step
        ref = R.load_dump(os.path.join(gold, "%s.bin" % name), n, n, c_out)
        diff = np.argwhere(sim != ref)
        print("  %-6s  %9d values  %d mismatches%s"
              % (name, ref.size, len(diff), "   (+bias, vs golden/flow.bin)" if name in bias else ""))
        for y, xx, c in diff[:5]:
            print("      (%d,%d,ch%d): sim %d  golden %d" % (y, xx, c, sim[y, xx, c], ref[y, xx, c]))
        ok &= len(diff) == 0
        if name == "flow":
            R.dump(sim, os.path.join(out_dir, "flow_sim.bin"))
            print("  flow: %d negative values reached the host -- ReLU was off for it"
                  % int(np.sum(v < 0)))
    print("PASS: whole network bit-exact, final flow == golden/flow.bin" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--export", default=R.DEFAULT_EXPORT)
    ap.add_argument("--out", default=os.path.join(REPO, "build", "net"))
    ap.add_argument("--check-sim", action="store_true")
    a = ap.parse_args()
    sys.exit(check_sim(a.export, a.out) if a.check_sim else generate(a.export, a.out))
