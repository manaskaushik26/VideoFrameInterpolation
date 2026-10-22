"""
check_golden -- does vfi_ref, run on an export's own weights, reproduce that
export's golden per-layer dumps bit-exactly?

    python model/check_golden.py                      # train/export_gpu
    python model/check_golden.py path/to/export

verify_export.py WRITES golden/<layer>.bin with vfi_ref; this READS them and
recomputes. The two only agree if the export's weights, shifts, bias, layer order
and the current vfi_ref are all the ones the dumps were made with -- so it catches
a stale golden folder, a mixed-up export, or a change to vfi_ref's arithmetic.

Also cross-checks host/vfi_driver.load_model against vfi_ref.load_export, since
the board reads the export through the driver and the reference through vfi_ref.

A full 112x112 forward pass takes a few minutes; that is vfi_ref being obvious
rather than fast.
"""

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "host"))
import vfi_ref as R      # noqa: E402
import vfi_driver as D   # noqa: E402


def main(export_dir):
    weights, shifts, bias, meta = R.load_export(export_dir)
    n = meta["img"]
    print("export  %s" % export_dir)
    print("shifts  %s" % shifts)
    print("bias    flow %s (host-side)" % bias["flow"].tolist())

    ok = True
    d_w, d_layers, d_bias = D.load_model(export_dir)
    same = (all(np.array_equal(a, weights[l.name]) for a, l in zip(d_w, d_layers))
            and [l.shift for l in d_layers] == [shifts[name] for name, *_ in R.NETWORK]
            and [l.name for l in d_layers] == [name for name, *_ in R.NETWORK]
            and np.array_equal(d_bias, bias["flow"]))
    print("vfi_driver.load_model == vfi_ref.load_export: %s" % ("yes" if same else "NO"))
    ok &= same

    gold = os.path.join(export_dir, "golden")
    x = R.load_dump(os.path.join(gold, "input.bin"), n, n, 6)
    print("\nlayer   shape            mismatches  saturated   range")
    t0 = time.time()
    for name, c_in, c_out, act in R.NETWORK:
        x, info = R.conv_stage(x, weights[name], shifts[name], act)
        if name in bias:
            x = R.add_bias(x, bias[name])
        ref = R.load_dump(os.path.join(gold, "%s.bin" % name), n, n, c_out)
        bad = int(np.sum(x != ref))
        ok &= bad == 0
        print("%-6s  %-15s  %10d  %8.3f%%   [%d, %d]"
              % (name, (n, n, c_out), bad, 100 * info["sat_frac"], x.min(), x.max()))
    print("(%.0f s)" % (time.time() - t0))

    torch_path = os.path.join(export_dir, "golden_flow_torch.bin")
    if os.path.exists(torch_path):
        t = R.load_dump(torch_path, n, n, 2).astype(np.int64)
        diff = np.abs(x.astype(np.int64) - t)
        print("flow vs quantised torch: max %d LSB, %.3f%% differ  (rounding mode only)"
              % (diff.max(), 100 * (diff > 0).mean()))
    neg = int(np.sum(x < 0))
    print("flow: %d of %d values negative -- these need ACT_MODE linear on the flow layer"
          % (neg, x.size))

    print("PASS  bit-exact on every layer" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else R.DEFAULT_EXPORT))
