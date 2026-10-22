"""
Tests for conv1_board.py against the mock PL, at the real 112x112x24 geometry.

Same principle as test_bringup.py: the happy path proves the script runs end to
end with no board, and the injected faults prove a failure is reported as one.

Needs the golden files: run `python model/golden_conv1.py` first.
Run: python host/test_conv1_board.py
"""

import io
import json
import os
import shutil
import sys
import tempfile
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import test_bringup as TB    # noqa: E402  (fake pynq + mock PL)
import conv1_board            # noqa: E402

EXPORT = os.path.join(REPO, "train", "export_gpu")
GOLDEN = os.path.join(REPO, "build", "conv1", "conv1_golden.bin")
FAILS = []


def check(name, cond):
    print("  %-4s %s" % ("PASS" if cond else "FAIL", name))
    if not cond:
        FAILS.append(name)


def data_dir():
    d = tempfile.mkdtemp()
    shutil.copy(os.path.join(EXPORT, "conv1_weight_q15.bin"), d)
    shutil.copy(os.path.join(EXPORT, "vfi_model.json"), d)
    shutil.copy(os.path.join(EXPORT, "golden", "input.bin"), d)
    shutil.copy(GOLDEN, d)
    return d


def run(**kw):
    bit = TB.build_env(**kw)
    d = data_dir()
    buf = io.StringIO()
    with redirect_stdout(buf):
        try:
            rc = conv1_board.main(["--bit", bit, "--data", d])
        except SystemExit as e:
            rc = e.code
    return rc, buf.getvalue(), d


def main():
    if not os.path.exists(GOLDEN):
        sys.exit("run `python model/golden_conv1.py` first")

    print("--- trained conv1 through the mock PL, two passes ---")
    rc, out, d = run()
    check("exit code 0", rc == 0)
    check("two passes", "pass 1: channels 24..31" in out)
    check("reports bit-exact on both inputs", "PASS  bit-exact on both inputs" in out)
    check("writes both result dumps", all(os.path.exists(os.path.join(d, f)) for f in ("conv1_board.bin", "conv1_board_dense.bin")))

    print("--- datapath returns constant zero ---")
    rc, out, _ = run(corrupt="zeros")
    check("exit code non-zero", rc == 1)
    check("names the constant-zero failure", "computed nothing" in out)

    print("--- one wrong value per pass ---")
    rc, out, _ = run(corrupt="one_value")
    check("exit code non-zero", rc == 1)
    check("counts exactly 2 mismatches", "FAIL  2 mismatches" in out)

    print("--- golden file from a different export ---")
    bit = TB.build_env()
    d = data_dir()
    cfg = json.load(open(os.path.join(d, "vfi_model.json")))
    cfg["shifts"]["conv1"] += 1
    json.dump(cfg, open(os.path.join(d, "vfi_model.json"), "w"))
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = conv1_board.main(["--bit", bit, "--data", d])
    check("refuses to compare against a mismatched golden", rc == 2)

    print("")
    print("FAIL  %d checks" % len(FAILS) if FAILS else "PASS  all checks")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
