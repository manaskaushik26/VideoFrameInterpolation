"""
Tests for network_board.py against the mock PL at the real 112x112x24 geometry,
with the real trained export and its golden dumps.

The happy path proves the script runs all six layers chained and matches every
golden; the injected faults prove a failure is reported, and where.

Run: python host/test_network_board.py
"""

import io
import os
import sys
import tempfile
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import test_bringup as TB    # noqa: E402  (fake pynq + mock PL)
import network_board         # noqa: E402

EXPORT = os.path.join(REPO, "train", "export_gpu")
FAILS = []


def check(name, cond):
    print("  %-4s %s" % ("PASS" if cond else "FAIL", name))
    if not cond:
        FAILS.append(name)


def run(**kw):
    bit = TB.build_env(**kw)
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = network_board.main(["--bit", bit, "--model", EXPORT, "--out-dir", tempfile.mkdtemp()])
    return rc, buf.getvalue()


def main():
    print("--- trained network through the mock PL, six layers chained ---")
    rc, out = run()
    check("exit code 0", rc == 0)
    check("all six layers ran with 0 mismatches",
          all(("%s " % n) in out for n in ("conv1", "conv2", "conv3", "conv4", "conv5", "flow"))
          and out.count("         0  ") >= 6)
    check("flow ran linear", "linear" in out)
    check("ends with PASS", "PASS  every layer bit-exact" in out)

    print("--- one wrong value per pass ---")
    rc, out = run(corrupt="one_value")
    check("exit code non-zero", rc == 1)
    check("stops at conv1, the first layer", "FAIL  conv1" in out and "conv2 " not in out)

    print("--- a bitstream from before CFG[14] ---")
    rc, out = run(no_linear=True)
    check("exit code non-zero", rc == 1)
    check("conv1..conv5 still pass, flow is refused", "FAIL  flow" in out and "predates" in out)

    print("")
    print("FAIL  %d checks" % len(FAILS) if FAILS else "PASS  all checks")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
