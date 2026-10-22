"""
make_kit -- assemble the board kit: everything the PYNQ-Z2 needs, in one folder.

    python host/make_kit.py --build %LOCALAPPDATA%\\vfi_build_20261023 \\
                            --kit   %USERPROFILE%\\OneDrive\\Desktop\\vfi_board_kit

Re-run after EVERY bitstream rebuild, or the board silently runs an old .bit.

What it refuses to package, because each one has already happened once:

  - a build whose post-route timing failed or was never reported: SUMMARY WNS
    and WHS must both be present and >= 0 (syn/build.tcl writes a .bit even
    when timing fails, so the .bit existing proves nothing). Judged on the
    numbers rather than on BUILD OK, because a later step (the XSA, which PYNQ
    does not use) can fail after a good bitstream -- it did on 2026-10-23;
  - a .hwh whose DMAs are not SG_LENGTH_WIDTH 26 (the 16,383-byte transfer cap);
  - a .hwh without the vfi / dma_fm / dma_wt cells the driver addresses.

The .hwh is copied as vfi.hwh: PYNQ's Overlay("vfi.bit") finds the handoff file by
the .bit's basename, and the build names it vfi_bd.hwh.

The old kit folder, if any, is moved aside to <kit>_old_<date>_<time> -- never
deleted -- and SHA256SUMS.txt covers every file so `sha256sum -c` on the board
proves the copy.
"""

import argparse
import datetime
import hashlib
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

SCRIPTS = ["host/vfi_driver.py", "host/bringup.py", "host/conv1_board.py",
           "host/network_board.py", "host/perf.py", "host/demo.py", "model/vfi_ref.py"]
EXPORT_FILES = ["vfi_model.json", "flow_bias_q15.bin", "LAYOUT.md"] + \
    ["%s_weight_q15.bin" % n for n in ("conv1", "conv2", "conv3", "conv4", "conv5", "flow")] + \
    ["golden/%s.bin" % n for n in ("input", "conv1", "conv2", "conv3", "conv4", "conv5", "flow")]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--build", required=True, help="syn/build.tcl output directory")
    ap.add_argument("--log", default=os.path.join(REPO, "build", "vivado_log",
                                                  "build_20261023_stdout.txt"),
                    help="build.tcl stdout, with its SUMMARY WNS / WHS lines")
    ap.add_argument("--kit", default=os.path.join(os.path.expanduser("~"), "OneDrive",
                                                  "Desktop", "vfi_board_kit"))
    ap.add_argument("--export", default=os.path.join(REPO, "train", "export_gpu"))
    ap.add_argument("--video", default=r"C:\Users\manas\Downloads\snnvfi\videos\sample.mp4")
    a = ap.parse_args(argv)

    bit = os.path.join(a.build, "vfi.bit")
    hwh = os.path.join(a.build, "vfi.gen", "sources_1", "bd", "vfi_bd", "hw_handoff", "vfi_bd.hwh")
    for p in (bit, hwh, a.log):
        if not os.path.exists(p):
            sys.exit("FAIL  missing %s" % p)

    log = open(a.log, encoding="utf-8", errors="replace").read()
    wns = re.search(r"^SUMMARY WNS (\S+)", log, re.M)
    whs = re.search(r"^SUMMARY WHS (\S+)", log, re.M)
    if not (wns and whs) or re.search(r"^SUMMARY TIMING FAILED", log, re.M):
        sys.exit("FAIL  %s: no post-route timing summary, or timing failed" % a.log)
    if float(wns.group(1)) < 0 or float(whs.group(1)) < 0:
        sys.exit("FAIL  post-route WNS %s / WHS %s -- negative slack" % (wns.group(1), whs.group(1)))
    if os.path.getmtime(bit) < os.path.getmtime(hwh):
        sys.exit("FAIL  vfi.bit is older than the .hwh -- not from this build")

    h = open(hwh, encoding="utf-8", errors="replace").read()
    sg = re.findall(r'NAME="c_sg_length_width" VALUE="(\d+)"', h)
    if sg != ["26", "26"]:
        sys.exit("FAIL  c_sg_length_width in the .hwh is %r, expected 26 on both DMAs" % sg)
    cells = set(re.findall(r'INSTANCE="([a-z_0-9]+)"', h))
    if not {"vfi", "dma_fm", "dma_wt"} <= cells:
        sys.exit("FAIL  .hwh cells %s lack vfi / dma_fm / dma_wt" % sorted(cells))
    fclk = set(re.findall(r'FREQ_HZ" VALUE="(\d+)"', h))

    if os.path.exists(a.kit):
        # Never delete: an earlier backup with the same name keeps it, and this
        # one gets the next free suffix.
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        old, i = "%s_old_%s" % (a.kit, stamp), 1
        while os.path.exists(old):
            old, i = "%s_old_%s_%d" % (a.kit, stamp, i), i + 1
        shutil.move(a.kit, old)
        print("previous kit moved to %s" % old)
    os.makedirs(os.path.join(a.kit, "export_gpu", "golden"))

    shutil.copy2(bit, os.path.join(a.kit, "vfi.bit"))
    shutil.copy2(hwh, os.path.join(a.kit, "vfi.hwh"))
    for s in SCRIPTS:
        shutil.copy2(os.path.join(REPO, s), a.kit)
    shutil.copy2(os.path.join(HERE, "BOARD_STEPS.md"), a.kit)
    for f in EXPORT_FILES:
        shutil.copy2(os.path.join(a.export, f), os.path.join(a.kit, "export_gpu", f))
    if os.path.exists(a.video):
        shutil.copy2(a.video, os.path.join(a.kit, "sample.mp4"))
    else:
        print("note: no %s -- copy a video to the board yourself" % a.video)

    files = []
    for root, _, names in os.walk(a.kit):
        for n in names:
            if n in ("SHA256SUMS.txt", "BOARD_STEPS.md"):
                continue
            files.append(os.path.relpath(os.path.join(root, n), a.kit).replace(os.sep, "/"))
    files.sort()
    with open(os.path.join(a.kit, "SHA256SUMS.txt"), "w", newline="\n") as f:
        for rel in files:
            f.write("%s *%s\n" % (sha256(os.path.join(a.kit, rel)), rel))

    print("kit  %s  (%d files + BOARD_STEPS.md + SHA256SUMS.txt)" % (a.kit, len(files)))
    print("bit  %s  sha256 %s" % (bit, sha256(bit)[:16]))
    print("post-route WNS %s ns, WHS %s ns; SG_LENGTH_WIDTH 26/26; FCLK %s Hz"
          % (wns.group(1), whs.group(1), "/".join(sorted(fclk))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
