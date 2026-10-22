# STATUS — import the trained model and take the design to the board

The running log of the six-step plan from 2026-10-23. Each step is written up here as it finishes, so
the results survive a lost terminal. Newest step last.

| # | Step | State |
|---|---|---|
| 1 | Import the GPU export | **done** |
| 2 | Runtime activation bit for the flow layer | **done** — new bitstream meets 75 MHz post-route (WNS +0.014 ns) |
| 3 | Full network in simulation | **done** — whole network bit-exact at full 112x112 |
| 4 | `host/demo.py` | **done** — pipeline verified; the model fails on real video (see below) |
| 5 | Rebuild the board kit | **done** — `Desktop\vfi_board_kit`, 26 files, mock-tested from a flat copy |
| 6 | Commit and push | **done** — pushed to GitHub `main` (`50879ae..1701425`) |

---

## Step 1 — import the GPU export (2026-10-23)

**Result: PASS. vfi_ref reproduces all six golden layers bit-exactly, and conv1 in xsim has 0
mismatches.**

### What was imported

`C:\Users\manas\Downloads\export.zip` was unzipped into `train/export_gpu/`, with `golden/` kept
inside. The file layout is described in `train/export_gpu/LAYOUT.md`. Nothing was retrained.

- Shifts: conv1 12, conv2 12, conv3 12, conv4 14, conv5 15, flow 18. Flow bias `[3, -2]`.
- Held-out score: 21.49 dB against 14.88 dB for a 50/50 blend; 26.35 dB at 112x112.
- Sanity checks on the files:
  - Every `.bin` has the right size.
  - The int weights equal `round(float * 2^shift)` in every layer.
  - `flow_bias_q15.bin` matches the JSON.
  - `golden/input.bin` is byte-identical to `golden_input.bin`.

### Code now pointing at it

| File | Change |
|---|---|
| `model/vfi_ref.py` | `DEFAULT_EXPORT = train/export_gpu`; new `load_export(dir)` returns weights, shifts, bias and meta. It refuses an export whose layer list differs from `NETWORK`. |
| `host/vfi_driver.py` | new `load_model(dir)` returns weights, `VfiLayer`s with the export's shifts, and the flow bias; new `add_flow_bias()`. |
| `model/golden_conv1.py` | `--export` defaults to `train/export_gpu`. It generates the testbench's stimulus, so `sim/tb_conv1_golden.sv` now runs the new weights and shift. |
| `host/test_conv1_board.py` | uses `train/export_gpu` |
| `model/check_golden.py` | **new.** Re-runs vfi_ref on the export's weights and diffs every layer against `golden/<layer>.bin`. It also checks that `vfi_driver.load_model` and `vfi_ref.load_export` read the same model. |
| `train/train_vfi.py` | one-line fix: `weight_absmax_int` used `np.abs` on int16, which overflows at -32768. Reporting only; the weights are unaffected. |

### Results

`python model/check_golden.py`:

```
vfi_driver.load_model == vfi_ref.load_export: yes
layer   shape            mismatches  saturated   range
conv1   (112, 112, 32)            0     0.890%   [0, 32767]
conv2   (112, 112, 64)            0     1.080%   [0, 32767]
conv3   (112, 112, 128)           0     3.916%   [0, 32767]
conv4   (112, 112, 64)            0     6.217%   [0, 32767]
conv5   (112, 112, 32)            0     8.489%   [0, 32767]
flow    (112, 112, 2)             0     0.000%   [-6618, 7870]
flow vs quantised torch: max 3 LSB, 1.814% differ  (rounding mode only)
flow: 12453 of 25088 values negative -- these need ACT_MODE linear on the flow layer
PASS  bit-exact on every layer
```

Other checks:

- `model/test_vfi_ref.py`: all 39 checks pass.
- `host/test_vfi_driver.py`: all pass.
- `host/test_bringup.py`: all pass.
- `host/test_conv1_board.py`: all pass on the new export.
- `model/golden_conv1.py`: conv1 at shift 12, scalar cross-check 0 / 2144, and 0 / 401,408
  mismatches against the export's `golden/conv1.bin`.

`sim/tb_conv1_golden.sv` was re-run in xsim with the new weights. Configuration: full 112x112,
MAX_CH=128, N_OUT=24, ReLU, 25% stalls on both streams, two passes:

```
pass 0: kernels 0..23  shift=12  status=03  sat_sticky=1 (golden 1)
pass 1: kernels 24..47  shift=12  status=03  sat_sticky=1 (golden 1)
protocol: S_AXIS 150528 beats / 2 pkts / 0 violations;  M_AXIS 602112 / 2 / 0
PASS  602112 beats match the golden model (8354 nonzero, 0 negative), 2 passes
```

`python model/golden_conv1.py --check-sim` → `PASS: bit-exact, 0 mismatches` (401,408 values,
Python's independent diff of the dumped beats). The xsim run itself took about 7 s, so a full-size
network is practical for step 3.

### Things to know

- **Correction, made during step 3: 45% of the real motion is negative.** The 12,453 figure above
  counts values after the bias. Most of this frame is black background with flow exactly 0, and
  the −2 bias on ch1 pushes those zeros negative. Before the bias, 789 of the 1,762 non-zero flow
  values (45%) are negative. On this held-out triplet, clamping them as the old bitstream would
  costs **6.6 dB**: 29.33 dB with the linear flow, 22.76 dB with the ReLU-clamped flow, against
  20.06 dB for the 50/50 blend.
- **conv4 and conv5 saturate on 6.2% and 8.5% of outputs.** That is above `verify_export.py`'s 5%
  warning. Training clamps activations to Q1.15 exactly as the RTL does, so the 21.49 dB score
  already includes this clipping. It is a candidate for a shift change in a future training run,
  not a bug.
- **52 conv5 weights (0.28%) are clipped at -32768.** The export used shift 15 where the float
  weights (max 1.025) call for 14. The quantised model that was scored used the same integers, so
  again the score describes what the board computes. Details are in `LAYOUT.md`.

---

## Step 2 — the flow layer's activation becomes a runtime bit (2026-10-23)

### Design

`CFG[14] linear` was added to the control register: 1 turns ReLU off for that layer. It is set for
the flow layer only.

```
0x04 CFG  RW  [7:0] channels   [13:8] shift   [14] linear (1 = ReLU off)
```

- **Path:** `axil_regs.cfg_linear` → `vfi_top` → `conv_stage_axis` → `conv_stage` → `act_unit.linear`.
  In `act_unit` the change is one extra term, `(v < 0 && !linear) ? 0 : v`.
- **Polarity (one deviation from "ReLU enable"):** the bit is a ReLU *bypass*, active-high, reset 0.
  Any host write of just channels and shift therefore still gets ReLU, which is what every existing
  caller (bring-up, the tests, older scripts) was written against. An enable-style bit would have
  silently turned ReLU off for them.
- **Old bitstreams are detected.** They read bit 14 back as 0:
  - `vfi_driver._configure` raises, rather than return a half-clamped flow;
  - `bringup.py` stage 2 now writes and reads the bit, and names the problem.
- **Documentation:**
  - `rtl/axis/axil_regs.sv` header;
  - the `CLAUDE.md` register map and `act_unit` entry;
  - `vfi_driver.py` (`CFG_LINEAR`, and `VfiLayer(..., linear=)` filled in from `vfi_model.json`'s
    `act` by `load_model`).

### Verification

- **Verilator lint:** all 14 modules clean.
- **`sim/tb_vfi_top.sv`:** two new linear frames, followed by ReLU again with no reset in between:
  ```
  ch=8  shift=22 relu    ... 0 negative outputs
  ch=8  shift=22 linear  ... 99 negative outputs
  ch=4  shift=20 linear  ... 88 negative outputs
  ch=2  shift=15 relu    ... 0 negative outputs
  PASS  (1152 beats, 6 layers via AXI-Lite, 2 linear)
  ```
  It fails if a linear frame returns no negatives, or a ReLU frame returns any.
- **Mutation check:** the same testbench run against a copy of `act_unit` with ReLU hardwired
  (the old behaviour) gives `FAIL 189 errors` — "linear layer produced no negative outputs". So the
  test does catch the bug.
- **`sim/tb_conv1_golden.sv` with `TB_ACT=0`:** now drives linear through CFG[14] on the
  ACT_MODE=1 build, i.e. the bitstream's configuration. Trained conv1 at full 112x112:
  `PASS 602112 beats (14648 negative)`, and the Python cross-check is bit-exact. The ReLU run was
  re-done and is also bit-exact.
- **Unchanged testbenches:** `tb_conv_stage_axis` and `tb_conv_chain` pass with the new port tied off.
- **Host tests, all passing:**
  - `test_vfi_driver.py` adds: linear returns negatives; the next ReLU layer clears the bit; an old
    bitstream is refused; `load_model` marks only flow linear.
  - `test_bringup.py` adds: an old bitstream stops at stage 2.
  - `test_conv1_board.py` passes unchanged.

### Rebuild at 75 MHz: timing MET, the RTL fix is used (no ARM fallback)

The sequence was `syn/package_ip.tcl`, then `syn/build.tcl` into `%LOCALAPPDATA%\vfi_build_20261023`.
The 2026-08-31 build in `vfi_build` is untouched.

| post-route, `clk_fpga_0` = 75 MHz | this build | 2026-08-31 build |
|---|---|---|
| WNS (setup) | **+0.014 ns** | +0.081 ns |
| TNS | **0.000 ns**, 0 failing of 85,306 endpoints | 0.000 |
| WHS (hold) | **+0.029 ns** | +0.021 ns |
| THS | 0.000 ns | 0.000 |
| DSP / BRAM / LUT / FF | 216 / 67 / 20,956 / 24,182 | — |

- **The worst path is unchanged.** It is `u_lb/chq_reg → g_mac[8].u_mac/prod_q_reg`, the known
  line-buffer→MAC broadcast route in `CLAUDE.md`, not the new logic. So the drop from +0.081 to
  +0.014 ns is place-and-route variation on that path. It still passes, but the margin is now 0.1%
  of the period. The real fix remains pipelining that broadcast (`CLAUDE.md`).
- **`cfg_linear` is real hardware.** In the routed netlist it is `u_regs/cfg_linear_reg`, which
  Vivado replicated as `_rep` and `_rep__0` to drive its fanout. So it was not optimised away.
- **Synthesis is clean for our RTL.** The only 8-7023 warnings come from Xilinx IP (ps7, DMAs,
  smartconnect), which `build.tcl` scopes out on purpose.
- **`build.tcl` exited 1 after the bitstream, in `write_hw_platform` (the XSA).** It could not
  delete its `.Xil` temp folder because I had launched Vivado from `build/vivado_log`, which is
  inside OneDrive — the trap `CLAUDE.md` documents.
  - **Fixed:** `build.tcl` now `cd`s into the output directory.
  - The XSA was then written by a separate batch, run from outside OneDrive:
    `%LOCALAPPDATA%\vfi_build_20261023\vfi.xsa`.
  - PYNQ needs only `.bit` + `.hwh`; the XSA is for Vitis.
- **Files:** `vfi.bit` (sha256 `fcc1cc041ac5306b…`), `vfi.xsa`, and the reports
  `timing_summary_routed.txt` and `utilization_routed.txt`, all in
  `%LOCALAPPDATA%\vfi_build_20261023`. The build log is
  `build/vivado_log/build_20261023_stdout.txt`.

---

## Step 3 — the whole network in simulation

**Result: PASS.** All six layers are bit-exact at full 112x112, chained through 17 hardware passes.
The final flow, after the host-side bias, equals `golden/flow.bin`. No reduced size was needed.

### How

- **Stimulus:** `model/golden_network.py` writes `build/net/`. That is per-layer kernels padded
  exactly as `run_layer` pads them, the expected output of every layer (flow before the bias), and
  the expected `sat_sticky` per pass. All of it comes from vfi_ref, which is first checked against
  the export's own `golden/<layer>.bin`.
- **Testbench:** `sim/tb_network_golden.sv` instantiates one `vfi_top` at the bitstream's
  configuration (112x112, MAX_CH=128, N_OUT=24, ACT_MODE=1). It drives every pass the way
  `vfi_driver.run_layer` does: write CFG and check the readback, clear saturation, DMA the kernels,
  DMA the map in and out, read STATUS. There are 20% random stalls on every stream and no reset
  between passes or layers.
- **It chains.** Each layer's input is the RTL's own stitched output of the previous layer, not a
  golden file.
- **The flow layer runs with `CFG[14]=1`.**
- **Checks:**
  - in-line, against the expected values; padded channels must be exactly 0;
  - TLAST on the last beat of every pass;
  - `sat_sticky` per pass against golden, and `FRAMES`;
  - protocol checkers on both streams;
  - every stitched layer is also dumped, and `golden_network.py --check-sim` diffs it
    independently.

### xsim output

```
conv1   6 -> 32  shift 12   relu 2 pass(es)   401408 values  0 mismatches  0 negative
conv2  32 -> 64  shift 12   relu 3 pass(es)   802816 values  0 mismatches  0 negative
conv3  64 -> 128 shift 12   relu 6 pass(es)  1605632 values  0 mismatches  0 negative
conv4 128 -> 64  shift 14   relu 3 pass(es)   802816 values  0 mismatches  0 negative
conv5  64 -> 32  shift 15   relu 2 pass(es)   401408 values  0 mismatches  0 negative
 flow  32 -> 2   shift 18 linear 1 pass(es)    25088 values  0 mismatches  789 negative
protocol: S_AXIS 12995584 beats / 17 pkts / 0 violations;  M_AXIS 5117952 / 17 / 0
PASS  all 6 layers bit-exact, 17 passes chained, FRAMES=17
```

`python model/golden_network.py --check-sim`:

```
conv1 0 / conv2 0 / conv3 0 / conv4 0 / conv5 0 mismatches
flow 0 mismatches (+bias, vs golden/flow.bin); 789 negative values reached the host
PASS: whole network bit-exact, final flow == golden/flow.bin
```

### Notes

- **The multi-pass layers are covered.** conv2 and conv4 are 64 out in 3 passes, conv3 is 128 out
  in 6, and all are bit-exact in the chain.
- **The flow layer's 789 negative values arrive intact.** Under the old hardwired ReLU every one of
  them would have been 0.
- **Simulated time:** 173 ms at the testbench's 100 MHz with 20% stalls, i.e. about 17.3M cycles
  for one frame pair. At 75 MHz with no stalls the PL floor is about 13.4M cycles, or **about
  0.18 s per frame pair**, before any PS/DMA overhead. `perf.py` will measure the real figure on
  the board.
- **Wall time:** 9 min 50 s in xsim.
- **Board equivalent:** `host/network_board.py` runs the same chained check layer by layer, with
  the same golden files. It is tested against the mock PL in `host/test_network_board.py`, all
  passing, including the "one wrong value" and "pre-CFG[14] bitstream" faults.

---

## Step 4 — `host/demo.py` (2026-10-23)

### What it does

`python host/demo.py VIDEO [--sim] [--frames N] [--start K] [--out X.mp4] [--png Y.png]`

1. For every consecutive pair (a, b), it preprocesses exactly as `train_vfi.py`'s video loader does:
   resize to 112 (INTER_AREA), BGR→RGB, /255, then Q1.15 clamped to 32767; a RGB ++ b RGB.
2. It runs `run_network` on the PL. `--sim` uses `vfi_ref.forward` instead, which is bit-exact
   with the PL.
3. It adds the flow bias, then converts to pixels: ×32/32768, ch0 = x, ch1 = y.
4. It upsamples the flow to the video size, scaling the vectors by W/112 and H/112.
5. It forms the midpoint `0.5·(warp(a, +f/2) + warp(b, −f/2))`, using cv2.remap with
   BORDER_REPLICATE, which is training's `grid_sample(align_corners=True, border)`.
6. It writes a, then the midpoint, at 2x fps.

`--png` writes four columns from a triplet: frame t | blend | model | true t+1, with PSNR.

### The pipeline is verified (`host/test_demo.py`, all pass)

- `preprocess(frame0, frame2)` equals the export's `golden_input.bin` value for value.
- The cv2 midpoint matches `verify_export.py`'s grid_sample-equivalent warp to 0 grey levels.
- On the held-out MNIST triplet: **model 29.33 dB vs blend 20.06 dB.**
- The warp sign convention is right: flow x = −3 moves content right.
- `--sim` runs end to end: 4 frames in gives 7 out, at 2x fps, plus the PNG.

### Run on `videos/sample.mp4`, `--sim`

Command: `--start 300 --frames 61`. That is 60 pairs at 1280x720, about 3.1 s per pair in numpy.

- `build/demo/sample_300_2x.mp4`: 121 frames at 59.94 fps.
- `build/demo/side_by_side.png`: synthesising frame 330 from 329 and 331, the model scores
  **25.47 dB** against **45.97 dB** for the blend.

Scored fairly over 15 triplets spread across the whole clip (`build/demo/eval_sample.txt`):

| | mean PSNR |
|---|---|
| model | **23.66 dB** |
| 50/50 blend | 58.92 dB |
| copy frame t | 59.00 dB |

The model loses to the blend on **15 of 15** triplets.

### What this means

- **The weights do not transfer to real video.** The model was trained on Moving MNIST only:
  grey digits on black, with no lit or textured background. On real frames it invents a
  near-constant flow of about 40 px at 1280 width (about 3.5 px at 112) and smears the image —
  the cup rim and logo in the PNG. This was predicted in `train/README.md` (item 8).
- **It is not a pipeline or hardware bug.** The same code gives +9.3 dB over the blend on the
  held-out MNIST triplet, and every stage is checked against training's own convention.
- **`sample.mp4` is also a poor demo clip.** It is near-static slow motion: copying the previous
  frame scores 59 dB. Even a good model has almost no motion to show off here.
- **To fix it** (not done; you said no retraining):
  - train with `train_vfi.py --videos <clips>` on real footage with visible motion;
  - demo on a clip with real motion.
  Everything downstream — RTL, bitstream, driver, demo — takes the new export unchanged.

---

## Step 5 — the board kit (2026-10-23)

**Location:** `C:\Users\manas\OneDrive\Desktop\vfi_board_kit\`, assembled by
`python host/make_kit.py --build %LOCALAPPDATA%\vfi_build_20261023`.

### Contents

| | |
|---|---|
| `vfi.bit`, `vfi.hwh` | the 2026-10-23 build; the `.hwh` is renamed from `vfi_bd.hwh` so `Overlay("vfi.bit")` finds it |
| `vfi_driver.py`, `vfi_ref.py` | driver (with CFG[14] and `load_model`) and the bit-exact reference |
| `bringup.py` | six stages; stage 2 now also checks CFG[14] |
| `conv1_board.py` | trained conv1 vs golden |
| `network_board.py` | **new:** the whole network on the PL, chained, every layer vs golden |
| `perf.py` | one pass, plus **new** full-network timing per layer against the PL floor |
| `demo.py` | the demo, with `sample.mp4` next to it |
| `export_gpu/` | trained weights, `vfi_model.json`, flow bias, `LAYOUT.md`, and `golden/` (input + all six layers) |
| `BOARD_STEPS.md` | the exact commands: bringup → conv1_board → network_board → perf → demo |
| `SHA256SUMS.txt` | all 26 files, checked on the board with `sha256sum -c` |

### What `make_kit.py` refuses to package

- a build without post-route `SUMMARY WNS`/`WHS`, or with either negative;
- a `.bit` older than its `.hwh`;
- a `.hwh` without `c_sg_length_width` 26 on both DMAs, or without the `vfi`/`dma_fm`/`dma_wt` cells.

It found 75 MHz FCLK, SG 26/26, and all three cells.

### Tested as the board will see it

The kit was copied to a temp folder and `sha256sum -c` passed on 26/26 files. Every script was then
run from that flat copy, importing the kit's own files, against the mock PL:

- `bringup.py`: rc 0
- `conv1_board.py`: `PASS bit-exact on both inputs`
- `network_board.py`: `PASS every layer bit-exact`
- `demo.py --sim`: rc 0; it finds `./export_gpu` by itself

### A mistake I made, and repaired

My first version of `make_kit.py` moved the existing kit to `vfi_board_kit_old_20261023`. When I
re-ran it the same day, it called `rmtree` on that backup — the original 2026-10-13 kit — and
deleted its files.

I rebuilt all 11 hashed files from their sources and they verify against that kit's own
SHA256SUMS: **11/11 OK, byte-identical.**

- the `.bit` and `.hwh` from `%LOCALAPPDATA%\vfi_build`;
- the scripts from git `3905b73`;
- `perf.py` and `conv1_board.py` by reversing this session's edits;
- the data from `train/export_cpu3k`.

`BOARD_BRINGUP.md` had no hash. I rewrote it from the full text I had printed earlier in this
session. If a byte of it differs, the original is in OneDrive's online recycle bin for 30 days.

`make_kit.py` now never deletes: an old kit is moved to `<kit>_old_<date>_<time>`, with a numeric
suffix if that name exists.

---

## Step 6 — commit and push (2026-10-23)

**Pushed: `origin/main` went `50879ae..1701425`, a fast-forward.** Nothing on GitHub was overwritten.

### The commits made today

| | |
|---|---|
| `da1918c` | Import the trained GPU export; point the reference model at it |
| `fe25e7b` | Make the flow layer's activation a runtime bit: CFG[14] linear |
| `97233d5` | Add demo.py |
| `dbe876a` | Verify the whole network in xsim, chained, and add its board twin |
| `6d5ade2` | Rebuild at 75 MHz with CFG[14]; add the board kit tooling |
| `9e1d04d` | Add STATUS.md |
| `1701425` | Merge GitHub main |

The push also carried the 26 earlier local commits that had never been pushed (the whole conv-engine
rebuild since August), and this STATUS commit follows.

### Two things found on the way

- **GitHub's `main` had moved.** It held one commit this clone had never fetched: `50879ae` "Add
  files via upload", 2026-08-04, which adds `snnvfi/` — notebooks, the untrained weights, the stale
  `top.bit` and `sample.mp4`. It touches no file of ours, so it merged with no conflicts and the
  push stayed a fast-forward.
- **Branches.** The work lived on local branch `reference-model`. Local `main` was fast-forwarded to
  the pushed commit, so the two now match. `reference-model` itself was not pushed as a separate
  branch.

---

## Where this leaves the project

**Verified in simulation and ready for the board:**

- the whole trained network at the bitstream's exact configuration — every layer bit-exact,
  chained;
- the flow layer's negatives preserved;
- a bitstream that meets 75 MHz;
- a kit with a staged, mock-tested script for each step.

**Not done, and why:**

1. **Nothing has run on silicon yet.** Run `BOARD_STEPS.md` in order.
2. **The model does not work on real video** (step 4): it loses to the blend on 15/15 triplets of
   `sample.mp4`. It needs training on real footage with motion (`train_vfi.py --videos`), which you
   asked me not to do now, and a demo clip that actually moves. Nothing else changes: RTL, bitstream,
   driver and demo all take a new export as-is.
3. **Timing margin is +0.014 ns.** It passes, but any RTL change needs a post-route re-check.
   Pipelining the line-buffer → MAC broadcast (`CLAUDE.md`) is the fix if it ever goes negative.
