# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Read this first

`PROJECT_CONTEXT.md`, next to this file, is a detailed handoff document covering the full defect
list, weight-file forensics, compute/memory budgets, and the project plan. **Read it before doing
RTL work.** This file covers only what you need to operate; it does not repeat it. (A byte-identical
copy circulates as `vfi_thirdyear.md`.)

## What this project is

SNN/CNN **video frame interpolation on a PYNQ-Z2** (Zynq-7020, `xc7z020clg400-1`, Vivado 2025.1).
The PL estimates an optical-flow field from a stacked frame pair; the PS (Python/PYNQ) warps and
blends to synthesise the in-between frame, doubling frame rate.

Goal priority, per the owner: **a finished, demonstrable portfolio piece** for job/internship
applications. A research paper is explicitly "a cherry on top, not the end goal." Favour advice that
is *working and verified* over *novel*.

## ⚠ The single most important fact

**The design currently computes a constant zero, and it synthesises cleanly while doing so.**

`spike_router`'s weight-memory port is tied off at `VFI FPGA/VideoFrameInterpolation/VideoFrameInterpolation.srcs/sources_1/new/top.v:128-129`
(`.bram_addr()`, `.bram_dout()`) and no weight memory exists in the project. That collapses the
whole datapath — `route_weight ≡ 0` → `syn_current ≡ 0` → LIF never crosses threshold → output ≡ 0 —
so Vivado proved the logic unobservable and deleted it. The final netlist is 67 LUT / 48 FF / 1 DSP
with **0 BRAM**, down from an RTL estimate of 280 BRAM.

Consequences for how you work here:

- **Treat the Vivado warning log as mandatory reading.** Every bug that produced this was a
  *warning*, not an error: width mismatches (`Synth 8-689`), unconnected ports (`8-7023`).
- **Do not trust `VideoFrameInterpolation/top.bit` or `VFI_wrapper.xsa`.** Both are stale, built
  from the superseded `VFI.bd`, and are the empty design.
- Top-level synth/impl for `VFI2` has **never completed** — `runs/synth_1/` and `runs/impl_1/` are
  empty.

## ⚠ The RTL implements a different architecture than the model

The Verilog is **not an unfinished version of the Python model** — it is a per-neuron spike-routing
(AER) design, while the model is a 6-layer convolutional net. There is no sliding window, no line
buffer, no channel accumulation, and exactly *one* `lif_neuron` instance standing in for ~380k
neurons. The input stream is 8-bit where the model needs int16; the output is one 32-bit word where
the model needs a 2×224×224 flow field.

**This is to be rebuilt as a line-buffered conv engine, not repaired.** Per §9 of the handoff doc,
this is settled: a per-source routing table would need 65 Mbit (13× the device) for what shared 3×3
kernels do in 13.8 Kbit.

Do not carry forward: `spike_router`, `synapse_fifo`, `synapse_accumulator`, `frame_buffer`,
`frame_assembler`, `input_adapter`, `axi_pixel_adapter`, `readout`, `neuron_array`.
Salvageable: `lif_neuron` (as a future activation unit) and the general AXI-Stream wrapper shape.

Also settled: `CHANNELS = 6` is **2 frames × 3 RGB**, concatenated host-side. The RTL's
`frame_buffer` / `pixel_prev` "fetch the previous frame" mechanism is redundant — both frames are
already in the input tensor.

## ⚠ The RTL has no bias

There is no bias anywhere in `rtl/`. `conv_mac_3x3`'s `round_bias` is the rounding constant and is
unrelated; `act_unit.sv`'s own header refers to the flow layer having "only a bias", but nothing
implements one.

The flow layer needs one — `flow_bias_q15.bin` is `[-1751, -414]`. Until that changes, **the PS adds
it after the DMA returns the feature map**, which is what `model/vfi_ref.py:add_bias` models: a
saturating add in Q1.15, applied after requantisation rather than inside the accumulator. That is
cheap (2 channels), costs at most 1 LSB on the final layer, and needs no change to a datapath that
has already closed timing. If the trained model turns out to need bias inside the accumulator, that
becomes an RTL change and the MAC's timing must be re-measured post-route.

## The trained model: `train/export_gpu/`

The model the board runs is `train/export_gpu/`, a full GPU run of `train/train_vfi.py`
(2026-10-22): held-out 21.49 dB vs 14.88 dB for a 50/50 blend, on Moving MNIST. **Shifts, flow bias
and weights come from its `vfi_model.json` and `.bin` files — never from this document.** It is
loaded by `vfi_ref.load_export()` and `vfi_driver.load_model()`, and `model/check_golden.py` proves
vfi_ref reproduces its `golden/<layer>.bin` dumps bit-exactly. Layout: `train/export_gpu/LAYOUT.md`.
Progress against it is logged in `STATUS.md`.

It was trained on Moving MNIST only and **does not transfer to real video** — on `sample.mp4` it
invents motion and loses to the plain blend (`STATUS.md` step 4). Real-footage training
(`train_vfi.py --videos`) is the open item for the demo.

The old `snnvfi/weights/*.bin` are a freshly-initialised, untrained network (§5 of the handoff doc);
ignore them.

## Environment

**Vivado 2025.1 is installed, in a non-standard location** — `C:\Users\manas\Downloads\2025.1\Vivado`
— with a licence at `C:\Users\manas\Downloads\Xilinx.lic`. It is **not on `PATH`** and there is no
`C:\Xilinx`, so `which vivado` finds nothing and a search of the usual directories comes up empty.
Do not conclude from that that it is missing. Put it on `PATH` first:

```bash
export PATH="/c/Users/manas/Downloads/2025.1/Vivado/bin:$PATH"
vivado -version          # v2025.1, verified working
```

`xvlog` / `xelab` / `xsim` live in the same `bin` directory. Vivado startup is slow (~1 min); give
batch invocations a generous timeout.

**`xsim` is the preferred simulator** — the owner works in Vivado and is most comfortable there.
Use `sim\run_xsim.ps1`, which sets the Vivado path, keeps all scratch output in `build\`, and works
around two Windows-specific traps documented in the script:

```powershell
.\sim\run_xsim.ps1 -Tb tb_line_buffer_3x3 -Sources rtl\conv\line_buffer_3x3.sv
.\sim\run_xsim.ps1 -Tb tb_line_buffer_3x3 -Sources rtl\conv\line_buffer_3x3.sv `
                   -Defines @{TB_IMG_W=224; TB_IMG_H=224}
```

Two failure modes `run_xsim.ps1` already handles — do not "simplify" them away:

- **`xelab`'s exit code is not trustworthy.** It builds the snapshot, then tries to delete its
  scratch `obj` directory, and under OneDrive that intermittently fails with "Access is denied",
  making it exit non-zero *after succeeding*. The script judges success by whether the snapshot
  exists. This is intermittent, so it looks like a flaky subset of configurations failing.
- **Never use `2>&1` on the Vivado executables.** PowerShell 5.1 wraps native stderr in
  `NativeCommandError` records, so any tool that merely prints a warning looks like a failure.

Do not hand-roll `xvlog` calls with defines on the command line. PowerShell 5.1 mangles native
arguments containing `=`, turning `-d W=224` into a define plus a phantom filename; and
`Set-Content -Encoding utf8` writes a BOM that corrupts the first line of an `-f` options file,
silently dropping `-sv` so every SystemVerilog source fails to parse. The script handles both.

**Lint with `sim\run_lint.ps1`** before committing any RTL. It exits non-zero on warnings, so it is
safe to gate on:

```powershell
.\sim\run_lint.ps1                        # every module under rtl\
.\sim\run_lint.ps1 -Modules line_buffer_3x3
```

Do not invoke `verilator` directly. Two traps, both of which fail *silently* and one of which
already produced a false "lint clean" claim in this repo's history:

- `oss-cad-suite\bin\verilator` is an extensionless **Perl script**, not an executable. From
  PowerShell it does not lint anything — Windows raises a "how do you want to open this file?"
  dialog and the shell reports **exit 0**, which is indistinguishable from a clean run. Use
  `verilator_bin.exe`.
- `environment.ps1` sets `VERILATOR_ROOT` to a Unix path (`/yosyshq/share/verilator`) that does not
  resolve on Windows, so `verilator_bin.exe` cannot find its waiver files and errors out. It must be
  overridden to `...\oss-cad-suite\share\verilator`.

Linting matters here specifically: implicit width truncation is what silently produced the
constant-zero design, and Verilator errors on it where Vivado only warned.

**oss-cad-suite also provides Icarus Verilog** (`iverilog`, `vvp`) as a fast second opinion —
seconds, versus ~1 min for an `xsim` elaborate. Its binaries will not run from the Bash tool at all
(exit 127); `which iverilog` resolving is not evidence that it works. From PowerShell:

```powershell
. "C:\Users\manas\Downloads\oss-cad-suite\environment.ps1"
iverilog -g2012 -o build\x.vvp sim\tb_x.sv rtl\x.sv;  vvp build\x.vvp
```

Python 3.14 is at `C:\Python314`. Installed: **numpy 2.5.2, torch 2.13.0+cpu, opencv 5.0.0**
(torch needs 2.9.0 or newer for a cp314 Windows wheel). **No CUDA** — `torch.cuda.is_available()`
is False and there is no NVIDIA GPU in this machine.

That matters less than it looks, because **none of the local work needs torch**:

- The **bit-exact reference model needs only numpy**. `model/vfi_ref.py` runs with no board and no
  torch, and torch would in fact be the wrong tool for it: it works in float, while the entire point
  of the reference is to reproduce the hardware's *integer* arithmetic exactly. A nearly-right answer
  is worse than useless when the question is "is this a real bug or a rounding difference?"
- Everything else done here — quantising weights to Q1.15, tuning the per-layer shifts, generating
  golden per-layer dumps — is numpy over weights that already exist.
- **Training belongs on a GPU elsewhere.** CPU training at 112x112 is slow enough not to be worth it.

**Export trained weights as `.npy` or raw int16, never as a `.pt` checkpoint.** A `.pt` can only be
reopened by torch, which drags the training dependency into the verification path for no reason;
`.npy` opens with numpy in one line. `model/vfi_ref.load_weights` already reads the raw
`(out, in, kh, kw)` int16 layout that `weight_loader.sv` expects, with no transpose.

Vivado commands (after putting it on `PATH` as above):

```bash
# Open a project in the GUI
vivado "VFI FPGA/VideoFrameInterpolation/VideoFrameInterpolation.xpr"

# Batch: synthesis + implementation + bitstream
vivado -mode batch -source build.tcl        # no build.tcl exists yet; write one if needed

# Batch Tcl against an existing project
vivado -mode tcl -source script.tcl "VFI FPGA/<proj>/<proj>.xpr"
```

Each per-module folder is its own Vivado project with its own testbench, runnable via the standalone
`xsim` flow:

```bash
xvlog -sv tb_lif_neuron.v lif_neuron.v
xelab -debug typical tb_lif_neuron -s tb_lif_neuron_sim
xsim tb_lif_neuron_sim -runall
```

Recommended guard once a build flow exists — promote the two message classes that let this project
build nothing into hard errors:

```tcl
set_msg_config -id {Synth 8-689}  -new_severity ERROR   ;# width mismatch
set_msg_config -id {Synth 8-7023} -new_severity ERROR   ;# port count mismatch
```

## Two copies of this project exist

This directory — `C:\Users\manas\OneDrive\Desktop\VFI_Third_Year\VFI FPGA` — is **the working
copy**, chosen by the owner. An older copy sits at `C:\Users\manas\VFI FPGA`.

Both are now clones of the same repo (`roaring-facet7/VideoFrameInterpolation`, branch `main`), with
identical history and remote; the RTL was already byte-identical between them. Work here, and treat
`C:\Users\manas\VFI FPGA` as a stale backup — it is safe to delete once you are confident this copy
is pushed.

**Nothing here has been pushed.** Commits made in this copy are local until someone runs
`git push origin main`, and the two clones will diverge silently in the meantime.

## Repository layout

### The conv-engine rebuild (new work, start here)

```
rtl/conv/     new streaming convolution RTL — the replacement design
sim/          testbenches + run_xsim.ps1
model/        bit-exact Python reference of the RTL datapath (numpy only)
build/        simulation scratch, gitignored
```

These are plain source directories, deliberately **not** a Vivado project — the RTL is developed and
verified standalone, then added to a project once it works. That inversion is the point: the old
design's defining failure was that the Vivado flow reported success on logic that computed nothing.

Built so far:

- **`line_buffer_3x3.sv`** — `same`-padded 3×3 window generator, full valid/ready handshaking.
  Verified tap-by-tap against an independently computed reference array, across three backpressure
  regimes and geometries from 1×1 to 224×224 (1.35M taps at full size).
- **`conv_mac_3x3.sv`** — nine Q1.15 multiplies (one DSP48E1 each) accumulated across input
  channels under caller-driven `first`/`last`, then rounded and saturated back to Q1.15. Verified
  against a 64-bit reference where wrapping is impossible.
- **`line_buffer_mc.sv`** — the multi-channel line buffer, and the block a real layer uses. Emits
  one window per channel tagged `m_ch` / `m_first` / `m_last_ch`, which is exactly the stream
  `conv_mac_3x3` consumes. Two-stage pipeline because the line stores must be BRAM at scale
  (2·112·128·16b = 0.46 Mbit) and BRAM reads are registered.
- **`conv_layer_1ch.sv`**, **`conv_layer_mc.sv`** — the two composed above, forming complete
  convolutions. `conv_layer_mc` is the N=1 case of a real layer.

- **`conv_layer.sv`** — the real layer: one `line_buffer_mc` driving N_OUT `conv_mac_3x3`
  instances, each with its own kernels, plus a weight write port.

**Post-synthesis, out-of-context** (`syn/ooc_synth.tcl`, `xc7z020clg400-1`, `conv_layer` at
112x112, CH_IN=6, **N_OUT=24**). Resource counts are reliable; the timing number is not — see
"What the design actually runs at" below:

| | Used | Available | |
|---|---|---|---|
| DSP48E1 | **216** | 220 | 98% — N_OUT=24 is the hard ceiling |
| LUT (logic) | 14,753 | 53,200 | 28% |
| Registers | 24,022 | 106,400 | 23% |
| Block RAM | 1 | 140 | grows with CH_IN |

**Post-synthesis timing at 100 MHz: +1.386 ns slack, 0 failing endpoints.** This is an OOC
estimate and the design does not ship at 100 MHz. That required three pipeline
stages in `conv_mac_3x3` (multiply -> accumulate -> requantise) and is not optional: the one-stage
version measured -10.6 ns and the two-stage -5.6 ns. Registering the products is what lets synthesis
use the DSP48E1's own output register. **If you restructure the MAC, re-measure post-route — an
OOC number cannot see this design's bottleneck.**

```powershell
$env:PATH = "C:\Users\manas\Downloads\2025.1\Vivado\bin;$env:PATH"
cd build\syn
vivado -mode batch -source ..\..\syn\ooc_synth.tcl  -tclargs conv_layer ..\..\rtl "IMG_W=112 IMG_H=112 CH_IN=6 N_OUT=24"
vivado -mode batch -source ..\..\syn\ooc_timing.tcl -tclargs conv_layer ..\..\rtl 10.0 "N_OUT=24"
```

Two Vivado gotchas, both encoded in those scripts: `-generic` **corrupts any parameter used as a
part-select width** (`x[i*W +: W]`), and synthesis then fails with "range width must be a positive
integer" — override only parameters used in plain arithmetic. And PowerShell silently drops an
empty-string argument, shifting every later `-tclargs` position.

- **`act_unit.sv`** — activation, combinational, `MODE` 0 linear / 1 ReLU, and in MODE 1 a runtime
  `linear` input (CFG[14]) that turns ReLU off for one layer — the flow layer, whose negative values
  are leftward/upward motion. Until 2026-10-23 ReLU was hardwired, so the shipped bitstream clamped
  about half of every flow field to zero. **This is the only module
  the CNN → SNN swap touches** (§10). Keep it separate and zero-latency; do not fold it into the MAC.
  MODE 2 (LIF) will need clk/rst and a timestep enable, at which point it stops being combinational
  and every integration testbench must be re-run.
- **`axis_serializer.sv`** — N parallel lanes → one value per cycle.
- **`conv_stage.sv`** — a complete layer: `conv_layer` → `act_unit` → `axis_serializer`.

**Layers chain with no glue.** `conv_stage`'s input and output are the same format —
`(row, col, channel)`, channel-minor — so one stage's output stream feeds the next stage's input
directly. `sim/tb_conv_chain.sv` proves it against a full two-layer software forward pass, including
backpressure propagating from stage 2 back through stage 1's serialiser into its MAC array and line
buffer. **Preserve that format.** It is also the shape the DMA path carries to and from DDR.

`conv_stage.sat_sticky` latches whether any output clipped, cleared by `sat_clear`. This is the
diagnostic for a wrong `OUT_SHIFT`: too small clips and destroys the activation, too large throws
away precision. Verified to work — the chain testbench reports clean at shift 20 and flags both
layers at shift 10. Use it to tune the per-layer shifts once the trained model exists.

### AXI-Stream layer (`rtl/axis/`)

- **`axis_skid.sv`** — skid buffer. Registers the data path, TVALID, and the upstream TREADY, so the
  ready chain does not run combinationally through the design.
- **`conv_stage_axis.sv`** — `S_AXIS → skid → conv_stage → skid → M_AXIS`. This is the unit the
  AXI-DMA talks to, replacing `top_axis.sv`.

TDATA is one Q1.15 value per beat in `(row, col, channel)` channel-minor order, identical on both
interfaces. **Exactly one TLAST per frame**, on the final beat — S2MM needs it to complete a
transfer, and its absence is the DMA hang in the old design. Input packet is
`IMG_W*IMG_H*CH_IN` beats, output `IMG_W*IMG_H*N_OUT`.

`s_axis_tlast` is deliberately accepted and ignored: frame length is a parameter and the line
buffer's counters define the frame, so a short or long DMA descriptor cannot silently desynchronise
the pipeline.

**`sim/axis_checker.sv` monitors any stream for protocol violations** — TVALID withdrawn before
TREADY, payload changing while stalled, wrong packet length, X on a qualified beat. Each rule maps to
a defect that shipped in the old design. **Instantiate it on both sides of any new AXI interface.**
It earned itself immediately by catching non-compliant stimulus in its own testbench's driver.

None of these bugs are observable at full rate — which is exactly why the old design passed its
testbenches and hung on hardware. **Always run AXI tests with stalls on both sides.**

**`weight_loader.sv`** streams a layer's kernels in over AXI-Stream, replacing per-value MMIO
writes (the host code did 186,624 writes per load, against an AXI-Lite interface the IP did not
have — §7). Beat order is **output channel major, then input channel, then the 9 taps**, which is
exactly the flattened `(out, in, kh, kw)` layout of `conv*_weight_q15.bin` (§4.3) — so the host DMAs
the file as-is. **Never add reordering here**; it would silently transpose every kernel.
`N_OUT*CH_IN*9` beats per layer; `weights_loaded` rises when a full set has landed.

### The complete IP: `rtl/axis/vfi_top.sv`

This is what the block design instantiates, replacing `top_axis.sv` / `vfi_top_0`.

```
S_AXIL   control and status (PS)
S_WT     kernels for the current layer   (DMA)
S_AXIS   input feature map               (DMA MM2S)
M_AXIS   output feature map              (DMA S2MM)
```

The PS runs the network by repeating, per layer:

```
write CFG (channels, shift, linear) -> DMA kernels -> DMA feature map in
                            -> DMA feature map out -> read STATUS
```

Layer output and input share the same `(row, col, channel)` channel-minor layout, so a result goes
straight back to DDR and returns as the next layer's input with no reformatting.

**AXI-Lite register map** (`rtl/axis/axil_regs.sv`) — the old IP had **no AXI-Lite at all**, so every
`ctrl.write()` in the host notebook was impossible as written (§7):

| Addr | | |
|---|---|---|
| 0x00 | ID | R — `0x56464931` ("VFI1"). **Read this first from the host.** The old host pointed at the wrong bitstream *and* the wrong instance name; a magic number turns both into an obvious failure instead of silently wrong output. |
| 0x04 | CFG | RW — `[7:0]` channels, `[13:8]` shift, `[14]` linear (1 = ReLU off; set for the flow layer only). Write before the frame that uses them; hold stable for its duration. `linear` resets to 0 and is active-high deliberately: a host that writes only channels and shift still gets ReLU. A bitstream older than 2026-10-23 reads bit 14 back as 0 — `vfi_driver` and `bringup.py` stage 2 both refuse it. |
| 0x08 | CTRL | W — `[0]` sat_clear, `[1]` soft_reset (both self-clearing pulses) |
| 0x0C | STATUS | R — `[0]` weights_loaded, `[1]` sat_sticky |
| 0x10 | FRAMES | R — completed frames; distinguishes "running" from "hung" |

Unmapped addresses respond rather than stall — an AXI-Lite slave that never answers locks the PS bus,
and a wrong address in host code is far easier to debug than a frozen board.

`soft_reset` resets the datapath only, deliberately: resetting `axil_regs` from its own write would
drop the AXI-Lite response mid-transaction and hang the PS.

### What the design actually runs at: 75 MHz, post-route

**The shipping build is 75 MHz.** `syn/ps7_config.tcl` sets `PCW_FPGA0_PERIPHERAL_FREQMHZ {75}`, and
that build meets timing post-route with 0 failing endpoints — but **only just**. The current build
(2026-10-23, which added CFG[14]) measured **WNS +0.014 ns, WHS +0.029 ns**, 0 failing of 85,306;
the 2026-08-31 build measured +0.081 / +0.021. Same worst path both times (the line-buffer → MAC
broadcast below), so the difference is placement variance, not the new bit. +0.014 ns is 0.1% of
a 13.333 ns period. It passes, and it is not a number to spend: assume any further
logic on this clock pushes it negative, and re-measure post-route rather than hoping. Every utilisation and timing table in
this file is **post-synthesis, out-of-context** — an estimate, and on this design a badly optimistic
one:

| Stage of the flow | Slack @ 100 MHz | |
|---|---|---|
| Post-synthesis, OOC | **+1.126 ns** | looks comfortable |
| Post-route, full block design | **−2.793 ns** | misses by 2.8 ns |

Estimated routing was optimistic by ~3.9 ns. The failing path is physical, not logical:

```
u_lb/chq_reg -> g_mac[16]/prod_q_reg
8.672 ns = 1.551 logic (18%) + 7.121 route (82%), 4 logic levels
```

The 144-bit window is broadcast to 24 MACs and 216 DSPs span the die, so **more pipelining inside the
MAC cannot fix this.** The real fix is pipelining the broadcast itself — two register stages between
`line_buffer_mc` and the MAC array, splitting the route. Worth doing when frame rate actually
matters; 75 MHz costs ~25% throughput (conv3 43 → 57 ms) and is not the binding constraint while
there is no trained model.

**Only post-route WNS counts.** `syn/ooc_synth.tcl` and `syn/ooc_timing.tcl` are screens for a design
that is nanoseconds off; they *cannot* see this bottleneck, because the bottleneck is routing. A
positive OOC number is not a pass. `syn/build.tcl` exits non-zero on negative post-route slack — and
it still writes a bitstream when timing fails, so **the `.bit` existing proves nothing.** That is the
same false success that produced the constant-zero design.

### Measured at the worst-case shape

`vfi_top` at 112x112, **MAX_CH=128, N_OUT=24** — the conv4 shape — synthesised OOC on
`xc7z020clg400-1`:

| | Used | Available | |
|---|---|---|---|
| DSP48E1 | **216** | 220 | 98% — N_OUT=24 is the hard ceiling |
| LUT | 17,239 | 53,200 | 32% |
| Registers | 17,065 | 106,400 | 16% |
| Block RAM | 64 | 140 | 46% |
| **Timing @ 100 MHz (post-synth)** | **+1.126 ns** | | post-route: **−2.793 ns**, hence 75 MHz |

`conv_mac_3x3` needs **all four** of its pipeline stages (multiply → accumulate → round →
shift/saturate). Each split was forced by measurement, not taste:

| Stages | Slack | What was in the critical path |
|---|---|---|
| 1 | -10.625 ns | accumulate + rounding add + saturate in series |
| 2 | -5.561 ns | BRAM → DSP → 9-way adder tree → accumulate |
| 3 | -0.320 ns | 48-bit add + barrel shift + compares, 24 levels / 17 CARRY4 |
| **4** | **+1.126 ns** | meets post-synthesis; still −2.793 ns post-route |

Two non-obvious requirements, both of which cost real slack when missed:

- **Products must be registered** (`prod_q`) so synthesis uses the DSP48E1's own output register.
- **`shift_q` carries `dont_touch` and `EQUIVALENT_REGISTER_REMOVAL = "no"`.** Every MAC's copy has
  identical inputs, so without those Vivado merges all 24 back into one and the long distribution
  route reappears — the "local" register ended up sourced from `g_mac[18]` feeding `g_mac[10]`, with
  the slack unchanged.

**If you restructure the MAC, re-measure post-route.** `LUT as Memory = 0` with a large register
count means the weight memories stopped inferring BRAM.

### Superseded note: the weight memories must stay BRAM

`conv_layer` at 112x112, **MAX_CH=128, N_OUT=24** — the conv4 shape, which is the hardest one —
synthesised OOC on `xc7z020clg400-1`:

| | Used | Available | |
|---|---|---|---|
| DSP48E1 | **216** | 220 | 98% — N_OUT=24 is the hard ceiling |
| LUT | 16,625 | 53,200 | 31% |
| Registers | 15,351 | 106,400 | 14% |
| Block RAM | 64 | 140 | 46% |
| **Timing @ 100 MHz (post-synth)** | **+0.524 ns** | | OOC estimate only |

**The weight memories must stay BRAM.** An earlier version declared them as arrays read
combinationally, expecting LUTRAM; Vivado inferred flip-flops and the same design measured
**138,248 LUT (260%) and 457,663 registers (430%)** — 24 x 128 x 9 x 16b is 442 Kbit of kernels.
Moving them to BRAM cut LUTs 8x and registers 30x. If you restructure `conv_layer.wmem`, re-measure;
`LUT as Memory = 0` alongside a huge register count is the signature of this going wrong again.

**The registered BRAM read costs no latency**, because it is issued from `line_buffer_mc`'s
*stage-A* index (`m_rd_ch`, enabled by `m_rd_en`), which runs exactly one cycle ahead of the window.
The BRAM output therefore lands in the same cycle as the window it belongs to, with no delay stage
and no alignment hazard. **Do not "simplify" this to read on `m_ch`** — that is a cycle late, and
every pixel would be convolved with the previous channel's kernel.

### Building a bitstream

```powershell
$env:PATH = "C:\Users\manas\Downloads\2025.1\Vivado\bin;$env:PATH"
vivado -mode batch -source syn\build.tcl                       # defaults: N_OUT=24, MAX_CH=128
vivado -mode batch -source syn\build.tcl -tclargs <dir> 8 8    # smaller, for a fast build
```

`syn/build.tcl` generates everything — project, block design, synthesis, implementation, bitstream,
XSA. The project directory is disposable; the sources of truth are `rtl/`, `syn/build.tcl` and
`syn/ps7_config.tcl`. The old project's state lived in a `.xpr` nobody could reproduce, its
top-level synthesis had never completed, and its committed bitstream was stale and from a different
block design.

```
PS7 --smc_ctrl-- vfi/s_axi, dma_fm lite, dma_wt lite      (control)
    --smc_mem--- dma_fm MM2S/S2MM, dma_wt MM2S -> HP0     (DDR)
    dma_wt MM2S -> vfi/s_wt      dma_fm MM2S -> vfi/s_axis      vfi/m_axis -> dma_fm S2MM
```

**It builds to `%LOCALAPPDATA%\vfi_build`, outside the repo, deliberately.** This repo is under
OneDrive; a Vivado project is tens of thousands of small files and the sync client holds handles on
them mid-build. The failure is `[Common 17-1293] The path '...vfi.cache/wt' already exists, is a
directory, but is not writable`. **`xelab` hits the same thing** on its scratch directories — that
is why `run_xsim.ps1` does not trust its exit code. If a tool fails on a file it just created, OneDrive
is the first suspect.

Four Vivado constraints this script encodes; do not "simplify" them away:

- **Replaying a full PS7 preset does not work.** Most of the 525 `PCW_*` properties are read-only
  derived values, and some MIO strings break Tcl parsing when replayed as a dict. It is also
  unnecessary — PYNQ applies DDR/MIO at boot, so an overlay needs only FCLK, the GP master and the
  HP slave. `syn/ps7_config.tcl` is that short list.
- **A SystemVerilog file cannot be the top of a BD module reference** (`[filemgmt 56-195]`), hence
  `rtl/axis/vfi_top_v.v`, a plain Verilog-2001 wrapper that adds no logic.
- **`apply_bd_automation` fails** with `[Ip 78-87] Error found in procedure get_rule_options` once
  more than one master is attached, even with `S_AXI_HP0` correctly enabled — verified by probing
  the configured PS7 cell. Both interconnects are built explicitly instead.
- **`Synth 8-689` (width mismatch) and `8-7023` (port count) are promoted to ERROR.** Those two
  warnings are what let the old design synthesise a constant-zero netlist and report success. There
  is deliberately no `allow_unconstrained_io.tcl`.

**The DMA Buffer Length Register must be widened.** `c_sg_length_width` defaults to **14**, which
caps a single transfer at 16,383 bytes — while every feature map here is far larger: 150,528 bytes in
(112x112x6), 602,112 out (112x112x24), and 3,211,264 for conv4's input. Only conv1's 2,592-byte kernel
set fits under the default. The first bitstream shipped with it, so bring-up would have passed the ID
check, CFG and the first weight load, then failed on the first feature map. Both DMAs are now set to
**26** (a 64 MB ceiling; 23 would cover the worst case) and the cost is a few registers.
`host/vfi_driver.py` issues one transfer per buffer with no chunking and **should stay that way** —
splitting a feature map in software would put a seam mid-frame that the line buffer's counters know
nothing about.

That is the second time this project produced a bitstream that builds clean and cannot work. Neither
Vivado nor simulation can catch this one: the limit lives in an IP parameter, and no testbench models
the DMA's length register. It was found by reading the generated `.hwh` against the driver's transfer
sizes — worth repeating for any other IP parameter left at its default.

No user XDC is needed: every physical pin belongs to PS7's fixed DDR/MIO interface, and the clock
comes from `FCLK_CLK0`.

### Assumed dataflow (decided 2026-08-23, revisit if wrong)

Intermediates round-trip through DDR via AXI-DMA, **one DMA transaction per layer pass**, with the PS
orchestrating. Chosen because a 112x112x64 map is 12.8 Mbit and cannot stay on chip, and because the
existing block design (PS7 → smartconnect → AXI-DMA) already supports it. This was the owner's call
to delegate; it is an assumption, not a measured result.

**A layer with more than N_OUT output channels runs as ceil(C_out/N_OUT) passes over the input.**
Where the input lives between passes — and between layers, since a 112x112x64 map is 12.8 Mbit and
cannot stay on chip — is the sequencer's problem and is **still an open design question**. It needs
a decision on DDR round-tripping via DMA before the sequencer can be built.

Two conventions these establish, both of which the old design violated:

1. **Saturation is evaluated at accumulator width, before any narrowing.** `synapse.v` compared the
   sum against `MAX_ACC` at the accumulator's own width, so it wrapped before the comparison could
   ever be true. Do not assume a clamp works because it is written down — `tb_conv_mac_3x3` counts
   how often each arm actually fires and fails if any never does.
2. **Everything crossing a module boundary is Q1.15.** The old `syn_current` was a raw integer while
   the LIF threshold was Q1.15, so the two were never comparable.

### Runtime configuration: `cfg_channels` and `cfg_shift`

**One bitstream must run all six layers**, which differ in input channel count (6, 32, 64, 128) and
in requantisation shift. Both are therefore **runtime inputs, not parameters**. `MAX_CH` only sizes
the memories; `cfg_channels` selects how many are used. Spatial size stays a parameter because every
layer is stride 1 with `same` padding, so `IMG_W`/`IMG_H` are constant across the network.

Both must be held stable while a frame is in flight — they define the frame.

`tb_conv_stage_axis` proves it: one elaborated instance runs six different (channels, shift)
configurations in sequence, each with its own DMA'd weights, all correct and protocol-clean.

At the neutral shift roughly 65% of random pixels saturate — a 9-tap sum over 128 channels of
full-scale Q1.15 operands simply does not fit int16 — so **each layer needs its own shift, derived
from the trained model.** Use `sat_sticky` to tune it.

**Trained weights are not needed to build or verify this.** Whether the hardware computes a
convolution correctly is an arithmetic property; random stimulus tests it perfectly well. RTL work
is therefore fully decoupled from the training critical path and the two can proceed in parallel —
which is the main reason not to wait on §11 Step 1.

### The old design

`VFI FPGA/` holds one integrated Vivado project plus nine per-module scratch projects. Each scratch
project (`lif_neuron/`, `readout/`, `spike_routing/`, `buffering/`, `synapse_accumulation/`,
`frame_assembler/`, `frame_buffer/`, `input_adapter/`, `FSM/`) has its own `.xpr` and testbench;
sources are then **copied** into the main project under
`VideoFrameInterpolation.srcs/sources_1/imports/`. Editing a scratch copy does not update the
integrated project, and vice versa — check which one you are in.

Main project: `VFI FPGA/VideoFrameInterpolation/`. Three block designs — `VFI2.bd` (current),
`VFI.bd` (superseded, delete), `PS.bd`. Module hierarchy:

```
top_axis  (top_axis.sv — AXI-Stream wrapper, 8-bit in / 32-bit out)
└── top   (top.v — the datapath below)
    ├── fsm_global          timestep_en / array_reset / frame_done
    ├── frame_assembler ─→ frame_buffer      (redundant, see above)
    ├── input_adapter       pixel_in vs pixel_prev delta → spike
    ├── spike_router ─→ synapse_fifo ─→ synapse_accumulator
    ├── lif_neuron
    └── readout             spike counter
```

Host/model side lives **outside this tree** at `C:\Users\manas\Downloads\snnvfi\` —
`jupyter.ipynb` (PYNQ pipeline), `notebooks/{preprocess,postprocess}.ipynb`, `weights/*.bin`,
`overlays/top.bit` (SHA-256 identical to the stale empty bitstream), `videos/sample.mp4`.

## Corrections to the handoff document

Verified against this copy on 2026-08-23; these items in `vfi_thirdyear.md` are **stale — do not
chase them**:

- §4.1 / §6.3 claim `readout/readout.srcs/sources_1/new/readout.v` was clobbered with an old
  `top.v`. It was not, in this copy: it is a correct 27-line `module readout`, byte-identical to the
  good copy in the main project.
- §6.3 claims `lif_neuron/` has stale `__synthesis_is_running__` / `.vivado.error.rst` lock files.
  No such files are present.

Still confirmed accurate in this copy: the tied-off `bram_dout` (`top.v:128-129`); empty
`runs/synth_1/` and `runs/impl_1/`; 0-byte `constrs_1/new/top.xdc`; `lif_neuron/` still targeting
Virtex-7 `xc7vx485tffg1157-1` instead of the Zynq; and all four width mismatches — notably
`synapse_accumulator` running on its `W_WIDTH=8` / `ACC_WIDTH=24` defaults while `top.v` connects
16-bit signals to both, and `readout`'s `CNT_WIDTH=8` `pixel_value` driving a 16-bit `out_value`
(upper 8 bits undriven). The accumulator's saturation logic is genuinely dead code: `syn_current +
mult_ext` wraps at `ACC_WIDTH` before the comparison against `MAX_ACC` can ever be true.

## Settled: 112x112

The §11 Step 2 scope-down is **decided — target 112×112**, owner's call, 2026-08-23. This sets the
DSP array size, so it is a precondition for the layer sequencer, not a later optimisation. New RTL
defaults to `IMG_W = IMG_H = 112`. At 224×224 the network is 9.36 GMAC/frame against 220 DSPs
(~0.3–2 fps); 112×112 is roughly 4× cheaper, and flow fields are smooth enough to upsample.

The per-layer `OUT_SHIFT` values now come from the trained export's `vfi_model.json`. Do not invent
or hand-tune them.

## Testing policy: blocks must be tested composed, not just alone

Owner's standing instruction, and it is the right one — the superseded design's modules each had a
passing unit testbench, and the integrated result computed a constant zero. Whenever a new block
lands, add or extend an integration testbench that runs it against the blocks it connects to and
checks the end-to-end result against a full software reference, not just the block's local
contract. Handshake mismatches, window-ordering errors and `first`/`last` sequencing bugs are only
visible there.

Current integration coverage:

- `sim/tb_conv_layer_1ch.sv` — `line_buffer_3x3` → `conv_mac_3x3`, checked against a 2D
  cross-correlation.
- `sim/tb_conv_layer_mc.sv` — `line_buffer_mc` → `conv_mac_3x3`, checked against a full 3D
  cross-correlation summed over input channels. Verified to 112×112×6.
- `model/test_vfi_ref.py` — the software reference against a naive from-the-definition
  implementation, and against the host driver's independent one. Not RTL, but the same principle:
  the thing you check against must itself be checked.

Both run frames back to back with **no reset between them**, because "drains cleanly once" and
"drains cleanly every time" are different properties and only the second one ships.

## The reference model: `model/vfi_ref.py`

The bit-exact software reference. PROJECT_CONTEXT.md section 13 calls this the highest-value thing to
build after the conv engine, because it is what turns "is the RTL right?" into a decidable question.
**It needs numpy only** — no torch, no cv2, no board — so it is not blocked on training and never was.

It reproduces `conv_mac_3x3` + `act_unit` exactly: int64 accumulation, round-half-up by
`1 << (shift-1)`, saturation evaluated at accumulator width *before* narrowing, then ReLU or linear.
Conventions were read off the RTL, not assumed — cross-correlation (tap `(kr,kc)` reads
`(y+kr-1, x+kc-1)`), `same` zero padding, `(row, col, channel)` channel-minor, kernels `(out,in,kh,kw)`
with no transpose.

`model/test_vfi_ref.py` — 39 checks, all passing. The model is not trusted on its own terms:

- every result is recomputed by a naive triple-loop written from the definition, the same way
  `sim/tb_conv_layer_mc.sv` writes its reference;
- padding is checked tap-by-tap with a delta kernel, so an edge off-by-one shows up as a displaced
  image rather than a slightly-wrong number;
- **both saturation arms are asserted to actually fire**, because the superseded design's clamp was
  unreachable code that no test noticed;
- it is cross-checked against `host/test_vfi_driver.py`'s independently written `ref_conv`, and the
  two agree exactly.

One deliberate difference from that host mock: it flags saturation with `>=`, over-reporting a value
that lands exactly on 32767 without clipping. `vfi_ref` follows the RTL's strict `>`.

`dump()` writes a feature map as raw little-endian int16 in AXI-Stream beat order, so per-layer
golden intermediates can be diffed against RTL output with no reordering. That per-layer comparison
is the verification story the portfolio rests on, and it is now possible.

**What it already shows.** Run on the real (untrained) weights at the neutral shift 15, activations
attenuate about 2.7x per layer — absmax `32767 -> 13191 -> 6588 -> 2456 -> 922 -> 196`, so the flow
layer uses 8 of its 15 bits. Concrete evidence that the per-layer `OUT_SHIFT` values matter and
cannot be guessed. They remain a property of the trained model: **do not invent them.**

## Board bring-up: `host/bringup.py`

Nothing in this project has run on silicon. `host/bringup.py` is the first thing to run on the
board, before anything else. Six stages, smallest to largest, each adding exactly one thing so a
failure names its own cause:

| | Stage | Proves |
|---|---|---|
| 1 | overlay loads, ID reads `0x56464931` | AXI-Lite reads, right bitstream, right instance |
| 2 | CFG writes and reads back | AXI-Lite writes |
| 3 | kernels DMA in, `weights_loaded` asserts | small transfer (2,592 B) |
| 4 | feature map round trip | **large** transfer (150,528 in / 602,112 out) |
| 5 | output matches `model/vfi_ref.py` exactly | the arithmetic |
| 6 | `FRAMES` advanced, saturation reported | housekeeping |

**Stage 3 vs stage 4 is the diagnostic split.** Stage 3 moves 2.5 KB and stage 4 moves 600 KB. Under
the old `c_sg_length_width = 14` (16,383-byte cap) stage 3 would pass and stage 4 fail — so that exact
pattern means an old bitstream, and `SG_LENGTH_WIDTH` in the `.hwh` should read **26**.

Copy to the board: `vfi.bit`, its matching `.hwh` (same basename), `host/vfi_driver.py`, and
`model/vfi_ref.py`. The reference model is numpy-only and runs there unchanged — which is the point
of having built it that way.

`host/test_bringup.py` runs the whole script against the mock PL from `test_vfi_driver.py`,
retargeted to the real 112x112x24 geometry. All six stages pass with an exact 301,056-value match,
**with no board**. It also injects three faults and requires the right diagnosis for each: an
unprogrammed PL (stops at 1), a constant-zero datapath (stops at 4, and names it as the superseded
design's exact failure), and a single wrong value out of 301,056 (stops at 5). A bring-up script
whose failure paths are broken is worse than none — it sends you hunting in the wrong place with a
board on loan.

## Direction of work

Per §11 of the handoff doc. Step 1 has since split in two, and only half of it is still blocked:
the **golden reference exists** (`model/vfi_ref.py`, numpy, no torch), while **training** does not.
(1) train the model in PyTorch with **ReLU**; (2) scope down to 112×112 and/or cap channels at 64 *before*
writing RTL, since it sets the DSP array size; (3) write the line-buffered conv engine, verified
layer-by-layer against golden intermediate dumps; (4) fix the DMA path; (5) demo — **stop here if
time is short**; (6) only then swap ReLU → LIF for the comparison paper.

The network as specified is 9.36 GMAC/frame against 220 DSPs — roughly 0.3–2 fps. `conv3`+`conv4`
alone are 79% of that. The widest feature map is 21× the whole device, so every layer must be
streamed through line buffers and never materialised.
