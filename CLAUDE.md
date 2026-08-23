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

## ⚠ The weights are not trained

`snnvfi/weights/*.bin` are a freshly-initialised PyTorch network dumped to disk — proven three ways
in §5 of the handoff doc (uniform distribution fingerprint, ranges matching `kaiming_uniform_(a=√5)`
to within 0.1%, flat histogram). There is **no trained model, no training script, and no golden
reference model** anywhere.

Never spend effort tuning quantisation or matching RTL bit-exactly against these numbers. Rebuilding
and training the model in PyTorch is Step 1 of the plan and the critical path, because it is also
the golden reference that makes verification possible at all.

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

Python 3.14 is at `C:\Python314`, but **`numpy`, `torch`, and `cv2` are all absent** — the reference
model and training work need those installed first. No CUDA.

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
model/        (empty) bit-exact Python reference, once numpy is available
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

**Measured, not estimated** (`syn/ooc_synth.tcl`, `xc7z020clg400-1`, `conv_layer` at 112x112,
CH_IN=6, **N_OUT=24**):

| | Used | Available | |
|---|---|---|---|
| DSP48E1 | **216** | 220 | 98% — N_OUT=24 is the hard ceiling |
| LUT (logic) | 14,753 | 53,200 | 28% |
| Registers | 24,022 | 106,400 | 23% |
| Block RAM | 1 | 140 | grows with CH_IN |

**Timing: met at 100 MHz, +1.386 ns slack, 0 failing endpoints.** That required three pipeline
stages in `conv_mac_3x3` (multiply -> accumulate -> requantise) and is not optional: the one-stage
version measured -10.6 ns and the two-stage -5.6 ns. Registering the products is what lets synthesis
use the DSP48E1's own output register. **If you restructure the MAC, re-measure — do not assume.**

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

- **`act_unit.sv`** — activation, combinational, `MODE` 0 linear / 1 ReLU. **This is the only module
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

`conv_mac_3x3.OUT_SHIFT` parameterises requantisation rather than hardcoding `>>> 15`. At the
neutral shift roughly 65% of random pixels saturate — a 9-tap sum over 128 channels of full-scale
Q1.15 operands simply does not fit int16 — so **each layer needs its own shift, derived from the
trained model.** Treat a hardcoded shift as a bug.

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

Still open, and genuinely blocked on the trained model: the **per-layer `OUT_SHIFT` values**. Do not
invent them.

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

Both run frames back to back with **no reset between them**, because "drains cleanly once" and
"drains cleanly every time" are different properties and only the second one ships.

## Direction of work

Per §11 of the handoff doc, in order: (1) rebuild and train the model in PyTorch with **ReLU**, which
also becomes the golden reference; (2) scope down to 112×112 and/or cap channels at 64 *before*
writing RTL, since it sets the DSP array size; (3) write the line-buffered conv engine, verified
layer-by-layer against golden intermediate dumps; (4) fix the DMA path; (5) demo — **stop here if
time is short**; (6) only then swap ReLU → LIF for the comparison paper.

The network as specified is 9.36 GMAC/frame against 220 DSPs — roughly 0.3–2 fps. `conv3`+`conv4`
alone are 79% of that. The widest feature map is 21× the whole device, so every layer must be
streamed through line buffers and never materialised.
