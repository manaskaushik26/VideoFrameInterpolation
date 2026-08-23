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

**Vivado is not installed on this machine** (not on `PATH`, absent from `C:\Xilinx` and every other
standard location; only a `C:` drive exists). You cannot build, synthesise, or simulate here — do
not propose running a flow and reporting results. Reason about the RTL by reading it and the
committed logs under `*.runs/`.

When commands do run, they are Vivado batch mode from a shell with the toolchain on `PATH`:

```bash
# Open a project in the GUI
vivado "VFI FPGA/VideoFrameInterpolation/VideoFrameInterpolation.xpr"

# Batch: synthesis + implementation + bitstream
vivado -mode batch -source build.tcl        # no build.tcl exists yet; write one if needed

# Batch Tcl against an existing project
vivado -mode tcl -source script.tcl "VFI FPGA/<proj>/<proj>.xpr"
```

Standalone simulation of one module (each per-module folder is its own Vivado project with its own
testbench — this is how individual modules get tested):

```bash
# Compile, elaborate, run one testbench with the standalone xsim flow
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

## Direction of work

Per §11 of the handoff doc, in order: (1) rebuild and train the model in PyTorch with **ReLU**, which
also becomes the golden reference; (2) scope down to 112×112 and/or cap channels at 64 *before*
writing RTL, since it sets the DSP array size; (3) write the line-buffered conv engine, verified
layer-by-layer against golden intermediate dumps; (4) fix the DMA path; (5) demo — **stop here if
time is short**; (6) only then swap ReLU → LIF for the comparison paper.

The network as specified is 9.36 GMAC/frame against 220 DSPs — roughly 0.3–2 fps. `conv3`+`conv4`
alone are 79% of that. The widest feature map is 21× the whole device, so every layer must be
streamed through line buffers and never materialised.
