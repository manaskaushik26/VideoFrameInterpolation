# SNN Video Frame Interpolation on FPGA — Full Project Context

> Handoff document. Everything established about this project as of **2026-08-03**.
> Written so another assistant can pick up with full context and no re-discovery.
> Facts here were verified by reading the actual files, Vivado logs, and weight binaries —
> not assumed. Where something is inferred rather than verified, it says so.

---

## 1. Goals

**Primary goal — portfolio piece for jobs / internships.** This needs to be a *finished, working,
demonstrable* project. That is the thing that matters. A modest design that runs end-to-end and is
properly verified beats an ambitious one that doesn't work.

**Secondary goal — a research paper.** Explicitly described by the owner as "a cherry on top,
not the end goal."

**Implication that has already been agreed:** the two goals point the same way. Build the
conventional-CNN version first (it's the portfolio piece AND the paper's control group), then swap
in the spiking version. Details in §10.

---

## 2. Hardware target

| | |
|---|---|
| Board | PYNQ-Z2 |
| Part | `xc7z020clg400-1` (Zynq-7020) |
| Tools | Vivado 2025.1 |
| LUTs | 53,200 |
| Flip-flops | 106,400 |
| BRAM | 140 × 36 Kb = **4.9 Mbit** (630 KB) |
| DSP48E1 | **220** |
| PS | Dual-core Cortex-A9, 512 MB DDR3 |
| Host | PYNQ (Python), `pynq.Overlay` + `pynq.lib.DMA` |

---

## 3. Intended system architecture

Video frame interpolation via **optical flow estimation**. The FPGA predicts a flow field; the host
warps and blends to synthesise the in-between frame, doubling the frame rate.

```
[PS / Python]
  read video → take consecutive frame pair (frame0, frame1)
  resize to 224×224, normalise to [-1,1], quantise to Q1.15 int16
  concat as 6 channels (frame0 RGB ++ frame1 RGB)  →  tensor (6,224,224) int16
       │
       │  AXI-DMA MM2S
       ▼
[PL / FPGA]  convolutional network
  conv1   6→32   3×3  + activation
  conv2  32→64   3×3  + activation
  conv3  64→128  3×3  + activation
  conv4 128→64   3×3  + activation
  conv5  64→32   3×3  + activation
  flow   32→2    3×3  + bias
       │
       │  AXI-DMA S2MM
       ▼
[PS / Python]
  flow (2,224,224) int16 Q1.15
  warp frame0 by +0.5·flow, warp frame1 by −0.5·flow  (cv2.remap)
  blend 50/50 → interpolated middle frame
  write [frame0, interpolated] to output video at 2× fps
```

**Activation:** currently intended as LIF spiking neurons (hence "SNN"), with `beta = 0.9`,
`threshold = 0.5`, reset-to-zero. See §10 for why this should start as ReLU.

**Confirmed:** `CHANNELS = 6` is **2 frames × 3 RGB**, concatenated by `preprocess_pair()`.
This matters — the RTL currently contains a separate `frame_buffer` / `pixel_prev` mechanism to
fetch a "previous frame," which is **redundant**. Both frames are already in the input tensor.

---

## 4. What exists — asset inventory

### 4.1 `VFI FPGA/` — Vivado projects (now in tag `legacy-spike-design`)

| Path | What it is |
|---|---|
| `VideoFrameInterpolation/` | **Main project.** Two block designs: `VFI2.bd` (current) and `VFI.bd` (older, should be deleted) |
| `block_des/` | Separate scratch project |
| `buffering/` | `synapse_fifo` module + testbench |
| `FSM/` | `fsm_global`, `lif_neuron`, `neuron_array` (unused), `top_snn` |
| `frame_assembler/` | `frame_assembler.sv` |
| `frame_buffer/` | `frame_buffer.sv` |
| `input_adapter/` | `input_adapter.v` |
| `lif_neuron/` | `lif_neuron.v`, `new_lif_neuron.v` |
| `readout/` | **BROKEN** — `readout.v` was overwritten with an old copy of `top.v`; project's TopModule is `readout`, which no longer exists in it |
| `spike_routing/` | `spike_router` (`router.v`), `fsm.sv` |
| `synapse_accumulation/` | `synapse_accumulator` (`synapse.v`) |
| `VideoFrameInterpolation - Copy/`, `.zip` | Backups |

RTL modules actually in the main project's source set:
`top_axis` → `top` → { `fsm_global`, `frame_assembler`, `frame_buffer`, `input_adapter`,
`spike_router`, `synapse_fifo`, `synapse_accumulator`, `lif_neuron`, `readout` }

### 4.2 `snnvfi/` — model + host code (now in tag `legacy-spike-design`)

```
jupyter.ipynb                 full PYNQ host pipeline (load bitstream, DMA, pre/post process)
notebooks/preprocess.ipynb    VideoPreprocessorQ15 class
notebooks/postprocess.ipynb   warp + blend
overlays/top.bit              ⚠ SHA-256 IDENTICAL to the stale VFI FPGA bitstream
videos/sample.mp4             2 MB test clip
weights/*.bin                 6 weight tensors + 1 bias + 5 LIF stat files, all int16/float32
```

### 4.3 Weight files — exact contents

All 3×3 kernels, stride 1, `same` padding (padding inferred, not verified).

| File | Bytes | int16 | Shape (out,in,kh,kw) |
|---|---|---|---|
| `conv1_weight_q15.bin` | 3,456 | 1,728 | 32 × 6 × 3 × 3 |
| `conv2_weight_q15.bin` | 36,864 | 18,432 | 64 × 32 × 3 × 3 |
| `conv3_weight_q15.bin` | 147,456 | 73,728 | 128 × 64 × 3 × 3 |
| `conv4_weight_q15.bin` | 147,456 | 73,728 | 64 × 128 × 3 × 3 |
| `conv5_weight_q15.bin` | 36,864 | 18,432 | 32 × 64 × 3 × 3 |
| `flow_weight_q15.bin` | 1,152 | 576 | 2 × 32 × 3 × 3 |
| `flow_bias_q15.bin` | 4 | 2 | `[-1751, -414]` = `[-0.0534, -0.0126]` in Q15 |
| `lif1..5_stats.bin` | 8 each | — | 2 × float32 = `(beta=0.9, threshold=0.5)`, **identical in all five** |

**Total weights: 186,624 int16 = 2.99 Mbit** — 61% of the device's BRAM. Fits, but tight.

---

## 5. ⚠ CRITICAL: the weights are NOT trained

**They are a freshly-initialised PyTorch network dumped to disk.** Proven three independent ways:

**Test 1 — the distribution is uniform.** `mean|w| / rms` fingerprints distribution shape:
uniform = √3/2 = 0.8660, Gaussian = 0.7979, Laplacian ≈ 0.707. Trained weights are Gaussian or
Laplacian (peaked at zero).

```
conv1 0.8671   conv2 0.8667   conv3 0.8667
conv4 0.8666   conv5 0.8667   flow  0.8706
```

Six independent tensors, all on the uniform constant.

**Test 2 — ranges match PyTorch's default `nn.Conv2d` init exactly.** `reset_parameters()` uses
`kaiming_uniform_(a=√5)`, which reduces to bound = `1/√fan_in`:

| layer | fan_in | predicted peak | actual | error |
|---|---|---|---|---|
| conv1 | 54 | 4459.2 | 4456 | −0.07% |
| conv2 | 288 | 1930.9 | 1931 | +0.01% |
| conv3 | 576 | 1365.3 | 1365 | −0.02% |
| conv4 | 1152 | 965.4 | 965 | −0.05% |
| conv5 | 576 | 1365.3 | 1365 | −0.02% |
| flow | 288 | 1930.9 | 1929 | −0.10% |

**Test 3 — flat histogram.** conv3 across 20 bins: every bin 4.7–5.2%. A trained layer is sharply
peaked at zero. This is a rectangle.

**Corroborating:** `lif1..5_stats.bin` are byte-identical `(0.9, 0.5)` — library defaults written
five times, not learned per-layer parameters.

**Consequence:** there is **no trained model, no training script, and no golden reference model**
anywhere. The owner does not know whether training was ever run. Do not spend any effort tuning
quantisation or matching RTL bit-exactly against these numbers — they are noise.

**Quantisation note for later:** peak |w| is only 965–4456 out of 32767, so only ~10 of 15
fractional bits are used and the top 4–5 bits are always sign extension. Per-layer scaling would
recover that precision, and **int8 with per-layer scales would very likely suffice**, halving
weight memory to ~1.5 Mbit.

---

## 6. ⚠ CRITICAL: the RTL implements a completely different design

The Verilog is **not an incomplete version of the model above**. It is a different architecture —
a per-neuron spike-routing design, not a convolution engine.

| | Python model | Current RTL |
|---|---|---|
| Computation | 6 conv layers, shared 3×3 kernels | 1 accumulator + per-source routing table |
| Convolution | yes | **none** — no sliding window, no line buffers, no channel accumulation |
| Neurons | 5 LIF layers, ~380k neurons | **one** `lif_neuron` instance |
| Input width | int16 | **8-bit** (`c_m_axis_mm2s_tdata_width = 8`) |
| Input size | 301,056 × int16 = 588 KB | 301,056 × int8 |
| Output | 2×224×224 int16 = 196 KB | **one 32-bit word** |
| Weights | 186,624 from DDR | none — `bram_dout` tied off |

The RTL needs to be **rebuilt as a line-buffered conv engine**, not repaired.
`spike_router`, `synapse_fifo`, `synapse_accumulator`, `frame_buffer`, `frame_assembler` and
`input_adapter` are all superseded by that design and should not be carried forward.

### 6.1 The headline RTL bug (explains the whole thing)

**The design synthesises with 0 errors and computes a constant zero.** Vivado proved it and deleted
the logic. From `VideoFrameInterpolation.runs/VFI2_top_axis_IP_0_0_synth_1/runme.log`:

| | RTL estimate | Final netlist |
|---|---|---|
| Frame buffer | `2352K Bit (301056 × 8)` | **gone** |
| Synapse FIFO | `384 Bit (16 × 24)` | **gone** |
| BRAM | `BRAMs: 280` | **0** |
| Total | — | 67 LUT, 48 FF, 1 DSP |

**Root cause:** `spike_router`'s weight-table interface is tied off in `top.v:128-129`
(`.bram_addr()`, `.bram_dout()`) and no weight memory exists in the project. The chain collapses:

```
bram_dout ≡ 0 → route_weight ≡ 0 → input_act × 0 = 0 → syn_current ≡ 0
  → LIF mem never reaches THRESH → spike ≡ 0 → out_value ≡ 0
```

Vivado then deleted the entire upstream datapath as unobservable. The project's own
`spike_routing` testbench already showed this: `ROUTE: dst=x weight=x` on every connection.

### 6.2 Full RTL defect list

**Width mismatches** (all emitted as *warnings*, so the build passed):

| Signal | Declared | Connected | Effect |
|---|---|---|---|
| `synapse_accumulator.weight` (`top.v:171`) | 8 | 16 | Q1.15 weight truncated to low 8 bits |
| `synapse_accumulator.syn_current` (`top.v:172`) | 24 | 16 | output truncated, saturation bypassed |
| `readout.pixel_value` (`top.v:202`) | 8 | 16 | **upper 8 bits of `out_value` undriven** |
| `spike_router.spike_src_id` (`top.v:124`) | 32 | 19 | zero-padded, unintended |

`W_WIDTH`/`ACC_WIDTH` are never overridden at instantiation — running on 8/24 defaults while
`top.v` assumes 16/16. `input_adapter` has 11 ports but only 8 connected: `channel_id`, `bit_id`,
`src_neuron_id` all dangling. `axi_pixel_adapter.sv` produces exactly the two missing inputs and
**is never instantiated anywhere**.

**AXI-Stream protocol violations:**
- TVALID deasserts after one cycle regardless of TREADY — illegal, beats are lost
- Data captured only when TREADY is already high — no skid buffer, drops on backpressure
- TLAST asserted every beat → every 32-bit word is its own packet
- `m_axis_tlast_reg` removed as unused → **S2MM never receives TLAST → DMA hangs**
- `s_axis_tlast` accepted then dropped (`unconnected or has no load`)
- `s_axis_tready` hardwired to `1'b1` — pipeline can never backpressure

**Structural / logic bugs:**
- `frame_done` is **double-driven** — both `fsm_global` and `frame_assembler` outputs on one wire
- `fsm_global` runs on default `PIXELS_PER_FRAME = 16`; the real frame is **301,056**
- FSM free-runs on `.start(1'b1)`, unrelated to the pixel stream; `timestep_en` is high 1 cycle in 4 → ≤25% of pixels processed
- `frame_buffer.rd_data` is registered but `pixel_in` isn't → compares `pixel_in[n]` vs `prev[n−1]`
- `input_adapter` diff overflows: range −255..+255 into an 8-bit signed
- Unit mismatch — `syn_current` is a raw integer, `THRESH` is Q1.15
- Accumulator saturation is **dead code**: `syn_current + mult_ext > MAX_ACC` wraps at 24 bits before the compare
- FIFO off-by-one: `rd_en` and `enable` share `~fifo_empty`, but `dout` is registered → always consumes the previous entry
- Router valid/data skew: `route_valid` combinational, `route_dst_id`/`weight` registered → data lags valid by a cycle
- `bram_addr` driven only during `READ`, forced to `'0` otherwise

### 6.3 Build state

| Item | Timestamp | Status |
|---|---|---|
| `runs/synth_1/` | — | **empty — never completed** |
| `runs/impl_1/` | — | **empty — never completed** |
| `top.bit` | 07-Feb 15:12 | **stale** — built from the old `VFI.bd` |
| `VFI_wrapper.xsa` | 07-Feb 16:27 | **stale** |
| VFI2 IP synth | 08-Feb 02:09 | current, IP-level only |

Top-level synthesis and implementation have **never completed** for `VFI2`.
`constrs_1/new/top.xdc` is **0 bytes**; the PYNQ-Z2 XDC has **0 uncommented lines**;
`allow_unconstrained_io.tcl` downgrades NSTD-1/UCIO-1 from error to warning.

**Block design connectivity is correct and is not the problem:**
PS7 → smartconnect → AXI-DMA (simple mode, `c_include_sg = 0`),
`M_AXIS_MM2S → top_axis_IP/s_axis` (8-bit), `top_axis_IP/m_axis → S_AXIS_S2MM` (32-bit).
Clocks and resets properly wired.

**Collateral damage:**
- `readout/readout.srcs/sources_1/new/readout.v` overwritten with an old `top.v` — that standalone project is broken. The good copy survives at `VideoFrameInterpolation.srcs/sources_1/imports/new/readout.v`
- `lif_neuron/` targets **Virtex-7 `xc7vx485t`**, not the Zynq; last synth died on a license error leaving stale `__synthesis_is_running__` / `.vivado.error.rst` lock files
- `frame_assembler`, `frame_buffer`, `input_adapter` have **never been simulated**

---

## 7. Host / Python defects

- `ctrl = overlay.snn_ctrl_0` — the BD instance is `top_axis_IP_0`. AttributeError on first run.
- **There is no AXI-Lite interface on the IP at all** — only `clk`, `rst_n`, `s_axis`, `m_axis`. Every `ctrl.write()` is impossible as designed.
- `load_weights()` is defined and **never called**.
- Only `lif1_stats` is loaded; lif2–5 ignored.
- `BIT_PATH = "/home/xilinx/overlays/snn.bit"`, actual file is `top.bit`.
- `overlays/top.bit` is **SHA-256 identical** to the stale VFI FPGA bitstream — it is the empty design.
- Weight loading is one MMIO write per value: 186,624 writes per frame. Must become a DMA'd blob.
- **`FLOW_SCALE` contradicts itself**: `1.0/32768` in `jupyter.ipynb`, `1.0` in `notebooks/postprocess.ipynb` with the comment "change to 1/32768 if PL outputs Q1.15". At `1/32768` maximum displacement is **±1 pixel**, which would produce nearly invisible interpolation. Must be resolved against the training convention — flow is probably normalised to image width, needing ×112 or ×224.
- Minor: `w0` is shadowed in `postprocess_and_write` (warped image, then reassigned to a width int). Works, but confusing.
- Warp sign convention (`map_x = x + dx`, backward warping) should be verified against the reference model once one exists.

---

## 8. Compute and memory analysis

**Per frame, per timestep, at 224×224, stride 1, `same` padding:**

| layer | MMAC | channels |
|---|---|---|
| conv1 | 86.7 | 6 → 32 |
| conv2 | 924.8 | 32 → 64 |
| conv3 | **3699.4** | 64 → 128 |
| conv4 | **3699.4** | 128 → 64 |
| conv5 | 924.8 | 64 → 32 |
| flow | 28.9 | 32 → 2 |
| **TOTAL** | **9.36 GMAC** | |

**conv3 + conv4 alone are 79% of the cost.**

Throughput on 220 DSPs at 100 MHz:

| | 100% DSP eff (fantasy) | 50% eff (realistic) |
|---|---|---|
| T=1 | 0.43 s/frame → 2.35 fps | 0.85 s → 1.17 fps |
| T=2 | 0.85 s → 1.17 fps | 1.70 s → 0.59 fps |
| T=4 | 1.70 s → 0.59 fps | 3.41 s → **0.29 fps** |

**Memory:**

| | |
|---|---|
| All weights | 2.99 Mbit = **61% of device BRAM** |
| Widest feature map (128×224×224 int16) | 102.8 Mbit = **21× the whole device** |
| 3-row line buffer @128ch | 1.38 Mbit — **feasible, this is the way** |

**Conclusions:** every layer must be streamed through line buffers, never materialised.
The network is too big for real-time at 224×224 — **scope down to 112×112 and/or cap channels at
64** for roughly 8×. Flow fields are smooth and survive downsampling; upsample the flow, not the
image.

---

## 9. Decisions already made

1. **Shared-kernel convolution, not per-source AER routing.** A per-source routing table (what
   `spike_router` implements) would need 301,056 × 9 × 24 bit = **65 Mbit**, 13× the device.
   Shared 3×3 conv kernels need **13.8 Kbit** for the same computation. This is settled — the
   trained model is already a conv net, so this is what to build.
2. **`CHANNELS = 6` is 2 frames × 3 RGB.** `frame_buffer` / `pixel_prev` in the RTL is redundant
   and should be deleted.
3. **Build the CNN version first, swap in the SNN later.** Reasoning in §10.

---

## 10. Strategy: why CNN first, SNN second

**The hardware is identical for both.** A line-buffered conv engine with a DSP array. The LIF is an
activation unit on the accumulator output, plus an outer timestep loop. Swapping ReLU → LIF later
touches one module and a loop; the conv engine, line buffers, weight memory, DMA and host code all
survive unchanged.

**Why the SNN is risky to start with:**
- Surrogate-gradient training is finicky and may not converge on optical flow
- `T` timesteps multiplies compute by `T`, when throughput is already 0.3–2 fps
- **Four-way debug ambiguity** — when output is wrong, is it the RTL, the quantisation, the
  training, or the fundamental approach? With a CNN it's a two-way question.

**Why the research angle is weaker than it appears:** SNNs earn their keep on *sparse,
event-driven* input. The credible work (Spike-FlowNet, EV-FlowNet lineage) uses **event cameras**,
where spikes come from the sensor. Feeding dense RGB frames means converting dense data into
spikes, paying full dense compute, multiplying by `T`, and getting all the training difficulty with
none of the sparsity benefit. Reviewers know this criticism. "SNN + dense RGB + well-trodden task"
is the weakest cell in that grid.

**The paper angle that actually works:** *"Is spiking worth it for dense vision on edge FPGAs?"*
Build both on the same conv engine, same board, same clock, same quantisation. Report accuracy,
throughput, resources and **measured power** for both. Very few people have measured this honestly
on identical silicon. That is a legitimate, modest, defensible contribution — **and it requires the
CNN as the control group.** Building the CNN first is not a detour from the paper; it is the first
half of the experiment.

**Practical warning:** the PYNQ-Z2 has no good onboard power instrumentation. An efficiency claim
with no measured power will be rejected. Decide early whether a shunt / current probe can be
attached to the board rail.

---

## 11. Plan

Realistically a **2–3 month project** at a serious pace.

**Step 1 — Rebuild the model and train it (CRITICAL PATH).**
Reconstruct in PyTorch from the shapes in §4.3, with **ReLU** activations. Train self-supervised:
predict flow from (frame0, frame1), warp both toward the midpoint, minimise reconstruction error
against the true middle frame. No ground-truth flow needed; trains on any video.
**This also becomes the golden reference model** — the thing that currently doesn't exist and
without which nothing can be verified.

**Step 2 — Scope down.** Move to 112×112 and/or cap channels at 64. Re-verify accuracy is still
acceptable. Do this *before* writing RTL, since it sets the DSP array size.

**Step 3 — Write the line-buffered conv engine.** Streaming convolution, 3-row line buffers, DSP
array, weights in BRAM. Verify **layer by layer** against the golden model's intermediate dumps.

**Step 4 — Fix the DMA path.** int16 stream width, correct TLAST at end-of-frame, proper TVALID
hold, input backpressure, weights loaded by DMA rather than 186k MMIO writes.

**Step 5 — Demo it.** Video in, visibly smoother video out.
**Stop here if time is short — this is the version that gets the job.**

**Step 6 — Swap ReLU → LIF.** Retrain with surrogate gradients, rerun the same measurements,
write the comparison paper.

### Housekeeping (cheap, do anytime)
- Delete stale `top.bit` / `VFI_wrapper.xsa` / `overlays/top.bit`; delete `VFI.bd`
- Restore the clobbered `readout/.../readout.v`; clear `lif_neuron/` lock files; retarget to `xc7z020`
- Promote Vivado messages `Synth 8-689` (width mismatch) and `8-7023` (port count) to **ERROR** so this class of bug can never pass silently again

### What NOT to carry forward
`spike_router`, `synapse_fifo`, `synapse_accumulator`, `frame_buffer`, `frame_assembler`,
`input_adapter`, `axi_pixel_adapter`, `readout`, `neuron_array`. All superseded by the conv engine.
Salvageable: `lif_neuron` (as the future activation unit), and the general AXI-Stream wrapper shape.

---

## 12. Open questions

1. **Was training ever run?** Owner doesn't know. Probably moot — the weights prove no useful
   training result exists, so it's a rebuild either way.
2. **Flow scale convention** — pixels, or normalised to image width? Determines `FLOW_SCALE`.
   Will be settled by whatever convention the new training uses.
3. **Padding** — assumed `same`, stride 1, no pooling. Inferred from weight shapes only; not verified.
4. **Training data** — is `videos/sample.mp4` the intended dataset, or just a test clip? Is there
   GPU access for training?
5. **Deadline** — unknown, and it should drive how hard to scope down in Step 2.

---

## 13. Notes for whoever picks this up

- **The verification story is the actual portfolio content.** A bit-exact reference model,
  per-layer intermediate dumps, and testbenches comparing RTL against them — that is what makes
  this look like professional engineering rather than a student project, and it is the only way
  it will ever get debugged. None of it exists yet. It is the highest-value thing to build after
  the conv engine.
- **Vivado will not stop you from building nothing.** This project synthesised clean, produced a
  bitstream, and computed a constant zero. Width mismatches and unconnected ports were all
  *warnings*. Treat the warning log as mandatory reading.
- **Don't trust the existing `.bit` or `.xsa`.** Both are stale and were built from a different,
  broken block design.
- Owner's context: wants this finished for job/internship applications; a paper is a bonus, not
  the objective. Advice should favour "working and verified" over "novel."
