# train/export_gpu — the trained model the board runs

Full GPU run of `train/train_vfi.py`, exported 2026-10-22 and imported from `export.zip` on
2026-10-23. It replaces `train/export_cpu3k` (the interim 3k-step CPU run). Not retrained here.

Held-out, quantised exactly as the board computes: **21.49 dB** against 14.88 dB for a 50/50 blend
(13.26 dB copy-frame-0). It scores 26.35 dB at the 112x112 hardware geometry.

All `.bin` files are raw little-endian int16.

| File | Contents |
|---|---|
| `vfi_model.json` | layer list, **per-layer shifts**, flow bias, flow scale, held-out scores. The source of truth for shifts. |
| `<layer>_weight_q15.bin` | kernels `(c_out, c_in, 3, 3)` flattened, the order `weight_loader.sv` takes. DMA as-is, no transpose. Layers: `conv1`..`conv5`, `flow`. |
| `<layer>_weight_float.npy` | the float weights before quantisation, for reference only |
| `flow_bias_q15.bin` | 2 values, `[3, -2]`, **added on the PS** after the flow layer (the RTL has no bias) |
| `golden_input.bin` | one held-out frame pair, 112x112x6 `(row, col, ch)`, frame0 RGB ++ frame2 RGB, [0,1] → Q1.15 |
| `golden_frames.npy` | float32 `(3, 3, 112, 112)`: frame0, frame1 (the true middle frame), frame2 |
| `golden_flow_torch.bin` | the quantised torch model's flow for that pair, 112x112x2. vfi_ref is within 3 LSB of it (rounding mode only). |
| `golden/input.bin` | same bytes as `golden_input.bin` |
| `golden/<layer>.bin` | vfi_ref's output of each layer for that input, `(row, col, ch)` beat order; `flow.bin` is **after** the bias. Diff RTL and board against these. |
| `vfi_model.json.bak` | metadata of an earlier checkpoint of the same run (bias `[1, 0]`). Not used. |

Shifts: conv1 12, conv2 12, conv3 12, conv4 14, conv5 15, flow 18. Activation: ReLU on conv1–5,
linear on flow. Flow in pixels = `value / 32768 * 32`; ch0 = x, ch1 = y.

Two quirks, neither needing action:

- **52 conv5 weights (0.28%) sit at -32768.** The float conv5 weights reach -1.025, and the exported
  shift is 15 rather than the 14 `weight_exponent` would choose, so those 52 clip. The quantised
  model that scored 21.49 dB used the same clipped integers, so the score describes what the board
  computes.
- `weight_absmax_int` for conv5 reads 31868 because `np.abs` of int16 -32768 overflows. The
  exporter is fixed (`train_vfi.py` casts to int32 first); this file was not regenerated.
