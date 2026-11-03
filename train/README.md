# train/ — training the flow CNN the PL actually runs

`train_vfi.py` trains the 6-layer network; `verify_export.py` checks the exported integers against
`model/vfi_ref.py` (numpy only). Both replace `Videoframeinterplotation.py` / `evaluate_only.py`
from `videoframeinterplotation.zip`.

## Audit of the original model (2026-10-11)

Reported: MSE 0.009501, PSNR 20.29 dB. Reproduced from `vfi_weights_dma.bin` on 300 random
triplets: **20.45 dB**, so the number is real. The problem is what it measures.

| Same 300 triplets, 112x112 | PSNR | |
|---|---|---|
| copy frame0 | 14.21 dB | |
| 50/50 blend of f0 and f2, **no model** | 15.97 dB | the bar any model must clear |
| original model, float, as trained | 20.45 dB | what was reported |
| original model, **without bias** (what the RTL computes) | **16.74 dB** | +0.8 dB over no model |
| original model, Q1.15 activation saturation | 20.36 dB | small loss |
| digit pixels only, original model | 11.88 dB | the background inflates the full-frame figure |

Defects, most serious first:

1. **Biases in conv1–conv5; the RTL has none.** Up to 0.40 in Q1.15. Removing them — which is what
   the board does — takes the model from 20.45 to 16.74 dB. The deployed model would be almost no
   better than averaging the two frames.
2. **Trained and evaluated on the same data.** All 10,000 sequences were used for both. The reported
   figure is a training-set number.
3. **Weights clipped at export.** conv3 and flow hit exactly 32767: the float weights exceeded the
   Q1.15 range and `to_q15` clamped them silently. A per-layer power-of-two scale, which the
   hardware's per-layer `shift` already provides, removes the clamp.
4. **Activations exceed Q1.15** — up to 6.4 in conv4 — and the MAC saturates at 1.0.
5. **`tanh` on the output, which nothing implements**, and golden dumps taken after it, from a
   float model with biases, on a synthetic ramp. They cannot match the RTL.
6. **Receptive field vs motion.** Six 3x3 layers see ±6 px. Upsampling 64x64 Moving MNIST to 112
   stretches f0→f2 motion to 12.4 px median (measured), right at that limit. At native 64x64 it is
   7.1 px.
7. Full-frame PSNR on mostly-black frames flatters any model; the same flow padded to 112x112 scores
   ~5 dB higher purely from more black border. Compare against the blend baseline, not in absolute dB.
8. Moving MNIST is grey digits on black. A model trained only on it will not interpolate
   `sample.mp4`. For the real-video demo, train with `--videos`.

## What `train_vfi.py` does instead

- conv1–5 with no bias; flow layer bias added after requantisation, as `vfi_ref.add_bias` does.
- Quantisation-aware training: int16 weights with per-layer power-of-two scale (→ the `shift`),
  activations rounded and clamped to Q1.15 after every ReLU, flow output saturated, no tanh.
- Flow in pixels (`FLOW_PX = 32` px at Q1.15 full scale), ch0 = x, ch1 = y — what `cv2.remap` wants.
- Moving MNIST at native 64x64. With no bias anywhere, a 64x64 frame zero-padded into the 112x112
  hardware frame gives the same flow over the digits.
- 10% of sequences (or videos) held out; per-frame PSNR reported against both baselines.
- Exports `conv1..5_weight_q15.bin`, `flow_weight_q15.bin`, `flow_bias_q15.bin` and `vfi_model.json`
  (shifts, flow scale, held-out scores). Layout `(out, in, 3, 3)`, the order `weight_loader.sv` takes.

`verify_export.py` runs `vfi_ref.forward` on a held-out pair and requires the flow to match the
quantised torch model (measured: max 1 LSB, from torch's half-to-even rounding vs the MAC's
half-up), checks saturation, and writes `golden/<layer>.bin` for RTL and board comparison.

## Measured result (CPU, 3,000 steps ≈ 0.6 epoch, `export_cpu3k/`)

Held-out sequences, quantised exactly as the board computes, 64x64:

| | PSNR | gain over 50/50 blend |
|---|---|---|
| 50/50 blend (no model) | 14.78 dB | — |
| **fixed model, quantised, no hidden bias** | **20.25 dB** | **+5.5 dB** |
| original model, float (not deployable) | 20.45 dB¹ | +4.5 dB |
| original model as the RTL would run it | 16.74 dB¹ | +0.8 dB |

¹ measured on the original's own 112x112-upsampled data, where the blend scores 15.97 dB, so
compare the gain column. `verify_export.py` on this export: vfi_ref matches torch to max 1 LSB,
worst saturation 0.37% (conv1), shifts `{conv1 16, conv2 15, conv3 16, conv4 16, conv5 16, flow 17}`.
This is an interim run; a full GPU run should do clearly better.

## Running

```bash
# on a GPU machine (pip install torch numpy; opencv-python for --videos)
python train_vfi.py --mnist mnist_test_seq.npy --out run_mnist          # ~30 epochs
python train_vfi.py --videos clips/ --out run_video                      # real footage
python verify_export.py run_mnist/export                                 # numpy only
```

Then use `vfi_model.json`'s `shifts` for each `VfiLayer`, add `flow_bias_q15` to the returned flow,
and convert to pixels as `q15 / 32768 * flow_px_per_unit` before `cv2.remap`.

## Real-video training data (2026-10-23)

The MNIST-trained export does not transfer to real footage. On `sample.mp4` it loses to the 50/50
blend on 15 of 15 triplets (`docs/STATUS.md`, step 4). It needs real video with real motion.

### Recommended: UCF101, filtered by `pick_clips.py`

- **Source:** <https://www.crcv.ucf.edu/data/UCF101/UCF101.rar>, 6.93 GB, a RAR of 13,320 `.avi`
  clips. The server's certificate chain is incomplete: download in a browser, or with
  `curl -k -O <url>`. Extract with 7-Zip.
- **Why it fits:**
  - real human motion, 101 action classes;
  - short single-shot clips, so few scene cuts;
  - 320x240 at 25 fps, so after the 112x112 resize typical motion between t and t+2 is a few
    pixels — inside the six-layer ±6 px receptive field (item 6 above);
  - plain video files, so `--videos` takes them with no new code.
- **Why it needs filtering:**
  - some clips barely move and teach nothing (like `sample.mp4`), and some move faster than the
    receptive field;
  - clips in one UCF group (`_gNN_`) share the actor and the room, so a random by-file split
    would score validation on near-duplicates of the training data.

```bash
python pick_clips.py UCF-101/ --out clips/          # measures every clip, ~minutes
python train_vfi.py --videos clips/train --out run_video
python verify_export.py run_video/export
```

### What `pick_clips.py` does

It measures each clip at 112x112, exactly as training resizes:

- **Motion:** the 90th-percentile Farneback flow between t and t+2 must be in [0.75, 6] px.
- **Too-fast motion:** caught separately by a warp-residual check. At 112x112 every flow
  estimator under-reports motion beyond ~10.5 px (14 px reads as ~5), so the number alone would
  let fast clips through.
- **Scene cuts:** by hue/value histogram distance between neighbouring frames.

Then it selects:

- **one clip per (class, group)**, round-robin over classes, up to `--max-frames` (default 100k
  frames, about 3.7 GB of RAM in `train_vfi.py`);
- **demo candidates** into `clips/demo/`, from groups never used in training, with clear 2–5 px
  motion. Use one for `host/demo.py` instead of `sample.mp4`.

It writes `clips/picks.csv` with every clip's numbers and verdict.

`test_pick_clips.py` is the known-answer test: synthetic clips with exact motion (0–14 px), a
scene cut, and same-group duplicates. All 20 checks pass, and motion is measured to within 0.05 px.

### Alternatives

- **Your own footage:** a phone at 25–30 fps, steady, landscape, with a few people or cars moving
  at walking speed. Put the files in one folder; `pick_clips.py` measures and filters them the same
  way, treating each file as its own group. You need at least 2 files.
- **Vimeo-90K triplets:** the standard VFI training set, and higher quality. But it ships as PNG
  frame triplets rather than videos, so `train_vfi.py` would need a loader for it. Mirrors list the
  archive at about 159 GB.
