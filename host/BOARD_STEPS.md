# VFI on the PYNQ-Z2: the commands, in order

Everything in this folder goes on the board. Assembled by `host/make_kit.py` from the repo. Re-run
that script after any rebuild, or the board silently runs an old bitstream.

**Bitstream:** built 2026-10-23 at 112x112, MAX_CH=128, N_OUT=24, FCLK 75 MHz. It adds `CFG[14]`, the
runtime ReLU-off bit for the flow layer.

- **Post-route timing:** met. WNS +0.014 ns, WHS +0.029 ns, 0 failing endpoints of 85,306.
- **DMA transfer length:** `SG_LENGTH_WIDTH` 26 on both DMAs.
- **Cell names:** `vfi`, `dma_fm`, `dma_wt`.

**Model:** `export_gpu/`, the trained model, with shifts 12, 12, 12, 14, 15, 18 and flow bias
`[3, -2]`. Its `golden/` holds every layer's expected output for one held-out frame pair.

## 0. One-time setup

(Unchanged from the previous kit's `BOARD_BRINGUP.md`.)

1. Flash PYNQ-Z2 **v3.1.1** to a 16 GB+ microSD.
2. Set the jumpers: **JP4 → SD**, and **JP5 → USB** (or REG with a 12 V supply).
3. Connect micro-USB and Ethernet, then power on. Wait for **DONE**, then for LD0–LD3 to flash.
4. Set the PC's Ethernet adapter to `192.168.2.1/24`.
5. Open <http://192.168.2.99:9090> (password `xilinx`), or `ssh xilinx@192.168.2.99`.

## 1. Copy the kit and check it

Copy this whole folder, **including the `export_gpu/` subfolder**, to
`/home/xilinx/jupyter_notebooks/vfi`. Use the network share `\\192.168.2.99\xilinx`, or Jupyter's
Upload button. In a Jupyter terminal (it runs as root). Over SSH as `xilinx`, run `sudo -i` first:
as a normal user, bring-up stage 1 fails with `RuntimeError: Could not open device with index '0'`
(XRT cannot open the PL), which looks like a broken overlay but is only permissions.

```bash
cd /home/xilinx/jupyter_notebooks/vfi
sha256sum -c SHA256SUMS.txt          # every line must say OK
python3 -c "import pynq, numpy, cv2; print(pynq.__version__, numpy.__version__, cv2.__version__)"
```

## 2. Bring-up: is the hardware alive and correct?

```bash
python3 bringup.py
```

Expect `PASS  all 6 stages`. The stages:

- **Stage 1** also prints `FCLK0`, read from `pynq.Clocks` after the overlay loads. **Write this
  number down.** PYNQ applies only the `.hwh`'s divisors (4 x 4) to the board's own IO PLL, so it is
  75 MHz only on a 1200 MHz IO PLL; 62.5 MHz (1000 MHz IO PLL) is timing-safe, just slower. Above
  75 MHz stage 1 fails: timing was closed at 75. Never set `Clocks.fclk0_mhz` by hand.
- **Stage 2** also writes `CFG[14]` and reads it back. If it fails with "CFG[14] (linear) reads
  back 0", the board is running a **pre-2026-10-23 bitstream**: copy `vfi.bit` and `vfi.hwh` again.
- **Stage 3 passes but 4 fails:** the old 16 KB DMA cap. The wrong `.bit` again.
- **Stage 5** requires all 301,056 values to match `vfi_ref.py` exactly.

## 3. conv1 with the trained weights

```bash
python3 conv1_board.py
```

conv1 runs in two hardware passes. It is checked against `export_gpu/golden/conv1.bin`, and against a
dense random input whose golden is computed on the board. Expect
`PASS  bit-exact on both inputs`.

## 4. The whole network against golden

```bash
python3 network_board.py
```

All six layers run chained: 17 hardware passes, with the flow layer's ReLU switched off through
CFG[14]. Each layer must match its `golden/<layer>.bin` exactly, and the flow must match
`golden/flow.bin` after the host-side bias. Expect `PASS  every layer bit-exact`. The same run
already passed in xsim (`sim/tb_network_golden.sv`).

On a failure, the script names the first layer that differs:

| First failing layer | Look at |
|---|---|
| conv1 | it passed in step 3, so suspect the copy: re-run `sha256sum -c` |
| conv2, conv3 or conv4, in some channels only | pass stitching across 3–6 passes in `vfi_driver.run_layer` |
| flow, with no negative values returned | the ReLU is still on: an old bitstream |

## 5. Timing

```bash
python3 perf.py
```

This reports one conv1 pass against the PL floor (1.00 ms at 75 MHz), then the full network:
milliseconds per frame pair, per-layer time, and the PL floor per layer. Floors use the measured
`FCLK0` (first line, `PL clock`); if it says `ASSUMED`, the clock couldn't be read and 75 MHz was used. The full-network number is the demo's
throughput.

## 6. The demo

```bash
python3 demo.py sample.mp4 --frames 61 --out sample_2x.mp4 --png side_by_side.png
```

`--frames 61` takes the first 61 frames (60 pairs). Drop it to process the whole clip (1,415 frames)
— check `perf.py`'s per-pair time first. The script writes `sample_2x.mp4` at 2x fps, plus a PNG
whose four columns are: frame t | 50/50 blend | model | true frame t+1.

Copy `sample_2x.mp4` and `side_by_side.png` back to the PC to watch them.

**Expect the model to look worse than the blend on this clip.** It was trained only on Moving MNIST
(grey digits on black) and invents motion on real footage — measured in simulation, see `docs/STATUS.md`
step 4. The board computes exactly what the simulation computes, so this is a property of the
weights, not the hardware. A model trained with `train_vfi.py --videos` on real clips is what fixes
it.

## If anything hangs

A hung DMA does not clear itself. Press Ctrl-C and re-run the script from the top: `Overlay()`
reprograms the PL and resets everything.
