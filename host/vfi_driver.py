"""
Host driver for the VFI convolution IP on a PYNQ-Z2.

The PL is passive: it transforms data that arrives on its streams and has no way
to start itself, fetch weights, or find an image. Everything below is what makes
it do work.

One `vfi_top` instance computes ONE convolution layer pass:

    write CFG (channels, shift) -> DMA kernels -> DMA feature map in
                                -> DMA feature map out -> read STATUS

The hardware has N_OUT MAC units (24, the DSP ceiling), so a layer with more
than N_OUT output channels runs as ceil(C_out / N_OUT) passes, and this module
stitches the partial results together. conv3 (64 -> 128) is six passes.

Tensor layout
-------------
Feature maps are int16 Q1.15 in (row, col, channel) order, channel MINOR --
identical on the input and output streams, which is what lets one layer's result
feed the next with no reformatting. In numpy that is simply an (H, W, C) array
in C order.

Kernels are (out, in, kh, kw) flattened, which is exactly the layout of the
conv*_weight_q15.bin files, so a weight file DMAs as-is with no reordering.

Q1.15 means a raw int16 v represents v / 2**15, i.e. the range [-1, 1).

What this cannot do yet
-----------------------
The per-layer `shift` values are a property of the trained model, which does not
exist. Use `sat_sticky` in the returned status to tune them: a layer that clips
has too small a shift, one that returns near-zero everywhere has too large a
one. Running against the untrained weights in snnvfi/weights/ exercises every
code path here and proves the plumbing, but the numbers are meaningless.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------- registers
# Must match rtl/axis/axil_regs.sv.
REG_ID = 0x00      # R   magic
REG_CFG = 0x04     # RW  [7:0] channels, [13:8] shift
REG_CTRL = 0x08    # W   [0] sat_clear, [1] soft_reset  (self-clearing)
REG_STATUS = 0x0C  # R   [0] weights_loaded, [1] sat_sticky
REG_FRAMES = 0x10  # R   completed frames since reset

ID_MAGIC = 0x56464931  # "VFI1"

CTRL_SAT_CLEAR = 1 << 0
CTRL_SOFT_RESET = 1 << 1

STATUS_WEIGHTS_LOADED = 1 << 0
STATUS_SAT_STICKY = 1 << 1

TAPS = 9  # 3x3


class VfiError(RuntimeError):
    pass


class VfiLayer:
    """One convolution layer's parameters, as the network defines it."""

    def __init__(self, ch_in: int, ch_out: int, shift: int, name: str = ""):
        self.ch_in = int(ch_in)
        self.ch_out = int(ch_out)
        self.shift = int(shift)
        self.name = name or f"{ch_in}->{ch_out}"


class Vfi:
    """
    Driver for one vfi_top instance plus its two DMAs.

    Instantiate against a loaded PYNQ Overlay:

        from pynq import Overlay
        ol  = Overlay("vfi.bit")
        vfi = Vfi(ol.vfi, ol.dma_fm, ol.dma_wt, img_w=112, img_h=112, n_out=24)

    The instance names are ol.vfi, ol.dma_fm and ol.dma_wt -- verified against the
    generated vfi_bd.hwh, and set by the cell names in syn/build.tcl. It is NOT
    ol.vfi_0: this docstring said so until the .hwh was checked, and following it
    gives an AttributeError on the first line of bring-up. That is the same class
    of mistake the ID-register check below exists to catch.
    """

    def __init__(self, ip, dma_fm, dma_wt, img_w=112, img_h=112, n_out=24,
                 max_ch=128, check_id=True):
        self.ip = ip
        self.dma_fm = dma_fm
        self.dma_wt = dma_wt
        self.img_w = int(img_w)
        self.img_h = int(img_h)
        self.n_out = int(n_out)
        self.max_ch = int(max_ch)

        if check_id:
            got = self.ip.read(REG_ID)
            if got != ID_MAGIC:
                raise VfiError(
                    f"ID register reads 0x{got:08X}, expected 0x{ID_MAGIC:08X}. "
                    "Wrong bitstream, or the overlay's IP instance name does not "
                    "refer to this core. This check exists because the previous "
                    "host code silently addressed the wrong instance."
                )

    # ------------------------------------------------------------ registers
    def status(self) -> dict:
        s = self.ip.read(REG_STATUS)
        return {
            "weights_loaded": bool(s & STATUS_WEIGHTS_LOADED),
            "saturated": bool(s & STATUS_SAT_STICKY),
            "raw": s,
        }

    def frames(self) -> int:
        return self.ip.read(REG_FRAMES)

    def clear_saturation(self) -> None:
        self.ip.write(REG_CTRL, CTRL_SAT_CLEAR)

    def soft_reset(self) -> None:
        self.ip.write(REG_CTRL, CTRL_SOFT_RESET)

    def _configure(self, channels: int, shift: int) -> None:
        if not 1 <= channels <= self.max_ch:
            raise VfiError(f"channels={channels} outside 1..{self.max_ch}")
        if not 0 <= shift <= 63:
            raise VfiError(f"shift={shift} outside 0..63")
        self.ip.write(REG_CFG, ((shift & 0x3F) << 8) | (channels & 0xFF))
        back = self.ip.read(REG_CFG)
        if (back & 0xFF) != channels or ((back >> 8) & 0x3F) != shift:
            raise VfiError(f"CFG readback 0x{back:04X} does not match what was written")

    # ------------------------------------------------------------ transfers
    def _send(self, dma_channel, data: np.ndarray) -> None:
        from pynq import allocate
        buf = allocate(shape=(data.size,), dtype=np.int16)
        buf[:] = data.reshape(-1)
        dma_channel.transfer(buf)
        dma_channel.wait()
        buf.freebuffer()

    def _run_pass(self, fmap: np.ndarray, kernels: np.ndarray, shift: int) -> np.ndarray:
        """
        One hardware pass: fmap (H, W, C_in) int16 -> (H, W, n_out) int16.

        kernels is (n_out, C_in, 3, 3) int16. Both are exactly what the streams
        expect once flattened in C order, which is why there is no reordering
        anywhere in this function.
        """
        from pynq import allocate

        h, w, c_in = fmap.shape
        if (h, w) != (self.img_h, self.img_w):
            raise VfiError(
                f"feature map is {h}x{w}, but the bitstream is built for "
                f"{self.img_h}x{self.img_w}. IMG_W/IMG_H are synthesis-time "
                "parameters, not runtime configuration."
            )
        if kernels.shape != (self.n_out, c_in, 3, 3):
            raise VfiError(f"kernels {kernels.shape}, expected {(self.n_out, c_in, 3, 3)}")

        self._configure(c_in, shift)
        self.clear_saturation()

        # Kernels first: the layer must not run before weights_loaded.
        self._send(self.dma_wt.sendchannel, kernels.astype(np.int16))
        st = self.status()
        if not st["weights_loaded"]:
            raise VfiError(
                "weights_loaded did not assert after the kernel transfer. "
                f"Expected {self.n_out * c_in * TAPS} beats."
            )

        out = allocate(shape=(h * w * self.n_out,), dtype=np.int16)
        self.dma_fm.recvchannel.transfer(out)
        self._send(self.dma_fm.sendchannel, fmap.astype(np.int16))
        self.dma_fm.recvchannel.wait()

        result = np.array(out).reshape(h, w, self.n_out)
        out.freebuffer()
        return result

    # ------------------------------------------------------------ layers
    def run_layer(self, fmap: np.ndarray, kernels: np.ndarray, layer: VfiLayer):
        """
        A full layer, in as many passes as the MAC array requires.

        fmap    (H, W, ch_in)               int16 Q1.15
        kernels (ch_out, ch_in, 3, 3)       int16 Q1.15
        returns (H, W, ch_out) int16, and a per-pass status list

        Output channels are produced n_out at a time. A layer whose ch_out is
        not a multiple of n_out has its final pass zero-padded and the surplus
        channels discarded -- the hardware always emits exactly n_out.
        """
        h, w, c_in = fmap.shape
        if c_in != layer.ch_in:
            raise VfiError(f"feature map has {c_in} channels, layer expects {layer.ch_in}")
        if kernels.shape != (layer.ch_out, layer.ch_in, 3, 3):
            raise VfiError(
                f"kernels {kernels.shape}, expected {(layer.ch_out, layer.ch_in, 3, 3)}")

        n_passes = -(-layer.ch_out // self.n_out)  # ceil
        out = np.zeros((h, w, layer.ch_out), dtype=np.int16)
        statuses = []

        for p in range(n_passes):
            lo = p * self.n_out
            hi = min(lo + self.n_out, layer.ch_out)

            k = np.zeros((self.n_out, layer.ch_in, 3, 3), dtype=np.int16)
            k[: hi - lo] = kernels[lo:hi]

            part = self._run_pass(fmap, k, layer.shift)
            out[:, :, lo:hi] = part[:, :, : hi - lo]

            st = self.status()
            st["pass"] = p
            st["channels"] = (lo, hi)
            statuses.append(st)

        return out, statuses

    def run_network(self, fmap: np.ndarray, weights: list, layers: list):
        """
        Run every layer in sequence, feeding each result into the next.

        This is the sequencer. It lives here rather than in RTL because the
        intermediate feature maps have to go through DDR anyway -- a 112x112x64
        map is 12.8 Mbit and cannot stay on chip -- so the loop costs nothing to
        run from the PS and is far easier to change.
        """
        if len(weights) != len(layers):
            raise VfiError("weights and layers must be the same length")

        x = fmap
        report = []
        for wgt, layer in zip(weights, layers):
            x, st = self.run_layer(x, wgt, layer)
            clipped = any(s["saturated"] for s in st)
            report.append({"layer": layer.name, "passes": len(st), "saturated": clipped})
            if clipped:
                print(f"  WARNING: layer {layer.name} saturated -- shift={layer.shift} "
                      "is too small for this layer's dynamic range")
        return x, report


# ---------------------------------------------------------------- weights
def load_weight_file(path: str, ch_out: int, ch_in: int) -> np.ndarray:
    """
    Read a conv*_weight_q15.bin into (ch_out, ch_in, 3, 3) int16.

    The file layout is flattened (out, in, kh, kw), which is byte-for-byte the
    order weight_loader expects on the stream, so this reshape is the only step
    and no transpose is ever correct here.
    """
    raw = np.fromfile(path, dtype=np.int16)
    expect = ch_out * ch_in * TAPS
    if raw.size != expect:
        raise VfiError(
            f"{path}: {raw.size} int16 values, expected {expect} "
            f"for {ch_out}x{ch_in}x3x3"
        )
    return raw.reshape(ch_out, ch_in, 3, 3)


def quantise_q15(x: np.ndarray) -> np.ndarray:
    """float in [-1, 1) -> int16 Q1.15, saturating."""
    v = np.rint(np.asarray(x, dtype=np.float64) * 32768.0)
    return np.clip(v, -32768, 32767).astype(np.int16)


def dequantise_q15(x: np.ndarray) -> np.ndarray:
    """int16 Q1.15 -> float."""
    return np.asarray(x, dtype=np.float64) / 32768.0
