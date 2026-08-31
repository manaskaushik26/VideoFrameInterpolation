"""
vfi_ref -- bit-exact software reference for the VFI convolution datapath.

This is the golden model PROJECT_CONTEXT.md section 13 calls the highest-value thing to
build after the conv engine: the thing that makes "is the RTL right?" a decidable
question instead of an opinion. It reproduces the hardware's arithmetic exactly --
same accumulator width, same rounding, same saturation, same activation -- so RTL
output can be compared against it value for value rather than eyeballed.

It needs numpy and nothing else. It is integer arithmetic end to end: no torch, no
cv2, no board. Training is a separate, still-blocked problem (see CLAUDE.md).

What it mirrors
---------------
rtl/conv/conv_mac_3x3.sv and rtl/conv/act_unit.sv, as of the 4-stage MAC:

    acc     = sum over (c, kr, kc) of  img[y+kr-1][x+kc-1][c] * kern[o][c][kr][kc]
    rsum    = acc + (0 if shift == 0 else 1 << (shift - 1))     # round half up
    rounded = rsum >> shift                                      # arithmetic
    out     = clamp(rounded, -32768, 32767)                      # saturation flagged
    out     = relu(out) if ACT_MODE == 1 else out

Every one of those steps is load-bearing, and the superseded design got three wrong:

  - Saturation is evaluated at accumulator width, BEFORE narrowing. synapse.v
    compared the sum against MAX_ACC at the accumulator's own width, so it wrapped
    before the comparison could ever be true and the clamp was unreachable code.
  - Everything crossing a boundary is Q1.15. The old syn_current was a raw integer
    while the LIF threshold was Q1.15, so the two were never comparable.
  - Rounding is half-up (add 1 << (shift-1), then arithmetic-shift), not truncation.
    Truncation biases every layer toward zero and compounds across six of them.

Conventions, all read off the RTL rather than assumed
-----------------------------------------------------
Cross-correlation, not flipped convolution: tap (kr, kc) reads input pixel
(y+kr-1, x+kc-1). This matches line_buffer_3x3's window layout, which is row-major
with index kr*3 + kc and [1][1] as the centre tap.

`same` padding with zeros, stride 1. Out-of-image taps read zero.

Tensor layout is (row, col, channel), channel-minor -- identical on a layer's input
and output, which is what lets conv_stage instances chain with no glue and what the
DMA path carries to and from DDR. Kernels are (out, in, kh, kw), which is the
flattened layout of conv*_weight_q15.bin and the beat order weight_loader.sv
expects. Do not transpose either one.

Known gap: the RTL has no bias
------------------------------
The flow layer has a bias (flow_bias_q15.bin), and act_unit.sv's own header refers to
it, but no RTL implements one -- conv_mac_3x3's `round_bias` is the rounding constant
and is unrelated. Until that is added, bias must be applied host-side. See add_bias().

Speed
-----
Deliberately written for obviousness over speed: int64 matmul has no BLAS path, so a
full 112x112 forward pass takes minutes. That is the right trade for a reference whose
only job is to be correct. Use small shapes in tests.
"""

import numpy as np

DATA_W   = 16
WEIGHT_W = 16
ACC_W    = 48                       # conv_mac_3x3 default

OUT_MAX = (1 << (DATA_W - 1)) - 1   #  32767
OUT_MIN = -(1 << (DATA_W - 1))      # -32768
ACC_MAX = (1 << (ACC_W - 1)) - 1
ACC_MIN = -(1 << (ACC_W - 1))

ACT_LINEAR = 0                      # act_unit MODE 0 -- flow layer, no activation
ACT_RELU   = 1                      # act_unit MODE 1 -- the CNN activation

N_OUT_MAX = 24                      # measured DSP ceiling: 24 * 9 = 216 of 220


# ---------------------------------------------------------------- fixed point

def to_q15(x):
    """Float in [-1, 1) -> Q1.15 int16, saturating, round half away from zero."""
    a = np.asarray(x, dtype=np.float64)
    v = np.where(a >= 0, np.floor(a * 32768.0 + 0.5), np.ceil(a * 32768.0 - 0.5))
    return np.clip(v, OUT_MIN, OUT_MAX).astype(np.int16)


def from_q15(v):
    """Q1.15 int16 -> float."""
    return np.asarray(v, dtype=np.int64).astype(np.float64) / 32768.0


# ---------------------------------------------------------------- convolution

def conv3x3_acc(img, kern):
    """3x3 cross-correlation, `same` zero padding, summed over input channels.

    img  (H, W, C_in)         int16-valued, Q1.15
    kern (C_out, C_in, 3, 3)  int16-valued, Q1.15
    ->   (H, W, C_out)        int64 accumulator, Q2.30, exact (not yet rounded)

    Exact in int64: the largest possible |acc| is 9 * C_in * 32768^2, which is
    1.24e12 at C_in=128 -- seven orders of magnitude inside int64's range.
    """
    img = np.asarray(img)
    kern = np.asarray(kern)
    if img.ndim != 3:
        raise ValueError("img must be (H, W, C_in), got %r" % (img.shape,))
    if kern.ndim != 4 or kern.shape[2:] != (3, 3):
        raise ValueError("kern must be (C_out, C_in, 3, 3), got %r" % (kern.shape,))

    H, W, c_in = img.shape
    c_out = kern.shape[0]
    if kern.shape[1] != c_in:
        raise ValueError("kernel C_in %d != image C_in %d" % (kern.shape[1], c_in))

    pad = np.zeros((H + 2, W + 2, c_in), dtype=np.int64)
    pad[1:H + 1, 1:W + 1, :] = img.astype(np.int64)

    k = kern.astype(np.int64)
    acc = np.zeros((H, W, c_out), dtype=np.int64)
    for kr in range(3):
        for kc in range(3):
            patch = pad[kr:kr + H, kc:kc + W, :].reshape(-1, c_in)
            acc += (patch @ k[:, :, kr, kc].T).reshape(H, W, c_out)
    return acc


def requantise(acc, shift):
    """Q2.30 accumulator -> Q1.15: round half up, then saturate.

    Mirrors conv_mac_3x3's rounding-add and shift/saturate stages, including the fact
    that both happen at accumulator width before any narrowing.

    -> (out int16 (H, W, C_out), sat bool (H, W, C_out))
    """
    acc = np.asarray(acc, dtype=np.int64)
    shift = int(shift)
    if not 0 <= shift < (1 << 6):
        raise ValueError("shift %d outside the SHIFT_W=6 range 0..63" % shift)

    rsum = acc if shift == 0 else acc + (np.int64(1) << (shift - 1))

    # The RTL holds rsum in a signed ACC_W register. Real shapes cannot come near
    # this, but a caller with an absurd channel count would otherwise diverge from
    # the hardware silently, which is the whole failure mode this model exists to
    # prevent.
    if rsum.size and (rsum.max() > ACC_MAX or rsum.min() < ACC_MIN):
        raise OverflowError("accumulator exceeds ACC_W=%d; the RTL would wrap" % ACC_W)

    rounded = rsum >> shift
    sat = (rounded > OUT_MAX) | (rounded < OUT_MIN)
    out = np.clip(rounded, OUT_MIN, OUT_MAX).astype(np.int16)
    return out, sat


def activate(x, mode):
    """act_unit: MODE 0 passthrough, MODE 1 ReLU.

    Applied after the clamp, exactly as conv_stage.sv wires it
    (conv_layer -> act_unit -> axis_serializer).
    """
    x = np.asarray(x, dtype=np.int16)
    if mode == ACT_LINEAR:
        return x
    if mode == ACT_RELU:
        return np.where(x < 0, np.int16(0), x).astype(np.int16)
    raise ValueError("ACT_MODE %r not implemented (2 = LIF is reserved)" % (mode,))


def conv_stage(img, kern, shift, act_mode=ACT_RELU):
    """One complete layer: conv_layer -> act_unit, matching rtl/conv/conv_stage.sv.

    -> (out int16 (H, W, C_out), info dict)

    info["sat_sticky"] is the module's sticky saturation flag -- the diagnostic for a
    wrong shift. sat_count and sat_frac are finer-grained than the hardware exposes,
    and are what you actually tune a shift with.
    """
    acc = conv3x3_acc(img, kern)
    out, sat = requantise(acc, shift)
    out = activate(out, act_mode)
    n = int(sat.sum())
    return out, {
        "sat_sticky": bool(n),
        "sat_count":  n,
        "sat_frac":   (n / sat.size) if sat.size else 0.0,
        "acc_absmax": int(np.abs(acc).max()) if acc.size else 0,
    }


# ------------------------------------------------------------ bias (host-side)

def add_bias(x, bias):
    """Add a per-output-channel Q1.15 bias, saturating.

    THE RTL DOES NOT DO THIS. There is no bias anywhere in rtl/ -- conv_mac_3x3's
    `round_bias` is the rounding constant, not a layer bias. So for the flow layer the
    PS must add it after the DMA returns the feature map, which is cheap (2 channels)
    and needs no change to a datapath that has already closed timing.

    Applied in Q1.15 after requantisation rather than in the accumulator domain. That
    is a deliberate difference from what a PyTorch conv does and costs at most 1 LSB on
    the final layer. If the trained model turns out to need the bias inside the
    accumulator, that is an RTL change, not a host one.
    """
    x = np.asarray(x, dtype=np.int64)
    b = np.asarray(bias, dtype=np.int64)
    if b.ndim != 1 or b.shape[0] != x.shape[-1]:
        raise ValueError("bias must be one value per output channel (%d), got %r"
                         % (x.shape[-1], b.shape))
    return np.clip(x + b, OUT_MIN, OUT_MAX).astype(np.int16)


# ------------------------------------------------------------ pass splitting

def output_passes(c_out, n_out=N_OUT_MAX):
    """Output-channel ranges for a layer run as ceil(c_out/n_out) hardware passes.

    The MAC array is N_OUT=24 wide (216 of 220 DSPs), so conv3's 128 output channels
    take six passes. The sequencer in host/vfi_driver.py stitches the partials; this
    mirrors the split so a reference dump can be checked per pass as well as whole.
    """
    if c_out <= 0 or n_out <= 0:
        raise ValueError("c_out and n_out must be positive")
    return [(s, min(s + n_out, c_out)) for s in range(0, c_out, n_out)]


# ---------------------------------------------------------------- the network

# (name, c_in, c_out, act_mode) -- shapes from PROJECT_CONTEXT.md section 4.3.
NETWORK = [
    ("conv1",   6,  32, ACT_RELU),
    ("conv2",  32,  64, ACT_RELU),
    ("conv3",  64, 128, ACT_RELU),
    ("conv4", 128,  64, ACT_RELU),
    ("conv5",  64,  32, ACT_RELU),
    ("flow",   32,   2, ACT_LINEAR),
]


def load_weights(path, c_out, c_in):
    """Read a conv*_weight_q15.bin as (C_out, C_in, 3, 3) with no transpose.

    The file layout, weight_loader.sv's beat order and this reshape are the same order
    by construction. A transpose here would silently rotate every kernel and still
    produce plausible-looking output, so there is deliberately none.
    """
    v = np.fromfile(path, dtype="<i2")
    want = c_out * c_in * 9
    if v.size != want:
        raise ValueError("%s: %d int16, expected %d for (%d, %d, 3, 3)"
                         % (path, v.size, want, c_out, c_in))
    return v.reshape(c_out, c_in, 3, 3)


def forward(x, weights, shifts, bias=None, dump_dir=None, verbose=False):
    """Run the full six-layer network.

    x       (H, W, 6) int16 Q1.15 -- frame0 RGB ++ frame1 RGB, channel-minor
    weights {name: (C_out, C_in, 3, 3)}
    shifts  {name: int} -- the per-layer requantisation shift. These are a property of
            the trained model and are NOT known yet; do not invent them. Use the
            returned sat_frac to tune them once a trained model exists.
    bias    {name: (C_out,)} or None -- host-side, see add_bias()
    dump_dir  if set, write each layer's output as raw int16 for RTL comparison

    -> (flow (H, W, 2) int16, [per-layer info dicts])
    """
    missing = [n for n, _, _, _ in NETWORK if n not in weights or n not in shifts]
    if missing:
        raise KeyError("no weights/shift for: %s" % ", ".join(missing))

    infos = []
    for name, c_in, c_out, act in NETWORK:
        if x.shape[-1] != c_in:
            raise ValueError("%s expects %d input channels, got %d"
                             % (name, c_in, x.shape[-1]))
        x, info = conv_stage(x, weights[name], shifts[name], act)
        if bias and name in bias:
            x = add_bias(x, bias[name])
        info["layer"], info["shape"] = name, x.shape
        infos.append(info)
        if verbose:
            print("  %-6s -> %-16s shift=%-3d saturated %6.2f%%"
                  % (name, x.shape, shifts[name], 100.0 * info["sat_frac"]))
        if dump_dir:
            dump(x, "%s/%s.bin" % (dump_dir, name))
    return x, infos


# ---------------------------------------------------------------- dumps

def dump(x, path):
    """Write a feature map as raw little-endian int16 in (row, col, channel)
    channel-minor order -- the exact beat order of the AXI-Stream interfaces, so a
    testbench or the host driver can read it back with no reordering."""
    np.asarray(x, dtype=np.int16).astype("<i2").tofile(path)


def load_dump(path, h, w, c):
    v = np.fromfile(path, dtype="<i2")
    if v.size != h * w * c:
        raise ValueError("%s: %d int16, expected %d" % (path, v.size, h * w * c))
    return v.reshape(h, w, c)
