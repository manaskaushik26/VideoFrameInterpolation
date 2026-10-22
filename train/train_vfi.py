"""
train_vfi.py -- train the 6-layer flow CNN so that what it learns is what the PL computes.

Replaces the original Videoframeinterplotation.py. That script trained a network the
hardware cannot run, so its 20.29 dB was a float-model number that does not survive
deployment. What changed, and why (each measured, see train/README.md):

  1. No bias in conv1..conv5.  The RTL has no bias anywhere (CLAUDE.md). The original
     model used biases up to 0.40 in every layer; deleting them, which is what the
     hardware does, drops it from 20.45 dB to 16.74 dB -- barely above a plain 50/50
     blend of the two frames (15.97 dB). The flow layer keeps its bias, added
     host-side after requantisation, exactly as model/vfi_ref.add_bias does.
  2. Quantisation-aware training. Weights are fake-quantised to int16 with a per-layer
     power-of-two scale, which is exactly what the per-layer `shift` provides
     (shift = 15 - k for weights scaled by 2^-k). Activations are clamped to Q1.15
     [0, 32767/32768] after every ReLU, as conv_mac_3x3 saturates. The original export
     silently clipped weights at +-1.0 (conv3 and flow hit 32767) and activations
     reached 6.4, all of which the hardware would clip.
  3. No tanh. The PL output is linear Q1.15, which already saturates at +-1, and the
     original golden dumps were taken after a tanh nothing implements.
  4. Flow in pixels, not grid units: flow_px = FLOW_PX * q15. Resolution independent,
     and what cv2.remap on the PS wants.
  5. Moving MNIST at its native 64x64. Upsampling to 112 adds no information and
     stretches f0->f2 motion to 12.4 px median, while six 3x3 layers see only +-6 px.
     At 64x64 it is 7.1 px. Because no layer has a bias, a 64x64 frame zero-padded
     into the 112x112 hardware frame produces exactly the same flow over the digit
     area, so this costs nothing on the board.
  6. Held-out validation. The original trained and scored on the same sequences.
     Here 10% of sequences are never trained on, and PSNR is per frame, against the
     no-model baselines, so the number means something.

Usage (GPU strongly recommended; see CLAUDE.md -- this machine has no CUDA):

    python train_vfi.py --mnist mnist_test_seq.npy --out run_mnist
    python train_vfi.py --videos path/to/mp4s       --out run_video    # real RGB video

Then, on any machine with numpy (no GPU):

    python verify_export.py run_mnist/export

Outputs in <out>/export/ are the files the board needs -- conv1..5_weight_q15.bin,
flow_weight_q15.bin (raw int16, (out, in, 3, 3), no transpose), flow_bias_q15.bin,
and vfi_model.json with the per-layer shifts. Nothing here needs torch to read.
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

Q = 32768.0
Q_MAX = 32767.0 / 32768.0
HW_SIZE = 112                    # the bitstream's IMG_W = IMG_H
FLOW_PX = 32.0                   # flow Q1.15 full scale, in pixels
LAYERS = [                       # (name, c_in, c_out) -- model/vfi_ref.NETWORK
    ("conv1", 6, 32), ("conv2", 32, 64), ("conv3", 64, 128),
    ("conv4", 128, 64), ("conv5", 64, 32), ("flow", 32, 2),
]


# ---------------------------------------------------------------- fixed point

def fake_quant(x, scale, lo, hi):
    """Round to the grid 1/scale and clamp to [lo, hi]; straight-through gradient
    inside the range, zero gradient where the hardware would saturate."""
    xc = torch.clamp(x, lo, hi)
    return xc + (torch.round(xc * scale) / scale - xc).detach()


def weight_exponent(w):
    """Smallest k with max|w| < 2^k, so w * 2^(15-k) fits int16. The layer's
    requantisation shift is then 15 - k."""
    m = float(w.detach().abs().max())
    return math.ceil(math.log2(m * (1 + 1e-6))) if m > 0 else 0


def quant_weight(w):
    k = weight_exponent(w)
    s = 2.0 ** (15 - k)
    return fake_quant(w, s, -32768 / s, 32767 / s), 15 - k


# ---------------------------------------------------------------- model

class VfiNet(nn.Module):
    """conv1..5: 3x3, no bias, ReLU, Q1.15 saturation. flow: 3x3 linear, saturate,
    then +bias (host-side), saturate again -- model/vfi_ref.forward exactly."""

    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList(
            nn.Conv2d(ci, co, 3, padding=1, bias=False) for _, ci, co in LAYERS)
        self.flow_bias = nn.Parameter(torch.zeros(2))

    def forward(self, x, quant=True):
        if quant:
            x = fake_quant(x, Q, -1.0, Q_MAX)
        for i, conv in enumerate(self.convs):
            w = quant_weight(conv.weight)[0] if quant else conv.weight
            x = F.conv2d(x, w, padding=1)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = fake_quant(x, Q, 0.0, Q_MAX) if quant else torch.clamp(x, 0, Q_MAX)
        b = self.flow_bias.view(1, 2, 1, 1)
        if quant:
            x = fake_quant(x, Q, -1.0, Q_MAX)
            x = fake_quant(x + fake_quant(b, Q, -1.0, Q_MAX), Q, -1.0, Q_MAX)
        else:
            x = torch.clamp(torch.clamp(x, -1.0, Q_MAX) + b, -1.0, Q_MAX)
        return x * FLOW_PX                                   # (B, 2, H, W) pixels


def warp(frame, flow_px):
    """Backward warp: sample frame at p + flow. flow_px[:, 0] is x, [:, 1] is y."""
    B, _, H, W = frame.shape
    gy, gx = torch.meshgrid(torch.linspace(-1, 1, H, device=frame.device),
                            torch.linspace(-1, 1, W, device=frame.device), indexing="ij")
    grid = torch.stack((gx, gy), -1).unsqueeze(0)
    d = torch.stack((flow_px[:, 0] * 2 / (W - 1), flow_px[:, 1] * 2 / (H - 1)), -1)
    return F.grid_sample(frame, grid + d, mode="bilinear", padding_mode="border",
                         align_corners=True)


def interpolate(model, f0, f2, quant=True):
    flow = model(torch.cat([f0, f2], 1), quant=quant)
    return 0.5 * (warp(f0, 0.5 * flow) + warp(f2, -0.5 * flow)), flow


# ---------------------------------------------------------------- data

class MovingMnist(Dataset):
    """(20, N, 64, 64) uint8. Triplets (t, t+1, t+2) from the given sequences.
    Grey is replicated to RGB so the 6-channel interface stays the one the PL takes."""

    def __init__(self, path, seqs, pad_to=None):
        self.d = np.load(path, mmap_mode="r")
        self.items = [(s, t) for s in seqs for t in range(self.d.shape[0] - 2)]
        self.pad_to = pad_to

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        s, t = self.items[i]
        f = torch.from_numpy(np.ascontiguousarray(self.d[t:t + 3, s])).float() / 255.0
        if self.pad_to:
            p = self.pad_to - f.shape[-1]
            f = F.pad(f, (p // 2, p - p // 2, p // 2, p - p // 2))
        f = f.unsqueeze(1).expand(3, 3, *f.shape[-2:])
        return f[0].contiguous(), f[1].contiguous(), f[2].contiguous()


class VideoTriplets(Dataset):
    """Every consecutive triplet of every video under `root`, resized to 112x112 RGB.
    Decoded once into memory: 37 KB per frame, so 100k frames is ~3.7 GB."""

    def __init__(self, files, size=HW_SIZE):
        import cv2
        self.frames, self.items = [], []
        for path in files:
            cap, clip = cv2.VideoCapture(path), []
            while True:
                ok, fr = cap.read()
                if not ok:
                    break
                fr = cv2.resize(fr, (size, size), interpolation=cv2.INTER_AREA)
                clip.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            if len(clip) >= 3:
                base = sum(len(c) for c in self.frames)
                self.items += [base + t for t in range(len(clip) - 2)]
                self.frames.append(np.stack(clip))
        if not self.frames:
            raise SystemExit("no decodable videos in %r" % (files,))
        self.frames = np.concatenate(self.frames)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        j = self.items[i]
        f = torch.from_numpy(self.frames[j:j + 3]).permute(0, 3, 1, 2).float() / 255.0
        return f[0], f[1], f[2]


def build_datasets(args):
    rng = np.random.default_rng(0)
    if args.mnist:
        n = np.load(args.mnist, mmap_mode="r").shape[1]
        perm = rng.permutation(n)
        n_val = max(1, int(n * args.val_frac))
        val, train = sorted(perm[:n_val]), sorted(perm[n_val:])
        return (MovingMnist(args.mnist, train),
                MovingMnist(args.mnist, val),
                MovingMnist(args.mnist, val, pad_to=HW_SIZE))
    files = sorted(os.path.join(dp, f) for dp, _, fs in os.walk(args.videos) for f in fs
                   if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv")))
    files = [files[i] for i in rng.permutation(len(files))]
    n_val = max(1, int(len(files) * args.val_frac))   # split by video, never by frame
    if len(files) < 2:
        raise SystemExit("need at least 2 videos so validation is held out")
    val = VideoTriplets(files[:n_val])
    return VideoTriplets(files[n_val:]), val, val


# ---------------------------------------------------------------- loss / eval

def charbonnier(x, y, eps=1e-6):
    return torch.mean(torch.sqrt((x - y) ** 2 + eps))


def tv(flow):
    return (flow[..., :, 1:] - flow[..., :, :-1]).abs().mean() + \
           (flow[..., 1:, :] - flow[..., :-1, :]).abs().mean()


def psnr(pred, tgt):
    mse = ((pred - tgt) ** 2).flatten(1).mean(1).clamp_min(1e-10)
    return 10 * torch.log10(1.0 / mse), mse                # per frame


@torch.no_grad()
def evaluate(model, loader, device, max_batches=None):
    """Per-frame mean PSNR of the quantised model and of the no-model baselines."""
    model.eval()
    acc = {"model": [], "blend": [], "copy": []}
    mses = []
    for b, (f0, f1, f2) in enumerate(loader):
        if max_batches and b >= max_batches:
            break
        f0, f1, f2 = f0.to(device), f1.to(device), f2.to(device)
        pred, _ = interpolate(model, f0, f2, quant=True)
        p, m = psnr(pred.clamp(0, 1), f1)
        acc["model"].append(p); mses.append(m)
        acc["blend"].append(psnr(0.5 * (f0 + f2), f1)[0])
        acc["copy"].append(psnr(f0, f1)[0])
    model.train()
    out = {k: float(torch.cat(v).mean()) for k, v in acc.items()}
    out["mse"] = float(torch.cat(mses).mean())
    return out


# ---------------------------------------------------------------- export

def export(model, out_dir, sample):
    """Write the board's files. Everything is raw little-endian int16 or JSON."""
    os.makedirs(out_dir, exist_ok=True)
    meta = {"layers": [], "flow_px_per_unit": FLOW_PX, "img": HW_SIZE,
            "input": "frame0 RGB ++ frame2 RGB, [0,1] -> Q1.15, (row, col, channel)",
            "flow": "Q1.15 after bias; pixels = value/32768 * flow_px_per_unit; ch0=x ch1=y; "
                    "frame_mid = 0.5*(remap(f0, p + 0.5*flow) + remap(f2, p - 0.5*flow))"}
    with torch.no_grad():
        for (name, ci, co), conv in zip(LAYERS, model.convs):
            k = weight_exponent(conv.weight)
            w = torch.round(conv.weight * 2.0 ** (15 - k)).clamp(-32768, 32767)
            w = w.cpu().numpy().astype("<i2")
            w.tofile(os.path.join(out_dir, "%s_weight_q15.bin" % name))
            np.save(os.path.join(out_dir, "%s_weight_float.npy" % name),
                    conv.weight.cpu().numpy())
            meta["layers"].append({"name": name, "c_in": ci, "c_out": co,
                                   "shift": 15 - k, "act": "linear" if name == "flow" else "relu",
                                   "weight_absmax_int": int(np.abs(w.astype(np.int32)).max())})
        b = torch.round(model.flow_bias * Q).clamp(-32768, 32767).cpu().numpy().astype("<i2")
        b.tofile(os.path.join(out_dir, "flow_bias_q15.bin"))
        meta["flow_bias_q15"] = b.tolist()
        meta["shifts"] = {l["name"]: l["shift"] for l in meta["layers"]}

        # A real held-out frame pair at the hardware geometry, plus what the
        # quantised torch model makes of it -- verify_export.py checks vfi_ref against it.
        f0, f1, f2 = (t.unsqueeze(0).to(next(model.parameters()).device) for t in sample)
        x = torch.cat([f0, f2], 1)
        x_q = torch.round(x.clamp(-1, Q_MAX) * Q).squeeze(0).permute(1, 2, 0)
        x_q.cpu().numpy().astype("<i2").tofile(os.path.join(out_dir, "golden_input.bin"))
        flow = model(x, quant=True) / FLOW_PX * Q
        flow.squeeze(0).permute(1, 2, 0).round().cpu().numpy().astype("<i2") \
            .tofile(os.path.join(out_dir, "golden_flow_torch.bin"))
        np.save(os.path.join(out_dir, "golden_frames.npy"),
                torch.cat([f0, f1, f2]).cpu().numpy())
    with open(os.path.join(out_dir, "vfi_model.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--mnist", help="mnist_test_seq.npy")
    src.add_argument("--videos", help="directory of video files (real RGB footage)")
    ap.add_argument("--out", default="run")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tv", type=float, default=0.002, help="flow smoothness weight (per px)")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--quant-after", type=int, default=2,
                    help="epochs of float warm-up before quantisation-aware training")
    ap.add_argument("--max-steps", type=int, default=0, help="stop early (smoke test)")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    torch.manual_seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds, val_ds, hw_ds = build_datasets(args)
    print("[*] %d train / %d held-out triplets on %s" % (len(train_ds), len(val_ds), dev))
    tl = DataLoader(train_ds, args.batch, shuffle=True, num_workers=args.workers,
                    drop_last=True, persistent_workers=args.workers > 0)
    vl = DataLoader(val_ds, 64, shuffle=False, num_workers=args.workers)
    hl = DataLoader(hw_ds, 64, shuffle=False, num_workers=args.workers)

    model = VfiNet().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total = args.epochs * len(tl) if not args.max_steps else args.max_steps
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=total,
                                                pct_start=max(0.05, 2.0 / total))  # >=1 warm-up step
    os.makedirs(args.out, exist_ok=True)

    step, best, t0 = 0, -1.0, time.time()
    max_val = 40 if args.max_steps else None
    quant_from = args.max_steps // 2 if args.max_steps else args.quant_after * len(tl)
    for ep in range(args.epochs):
        for f0, f1, f2 in tl:
            quant = step >= quant_from
            f0, f1, f2 = f0.to(dev), f1.to(dev), f2.to(dev)
            pred, flow = interpolate(model, f0, f2, quant=quant)
            loss = charbonnier(pred, f1) + args.tv * tv(flow)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(); sched.step(); step += 1
            if step % 200 == 0 or step == 1:
                print("  ep %2d step %6d  loss %.5f  %s  %.0fs"
                      % (ep, step, loss.item(), "QAT" if quant else "float", time.time() - t0))
            if args.max_steps and step >= args.max_steps:
                break
        r = evaluate(model, vl, dev, max_val)
        print(">>> epoch %d  held-out PSNR %.2f dB (MSE %.6f)   | 50/50 blend %.2f  copy-f0 %.2f"
              % (ep, r["model"], r["mse"], r["blend"], r["copy"]))
        if quant and r["model"] > best:
            best = r["model"]
            torch.save(model.state_dict(), os.path.join(args.out, "best.pt"))
        if args.max_steps and step >= args.max_steps:
            break

    if os.path.exists(os.path.join(args.out, "best.pt")):
        model.load_state_dict(torch.load(os.path.join(args.out, "best.pt"), map_location=dev))
    r = evaluate(model, vl, dev, max_val)
    rh = evaluate(model, hl, dev, max_val) if hl.dataset is not vl.dataset else r
    print("\n=========== held-out, quantised (what the board computes) ===========")
    print("  model          PSNR %.2f dB   MSE %.6f" % (r["model"], r["mse"]))
    print("  50/50 blend    PSNR %.2f dB   (no model -- the bar to clear)" % r["blend"])
    print("  copy frame0    PSNR %.2f dB" % r["copy"])
    print("  at %dx%d hardware geometry: %.2f dB" % (HW_SIZE, HW_SIZE, rh["model"]))
    meta = export(model, os.path.join(args.out, "export"), hw_ds[len(hw_ds) // 2])
    meta["heldout"] = {"model_psnr": r["model"], "blend_psnr": r["blend"],
                       "copy_psnr": r["copy"], "hw_geometry_psnr": rh["model"], "mse": r["mse"]}
    with open(os.path.join(args.out, "export", "vfi_model.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("[*] shifts:", meta["shifts"], " flow bias:", meta["flow_bias_q15"])
    print("[*] exported to %s -- next: python verify_export.py %s"
          % (os.path.join(args.out, "export"), os.path.join(args.out, "export")))


if __name__ == "__main__":
    main()
