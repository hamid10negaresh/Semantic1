# scripts/train_cifar10.py
import argparse, os, random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from deepsc_ri.models import DeepSCRI
from deepsc_ri.channel import rician_H, noise_std_from_snr_db_unit_power
from deepsc_ri.metrics import batch_psnr, lpips_score

torch.set_float32_matmul_precision("medium")

# ---------------------------
# Reproducibility
# ---------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# ---------------------------
# Checkpoint I/O
# ---------------------------
def save_ckpt(path, epoch, model, opt):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": opt.state_dict(),
        },
        path,
    )

def load_ckpt(path, model, opt, device):
    ckpt = torch.load(path, map_location=device)

    # warm-up in case the model has lazy init
    img_hw = getattr(model, "img_hw", 32)
    with torch.no_grad():
        _ = model(torch.zeros(1, 3, img_hw, img_hw, device=device), H=None, noise_std=0.0)

    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state, strict=True)

    if opt is not None and "optimizer_state" in ckpt:
        opt.load_state_dict(ckpt["optimizer_state"])

    return ckpt.get("epoch", -1) + 1

# ---------------------------
# Dataset: (Ic -> Iu) pairs from cache
# ---------------------------
def _list_pt(dir_path: str) -> List[str]:
    if not os.path.isdir(dir_path):
        return []
    return [f for f in os.listdir(dir_path) if f.endswith(".pt")]

def _stem(pt_name: str) -> str:
    return os.path.splitext(pt_name)[0]

class SemanticImpairmentPairs(torch.utils.data.Dataset):
    """
    Paper-style training data:
      input  = Ic (corrupted image with semantic impairments)
      target = Iu (uncorrupted image)

    Expects:
      cache_dir/Ic/*.pt
      cache_dir/Iu/*.pt
    with aligned filename stems.
    """
    def __init__(self, cache_dir: str):
        super().__init__()
        self.cache_dir = cache_dir
        self.ic_dir = os.path.join(cache_dir, "Ic")
        self.iu_dir = os.path.join(cache_dir, "Iu")

        ic_files = _list_pt(self.ic_dir)
        iu_files = _list_pt(self.iu_dir)

        ic_map = {_stem(f): os.path.join(self.ic_dir, f) for f in ic_files}
        iu_map = {_stem(f): os.path.join(self.iu_dir, f) for f in iu_files}

        keys = sorted(set(ic_map.keys()).intersection(set(iu_map.keys())))
        if len(keys) == 0:
            raise FileNotFoundError(
                "No (Ic,Iu) pairs were found. Expected structure:\n"
                "  CACHE_DIR/Ic/*.pt\n"
                "  CACHE_DIR/Iu/*.pt\n"
                "and matching filename stems."
            )

        self.ic_paths = [ic_map[k] for k in keys]
        self.iu_paths = [iu_map[k] for k in keys]

    def __len__(self):
        return len(self.ic_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        Ic = torch.load(self.ic_paths[idx])
        Iu = torch.load(self.iu_paths[idx])
        if not (isinstance(Ic, torch.Tensor) and isinstance(Iu, torch.Tensor)):
            raise TypeError("Cache .pt files must contain torch.Tensors.")
        return Ic.float(), Iu.float()

# ---------------------------
# Helpers
# ---------------------------
def infer_channel_dim(model: nn.Module, device: str) -> int:
    with torch.no_grad():
        dummy = torch.zeros(
            2, 3, getattr(model, "img_hw", 32), getattr(model, "img_hw", 32), device=device
        )
        out = model(dummy, H=None, noise_std=0.0)
        Tx = out[1]
    return int(Tx.size(1))

def pixels_to_labels_0_255(img01: torch.Tensor) -> torch.Tensor:
    img01 = img01.clamp(0.0, 1.0)
    lab = torch.round(img01 * 255.0).to(torch.int64)
    return lab

def logits_to_img01(pred: torch.Tensor) -> torch.Tensor:
    if pred.dim() == 5 and pred.size(2) == 256:
        logits = pred
    elif pred.dim() == 4 and pred.size(1) == 3 * 256:
        B, C, H, W = pred.shape
        logits = pred.view(B, 3, 256, H, W)
    else:
        raise ValueError(
            "For paper-style pixel CE, model output must be pixel logits: "
            "(B,3,256,H,W) or (B,3*256,H,W)."
        )

    probs = torch.softmax(logits, dim=2)
    bins = torch.linspace(
        0.0, 1.0, 256, device=probs.device, dtype=probs.dtype
    ).view(1, 1, 256, 1, 1)
    img = (probs * bins).sum(dim=2)
    return img.clamp(0.0, 1.0)

class PixelCrossEntropy(nn.Module):
    """Pixel-wise CE by 256-level quantization."""
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, pred_logits: torch.Tensor, target_img01: torch.Tensor) -> torch.Tensor:
        target = pixels_to_labels_0_255(target_img01)

        if pred_logits.dim() == 5 and pred_logits.size(2) == 256:
            B, C3, K, H, W = pred_logits.shape
            logits = pred_logits.permute(0, 1, 3, 4, 2).contiguous().view(B * 3 * H * W, 256)
            y = target.view(B * 3 * H * W)
            return self.ce(logits, y)

        if pred_logits.dim() == 4 and pred_logits.size(1) == 3 * 256:
            B, _, H, W = pred_logits.shape
            logits = pred_logits.view(B, 3, 256, H, W)
            logits = logits.permute(0, 1, 3, 4, 2).contiguous().view(B * 3 * H * W, 256)
            y = target.view(B * 3 * H * W)
            return self.ce(logits, y)

        raise ValueError(
            "For paper-style pixel CE, model output must be pixel logits: "
            "(B,3,256,H,W) or (B,3*256,H,W)."
        )

# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()

    # training
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--amp", action="store_true")

    # Paper: training over multiple SNRs
    ap.add_argument("--snrs", type=float, nargs="+", default=[-15, -10, -5, 0, 5, 10, 15, 18])
    ap.add_argument("--snr_sampling", type=str, choices=["random", "cycle"], default="random")

    # channel
    ap.add_argument("--channel", type=str, choices=["awgn", "rician"], default="awgn")
    ap.add_argument("--kdb", type=float, default=7.0)

    # data: separate train/test impairment caches
    ap.add_argument(
        "--impairment_cache_train", type=str, required=True,
        help="Path to TRAIN cache dir containing Ic/ and Iu/ .pt files."
    )
    ap.add_argument(
        "--impairment_cache_test", type=str, required=True,
        help="Path to TEST/VAL cache dir containing Ic/ and Iu/ .pt files."
    )

    # loss weight in Eq.(26)
    ap.add_argument(
        "--alpha_chan", type=float, default=1e-4,
        help="alpha in L_total = L_CE + alpha * MSE(Tx,Rx)"
    )

    # ckpt
    ap.add_argument("--save-dir", type=str, default="./checkpoints")
    ap.add_argument("--resume", type=str, default=None)

    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataset: (Ic, Iu)
    train_ds = SemanticImpairmentPairs(args.impairment_cache_train)
    test_ds  = SemanticImpairmentPairs(args.impairment_cache_test)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    model = DeepSCRI(img_hw=32).to(device)

    opt = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)

    # Losses (paper Eq.(26))
    l_ce  = PixelCrossEntropy()
    l_mse = nn.MSELoss()

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        start_epoch = load_ckpt(args.resume, model, opt, device=device)
        print(f"🔁 Resumed from {args.resume} at epoch {start_epoch}")

    ch_dim = infer_channel_dim(model, device=device)
    print(f"ℹ️  inferred channel dim = {ch_dim}")

    # helper for snr picking
    def pick_snr(step_idx: int) -> float:
        if args.snr_sampling == "random":
            return random.choice(args.snrs)
        return args.snrs[step_idx % len(args.snrs)]

    # ---------------------------
    # Train
    # ---------------------------
    global_step = 0
    for epoch in range(start_epoch, args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}")

        for Ic, Iu in pbar:
            Ic = Ic.to(device, non_blocking=True)
            Iu = Iu.to(device, non_blocking=True)

            snr = pick_snr(global_step)
            global_step += 1

            noise_std = noise_std_from_snr_db_unit_power(
                snr, device=torch.device(device), dtype=torch.float32
            )

            H = None
            if args.channel == "rician":
                H = rician_H((Ic.size(0), ch_dim), K_db=args.kdb, device=device)

            with torch.cuda.amp.autocast(enabled=args.amp):
                pred, Tx, Rx = model(Ic, H=H, noise_std=noise_std)
                loss_rec = l_ce(pred, Iu)
                loss_chan = l_mse(Tx, Rx)
                loss = loss_rec + args.alpha_chan * loss_chan

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss! loss={loss.item()}, "
                    f"Lce={loss_rec.item()}, Lch={loss_chan.item()}"
                )

            pbar.set_postfix(
                snr=snr,
                loss=float(loss.detach().cpu()),
                Lce=float(loss_rec.detach().cpu()),
                Lch=float(loss_chan.detach().cpu()),
            )

        ckpt_path = os.path.join(args.save_dir, f"checkpoint_epoch{epoch}.pt")
        save_ckpt(ckpt_path, epoch, model, opt)
        print(f"✅ Saved: {ckpt_path}")

        # ---------------------------
        # Eval (PSNR/LPIPS) vs SNR
        # ---------------------------
        model.eval()
        for snr in args.snrs:
            noise_std = noise_std_from_snr_db_unit_power(
                snr, device=torch.device(device), dtype=torch.float32
            )

            psnr_vals = []
            lpips_vals = []

            with torch.no_grad():
                for Ic, Iu in tqdm(test_loader, desc=f"eval snr {snr} dB"):
                    Ic = Ic.to(device, non_blocking=True)
                    Iu = Iu.to(device, non_blocking=True)

                    H = None
                    if args.channel == "rician":
                        H = rician_H((Ic.size(0), ch_dim), K_db=args.kdb, device=device)

                    pred, Tx, Rx = model(Ic, H=H, noise_std=noise_std)
                    I_hat = logits_to_img01(pred)

                    psnr_vals.append(batch_psnr(I_hat, Iu))
                    lp = lpips_score(I_hat, Iu)
                    if lp is not None:
                        lpips_vals.append(lp)

            print(
                f"SNR={snr}dB :: "
                f"PSNR={sum(psnr_vals)/len(psnr_vals):.3f}  "
                f"LPIPS={'N/A' if not lpips_vals else sum(lpips_vals)/len(lpips_vals):.4f}"
            )

if __name__ == "__main__":
    main()
