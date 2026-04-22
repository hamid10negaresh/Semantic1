import os, argparse, json
import torch
import torch.nn as nn
from tqdm import tqdm

from deepsc_ri.models import DeepSCRI
from deepsc_ri.data import ADE20KStub
from deepsc_ri.channel import rician_H, noise_std_from_snr_db_unit_power
from deepsc_ri.metrics import miou
from deepsc_ri.attacks.isii_pgd import make_isii_batch_via_pgd


# --------------------------
# Simple segmentation proxy
# (برای نتایج مقاله باید مدل segmentation از پیش آموزش‌دیده استفاده شود)
# --------------------------
class SimpleSegHead(nn.Module):
    def __init__(self, in_ch=3, num_classes=150):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 1),
        )

    def forward(self, x):
        return self.net(x)


def load_checkpoint_strict(model: nn.Module, ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state, strict=True)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--ade20k_dir", type=str, default=os.environ.get("ADE20K_DIR", "./ade20k_stub"))
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=2)

    ap.add_argument("--snrs", type=float, nargs="+", default=[-5, 0, 5, 10, 15, 18])
    ap.add_argument("--isiis", type=float, nargs="+", default=[0.2,0.3,0.4,0.5,0.6,0.7,0.8])

    ap.add_argument("--channel", type=str, choices=["awgn", "rician"], default="awgn")
    ap.add_argument("--kdb", type=float, default=7.0)

    # DeepSC-RI checkpoint (pretrained)
    ap.add_argument("--deepsc_ckpt", type=str, required=True, help="Path to pretrained DeepSCRI checkpoint")

    # downstream segmentation model
    ap.add_argument("--seg_ckpt", type=str, default=None, help="Optional: pretrained seg head weights (state_dict)")
    ap.add_argument("--num_classes", type=int, default=150)
    ap.add_argument("--ignore_index", type=int, default=255)

    # PGD params
    ap.add_argument("--eps_max", type=float, default=8/255)
    ap.add_argument("--pgd_steps", type=int, default=10)
    ap.add_argument("--search_iters", type=int, default=8)
    ap.add_argument("--tol", type=float, default=1e-3)

    # AMP (پیش‌فرض خاموش برای پایداری)
    ap.add_argument("--amp", action="store_true", help="Enable AMP (not recommended for low SNR)")

    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --------------------------
    # Data
    # --------------------------
    ds = ADE20KStub(args.ade20k_dir)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    # --------------------------
    # DeepSC-RI (pretrained, fixed)
    # --------------------------
    model = DeepSCRI(img_hw=256).to(device)

    # warm-up (اگر pos embed یا ماژول‌های lazy دارید)
    with torch.no_grad():
        _ = model(torch.zeros(1, 3, 256, 256, device=device), H=None, noise_std=0.0)

    load_checkpoint_strict(model, args.deepsc_ckpt, device=device)
    model.eval()

    # infer channel dim from model output (به جای هاردکد 256)
    with torch.no_grad():
        _Ir, _Tx, _Rx = model(torch.zeros(1, 3, 256, 256, device=device), H=None, noise_std=0.0)
        ch_dim = int(_Tx.shape[1])

    # --------------------------
    # Downstream segmentation model (برای paper باید pretrained و ثابت باشد)
    # --------------------------
    seg = SimpleSegHead(in_ch=3, num_classes=args.num_classes).to(device).eval()
    if args.seg_ckpt is not None and os.path.isfile(args.seg_ckpt):
        st = torch.load(args.seg_ckpt, map_location=device)
        st = st.get("state_dict", st)
        seg.load_state_dict(st, strict=False)
        seg.eval()
        print(f"✅ Loaded seg_ckpt: {args.seg_ckpt}")
    else:
        print("⚠️ seg_ckpt not provided. Using an UNTRAINED proxy seg head (paper numbers will NOT match).")

    # IMPORTANT: همین pipeline باید هم برای PGD و هم برای mIoU استفاده شود
    def seg_forward_fn(x01: torch.Tensor) -> torch.Tensor:
        # اگر مدل pretrained واقعی استفاده کردی و normalization لازم داشت،
        # دقیقاً همینجا اعمال کن تا PGD و eval یکی باشند.
        return seg(x01)

    use_amp = (args.amp and device == "cuda")

    results = {}
    for target_isii in args.isiis:
        key_isii = str(float(target_isii))
        results[key_isii] = {}

        for snr in args.snrs:
            # Tx واحد-توان => std فقط تابع snr است
            noise_std = noise_std_from_snr_db_unit_power(
                snr_db=float(snr),
                device=torch.device(device),
                dtype=torch.float32
            )

            miou_vals = []
            achieved_isii_vals = []
            eps_vals = []

            for imgs, gts in tqdm(loader, desc=f"ISII {target_isii} | SNR {snr} dB"):
                imgs = imgs.to(device, non_blocking=True).float().clamp(0.0, 1.0)
                gts  = gts.to(device, non_blocking=True).long()

                # ---- build Ic via PGD on downstream segmentation model ----
                Ic, eps_used, isii_mean = make_isii_batch_via_pgd(
                    clean_imgs=imgs,
                    target_isii=float(target_isii),
                    downstream_targets=gts,
                    forward_fn=seg_forward_fn,
                    task="seg",
                    eps_max=float(args.eps_max),
                    search_iters=int(args.search_iters),
                    pgd_steps=int(args.pgd_steps),
                    alpha=None,                 # همان سیاست قبلی: alpha پیش‌فرض داخل ماژول
                    tol=float(args.tol),
                    ignore_index=int(args.ignore_index),
                )

                achieved_isii_vals.append(float(isii_mean))
                eps_vals.append(float(eps_used))

                # ---- channel ----
                if args.channel == "awgn":
                    H = None
                else:
                    H = rician_H((imgs.size(0), ch_dim), K_db=args.kdb, device=device)

                # ---- transmit/reconstruct ----
                with torch.no_grad():
                    if use_amp:
                        with torch.cuda.amp.autocast(True):
                            Ir, Tx, Rx = model(Ic, H=H, noise_std=noise_std)
                    else:
                        Ir, Tx, Rx = model(Ic, H=H, noise_std=noise_std)

                Ir = Ir.float().clamp(0.0, 1.0)

                # ---- downstream inference & mIoU ----
                logits = seg_forward_fn(Ir)      # (B,C,H,W)
                preds  = logits.argmax(dim=1)    # (B,H,W)

                mi = miou(preds, gts, num_classes=args.num_classes, ignore_index=args.ignore_index)
                miou_vals.append(float(mi))

            results[key_isii][str(float(snr))] = {
                "mIoU": float(sum(miou_vals) / max(1, len(miou_vals))),
                "avg_eps": float(sum(eps_vals) / max(1, len(eps_vals))),
                "achieved_isii": float(sum(achieved_isii_vals) / max(1, len(achieved_isii_vals))),
            }

            print(
                f"ISII={target_isii} SNR={snr} :: "
                f"mIoU={results[key_isii][str(float(snr))]['mIoU']:.4f} "
                f"(eps~{results[key_isii][str(float(snr))]['avg_eps']:.5f}, "
                f"isii~{results[key_isii][str(float(snr))]['achieved_isii']:.3f})"
            )

    with open("ade20k_isii_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("✅ Saved to ade20k_isii_results.json")


if __name__ == "__main__":
    main()
