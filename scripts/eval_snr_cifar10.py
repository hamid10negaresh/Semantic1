import argparse, os, json
import torch
from tqdm import tqdm
from torchvision import datasets, transforms, models

from deepsc_ri.models import DeepSCRI
from deepsc_ri.metrics import batch_psnr, lpips_score
from deepsc_ri.channel import rician_H
from deepsc_ri.attacks.isii_pgd import make_isii_batch_via_pgd


def noise_std_from_snr_db(snr_db: float, device, dtype=torch.float32):
    # چون Tx در مدل شما power-normalized است (RMS=1)،
    # sigma = sqrt(1/SNR_lin) = 10^(-snr_db/20)
    return torch.tensor(10.0 ** (-snr_db / 20.0), device=device, dtype=dtype)


def main():
    ap = argparse.ArgumentParser()
    # تغییر 1: به جای یک SNR و چند ISII، حالا چند SNR و یک ISII داریم
    ap.add_argument("--snrs", type=float, nargs="+", default=[-5.0, 0.0, 5.0, 10.0, 15.0, 20.0])
    ap.add_argument("--target_isii", type=float, default=0.5)
    
    ap.add_argument("--channel", type=str, choices=["awgn","rician"], default="awgn")
    ap.add_argument("--kdb", type=float, default=7.0)

    # downstream classifier
    ap.add_argument("--clf_arch", type=str, default="robustbench",
                    choices=["robustbench","resnet18_cifar"])
    ap.add_argument("--rb_model", type=str, default="Carmon2019Unlabeled")
    ap.add_argument("--clf_path", type=str, default=None)
    ap.add_argument("--apply_cifar_norm", action="store_true",
                    help="اگر مدل پایین‌دستی normalization داخلی ندارد، روشن کنید (PGD و ACC هر دو با همین انجام می‌شوند).")

    # PGD / ISII search params (paper-style defaults)
    ap.add_argument("--eps_max", type=float, default=8/255)
    ap.add_argument("--search_iters", type=int, default=8)
    ap.add_argument("--pgd_steps", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.0, help="اگر 0 باشد، alpha پیش‌فرض داخل ماژول (2/255) استفاده می‌شود.")
    ap.add_argument("--tol", type=float, default=1e-3)

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=2)

    ap.add_argument("--ckpt", type=str, default="./checkpoints/checkpoint_epoch1.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # CIFAR10 loader (Iu in [0,1])
    tf = transforms.Compose([transforms.ToTensor()])
    test = datasets.CIFAR10(root="./data", train=False, download=True, transform=tf)
    loader = torch.utils.data.DataLoader(
        test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=True
    )

    # ---------- Downstream classifier ----------
    if args.clf_arch == "robustbench":
        try:
            from robustbench.utils import load_model
        except Exception as e:
            raise RuntimeError("robustbench نصب نیست. pip install robustbench") from e
        clf = load_model(model_name=args.rb_model, dataset="cifar10", threat_model="Linf")
    else:
        clf = models.resnet18(num_classes=10)
        clf.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        clf.maxpool = torch.nn.Identity()
        if args.clf_path is None or (not os.path.isfile(args.clf_path)):
            raise FileNotFoundError("--clf_path لازم است و باید به فایل وزن CIFAR-10 اشاره کند.")
        state = torch.load(args.clf_path, map_location="cpu")
        state = state.get("state_dict", state)
        clf.load_state_dict(state, strict=False)

    clf = clf.to(device).eval()

    # CIFAR10 normalization (اگر لازم دارید)
    CIFAR10_MEAN = torch.tensor([0.4914,0.4822,0.4465], device=device).view(1,3,1,1)
    CIFAR10_STD  = torch.tensor([0.2470,0.2435,0.2616], device=device).view(1,3,1,1)

    def clf_forward(x01: torch.Tensor) -> torch.Tensor:
        # IMPORTANT: همین مسیر هم در PGD استفاده می‌شود هم در ACC
        if args.apply_cifar_norm:
            x = (x01 - CIFAR10_MEAN) / CIFAR10_STD
        else:
            x = x01
        return clf(x)

    # ---------- DeepSC-RI ----------
    model = DeepSCRI(img_hw=32).to(device)

    # warm-up (اگر ماژول‌های lazy دارید)
    with torch.no_grad():
        _ = model(torch.zeros(2,3,32,32, device=device), H=None, noise_std=0.0)

    # load checkpoint
    state = torch.load(args.ckpt, map_location=device)
    if "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state, strict=True)
    model.eval()

    # infer channel dim (به جای هاردکد 256)
    with torch.no_grad():
        _Ir, _Tx, _Rx = model(torch.zeros(2,3,32,32, device=device), H=None, noise_std=0.0)
        ch_dim = int(_Tx.shape[1])

    # alpha handling
    alpha = None if args.alpha == 0.0 else float(args.alpha)

    # 1. تغییر 2: نام فایل خروجی را پویا کردیم تا نتایج SNR ذخیره شود
    out_file = f"snr_results_isii{args.target_isii}_{args.channel}.json"
    results = {}
    if os.path.exists(out_file):
        with open(out_file, "r") as f:
            results = json.load(f)
            print(f"Loaded existing results: {list(results.keys())}")

    # تغییر 3: شروع حلقه اصلی روی مقادیر SNR به جای ISII
    for snr in args.snrs:
        if str(snr) in results:
            print(f"Skipping SNR={snr}, already computed.")
            continue
            
        # محاسبه نویز مخصوص همین SNR
        noise_std = noise_std_from_snr_db(snr, device=device)
        
        # محاسبه Drop Ratio بر اساس نویز برای ذخیره در فایل
        n_val = noise_std.item() if hasattr(noise_std, 'item') else float(noise_std)
        if n_val < 0.18:
            drop_ratio = 0.05
        elif n_val < 0.32:
            drop_ratio = 0.15
        else:
            drop_ratio = 0.35

        psnrs, lpips_vals, accs = [], [], []

        # -- شروع پردازش دسته‌های تصویر برای SNR فعلی --
        for i, (Iu, labels) in enumerate(tqdm(loader, desc=f"SNR {snr} dB")):
            
            # (برای گرفتن نتایج نهایی پایان‌نامه، این دو خط را پاک کنید)
            if i >= 2:  
                break
                
            Iu = Iu.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # --------- Build Ic via PGD on downstream model (paper protocol) ---------
            Ic, eps_used, isii_mean = make_isii_batch_via_pgd(
                clean_imgs=Iu,
                target_isii=float(args.target_isii), # اینجا از ISII ثابت استفاده می‌کنیم
                downstream_targets=labels,
                forward_fn=clf_forward,
                task="cls",
                eps_max=float(args.eps_max),
                search_iters=int(args.search_iters),
                pgd_steps=int(args.pgd_steps),
                alpha=alpha,
                tol=float(args.tol),
            )

            # channel
            if args.channel == "awgn":
                H = None
            else:
                H = rician_H((Iu.size(0), ch_dim), K_db=args.kdb, device=device)

            # --------- Run DeepSC-RI on Ic ---------
            with torch.no_grad():
                # ۱. دریافت خروجی خام مدل (logits)
                logits, Tx, Rx = model(Ic, H=H, noise_std=noise_std)

                # ۲. تبدیل logits به تصویر با روش میانگین وزنی احتمالات (Expectation)
                B, _, H_img, W_img = logits.shape
                logits_reshaped = logits.view(B, 3, 256, H_img, W_img)
                
                # محاسبه احتمال هر سطح از رنگ با تابع سافت‌مکس
                probs = logits_reshaped.softmax(dim=2)
                
                # ایجاد یک بردار از 0 تا 1 با 256 پله
                bin_centers = torch.linspace(0, 1, 256, device=logits.device)
                
                # ضرب احتمال در مقدار رنگ و جمع زدن (برای تولید رنگ نهایی پیکسل)
                Ir = (probs * bin_centers.view(1, 1, 256, 1, 1)).sum(dim=2)
                
                # محدود کردن مقادیر برای اطمینان از قرارگیری در بازه استاندارد تصویر
                Ir = Ir.clamp(0.0, 1.0)

                # PSNR/LPIPS between Iu (gt) and Ir (received)
                psnrs.append(batch_psnr(Ir, Iu))
                lp = lpips_score(Ir, Iu)
                if lp is not None:
                    lpips_vals.append(lp)

                # ACC on received image
                clf_logits = clf_forward(Ir)
                accs.append((clf_logits.argmax(dim=1) == labels).float().mean().item())
        # -- پایان پردازش تصاویر این مرحله --

        # ذخیره مقادیر به همراه درصد Drop Ratio
        results[str(snr)] = {
            "PSNR": float(sum(psnrs)/len(psnrs)) if psnrs else 0.0,
            "LPIPS": float(sum(lpips_vals)/len(lpips_vals)) if lpips_vals else 0.0,
            "ACC": float(sum(accs)/len(accs)) if accs else 0.0,
            "DropRatio": drop_ratio
        }
        
        print(f"SNR={snr} dB  Metrics: {results[str(snr)]}")

        # ذخیره‌سازی مرحله‌به‌مرحله پس از پایان هر SNR
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved progress for SNR={snr} to {out_file}")

if __name__ == "__main__":
    main()
