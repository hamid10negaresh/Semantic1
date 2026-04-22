# scripts/build_isii_cache_cifar10.py
import argparse, os, json
import torch
from tqdm import tqdm
from torchvision import datasets, transforms, models

from deepsc_ri.attacks.isii_pgd import (
    make_isii_batch_via_pgd,
    make_cifar10_forward_fn,
)

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _save_meta(cache_dir: str, meta: dict):
    with open(os.path.join(cache_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

def _build_classifier(args, device: torch.device):
    """
    Matches the logic of scripts/eval_isii_cifar10.py in your project:
      - robustbench (if installed and usable)
      - or a local CIFAR-10 ResNet18 checkpoint
    """
    if args.clf_arch == "robustbench":
        try:
            from robustbench.utils import load_model
        except Exception as e:
            raise RuntimeError(
                "robustbench could not be imported/used. Install it with: pip install robustbench\n"
                "Or use --clf_arch resnet18_cifar and provide --clf_path to a CIFAR-10 weight file."
            ) from e

        clf = load_model(model_name=args.rb_model, dataset="cifar10", threat_model="Linf")
        clf = clf.to(device).eval()
        return clf

    # resnet18_cifar
    clf = models.resnet18(num_classes=10)
    clf.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    clf.maxpool = torch.nn.Identity()

    if args.clf_path is None or (not os.path.isfile(args.clf_path)):
        raise FileNotFoundError(
            "--clf_path is required and must point to a CIFAR-10 classifier checkpoint file."
        )

    state = torch.load(args.clf_path, map_location="cpu")
    state = state.get("state_dict", state)
    clf.load_state_dict(state, strict=False)

    clf = clf.to(device).eval()
    return clf

@torch.no_grad()
def _sanity_tensor(x: torch.Tensor, name: str):
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if x.dim() != 4 or x.size(1) != 3:
        raise ValueError(f"{name} must have shape (B,3,32,32), but got {tuple(x.shape)}.")

def main():
    ap = argparse.ArgumentParser()

    # -----------------------------------------------------
    # Output cache (same as before)
    # -----------------------------------------------------
    ap.add_argument(
        "--out_dir", type=str, required=True,
        help="Output cache directory. It will contain Iu/Ic/y/isii/meta.json."
    )

    # -----------------------------------------------------
    # NEW: explicit split directories (train/test caches)
    # This makes it easy to build separate caches for train and test.
    # If --split is not provided, it will be inferred from the output directory name.
    # -----------------------------------------------------
    ap.add_argument(
        "--split", type=str, choices=["train", "test"], default=None,
        help="Which CIFAR-10 split to build. If omitted, inferred from out_dir name (train/test)."
    )

    # data/loader
    ap.add_argument("--data_root", type=str, default="./data")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument(
        "--max_samples", type=int, default=0,
        help="If >0, build only this many samples (useful for quick tests)."
    )

    # ISII targets (like eval_isii_cifar10.py)
    ap.add_argument(
        "--isiis", type=float, nargs="+",
        default=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        help="List of target ISII levels. Applied per-batch (cycled by default)."
    )
    ap.add_argument(
        "--isii_schedule", type=str, choices=["cycle", "fixed"], default="cycle",
        help="cycle: rotate across batches. fixed: always use the first value in --isiis."
    )

    # PGD/ISII search params
    ap.add_argument("--eps_max", type=float, default=8/255)
    ap.add_argument("--search_iters", type=int, default=8)
    ap.add_argument("--pgd_steps", type=int, default=10)
    ap.add_argument(
        "--alpha", type=float, default=0.0,
        help="If 0 => alpha=None and the module's default (2/255) will be used."
    )
    ap.add_argument("--tol", type=float, default=1e-3)

    # downstream classifier
    ap.add_argument(
        "--clf_arch", type=str, default="resnet18_cifar",
        choices=["robustbench", "resnet18_cifar"]
    )
    ap.add_argument("--rb_model", type=str, default="Carmon2019Unlabeled")
    ap.add_argument("--clf_path", type=str, default=None)

    # misc
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    # -----------------------------------------------------
    # NEW: infer split if not explicitly provided
    # -----------------------------------------------------
    if args.split is None:
        out_lower = os.path.basename(os.path.normpath(args.out_dir)).lower()
        if "train" in out_lower:
            args.split = "train"
        elif "test" in out_lower:
            args.split = "test"
        else:
            raise ValueError(
                "--split was not provided and could not be inferred from --out_dir. "
                "Please pass --split train or --split test."
            )

    # reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # prepare cache dirs
    _ensure_dir(args.out_dir)
    _ensure_dir(os.path.join(args.out_dir, "Iu"))
    _ensure_dir(os.path.join(args.out_dir, "Ic"))
    _ensure_dir(os.path.join(args.out_dir, "y"))
    _ensure_dir(os.path.join(args.out_dir, "isii"))

    # CIFAR10 in [0,1]
    tf = transforms.Compose([transforms.ToTensor()])
    is_train = (args.split == "train")
    ds = datasets.CIFAR10(root=args.data_root, train=is_train, download=True, transform=tf)

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # downstream model + forward_fn wrapper
    clf = _build_classifier(args, device=device)
    clf_forward = make_cifar10_forward_fn(clf, device=device)

    alpha = None if float(args.alpha) == 0.0 else float(args.alpha)

    # meta
    meta = {
        "dataset": "cifar10",
        "split": args.split,
        "num_total": len(ds),
        "out_dir": args.out_dir,
        "isiis": list(map(float, args.isiis)),
        "isii_schedule": args.isii_schedule,
        "eps_max": float(args.eps_max),
        "search_iters": int(args.search_iters),
        "pgd_steps": int(args.pgd_steps),
        "alpha": None if alpha is None else float(alpha),
        "tol": float(args.tol),
        "clf_arch": args.clf_arch,
        "rb_model": args.rb_model if args.clf_arch == "robustbench" else None,
        "clf_path": args.clf_path if args.clf_arch != "robustbench" else None,
        "seed": int(args.seed),
    }
    _save_meta(args.out_dir, meta)

    # build
    global_idx = 0
    pbar = tqdm(loader, desc=f"build cache [{args.split}] -> {args.out_dir}")

    for batch_idx, (Iu, y) in enumerate(pbar):
        if args.max_samples > 0 and global_idx >= args.max_samples:
            break

        Iu = Iu.to(device, non_blocking=True).float().clamp(0.0, 1.0)
        y = y.to(device, non_blocking=True).long()

        _sanity_tensor(Iu, "Iu")

        # pick target ISII for this batch
        if args.isii_schedule == "fixed":
            target_isii = float(args.isiis[0])
        else:
            target_isii = float(args.isiis[batch_idx % len(args.isiis)])

        # -------- Build Ic via PGD on downstream model (paper protocol) --------
        Ic, eps_used, isii_mean, isii_vals = make_isii_batch_via_pgd(
            clean_imgs=Iu,
            target_isii=target_isii,
            downstream_targets=y,
            forward_fn=clf_forward,
            task="cls",
            eps_max=float(args.eps_max),
            search_iters=int(args.search_iters),
            pgd_steps=int(args.pgd_steps),
            alpha=alpha,  # None => module default (2/255)
            tol=float(args.tol),
            return_isii_vals=True,  # request per-sample ISII values
        )

        # save each sample with aligned stems
        B = Iu.size(0)
        for i in range(B):
            if args.max_samples > 0 and global_idx >= args.max_samples:
                break

            stem = f"{global_idx:06d}"
            torch.save(Iu[i].detach().cpu(), os.path.join(args.out_dir, "Iu", f"{stem}.pt"))
            torch.save(Ic[i].detach().cpu(), os.path.join(args.out_dir, "Ic", f"{stem}.pt"))
            torch.save(y[i].detach().cpu(),  os.path.join(args.out_dir, "y",  f"{stem}.pt"))

            # Save the exact per-sample ISII value for this image.
            torch.save(isii_vals[i].detach().cpu(), os.path.join(args.out_dir, "isii", f"{stem}.pt"))

            global_idx += 1

        pbar.set_postfix(
            target_isii=target_isii,
            eps_used=float(eps_used),
            isii_mean=float(isii_mean),
            saved=global_idx,
        )

    print(f"✅ Done. Saved {global_idx} samples to: {args.out_dir}")

if __name__ == "__main__":
    main()
