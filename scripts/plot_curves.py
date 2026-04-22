import json, argparse
import matplotlib.pyplot as plt

def _to_float_keys(data: dict):
    """
    Robustly map json keys to float, keeping original key strings for lookup.
    Returns list of tuples: (key_float, key_str)
    """
    out = []
    for k in data.keys():
        try:
            out.append((float(k), k))
        except Exception:
            # ignore non-numeric keys
            pass
    out.sort(key=lambda t: t[0])
    return out

def _get_by_float_key(data: dict, kf: float):
    """
    Robust lookup: tries several string formats to find the matching entry.
    """
    # direct exact float->str may fail; try common formats
    candidates = [
        str(kf),
        f"{kf:.1f}",
        f"{kf:.2f}",
        f"{kf:.3f}",
    ]
    # also try any key that parses to same float within tolerance
    for ks in data.keys():
        try:
            if abs(float(ks) - kf) < 1e-9:
                candidates.insert(0, ks)
                break
        except Exception:
            continue

    for c in candidates:
        if c in data:
            return data[c]
    raise KeyError(f"Could not find key for ISII={kf}. Available keys: {list(data.keys())[:10]} ...")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--isii_json", type=str, default="isii_results.json")
    ap.add_argument("--prefix", type=str, default="", help="optional filename prefix")
    args = ap.parse_args()

    with open(args.isii_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    key_pairs = _to_float_keys(data)
    xs = [kf for (kf, _) in key_pairs]

    psnr = []
    acc = []
    lpips_x = []
    lpips_y = []

    for kf in xs:
        row = _get_by_float_key(data, kf)
        psnr.append(row.get("PSNR", None))
        acc.append(row.get("ACC", None))

        lp = row.get("LPIPS", None)
        if lp is not None:
            lpips_x.append(kf)
            lpips_y.append(lp)

    # ---- PSNR vs ISII ----
    plt.figure()
    plt.plot(xs, psnr, marker="o")
    plt.xlabel("ISII")
    plt.ylabel("PSNR (dB)")
    plt.title("PSNR vs ISII")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{args.prefix}psnr_vs_isii.png")

    # ---- ACC vs ISII ----
    plt.figure()
    plt.plot(xs, acc, marker="o")
    plt.xlabel("ISII")
    plt.ylabel("Accuracy")
    plt.title("ACC vs ISII")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{args.prefix}acc_vs_isii.png")

    # ---- LPIPS vs ISII (plot only available points) ----
    if len(lpips_x) > 0:
        plt.figure()
        plt.plot(lpips_x, lpips_y, marker="o")
        plt.xlabel("ISII")
        plt.ylabel("LPIPS")
        plt.title("LPIPS vs ISII")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{args.prefix}lpips_vs_isii.png")

    print(
        f"Saved plots: {args.prefix}psnr_vs_isii.png, {args.prefix}acc_vs_isii.png"
        + (f", {args.prefix}lpips_vs_isii.png" if len(lpips_x) > 0 else " (LPIPS not available)")
    )

if __name__ == "__main__":
    main()
