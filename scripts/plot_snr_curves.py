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
        f"{int(kf)}" if kf.is_integer() else str(kf)
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
    raise KeyError(f"Could not find key for SNR={kf}. Available keys: {list(data.keys())[:10]} ...")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snr_json", type=str, required=True, help="Path to the SNR results JSON file")
    ap.add_argument("--prefix", type=str, default="", help="optional filename prefix")
    args = ap.parse_args()

    with open(args.snr_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    key_pairs = _to_float_keys(data)
    xs = [kf for (kf, _) in key_pairs] # These are SNR values

    psnr = []
    acc = []
    drop_ratio = []
    lpips_x = []
    lpips_y = []

    for kf in xs:
        row = _get_by_float_key(data, kf)
        psnr.append(row.get("PSNR", None))
        acc.append(row.get("ACC", None))
        drop_ratio.append(row.get("DropRatio", None))

        lp = row.get("LPIPS", None)
        if lp is not None:
            lpips_x.append(kf)
            lpips_y.append(lp)

    # ---- PSNR vs SNR ----
    plt.figure()
    plt.plot(xs, psnr, marker="o", color="blue")
    plt.xlabel("SNR (dB)")
    plt.ylabel("PSNR (dB)")
    plt.title("PSNR vs SNR")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{args.prefix}psnr_vs_snr.png")

    # ---- ACC vs SNR ----
    plt.figure()
    plt.plot(xs, acc, marker="o", color="green")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Accuracy")
    plt.title("ACC vs SNR")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{args.prefix}acc_vs_snr.png")
    
    # ---- Drop Ratio vs SNR (Innovation) ----
    plt.figure()
    plt.plot(xs, drop_ratio, marker="s", color="orange")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Drop Ratio")
    plt.title("Adaptive Token Drop Ratio vs SNR")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{args.prefix}drop_ratio_vs_snr.png")

    # ---- LPIPS vs SNR (plot only available points) ----
    if len(lpips_x) > 0:
        plt.figure()
        plt.plot(lpips_x, lpips_y, marker="o", color="red")
        plt.xlabel("SNR (dB)")
        plt.ylabel("LPIPS")
        plt.title("LPIPS vs SNR")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{args.prefix}lpips_vs_snr.png")

    saved_files = f"{args.prefix}psnr_vs_snr.png, {args.prefix}acc_vs_snr.png, {args.prefix}drop_ratio_vs_snr.png"
    if len(lpips_x) > 0:
        saved_files += f", {args.prefix}lpips_vs_snr.png"
    else:
        saved_files += " (LPIPS not available)"
        
    print(f"Saved plots: {saved_files}")

if __name__ == "__main__":
    main()
