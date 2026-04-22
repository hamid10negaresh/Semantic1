# DeepSC-RI (Reproduction Pack)

This pack provides runnable PyTorch code to **simulate** and **reproduce** the evaluation protocol from
“A Robust Image Semantic Communication System With Multi-Scale Vision Transformer (DeepSC-RI)” (J-SAC 2025).

## What’s inside
- `deepsc_ri/models.py` — Multi-scale ViT-based semantic encoder (fine/coarse) + fusion + channel codec + semantic decoder.
- `deepsc_ri/channel.py` — AWGN & Rician channel simulators; SNR ↔ noise std helpers.
- `deepsc_ri/metrics.py` — PSNR/ACC/mIoU/LPIPS (LPIPS optional).
- `deepsc_ri/attacks/isii_pgd.py` — Targeted-PGD to reach **desired ISII** (cosine distance in VGG feature space).
- `deepsc_ri/data.py` — CIFAR-10 & ADE20K loaders and wrappers (ADE20K requires local path).
- `scripts/train_cifar10.py` — Train & evaluate vs **SNR** on CIFAR-10 under AWGN/Rician.
- `scripts/eval_isii_cifar10.py` — Evaluate vs **ISII** at fixed SNR=18 dB (AWGN & Rician).
- `scripts/train_ade20k.py` — Evaluate **mIoU** on ADE20K (semantic segmentation) at ISII=0.4 across SNRs.
- `scripts/plot_curves.py` — Plot curves similar to the paper figures.

> Note: This pack avoids internet downloads. Place ADE20K locally and set env var `ADE20K_DIR`.

## Quick start (CIFAR-10, AWGN)
```bash
python -m scripts.train_cifar10 --epochs 2 --snrs -5 0 5 10 15 --channel awgn
```

## ISII dataset (targeted)
```bash
python -m scripts.eval_isii_cifar10 --snr 18 --isiis 0.2 0.3 0.4 0.5 0.6 0.7 0.8
```

## ADE20K (segmentation)
Prepare a pretrained segmentation model or use the simple PSPNet-lite stub provided. Set `ADE20K_DIR` to the dataset root.



AAAAAAA