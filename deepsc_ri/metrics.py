import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    import lpips  # optional
    _has_lpips = True
except Exception:
    _has_lpips = False


# ---------------------------
# (27)-(28) PSNR per paper
# PSNR = 10 * log10( Vmax^2 / MSE )
# ---------------------------

@torch.no_grad()
def mse_torch(img: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    Mean squared error over pixels (and channels).
    img/ref: (B,C,H,W) or (C,H,W)
    returns: (B,) or scalar
    """
    img = img.float()
    ref = ref.float()
    if img.dim() == 3:
        return (img - ref).pow(2).mean()
    elif img.dim() == 4:
        return (img - ref).pow(2).mean(dim=(1, 2, 3))
    else:
        raise ValueError("img/ref must be (C,H,W) or (B,C,H,W)")

@torch.no_grad()
def psnr_torch(img: torch.Tensor, ref: torch.Tensor, vmax: float = 1.0, eps: float = 1e-12) -> torch.Tensor:
    """
    PSNR = 10log10(Vmax^2 / MSE)
    - if tensors in [0,1], vmax=1
    - if tensors in [0,255], vmax=255
    """
    mse = mse_torch(img, ref).clamp_min(eps)
    v = torch.as_tensor(vmax, device=mse.device, dtype=mse.dtype)
    return 10.0 * torch.log10((v * v) / mse)

@torch.no_grad()
def batch_psnr(imgs: torch.Tensor, refs: torch.Tensor, vmax: float = 1.0, eps: float = 1e-12) -> float:
    """
    imgs/refs: (B,C,H,W)
    """
    ps = psnr_torch(imgs, refs, vmax=vmax, eps=eps)
    return float(ps.mean().item())


# ---------------------------
# (29) LPIPS per paper
# (standard LPIPS; expects input in [-1,1])
# ---------------------------

_lpips_model = None

@torch.no_grad()
def lpips_score(imgs: torch.Tensor, refs: torch.Tensor, net: str = "alex") -> float | None:
    """
    imgs/refs: (B,C,H,W) in [0,1]
    LPIPS expects [-1,1]
    """
    global _lpips_model
    if not _has_lpips:
        return None

    dev = imgs.device
    if (_lpips_model is None) or (next(_lpips_model.parameters()).device != dev):
        _lpips_model = lpips.LPIPS(net=net).to(dev).eval()

    imgs = imgs.float().clamp(0.0, 1.0)
    refs = refs.float().clamp(0.0, 1.0)

    imgs_ = imgs * 2.0 - 1.0
    refs_ = refs * 2.0 - 1.0

    val = _lpips_model(imgs_, refs_).mean()
    return float(val.item())


# ---------------------------
# (30) Accuracy per paper
# ---------------------------

@torch.no_grad()
def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return float((preds == targets).float().mean().item())


# ---------------------------
# (31) mIoU per paper
# mIoU = (1/k) * sum_i TP_i / (FN_i + FP_i + TP_i)
# ---------------------------

@torch.no_grad()
def miou(
    preds: torch.Tensor,
    gts: torch.Tensor,
    num_classes: int,
    ignore_index: int | None = None,
    avg_over_present: bool = False,   # اگر مقاله گفته فقط کلاس‌های حاضر، True کن
) -> float:
    """
    preds:
      - (B,H,W) indices
      - (B,C,H,W) logits/probs
    gts: (B,H,W)
    """
    if preds.dim() == 4:
        preds = preds.argmax(dim=1)
    preds = preds.long()
    gts = gts.long()

    if ignore_index is not None:
        mask = (gts != ignore_index)
        preds = preds[mask]
        gts = gts[mask]

    if preds.numel() == 0:
        return 0.0

    k = int(num_classes)
    idx = gts * k + preds
    cm = torch.bincount(idx, minlength=k * k).reshape(k, k).float()

    TP = torch.diag(cm)
    FP = cm.sum(dim=0) - TP
    FN = cm.sum(dim=1) - TP

    denom = TP + FP + FN
    iou = torch.where(denom > 0, TP / denom, torch.zeros_like(denom))

    if avg_over_present:
        present = (denom > 0)
        if present.any():
            return float(iou[present].mean().item())
        return 0.0

    return float(iou.mean().item())


# ---------------------------
# (6) ISII per paper
# ISII = 1 - cos( VΦ(Iu), VΦ(Ic) )
# ---------------------------

_vgg = None

def _get_vgg(device: torch.device):
    global _vgg
    if _vgg is not None and next(_vgg.parameters()).device == device:
        return _vgg

    try:
        from torchvision.models import vgg16, VGG16_Weights
    except Exception as e:
        raise RuntimeError("torchvision is required for ISII (VGG-based) computation.") from e

    base = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).to(device).eval()
    for p in base.parameters():
        p.requires_grad = False

    class VGGVec(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.features = m.features
            self.avgpool = m.avgpool
        def forward(self, x):
            x = self.features(x)
            x = self.avgpool(x)
            return x.flatten(1)

    _vgg = VGGVec(base)
    return _vgg

@torch.no_grad()
def isii_vgg(Iu: torch.Tensor, Ic: torch.Tensor, eps: float = 1e-12) -> float:
    """
    Returns mean ISII over batch.
    Iu/Ic expected in [0,1].
    """
    dev = Iu.device
    vgg = _get_vgg(dev)

    Iu = Iu.float().clamp(0.0, 1.0)
    Ic = Ic.float().clamp(0.0, 1.0)

    Iu_ = F.interpolate(Iu, size=(224, 224), mode="bilinear", align_corners=False)
    Ic_ = F.interpolate(Ic, size=(224, 224), mode="bilinear", align_corners=False)

    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    Iu_ = (Iu_ - mean) / std
    Ic_ = (Ic_ - mean) / std

    fu = vgg(Iu_)
    fc = vgg(Ic_)

    fu = fu / fu.norm(dim=1, keepdim=True).clamp_min(eps)
    fc = fc / fc.norm(dim=1, keepdim=True).clamp_min(eps)

    cos = (fu * fc).sum(dim=1)
    isii = 1.0 - cos
    return float(isii.mean().item())
