import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Callable, Optional, Tuple, Union


# =========================================================
# (6) ISII = 1 - cosine( VΦ(Iu), VΦ(Ic) )
# VΦ(.) : pretrained VGG (ImageNet)
# =========================================================

class VGGEmbed(nn.Module):
    """
    VGG-based embedding for ISII measurement (paper Eq.6).
    - VGG16 pretrained on ImageNet
    - resize to 224x224
    - ImageNet normalization
    - embedding = flatten( avgpool( features(x) ) )
    """
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).eval()
        for p in vgg.parameters():
            p.requires_grad = False

        self.features = vgg.features
        self.avgpool = vgg.avgpool

        # ImageNet normalization buffers
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,3,H,W) in [0,1]
        x = x.clamp(0.0, 1.0).float()
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        f = self.features(x)
        f = self.avgpool(f)
        return f.flatten(1)  # (B, D)


@torch.no_grad()
def isii_from_emb(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    a,b: (B,D)
    returns ISII per sample: (B,) = 1 - cosine(a,b)
    """
    a = a.float()
    b = b.float()
    a = a / a.norm(dim=1, keepdim=True).clamp_min(eps)
    b = b / b.norm(dim=1, keepdim=True).clamp_min(eps)
    cos = (a * b).sum(dim=1)
    return 1.0 - cos


@torch.no_grad()
def isii(vgg: VGGEmbed, Iu: torch.Tensor, Ic: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Convenience wrapper: computes embeddings and returns ISII per sample (B,).
    """
    fu = vgg(Iu)
    fc = vgg(Ic)
    return isii_from_emb(fu, fc, eps=eps)


# =========================================================
# Linf-PGD on downstream task model (paper corruption generation)
# =========================================================

def pgd_linf(
    x_clean: torch.Tensor,
    y: torch.Tensor,
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    eps: float,
    alpha: float,
    steps: int,
    random_start: bool = True,
) -> torch.Tensor:
    """
    Untargeted Linf-PGD: maximize loss_fn(forward_fn(x), y)
    s.t. ||x - x_clean||_inf <= eps, x in [0,1]

    NOTE:
    - forward_fn must be differentiable wrt x (i.e., contains downstream model forward).
    - This function DOES NOT do any normalization itself. Do it inside forward_fn.
    """
    x0 = x_clean.detach()

    if random_start:
        x = (x0 + torch.empty_like(x0).uniform_(-eps, eps)).clamp(0.0, 1.0)
    else:
        x = x0.clone()

    for _ in range(int(steps)):
        x = x.detach().requires_grad_(True)
        out = forward_fn(x)
        loss = loss_fn(out, y)
        grad = torch.autograd.grad(loss, x, retain_graph=False, create_graph=False)[0]

        with torch.no_grad():
            x = x + alpha * grad.sign()
            x = torch.max(torch.min(x, x0 + eps), x0 - eps)
            x.clamp_(0.0, 1.0)

    return x.detach()


# =========================================================
# Paper-style ISII-targeted corruption builder
# (binary search eps to match target ISII, using PGD on downstream model)
# =========================================================

def make_isii_batch_via_pgd(
    clean_imgs: torch.Tensor,
    target_isii: float,
    downstream_targets: torch.Tensor,
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    task: str = "cls",                         # "cls" or "seg"
    vgg: Optional[VGGEmbed] = None,
    eps_max: float = 8/255,
    search_iters: int = 8,                     # binary search iterations
    pgd_steps: int = 10,
    alpha: Optional[float] = None,             # default: 2/255 unless explicitly provided
    tol: float = 1e-3,
    ignore_index: Optional[int] = None,
    cache_clean_vgg: bool = True,              # speed: cache VGG embedding of clean
    return_isii_vals: bool = False,            # NEW: return per-sample ISII for best eps
) -> Union[
    Tuple[torch.Tensor, float, float],
    Tuple[torch.Tensor, float, float, torch.Tensor]
]:
    """
    Returns (default):
      Ic: corrupted batch (PGD on downstream model)
      eps_best: epsilon used
      isii_mean: achieved mean ISII (over the batch)

    If return_isii_vals=True, returns additionally:
      isii_vals: per-sample ISII tensor of shape (B,) for the selected best eps

    IMPORTANT for matching paper protocol:
      - forward_fn MUST include the downstream preprocessing used in evaluation (e.g., CIFAR10 normalization).
      - PGD is computed on the downstream model loss (CE for cls/seg).

    Defaults aligned with typical robust benchmarks:
      eps_max=8/255, pgd_steps=10, alpha=2/255 (if alpha is None)
    """
    device = clean_imgs.device
    clean_imgs = clean_imgs.clamp(0.0, 1.0)

    if vgg is None:
        vgg = VGGEmbed().to(device).eval()

    # downstream loss
    if task == "cls":
        def loss_fn(logits, y):
            return F.cross_entropy(logits, y)
    elif task == "seg":
        def loss_fn(logits, ymask):
            return F.cross_entropy(logits, ymask, ignore_index=ignore_index)
    else:
        raise ValueError("task must be 'cls' or 'seg'")

    # cache VGG embedding of clean batch once (much faster and stable)
    if cache_clean_vgg:
        with torch.no_grad():
            fu = vgg(clean_imgs)
    else:
        fu = None

    lo, hi = 0.0, float(eps_max)
    best_Ic: Optional[torch.Tensor] = None
    best_eps: float = 0.0
    best_isii: Optional[float] = None
    best_isii_vals: Optional[torch.Tensor] = None  # NEW

    for _ in range(int(search_iters)):
        mid = (lo + hi) / 2.0

        # common step size if not specified
        step_alpha = float(2/255) if alpha is None else float(alpha)

        Ic = pgd_linf(
            x_clean=clean_imgs,
            y=downstream_targets,
            forward_fn=forward_fn,
            loss_fn=loss_fn,
            eps=mid,
            alpha=step_alpha,
            steps=pgd_steps,
            random_start=True,
        )

        with torch.no_grad():
            fu_now = vgg(clean_imgs) if fu is None else fu
            fc = vgg(Ic)
            isii_vals = isii_from_emb(fu_now, fc)         # (B,)
            isii_mean = float(isii_vals.mean().item())

        # keep best-so-far
        if (best_isii is None) or (abs(isii_mean - target_isii) < abs(best_isii - target_isii)):
            best_Ic, best_eps, best_isii = Ic, mid, isii_mean
            best_isii_vals = isii_vals.detach()          # NEW

        # binary-search update
        if isii_mean < target_isii - tol:
            lo = mid
        else:
            hi = mid

    assert best_Ic is not None and best_isii is not None

    if return_isii_vals:
        assert best_isii_vals is not None
        return best_Ic, float(best_eps), float(best_isii), best_isii_vals

    return best_Ic, float(best_eps), float(best_isii)


# =========================================================
# Ready-to-use forward_fn wrappers (to avoid mistakes)
# =========================================================

def make_cifar10_forward_fn(model: nn.Module, device: torch.device) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Returns a forward_fn(x) that applies CIFAR-10 normalization exactly like evaluation.
    x expected in [0,1], shape (B,3,32,32).
    """
    model.eval()

    mean = torch.tensor([0.4914, 0.4822, 0.4465], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.2470, 0.2435, 0.2616], device=device).view(1, 3, 1, 1)

    def forward_fn(x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(0.0, 1.0)
        x = (x - mean) / std
        return model(x)

    return forward_fn
