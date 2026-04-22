import os
import json
from typing import Optional, Tuple, List, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


# -------------------------------
# Helpers: deterministic file ops
# -------------------------------

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _list_pt(dir_path: str) -> List[str]:
    if not os.path.isdir(dir_path):
        return []
    return [f for f in os.listdir(dir_path) if f.endswith(".pt")]

def _stem(pt_name: str) -> str:
    # "000123.pt" -> "000123"
    return os.path.splitext(pt_name)[0]

def _paired_pt_lists(cache_dir: str, return_isii: bool) -> Dict[str, List[str]]:
    """
    Ensures Iu/Ic/y/(isii) are aligned by filename stem intersection.
    Returns dict with absolute paths.
    """
    iu_dir = os.path.join(cache_dir, "Iu")
    ic_dir = os.path.join(cache_dir, "Ic")
    y_dir  = os.path.join(cache_dir, "y")
    is_dir = os.path.join(cache_dir, "isii")

    iu = _list_pt(iu_dir)
    ic = _list_pt(ic_dir)
    yy = _list_pt(y_dir)

    iu_stems = set(map(_stem, iu))
    ic_stems = set(map(_stem, ic))
    y_stems  = set(map(_stem, yy))

    common = iu_stems & ic_stems & y_stems
    if len(common) == 0:
        raise RuntimeError(
            f"No common stems found in cache: {cache_dir}\n"
            f"Check that Iu/Ic/y filenames match (e.g., 000001.pt)."
        )

    stems_sorted = sorted(common)

    out = {
        "Iu": [os.path.join(iu_dir, s + ".pt") for s in stems_sorted],
        "Ic": [os.path.join(ic_dir, s + ".pt") for s in stems_sorted],
        "y":  [os.path.join(y_dir,  s + ".pt") for s in stems_sorted],
    }

    # ---------------------------
    # IMPROVEMENT (paper-style):
    # if return_isii=True => isii/ MUST exist and be complete
    # ---------------------------
    if return_isii:
        if not os.path.isdir(is_dir):
            raise RuntimeError(
                f"return_isii=True but missing isii/ directory in cache: {cache_dir}\n"
                f"Expected: {is_dir}"
            )

        is_files = _list_pt(is_dir)
        is_stems = set(map(_stem, is_files))
        common2 = common & is_stems

        if len(common2) != len(common):
            missing = sorted(common - is_stems)[:10]
            raise RuntimeError(
                f"ISII cache incomplete in {cache_dir}. Missing examples like: {missing}"
            )

        out["isii"] = [os.path.join(is_dir, s + ".pt") for s in stems_sorted]

    return out


# -----------------------------------------
# CIFAR-10 base dataset (Iu in [0,1], y)
# -----------------------------------------

class CIFAR10Base(Dataset):
    """
    Returns (paper-style interface):
      Iu: (3,32,32) float in [0,1]
      Ic: (3,32,32) float in [0,1]  (fallback = Iu)
      y : torch.long scalar
    """
    def __init__(self, root="./data", train=True, download=True):
        super().__init__()
        tf = transforms.Compose([transforms.ToTensor()])
        self.ds = datasets.CIFAR10(root=root, train=train, download=download, transform=tf)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        Iu, y = self.ds[idx]
        Iu = Iu.float().clamp(0.0, 1.0)
        Ic = Iu.clone()  # fallback (NOT paper-accurate corruption)
        y = torch.tensor(y, dtype=torch.long)
        return Iu, Ic, y


# -------------------------------------------------------
# Cached semantic corruption dataset (paper-style)
# -------------------------------------------------------

class SemanticCorruptionCacheDataset(Dataset):
    """
    Cache-backed dataset for (Iu, Ic, y[, isii]).

    Expected:
      cache_dir/
        meta.json (optional)
        Iu/*.pt
        Ic/*.pt
        y/*.pt
        isii/*.pt (optional)
    Filenames MUST match by stem (e.g. 000123.pt).
    """
    def __init__(self, cache_dir: str, return_isii: bool = False):
        super().__init__()
        self.cache_dir = cache_dir
        self.return_isii = return_isii

        paths = _paired_pt_lists(cache_dir, return_isii=return_isii)
        self.Iu_files = paths["Iu"]
        self.Ic_files = paths["Ic"]
        self.y_files  = paths["y"]
        self.isii_files = paths.get("isii", None)

        meta_path = os.path.join(cache_dir, "meta.json")
        self.meta = None
        if os.path.isfile(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                self.meta = json.load(f)

    def __len__(self):
        return len(self.Iu_files)

    def __getitem__(self, idx):
        Iu = torch.load(self.Iu_files[idx]).float().clamp(0.0, 1.0)
        Ic = torch.load(self.Ic_files[idx]).float().clamp(0.0, 1.0)
        y  = torch.load(self.y_files[idx])

        # unify label type:
        # - CIFAR: scalar => torch.long scalar
        # - Segmentation: (H,W) => torch.long
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y)
        y = y.long()

        if y.numel() == 1:
            y = y.view(())  # scalar tensor

        if self.return_isii and (self.isii_files is not None):
            isii = torch.load(self.isii_files[idx])
            if not isinstance(isii, torch.Tensor):
                isii = torch.tensor(isii)
            isii = isii.float().view(())  # scalar
            return Iu, Ic, y, isii

        return Iu, Ic, y


# -------------------------------------------------------
# Paper-style loaders for CIFAR-10
# -------------------------------------------------------

def cifar10_loaders_paper(
    batch_size: int = 64,
    num_workers: int = 2,
    root: str = "./data",
    cache_dir: Optional[str] = None,
    split_isii: Optional[float] = None,
    return_isii: bool = False,
    pin_memory: Optional[bool] = None,
):
    """
    Paper-style: training/eval on pairs (Iu, Ic, y[, isii]).

    If cache_dir + split_isii provided:
      cache_dir/cifar10/train/isii_0.4/...
      cache_dir/cifar10/test/isii_0.4/...

    Else:
      fallback to base CIFAR10 but still returns (Iu, Ic, y).
    """
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    if cache_dir is None or split_isii is None:
        train_ds = CIFAR10Base(root=root, train=True, download=True)
        test_ds  = CIFAR10Base(root=root, train=False, download=True)
    else:
        isii_tag = f"isii_{split_isii:.1f}"
        train_cache = os.path.join(cache_dir, "cifar10", "train", isii_tag)
        test_cache  = os.path.join(cache_dir, "cifar10", "test",  isii_tag)

        train_ds = SemanticCorruptionCacheDataset(train_cache, return_isii=return_isii)
        test_ds  = SemanticCorruptionCacheDataset(test_cache,  return_isii=return_isii)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False
    )
    return train_loader, test_loader


# -------------------------------------------------------
# ADE20K: cache-backed (recommended)
# -------------------------------------------------------

class ADE20KCacheDataset(Dataset):
    """
    Paper-style for segmentation:
      returns (Iu, Ic, mask) or (Iu, Ic, mask, isii)
    """
    def __init__(self, cache_dir: str, return_isii: bool = False):
        super().__init__()
        self.base = SemanticCorruptionCacheDataset(cache_dir, return_isii=return_isii)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        return self.base[idx]


# -------------------------------------------------------
# Non-cache fallback for ADE20K tensors (NOT paper-accurate)
# -------------------------------------------------------

class ADE20KStub(Dataset):
    def __init__(self, data_dir: str):
        super().__init__()
        self.img_dir = os.path.join(data_dir, "images")
        self.lbl_dir = os.path.join(data_dir, "labels")

        self.imgs = sorted([os.path.join(self.img_dir, f) for f in os.listdir(self.img_dir) if f.endswith(".pt")])
        self.lbls = sorted([os.path.join(self.lbl_dir, f) for f in os.listdir(self.lbl_dir) if f.endswith(".pt")])

        assert len(self.imgs) == len(self.lbls), "images/ and labels/ length mismatch"

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        x = torch.load(self.imgs[idx]).float().clamp(0.0, 1.0)  # (3,H,W)
        y = torch.load(self.lbls[idx])
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y)
        y = y.long()
        return x, y
