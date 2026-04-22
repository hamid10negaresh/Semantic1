import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================
# Basic blocks
# =========================================================

class MLP(nn.Module):
    def __init__(self, d_in: int, d_h: int, d_out: int, act=nn.ReLU):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_h),
            act(),
            nn.Linear(d_h, d_out),
        )

    def forward(self, x):
        return self.net(x)


class PatchEmbed(nn.Module):
    """Conv patch embedding: (B,3,H,W) -> (B,N,D), N=(H/p)*(W/p)"""
    def __init__(self, in_ch=3, patch=4, dim=256):
        super().__init__()
        self.patch = int(patch)
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=self.patch, stride=self.patch)

    def forward(self, x):
        x = self.proj(x)                       # (B,D,H',W')
        B, D, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B,N,D)
        return tokens, (H, W)


class PosEmbed(nn.Module):
    """Learnable absolute positional embedding (fixed-length)."""
    def __init__(self, n_patches: int, dim: int):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, n_patches, dim))
        nn.init.trunc_normal_(self.pe, std=0.02)

    def forward(self, x):
        if x.size(1) != self.pe.size(1):
            raise ValueError(f"PosEmbed mismatch: got N={x.size(1)} expected {self.pe.size(1)}")
        return x + self.pe


# =========================================================
# Attention with EXACT Md on logits (Eq.9-10)
# =========================================================

class MaskedMHA(nn.Module):
    """
    Multi-head self-attention where Md is applied as:
      attn = softmax( (QK^T)/sqrt(dk) + Md )  then attn @ V
    Md masks *keys* j in C by setting column j to -inf for all queries i (Eq.9-10).
    """
    def __init__(self, dim=256, nhead=8, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % nhead == 0
        self.dim = dim
        self.nhead = nhead
        self.dk = dim // nhead
        self.scale = 1.0 / math.sqrt(self.dk)

        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, md_key_mask: torch.Tensor | None = None):
        """
        x: (B,N,D)
        md_key_mask: (B,N) bool, True => key index is in C (mask column to -inf)
        """
        B, N, D = x.shape
        qkv = self.qkv(x)                          # (B,N,3D)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, N, self.nhead, self.dk).transpose(1, 2)  # (B,h,N,dk)
        k = k.view(B, N, self.nhead, self.dk).transpose(1, 2)
        v = v.view(B, N, self.nhead, self.dk).transpose(1, 2)

        attn_logits = (q @ k.transpose(-2, -1)) * self.scale  # (B,h,N,N)

        if md_key_mask is not None:
            md = md_key_mask[:, None, None, :].to(torch.bool)  # (B,1,1,N)
            attn_logits = attn_logits.masked_fill(md, float("-inf"))

        attn = torch.softmax(attn_logits, dim=-1)  # SoftMax(At + Md)  (Eq.10)
        attn = self.attn_drop(attn)

        out = attn @ v                              # (B,h,N,dk)
        out = out.transpose(1, 2).contiguous().view(B, N, D)  # (B,N,D)
        out = self.proj_drop(self.proj(out))
        return out


class SelfAttnBlockExactMd(nn.Module):
    """Pre-LN Transformer block using MaskedMHA (for fine branch)."""
    def __init__(self, dim=256, nhead=8, mlp_ratio=4.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = MaskedMHA(dim=dim, nhead=nhead)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x, md_key_mask=None):
        x = x + self.attn(self.ln1(x), md_key_mask=md_key_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class SelfAttnBlock(nn.Module):
    """Pre-LN Transformer block (no Md), for coarse branch."""
    def __init__(self, dim=256, nhead=8, mlp_ratio=4.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, nhead, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x):
        h = self.ln1(x)
        y, _ = self.attn(h, h, h)
        x = x + y
        x = x + self.mlp(self.ln2(x))
        return x


# =========================================================
# Fine-grained semantic extractor (Eq.7-11)
# =========================================================

class FineGrainedExtractor(nn.Module):
    """
    Table II: patch=2, layers=3, heads=8
    Importance evaluator: backbone=256 + sigmoid (Eq.8)
    Dynamic mask Md: add -inf on masked key columns before softmax (Eq.9-10)
    """
    def __init__(self, img_hw=32, in_ch=3, dim=256, patch=2, depth=3, nhead=8, k_drop_ratio=0.15):
        super().__init__()
        assert img_hw % patch == 0
        self.patch = PatchEmbed(in_ch=in_ch, patch=patch, dim=dim)
        n_patches = (img_hw // patch) * (img_hw // patch)
        self.pos = PosEmbed(n_patches=n_patches, dim=dim)

        self.blocks = nn.ModuleList([SelfAttnBlockExactMd(dim=dim, nhead=nhead) for _ in range(depth)])

        # f_epsilon (Eq.8): 256 backbone + sigmoid (Table II)
        self.importance = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.k_drop_ratio = float(k_drop_ratio)
        self.out_ln = nn.LayerNorm(dim)  # f_lambda (Eq.11) as LN + mean pool

    def forward(self, x, dynamic_drop_ratio=None):
        tokens, _ = self.patch(x)
        tokens = self.pos(tokens)
        B, N, D = tokens.shape

        scores = self.importance(tokens).squeeze(-1)
        
        # --- نوآوری: استفاده از نرخ هرس تطبیقی ---
        current_ratio = dynamic_drop_ratio if dynamic_drop_ratio is not None else self.k_drop_ratio
        k = max(0, int(current_ratio * N)) # اگر 0 باشد هیچ توکنی حذف نمی‌شود

        if k > 0:
            idx = scores.topk(k, dim=1, largest=False).indices
            md_key_mask = torch.zeros(B, N, dtype=torch.bool, device=tokens.device)
            md_key_mask.scatter_(1, idx, True)
        else:
            md_key_mask = None # بدون هرس

        for blk in self.blocks:
            tokens = blk(tokens, md_key_mask=md_key_mask)

        tokens = self.out_ln(tokens)
        return tokens.mean(dim=1)


# =========================================================
# Coarse-grained semantic extractor (Eq.12-16)
# =========================================================

class CoarseGrainedExtractor(nn.Module):
    """
    Table II: patch=4, layers=3, heads=8
    3-level pooling then concat then Linear+Sigmoid head (Eq.14-16)
    """
    def __init__(self, img_hw=32, in_ch=3, dim=256, patch=4, depth=3, nhead=8):
        super().__init__()
        assert img_hw % patch == 0
        self.patch = PatchEmbed(in_ch=in_ch, patch=patch, dim=dim)
        Hc = img_hw // patch
        Wc = img_hw // patch
        self.pos = PosEmbed(n_patches=Hc * Wc, dim=dim)

        self.blocks = nn.ModuleList([SelfAttnBlock(dim=dim, nhead=nhead) for _ in range(depth)])

        # Eq.(16): Linear + Sigmoid on concat(S1,S2,S3) where each Si is (B,D)
        self.head = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.Sigmoid()
        )

    @staticmethod
    def _spatial_mean(feat_map):
        return feat_map.flatten(2).mean(dim=-1)  # (B,D,H,W)->(B,D)

    def forward(self, x):
        tokens, (H, W) = self.patch(x)  # Eq.(12)
        tokens = self.pos(tokens)

        for blk in self.blocks:         # Eq.(13)
            tokens = blk(tokens)

        B, N, D = tokens.shape
        Sv = tokens.transpose(1, 2).reshape(B, D, H, W)  # (B,D,Hc,Wc)

        # Eq.(14) pooling levels
        l1 = Sv
        l2 = F.avg_pool2d(l1, kernel_size=2, stride=2) if (H >= 2 and W >= 2) else l1
        H2, W2 = l2.shape[-2], l2.shape[-1]
        l3 = F.avg_pool2d(l2, kernel_size=2, stride=2) if (H2 >= 2 and W2 >= 2) else l2

        S1 = self._spatial_mean(l1)
        S2 = self._spatial_mean(l2)
        S3 = self._spatial_mean(l3)

        S = torch.cat([S1, S2, S3], dim=-1)  # Eq.(15)
        Sc = self.head(S)                    # Eq.(16)
        return Sc


# =========================================================
# Semantic fusion module (Eq.17-22) — EXACT per paper
# =========================================================

class ProjLayer(nn.Module):
    """
    GroupNorm + Swish + 1x1 Conv2D + Dropout(0.1)
    Input/output are vectors (B,C). For Conv2D we reshape to (B,C,1,1).
    """
    def __init__(self, in_ch: int, gn_groups: int, out_ch: int, drop: float = 0.1):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups=gn_groups, num_channels=in_ch)
        self.act = nn.SiLU()                 # Swish
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.drop = nn.Dropout2d(drop)

    def forward(self, x):  # x: (B,in_ch)
        x = x.unsqueeze(-1).unsqueeze(-1)    # (B,C,1,1)
        x = self.drop(self.conv(self.act(self.gn(x))))
        return x.squeeze(-1).squeeze(-1)     # (B,out_ch)


class FusionModule(nn.Module):
    """
    Implements Eq.(17)-(22) using Table II settings:
    - Fine projection: GN(256) + Swish + Conv2D(128) + Dropout(0.1)
    - Coarse projection: GN(64) + Swish + Conv2D(128) + Dropout(0.1)
    - Cross-Attention: 4 heads, embed_dim=256
    - Final projection: Linear(256) + Swish
    """
    def __init__(self, dim=256, nhead=4, drop=0.1):
        super().__init__()

        # (17) and (18) projection layers (Table II)
        self.proj_f = ProjLayer(in_ch=dim, gn_groups=256, out_ch=128, drop=drop)  # Fine
        self.proj_c = ProjLayer(in_ch=dim, gn_groups=64,  out_ch=128, drop=drop)  # Coarse

        # (19)-(21) Q,K,V
        self.q = nn.Linear(128, dim)
        self.k = nn.Linear(128, dim)
        self.v = nn.Linear(256, dim)  # concat(128,128)=256 -> 256

        # Cross Attention (4 heads) (Table II)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=nhead, batch_first=True)

        # (22) final projection layer: Linear + Swish (Table II)
        self.out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),   # Swish
        )

    def forward(self, Sf, Sc):
        # Sf, Sc: (B,256)
        Sf_ = self.proj_f(Sf)   # (B,128)
        Sc_ = self.proj_c(Sc)   # (B,128)

        Q = self.q(Sf_).unsqueeze(1)                            # (B,1,256)
        K = self.k(Sf_).unsqueeze(1)                            # (B,1,256)
        V = self.v(torch.cat([Sf_, Sc_], dim=-1)).unsqueeze(1)  # (B,1,256)

        y, _ = self.attn(Q, K, V)        # softmax(QK^T)V
        Sm = self.out(y.squeeze(1))      # (B,256)
        return Sm


# =========================================================
# Channel codec (Table II)
# =========================================================

class ChannelEncoder(nn.Module):
    def __init__(self, dim=256, ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(),
            nn.Linear(512, ch),
            nn.LayerNorm(ch),
        )

    def forward(self, Sm):
        Tx = self.net(Sm)          # (B,64)
        # Power constraint: normalize RMS power to 1
        rms = Tx.pow(2).mean(dim=1, keepdim=True).sqrt().clamp_min(1e-8)
        Tx = Tx / rms
        return Tx


class ChannelDecoder(nn.Module):
    def __init__(self, ch: int = 64, dim: int = 256):
        super().__init__()  # <-- MUST be first before assigning submodules

        self.ch = ch
        self.dim = dim

        self.net = nn.Sequential(
            nn.Linear(ch, 512),
            nn.ReLU(),
            # Paper Eq.(4): output \hat{S}_m with same dim as S_m
            nn.Linear(512, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, y):
        return self.net(y)

# =========================================================
# Semantic decoder (3 ResBlocks + self-attention + projection)
# =========================================================

def _gn(ch: int, groups: int = 8) -> nn.GroupNorm:
    # Table II does not specify GN group count; 8 is a common choice.
    assert ch % groups == 0, f"channels={ch} must be divisible by groups={groups}"
    return nn.GroupNorm(groups, ch)


class SpatialAttention(nn.Module):
    """Attention module (Table II): attention over spatial axis (H*W)."""
    def __init__(self, ch: int = 256, nhead: int = 8):
        super().__init__()
        assert ch % nhead == 0
        self.attn = nn.MultiheadAttention(embed_dim=ch, num_heads=nhead, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,H,W)
        B, C, H, W = x.shape
        t = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        y, _ = self.attn(t, t, t)         # (B, HW, C)
        t = t + y                         # residual
        return t.transpose(1, 2).reshape(B, C, H, W)


class ResBlock(nn.Module):
    """
    Table II ResBlock:
      GroupNorm x4 (Swish)
      Conv2D    x4
      Dropout
      Residual connection
    """
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.1, gn_groups: int = 8):
        super().__init__()
        self.act = nn.SiLU()  # Swish

        self.gn1 = _gn(in_ch, gn_groups)
        self.c1  = nn.Conv2d(in_ch,  out_ch, 3, padding=1)

        self.gn2 = _gn(out_ch, gn_groups)
        self.c2  = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.gn3 = _gn(out_ch, gn_groups)
        self.c3  = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.gn4 = _gn(out_ch, gn_groups)
        self.c4  = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.drop = nn.Dropout2d(p=dropout)

        # residual path (if channels change)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.c1(self.act(self.gn1(x)))
        h = self.c2(self.act(self.gn2(h)))
        h = self.c3(self.act(self.gn3(h)))
        h = self.c4(self.act(self.gn4(h)))
        h = self.drop(h)
        return self.skip(x) + h


class SemanticDecoder(nn.Module):
    """
    Paper-faithful semantic decoder for pixel-wise CE:
    Head Layer: Conv2D -> 256
    ResBlock(256) + Attention(256)
    ResBlock -> 128
    ResBlock -> 64
    Upsampling: Interpolation x2 + Conv2D(64) (until img_hw)
    Projection: GroupNorm(256)+Swish + Conv2D -> (3*256) pixel logits
    """
    def __init__(
        self,
        dim: int = 64,
        img_ch: int = 3,
        img_hw: int = 32,
        base_hw: int = 4,
        nhead: int = 8,
        dropout: float = 0.1,
        gn_groups: int = 8,
        num_bins: int = 256,
    ):
        super().__init__()
        assert img_hw % base_hw == 0

        self.dim = dim
        self.img_hw = img_hw
        self.base_hw = base_hw
        self.img_ch = img_ch
        self.num_bins = int(num_bins)

        # (B,dim) -> (B,dim,base,base)
        self.fc = nn.Linear(dim, dim * base_hw * base_hw)

        # Head Layer: Conv2D -> 256
        self.head = nn.Conv2d(dim, 256, kernel_size=3, padding=1)

        # ResBlock(256) + Attention
        self.rb256 = ResBlock(256, 256, dropout=dropout, gn_groups=gn_groups)
        self.attn = SpatialAttention(ch=256, nhead=nhead)

        # ResBlock -> 128 -> 64
        self.rb128 = ResBlock(256, 128, dropout=dropout, gn_groups=gn_groups)
        self.rb64  = ResBlock(128,  64, dropout=dropout, gn_groups=gn_groups)

        # Upsampling conv: Conv2D -> 64
        self.up_conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Projection: GN(256)+Swish + Conv2D -> (3*256) logits
        self.pre_proj_64_to_256 = nn.Conv2d(64, 256, kernel_size=1)
        self.proj_gn  = _gn(256, gn_groups)
        self.proj_act = nn.SiLU()  # Swish
        self.proj     = nn.Conv2d(256, img_ch * self.num_bins, kernel_size=3, padding=1)

    def forward(self, Sm_hat: torch.Tensor) -> torch.Tensor:
        B = Sm_hat.size(0)
        feat = self.fc(Sm_hat).view(B, self.dim, self.base_hw, self.base_hw)  # (B,dim,base,base)
        feat = self.head(feat)                                               # (B,256,base,base)

        feat = self.rb256(feat)                                              # ResBlock(256)
        feat = self.attn(feat)                                               # Attention module

        feat = self.rb128(feat)                                              # -> 128
        feat = self.rb64(feat)                                               # -> 64

        while feat.shape[-1] < self.img_hw:
            feat = F.interpolate(feat, scale_factor=2, mode="nearest")       # Interpolation x2
            feat = self.up_conv(feat)                                        # Conv2D -> 64

        feat = self.pre_proj_64_to_256(feat)                                 # (B,256,H,W)
        logits = self.proj(self.proj_act(self.proj_gn(feat)))                # (B,3*256,H,W)
        return logits


# =========================================================
# Full DeepSC-RI (Eq.1-5 / Algorithm)
# =========================================================

class DeepSCRI(nn.Module):
    def __init__(self, img_hw=32, dim=256, ch=64):
        super().__init__()
        self.fine = FineGrainedExtractor(img_hw=img_hw, dim=dim, patch=2, depth=3, nhead=8, k_drop_ratio=0.15)
        self.coarse = CoarseGrainedExtractor(img_hw=img_hw, dim=dim, patch=4, depth=3, nhead=8)
        self.fuse = FusionModule(dim=dim, nhead=4)

        self.chan_enc = ChannelEncoder(dim=dim, ch=ch)
        self.chan_dec = ChannelDecoder(ch=ch, dim=dim)

        # Decoder outputs pixel logits for 256 bins per channel (paper CE)
        self.sem_dec = SemanticDecoder(
            # Paper Eq.(4): SemanticDecoder consumes \hat{S}_m (dim=256)
            dim=dim, img_ch=3, img_hw=img_hw, base_hw=4, nhead=8, dropout=0.1, gn_groups=8, num_bins=256
        )

    def forward(self, I, H=None, noise_std=0.0):
        # --- محاسبه نرخ هرس بر اساس نویز ---
        # یک فرمول ساده: هرچه نویز بیشتر، هرس بیشتر (برای تمرکز روی ویژگی‌های مهم)
        # این فرمول می‌تواند در پایان‌نامه شما یک جدول کامل از آزمایشات باشد
        if noise_std < 0.18:
            drop_ratio = 0.05  # کانال خوب: 5% هرس
        elif noise_std < 0.32:
            drop_ratio = 0.15  # کانال متوسط: 15% هرس (مثل مقاله اصلی)
        else:
            drop_ratio = 0.35  # کانال بد: 35% هرس برای مقاومت بالا

        #  چاپ دیباگ همینجا انجام شود (چون هم noise_std و هم drop_ratio اینجا در دسترس هستند)
        if not self.training:
            # استخراج مقدار عددی نویز (چه تنسور باشد چه فلوت)
            n_val = noise_std.item() if hasattr(noise_std, 'item') else float(noise_std)
            print(f"[Debug] Noise Std: {n_val:.3f} --> Drop Ratio: {drop_ratio:.2f}")

        Sf = self.fine(I, dynamic_drop_ratio=drop_ratio)
        Sc = self.coarse(I)
        Sm = self.fuse(Sf, Sc)

        Tx = self.chan_enc(Sm)
        if H is None:
            H = torch.ones_like(Tx)

        Rx = H * Tx + noise_std * torch.randn_like(Tx)

        Sm_hat = self.chan_dec(Rx)
        logits = self.sem_dec(Sm_hat)
        return logits, Tx, Rx
