# deepsc_ri/channel.py
import torch
from typing import Optional

__all__ = [
    "snr_db_to_noise_std",
    "noise_std_from_snr_db_unit_power",
    "snr_to_noise_std",
    "awgn_noise_like",
    "rician_H_real",
    "rician_H",
]

# ============================================================
# Noise Std from SNR
# ============================================================

def snr_db_to_noise_std(
    snr_db: float,
    tx_power: float = 1.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    If E[Tx^2] = tx_power then:
      sigma^2 = tx_power / SNR_lin
      sigma   = sqrt(tx_power / SNR_lin)
    """
    snr_lin = 10.0 ** (snr_db / 10.0)
    sigma = (tx_power / snr_lin) ** 0.5
    return torch.tensor(float(sigma), device=device, dtype=dtype)


def noise_std_from_snr_db_unit_power(
    snr_db: float,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    If Tx is power-normalized: E[Tx^2]=1
      sigma = sqrt(1 / SNR_lin) = 10^(-snr_db/20)
    """
    sigma = 10.0 ** (-snr_db / 20.0)
    return torch.tensor(float(sigma), device=device, dtype=dtype)


def snr_to_noise_std(signal: torch.Tensor, snr_db: float) -> torch.Tensor:
    """
    Backward compatible:
      std = sqrt( E[signal^2] / SNR_lin )
    If signal is already power-normalized, this equals unit_power formula.
    """
    with torch.no_grad():
        tx_power = float(signal.detach().float().pow(2).mean().cpu().item())
    return snr_db_to_noise_std(snr_db, tx_power=tx_power, device=signal.device, dtype=signal.dtype)


# ============================================================
# AWGN
# ============================================================

def awgn_noise_like(x: torch.Tensor, noise_std: torch.Tensor) -> torch.Tensor:
    return noise_std * torch.randn_like(x)


# ============================================================
# Rician fading (real-valued)
# ============================================================

def rician_H_real(
    shape,
    K_db: float = 7.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Real-valued Rician fading with avg power ~1:
      H = sqrt(K/(K+1)) + sqrt(1/(K+1)) * N(0,1)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    K = 10.0 ** (K_db / 10.0)
    s = (K / (K + 1.0)) ** 0.5
    sigma = (1.0 / (K + 1.0)) ** 0.5

    LoS = s * torch.ones(shape, device=device, dtype=dtype)
    NLoS = sigma * torch.randn(shape, device=device, dtype=dtype)
    return LoS + NLoS


def rician_H(
    shape,
    K_db: float = 7.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Alias for backward-compatible imports."""
    return rician_H_real(shape, K_db=K_db, device=device, dtype=dtype)
