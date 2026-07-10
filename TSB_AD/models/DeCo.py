# -*- coding: utf-8 -*-
"""
DeCo with RATFD-v3-no-orth-flatmix + DA-RDA attention for TSB-AD.

Core:
1. RATFD-v3-no-orth-flatmix:
   X = T + S + N
   T: robust stable trend
   S: hard spectral-peak seasonal component
   N: residual noise
   Nw: flat-mix whitened residual representation for scoring only

2. DA-RDA:
   Decomposition-Aware Role-specific Dual Attention
   Trend branch    : Channel Attention dominant + Temporal Attention auxiliary
   Seasonal branch : Temporal Attention dominant + Channel Attention auxiliary

3. Training target:
   reconstruct T, S, and X_clean = T + S.
   Do not reconstruct original X directly.

4. Score:
   RobustZ(clean reconstruction error) + beta * RobustZ(white-noise energy)
"""

import math
import os
from typing import Dict, Tuple, Optional

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class RATFDv3NoOrthFlatMix(nn.Module):
    def __init__(
        self,
        trend_lam: float = 120.0,
        irls_iters: int = 5,
        huber_delta: float = 1.2,
        peak_z: float = 1.00,
        topk_ratio: float = 0.10,
        min_seasonal_freq: float = 0.05,
        max_seasonal_freq: float = 1.00,
        band_width: int = 1,
        noise_corr_lam: float = 35.0,
        lowcorr_alpha: float = 0.75,
        second_peak_z: float = 0.85,
        second_topk_ratio: float = 0.08,
        white_alpha: float = 0.35,
        use_second_stage: bool = False,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.trend_lam = float(trend_lam)
        self.irls_iters = int(irls_iters)
        self.huber_delta = float(huber_delta)

        self.peak_z = float(peak_z)
        self.topk_ratio = float(topk_ratio)
        self.min_seasonal_freq = float(min_seasonal_freq)
        self.max_seasonal_freq = float(max_seasonal_freq)
        self.band_width = int(band_width)

        self.noise_corr_lam = float(noise_corr_lam)
        self.lowcorr_alpha = float(lowcorr_alpha)

        self.second_peak_z = float(second_peak_z)
        self.second_topk_ratio = float(second_topk_ratio)

        self.white_alpha = float(white_alpha)
        self.use_second_stage = bool(use_second_stage)
        self.eps = float(eps)

    def hp_filter_fft(self, x: torch.Tensor, lam: float) -> torch.Tensor:
        """
        Frequency-domain second-order HP-like smoothing.

        Approximate:
            min_T ||X - T||_2^2 + lam * ||D2 T||_2^2

        x: [B, L, C]
        """
        B, L, C = x.shape

        x_fft = torch.fft.rfft(x, dim=1)
        freq_len = x_fft.shape[1]

        freq = torch.arange(freq_len, device=x.device, dtype=x.dtype) / max(freq_len - 1, 1)
        omega = math.pi * freq

        d2_gain = (2.0 - 2.0 * torch.cos(omega)) ** 2
        smooth_gain = 1.0 / (1.0 + float(lam) * d2_gain)

        t_fft = x_fft * smooth_gain.view(1, freq_len, 1)
        trend = torch.fft.irfft(t_fft, n=L, dim=1)

        return trend

    def robust_trend(self, x: torch.Tensor) -> torch.Tensor:
        """
        Robust IRLS trend estimation.
        Large residual points get lower weights.
        """
        trend = self.hp_filter_fft(x, self.trend_lam)

        for _ in range(self.irls_iters):
            residual = x - trend

            med = residual.median(dim=1, keepdim=True).values
            mad = (residual - med).abs().median(dim=1, keepdim=True).values
            scale = 1.4826 * mad + self.eps

            u = residual / (self.huber_delta * scale)
            weight = 1.0 / (1.0 + u.abs() ** 2)

            x_weighted = weight * x + (1.0 - weight) * trend
            trend = self.hp_filter_fft(x_weighted, self.trend_lam)

        return trend

    def hard_periodic_extract(
        self,
        residual: torch.Tensor,
        peak_z: float,
        topk_ratio: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract structured seasonal/periodic component by hard spectral peak mask.

        residual: [B, L, C]
        """
        B, L, C = residual.shape

        r_fft = torch.fft.rfft(residual, dim=1)
        amp = torch.abs(r_fft) + self.eps
        log_amp = torch.log(amp)

        freq_len = log_amp.shape[1]
        freq = torch.arange(freq_len, device=residual.device, dtype=residual.dtype) / max(freq_len - 1, 1)

        valid_freq = (
            (freq >= self.min_seasonal_freq) &
            (freq <= self.max_seasonal_freq)
        ).view(1, freq_len, 1)

        z = log_amp.permute(0, 2, 1)
        z_pad = F.pad(z, (2, 2), mode="replicate")
        bg = F.avg_pool1d(z_pad, kernel_size=5, stride=1)
        bg = bg.permute(0, 2, 1)

        peak_score = log_amp - bg

        med = peak_score.median(dim=1, keepdim=True).values
        mad = (peak_score - med).abs().median(dim=1, keepdim=True).values
        zscore = (peak_score - med) / (1.4826 * mad + self.eps)

        zscore = zscore.masked_fill(~valid_freq, -1e9)

        threshold_mask = (zscore >= float(peak_z)).float()

        k = max(1, int((freq_len - 1) * float(topk_ratio)))
        k = min(k, freq_len)

        top_idx = torch.topk(zscore, k=k, dim=1).indices
        top_mask = torch.zeros_like(zscore)
        top_mask.scatter_(1, top_idx, 1.0)

        mask = torch.maximum(threshold_mask, top_mask)
        mask = mask * valid_freq.float()
        mask[:, 0:1, :] = 0.0

        if self.band_width > 0:
            m = mask.permute(0, 2, 1)
            for _ in range(self.band_width):
                m = F.max_pool1d(
                    F.pad(m, (1, 1), mode="replicate"),
                    kernel_size=3,
                    stride=1,
                )
            mask = m.permute(0, 2, 1)

        seasonal_fft = r_fft * mask
        seasonal = torch.fft.irfft(seasonal_fft, n=L, dim=1)

        return seasonal, mask

    def whiten_noise_freq(self, noise: torch.Tensor) -> torch.Tensor:
        """
        Spectral flat-mix whitening.

        p_white = (1 - alpha) * p + alpha * uniform

        It does not modify trend or seasonal.
        It does not participate in exact reconstruction.
        It preserves residual RMS energy.
        """
        alpha = max(0.0, min(float(self.white_alpha), 1.0))

        if alpha <= 0.0:
            return noise

        B, L, C = noise.shape

        noise_mean = noise.mean(dim=1, keepdim=True)
        noise_centered = noise - noise_mean

        n_fft = torch.fft.rfft(noise_centered, dim=1)
        amp = torch.abs(n_fft) + self.eps
        phase = n_fft / amp

        power = amp ** 2
        freq_len = power.shape[1]

        if freq_len <= 2:
            return noise

        power_non_dc = power[:, 1:, :]
        total_power = power_non_dc.sum(dim=1, keepdim=True).clamp_min(self.eps)

        prob = power_non_dc / total_power
        uniform = torch.full_like(prob, 1.0 / prob.shape[1])

        prob_white = (1.0 - alpha) * prob + alpha * uniform
        power_white_non_dc = prob_white * total_power

        power_white = torch.cat(
            [torch.zeros_like(power[:, 0:1, :]), power_white_non_dc],
            dim=1,
        )

        amp_white = torch.sqrt(power_white.clamp_min(self.eps))
        white_fft = amp_white * phase
        white_centered = torch.fft.irfft(white_fft, n=L, dim=1)

        old_rms = torch.sqrt((noise_centered ** 2).mean(dim=1, keepdim=True) + self.eps)
        new_rms = torch.sqrt((white_centered ** 2).mean(dim=1, keepdim=True) + self.eps)

        white_centered = white_centered * (old_rms / new_rms)

        white_noise = white_centered + noise_mean
        return white_noise

        self.rga_enabled = bool(rga_enabled)

        self.trend_rga = ResidualGatedAttentionAdapter(
            channels=channels,
            seq_len=seq_len,
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            role="trend",
            gamma_init=gamma_init,
            rga_gate_strength=rga_gate_strength,
            rga_max_scale=rga_max_scale,
        )

        self.seasonal_rga = ResidualGatedAttentionAdapter(
            channels=channels,
            seq_len=seq_len,
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            role="seasonal",
            gamma_init=gamma_init,
            rga_gate_strength=rga_gate_strength,
            rga_max_scale=rga_max_scale,
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        trend = self.robust_trend(x)

        residual = x - trend
        seasonal_1, mask_1 = self.hard_periodic_extract(
            residual,
            peak_z=self.peak_z,
            topk_ratio=self.topk_ratio,
        )

        noise = x - trend - seasonal_1

        low_corr = self.hp_filter_fft(noise, self.noise_corr_lam)
        trend = trend + self.lowcorr_alpha * low_corr
        noise = noise - self.lowcorr_alpha * low_corr

        if self.use_second_stage:
            seasonal_2, mask_2 = self.hard_periodic_extract(
                noise,
                peak_z=self.second_peak_z,
                topk_ratio=self.second_topk_ratio,
            )
        else:
            seasonal_2 = torch.zeros_like(seasonal_1)
            mask_2 = torch.zeros_like(mask_1)

        seasonal = seasonal_1 + seasonal_2

        noise = x - trend - seasonal
        white_noise = self.whiten_noise_freq(noise)

        return {
            "trend": trend,
            "seasonal": seasonal,
            "noise": noise,
            "white_noise": white_noise,
            "clean": trend + seasonal,
            "seasonal_gate": torch.maximum(mask_1, mask_2),
        }


class TemporalAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        d_model: int = 64,
        n_heads: int = 4,
        dropout: float = 0.05,
    ):
        super().__init__()

        self.in_proj = nn.Linear(channels, d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.out_proj = nn.Linear(d_model, channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.in_proj(x)
        z, _ = self.attn(z, z, z, need_weights=False)
        z = self.out_proj(z)
        return self.drop(z)


class ChannelAttention(nn.Module):
    def __init__(
        self,
        seq_len: int,
        d_model: int = 64,
        n_heads: int = 4,
        dropout: float = 0.05,
    ):
        super().__init__()

        self.in_proj = nn.Linear(seq_len, d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.out_proj = nn.Linear(d_model, seq_len)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.transpose(1, 2)
        z = self.in_proj(z)
        z, _ = self.attn(z, z, z, need_weights=False)
        z = self.out_proj(z)
        z = z.transpose(1, 2)
        return self.drop(z)


class RoleSpecificDualAttentionBlock(nn.Module):
    """
    DA-RDA block.

    trend:
        Channel Attention dominant + Temporal Attention auxiliary

    seasonal:
        Temporal Attention dominant + Channel Attention auxiliary
    """
    def __init__(
        self,
        channels: int,
        seq_len: int,
        role: str,
        d_model: int = 64,
        n_heads: int = 4,
        dropout: float = 0.05,
        gamma_init: float = 0.75,
        rsda_mode: str = "full",
    ):
        super().__init__()

        assert role in ["trend", "seasonal"]
        self.role = role
        self.rsda_mode = str(rsda_mode)

        self.temporal = TemporalAttention(
            channels=channels,
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
        )

        self.channel = ChannelAttention(
            seq_len=seq_len,
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
        )

        gamma_init = min(max(gamma_init, 1e-4), 1.0 - 1e-4)
        self.gamma_logit = nn.Parameter(
            torch.tensor(
                math.log(gamma_init / (1.0 - gamma_init)),
                dtype=torch.float32,
            )
        )

        self.norm1 = nn.LayerNorm(channels)

        hidden = max(16, channels * 2)
        self.ffn = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, channels),
            nn.Dropout(dropout),
        )

        self.norm2 = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        temporal_out = self.temporal(x)
        channel_out = self.channel(x)

        mode = self.rsda_mode

        if mode == "temporal_only":
            mixed = temporal_out
        elif mode == "channel_only":
            mixed = channel_out
        elif mode == "equal":
            mixed = 0.5 * channel_out + 0.5 * temporal_out
        else:
            gamma = torch.sigmoid(self.gamma_logit)
            role = self.role
            if mode == "swapped":
                role = "seasonal" if self.role == "trend" else "trend"

            if role == "trend":
                mixed = gamma * channel_out + (1.0 - gamma) * temporal_out
            else:
                mixed = gamma * temporal_out + (1.0 - gamma) * channel_out

        x = self.norm1(x + mixed)
        x = self.norm2(x + self.ffn(x))

        return x


class BranchReconstructor(nn.Module):
    def __init__(
        self,
        channels: int,
        seq_len: int,
        role: str,
        d_model: int = 64,
        n_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.05,
        gamma_init: float = 0.75,
        rsda_mode: str = "full",
    ):
        super().__init__()

        self.input_proj = nn.Linear(channels, channels)

        self.blocks = nn.ModuleList([
            RoleSpecificDualAttentionBlock(
                channels=channels,
                seq_len=seq_len,
                role=role,
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
                gamma_init=gamma_init,
                rsda_mode=rsda_mode,
            )
            for _ in range(num_layers)
        ])

        self.output_proj = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.input_proj(x)

        for block in self.blocks:
            z = block(z)

        out = self.output_proj(z)
        return out



class SimpleReconstructor(nn.Module):
    """
    Plain non-RSDA reconstructor used only for the w/o RSDA-R main ablation.
    It keeps the same input/output interface but removes role-specialized
    temporal-channel attention fusion.
    """
    def __init__(
        self,
        channels: int,
        seq_len: int,
        d_model: int = 64,
        n_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.05,
    ):
        super().__init__()
        hidden = max(d_model, channels * 2, 16)
        layers = [nn.Linear(channels, hidden), nn.GELU(), nn.Dropout(dropout)]
        for _ in range(max(0, num_layers - 1)):
            layers += [nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout)]
        layers += [nn.Linear(hidden, channels)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RGATemporalAdapter(nn.Module):
    """
    Residual-gated temporal adapter.
    It produces only a small residual correction, not a replacement.
    """
    def __init__(self, channels: int, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.in_proj = nn.Linear(channels, d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
        )
        self.out_proj = nn.Linear(d_model, channels)

        # Zero-init keeps the whole model initially close to ENVHP_BEST.
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x):
        h = self.in_proj(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        h = self.norm(h + a)
        h = self.norm(h + self.ffn(h))
        return self.out_proj(h)


class RGAChannelAdapter(nn.Module):
    """
    Residual-gated channel adapter.
    Channels are treated as tokens; time is token feature.
    """
    def __init__(self, channels: int, seq_len: int, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.in_proj = nn.Linear(seq_len, d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
        )
        self.out_proj = nn.Linear(d_model, seq_len)

        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x):
        # [B, L, C] -> [B, C, L]
        h = x.transpose(1, 2)
        h = self.in_proj(h)
        a, _ = self.attn(h, h, h, need_weights=False)
        h = self.norm(h + a)
        h = self.norm(h + self.ffn(h))
        h = self.out_proj(h)
        return h.transpose(1, 2)


class ResidualGatedAttentionAdapter(nn.Module):
    """
    RGA: Residual-Gated Attention Adapter.

    It still uses two attention primitives:
      1. channel adapter
      2. temporal adapter

    It does NOT replace DA-RDA.
    It only adds a small residual correction:
        output = base_output + scale * reliability * correction

    white_noise is only used to build reliability, not as value to reconstruct.
    """
    def __init__(
        self,
        channels: int,
        seq_len: int,
        d_model: int,
        n_heads: int,
        dropout: float,
        role: str,
        gamma_init: float = 0.75,
        rga_gate_strength: float = 1.0,
        rga_max_scale: float = 0.10,
    ):
        super().__init__()
        assert role in ["trend", "seasonal"]
        self.role = role
        self.rga_gate_strength = float(rga_gate_strength)
        self.rga_max_scale = float(rga_max_scale)

        self.channel_adapter = RGAChannelAdapter(
            channels=channels,
            seq_len=seq_len,
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.temporal_adapter = RGATemporalAdapter(
            channels=channels,
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
        )

        # CA/TA role prior. Trend prefers channel; seasonal prefers temporal.
        gamma_init = max(1e-4, min(float(gamma_init), 1.0 - 1e-4))
        if role == "trend":
            self.gamma = nn.Parameter(torch.tensor(gamma_init, dtype=torch.float32))
        else:
            self.gamma = nn.Parameter(torch.tensor(1.0 - gamma_init, dtype=torch.float32))

        # Learnable scale, but bounded. Init is small and safe.
        init_ratio = 0.5
        init_logit = torch.logit(torch.tensor(init_ratio, dtype=torch.float32))
        self.scale_logit = nn.Parameter(init_logit)

    def _reliability(self, white_noise):
        """
        Reliability is high where white residual energy is low.
        white_noise: [B, L, C]
        """
        r = white_noise.pow(2)
        mu = r.mean(dim=(1, 2), keepdim=True)
        std = r.std(dim=(1, 2), keepdim=True, unbiased=False) + 1e-6
        z = (r - mu) / std

        k = max(float(self.rga_gate_strength), 1e-6)
        reliability = torch.sigmoid(-k * z)

        return reliability

    def forward(self, component, base_output, white_noise):
        reliability = self._reliability(white_noise)

        safe_component = component * reliability

        c_delta = self.channel_adapter(safe_component)
        t_delta = self.temporal_adapter(safe_component)

        gamma = torch.clamp(self.gamma, 0.0, 1.0)

        if self.role == "trend":
            delta = gamma * c_delta + (1.0 - gamma) * t_delta
        else:
            delta = gamma * t_delta + (1.0 - gamma) * c_delta

        scale = self.rga_max_scale * torch.sigmoid(self.scale_logit)

        return base_output + scale * reliability * delta



class RATFD_DARDA_Model(nn.Module):
    def __init__(
        self,
        channels: int,
        seq_len: int,
        d_model: int = 64,
        n_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.05,
        gamma_init: float = 0.75,
        rga_enabled: bool = True,
        rga_gate_strength: float = 1.0,
        rga_max_scale: float = 0.10,
        ablation_main: str = "full",
        rsda_mode: str = "full",
        **decomp_kwargs,
    ):
        super().__init__()

        self.ablation_main = str(ablation_main)
        self.rsda_mode = str(rsda_mode)
        self.decomp = RATFDv3NoOrthFlatMix(**decomp_kwargs)

        branch_cls = SimpleReconstructor if self.ablation_main == "no_rsda" else BranchReconstructor

        if self.ablation_main == "no_rsda":
            self.trend_branch = SimpleReconstructor(
                channels=channels,
                seq_len=seq_len,
                d_model=d_model,
                n_heads=n_heads,
                num_layers=num_layers,
                dropout=dropout,
            )
            self.seasonal_branch = SimpleReconstructor(
                channels=channels,
                seq_len=seq_len,
                d_model=d_model,
                n_heads=n_heads,
                num_layers=num_layers,
                dropout=dropout,
            )
        else:
            self.trend_branch = BranchReconstructor(
                channels=channels,
                seq_len=seq_len,
                role="trend",
                d_model=d_model,
                n_heads=n_heads,
                num_layers=num_layers,
                dropout=dropout,
                gamma_init=gamma_init,
                rsda_mode=self.rsda_mode,
            )
            self.seasonal_branch = BranchReconstructor(
                channels=channels,
                seq_len=seq_len,
                role="seasonal",
                d_model=d_model,
                n_heads=n_heads,
                num_layers=num_layers,
                dropout=dropout,
                gamma_init=gamma_init,
                rsda_mode=self.rsda_mode,
            )

        # RGA_FIX_BEGIN
        # Robust RGA initialization.
        # This block is intentionally inserted at the end of __init__,
        # right before forward(), so the adapter attributes always exist.
        _rga_kwargs = locals().get("decomp_kwargs", {})
        if not isinstance(_rga_kwargs, dict):
            _rga_kwargs = {}

        self.rga_enabled = bool(locals().get("rga_enabled", _rga_kwargs.get("rga_enabled", True)))
        if self.ablation_main == "no_wrga":
            self.rga_enabled = False
        _rga_gate_strength = float(locals().get("rga_gate_strength", _rga_kwargs.get("rga_gate_strength", 1.0)))
        _rga_max_scale = float(locals().get("rga_max_scale", _rga_kwargs.get("rga_max_scale", 0.10)))

        self.trend_rga = ResidualGatedAttentionAdapter(
            channels=channels,
            seq_len=seq_len,
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            role="trend",
            gamma_init=gamma_init,
            rga_gate_strength=_rga_gate_strength,
            rga_max_scale=_rga_max_scale,
        )

        self.seasonal_rga = ResidualGatedAttentionAdapter(
            channels=channels,
            seq_len=seq_len,
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            role="seasonal",
            gamma_init=gamma_init,
            rga_gate_strength=_rga_gate_strength,
            rga_max_scale=_rga_max_scale,
        )
        # RGA_FIX_END

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            if self.ablation_main == "no_ratfd":
                trend = x
                seasonal = torch.zeros_like(x)
                clean = x
                noise = torch.zeros_like(x)
                white_noise = torch.zeros_like(x)
            else:
                dec = self.decomp(x)
                trend = dec["trend"]
                seasonal = dec["seasonal"]
                clean = dec["clean"]
                noise = dec["noise"]
                white_noise = dec["white_noise"]

        trend_hat = self.trend_branch(trend)
        seasonal_hat = self.seasonal_branch(seasonal)

        # RGA residual correction. It does not replace DA-RDA.
        # If the adapter learns nothing, the model remains close to ENVHP_BEST.
        if self.rga_enabled:
            trend_hat = self.trend_rga(trend, trend_hat, white_noise)
            seasonal_hat = self.seasonal_rga(seasonal, seasonal_hat, white_noise)

        clean_hat = trend_hat + seasonal_hat

        return {
            "trend": trend,
            "seasonal": seasonal,
            "clean": clean,
            "input": x,
            "noise": noise,
            "white_noise": white_noise,
            "trend_hat": trend_hat,
            "seasonal_hat": seasonal_hat,
            "clean_hat": clean_hat,
        }


class DeCo:
    """
    TSB-AD compatible detector wrapper.

    API:
        fit(X)
        decision_function(X)
        anomaly_score(X)
        fit_predict(X)
        predict(X)
    """
    def __init__(self, **kwargs):
        def _env(name, default, cast):
            value = os.environ.get(name, None)
            if value is None or str(value).strip() == "":
                return cast(default)
            return cast(value)

        self.seed = _env("DECO_SEED", kwargs.get("seed", 2026), int)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        self.seq_len = int(
            kwargs.get("seq_len", kwargs.get("win_size", kwargs.get("window_size", 96)))
        )
        self.stride = int(kwargs.get("stride", max(1, self.seq_len // 2)))

        self.batch_size = _env("DECO_BATCH_SIZE", kwargs.get("batch_size", 128), int)
        self.epochs = _env("DECO_EPOCHS", kwargs.get("epochs", kwargs.get("num_epochs", kwargs.get("epoch", 5))), int)

        self.lr = _env("DECO_LR", kwargs.get("lr", kwargs.get("learning_rate", 1e-3)), float)
        self.weight_decay = _env("DECO_WEIGHT_DECAY", kwargs.get("weight_decay", 1e-4), float)

        self.d_model = _env("DECO_D_MODEL", kwargs.get("d_model", 64), int)
        self.n_heads = _env("DECO_N_HEADS", kwargs.get("n_heads", 4), int)

        if self.d_model % self.n_heads != 0:
            self.n_heads = 1

        self.num_layers = _env("DECO_NUM_LAYERS", kwargs.get("e_layers", kwargs.get("num_layers", 2)), int)
        self.dropout = _env("DECO_DROPOUT", kwargs.get("dropout", kwargs.get("fc_dropout", 0.05)), float)
        self.gamma_init = _env("DECO_GAMMA_INIT", kwargs.get("gamma_init", 0.75), float)

        self.beta = _env("DECO_NOISE_BETA", kwargs.get("noise_beta", kwargs.get("beta", 0.10)), float)

        self.ablation_main = str(os.environ.get("DECO_MAIN_ABLATION", kwargs.get("ablation_main", "full"))).strip()
        self.ratfd_ablation = str(os.environ.get("DECO_RATFD_ABLATION", kwargs.get("ratfd_ablation", "full"))).strip()
        self.rsda_mode = str(os.environ.get("DECO_RSDA_MODE", kwargs.get("rsda_mode", "full"))).strip()
        self.score_mode = str(os.environ.get("DECO_SCORE_MODE", kwargs.get("score_mode", "clean_white"))).strip()

        if self.ablation_main == "no_crjas":
            self.score_mode = "clean"
            self.beta = 0.0

        # RGA residual adapter parameters.
        self.rga_enabled = bool(int(_env("DECO_RGA_ENABLED", kwargs.get("rga_enabled", 1), int)))
        self.rga_gate_strength = _env("DECO_RGA_GATE_STRENGTH", kwargs.get("rga_gate_strength", 1.0), float)
        self.rga_max_scale = _env("DECO_RGA_MAX_SCALE", kwargs.get("rga_max_scale", 0.10), float)

        self.max_train_windows = _env("DECO_MAX_TRAIN_WINDOWS", kwargs.get("max_train_windows", 4096), int)

        self.verbose = bool(kwargs.get("verbose", False))

        dev = kwargs.get("device", None)
        self.device = torch.device(
            dev if dev is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        if self.verbose:
            print(
                "[DeCo-ENVHP] "
                f"seed={self.seed}, epochs={self.epochs}, lr={self.lr}, "
                f"batch_size={self.batch_size}, d_model={self.d_model}, "
                f"n_heads={self.n_heads}, num_layers={self.num_layers}, "
                f"dropout={self.dropout}, beta={self.beta}, "
                f"rga_enabled={self.rga_enabled}, "
                f"rga_gate_strength={self.rga_gate_strength}, "
                f"rga_max_scale={self.rga_max_scale}, "
                f"ablation_main={self.ablation_main}, "
                f"ratfd_ablation={self.ratfd_ablation}, "
                f"rsda_mode={self.rsda_mode}, "
                f"score_mode={self.score_mode}, "
                f"max_train_windows={self.max_train_windows}"
            )

        self.model: Optional[RATFD_DARDA_Model] = None

        self.center_ = None
        self.scale_ = None

        self.err_med_ = 0.0
        self.err_mad_ = 1.0
        self.noise_med_ = 0.0
        self.noise_mad_ = 1.0

    @staticmethod
    def _to_numpy(X):
        if hasattr(X, "values"):
            X = X.values

        X = np.asarray(X, dtype=np.float32)

        if X.ndim == 1:
            X = X[:, None]

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X = X.astype(np.float32)

        return X

    def _fit_normalizer(self, X):
        med = np.median(X, axis=0, keepdims=True)

        q1 = np.percentile(X, 25, axis=0, keepdims=True)
        q3 = np.percentile(X, 75, axis=0, keepdims=True)
        iqr = q3 - q1

        iqr = np.where(iqr < 1e-6, 1.0, iqr)

        self.center_ = med.astype(np.float32)
        self.scale_ = iqr.astype(np.float32)

    def _transform(self, X):
        X = self._to_numpy(X)

        if self.center_ is None:
            self._fit_normalizer(X)

        X = (X - self.center_) / self.scale_
        X = np.clip(X, -20.0, 20.0)

        return X.astype(np.float32)

    def _make_windows(self, X):
        T, C = X.shape

        if T <= 0:
            return np.zeros((0, self.seq_len, C), dtype=np.float32), []

        if T < self.seq_len:
            pad_len = self.seq_len - T
            pad = np.repeat(X[-1:, :], pad_len, axis=0)
            Xp = np.concatenate([X, pad], axis=0)
            return Xp[None, :, :].astype(np.float32), [0]

        starts = list(range(0, T - self.seq_len + 1, self.stride))

        if starts[-1] != T - self.seq_len:
            starts.append(T - self.seq_len)

        windows = np.stack(
            [X[s:s + self.seq_len] for s in starts],
            axis=0,
        ).astype(np.float32)

        return windows, starts

    def _subsample_train_windows(self, W):
        if self.max_train_windows > 0 and len(W) > self.max_train_windows:
            idx = np.linspace(0, len(W) - 1, self.max_train_windows).round().astype(np.int64)
            W = W[idx]
        return W

    def fit(self, X, y=None):
        X = self._to_numpy(X)

        self._fit_normalizer(X)
        Xn = self._transform(X)

        W, _ = self._make_windows(Xn)
        W = self._subsample_train_windows(W)

        if len(W) == 0:
            return self

        channels = W.shape[-1]

        irls_iters = 0 if self.ratfd_ablation == "no_robust" else 5
        use_second_stage = False  # final released DARA-AD uses single-stage RATFD
        lowcorr_alpha = 0.0 if self.ratfd_ablation == "no_lowfreq" else 0.75
        white_alpha = 0.0 if self.ratfd_ablation == "no_whitening" else 0.35

        self.model = RATFD_DARDA_Model(
            channels=channels,
            seq_len=self.seq_len,
            d_model=self.d_model,
            n_heads=self.n_heads,
            num_layers=self.num_layers,
            dropout=self.dropout,
            gamma_init=self.gamma_init,
            rga_enabled=self.rga_enabled,
            rga_gate_strength=self.rga_gate_strength,
            rga_max_scale=self.rga_max_scale,
            ablation_main=self.ablation_main,
            rsda_mode=self.rsda_mode,

            trend_lam=120.0,
            irls_iters=irls_iters,
            huber_delta=1.2,
            peak_z=1.00,
            topk_ratio=0.10,
            min_seasonal_freq=0.05,
            max_seasonal_freq=1.00,
            band_width=1,
            noise_corr_lam=35.0,
            lowcorr_alpha=lowcorr_alpha,
            second_peak_z=0.85,
            second_topk_ratio=0.08,
            white_alpha=white_alpha,
            use_second_stage=False,
        ).to(self.device)

        ds = TensorDataset(torch.from_numpy(W))
        dl = DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

        opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        self.model.train()

        for ep in range(self.epochs):
            losses = []

            for (xb,) in dl:
                xb = xb.to(self.device)

                out = self.model(xb)

                loss_t = F.mse_loss(out["trend_hat"], out["trend"])
                loss_s = F.mse_loss(out["seasonal_hat"], out["seasonal"])
                clean_target = out["input"] if self.ablation_main == "no_clean_target" else out["clean"]
                loss_c = F.mse_loss(out["clean_hat"], clean_target)

                loss = 0.5 * loss_t + 0.5 * loss_s + loss_c

                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()

                losses.append(float(loss.detach().cpu()))

            if self.verbose:
                print(
                    f"[DeCo-RATFDv3-DA-RDA] "
                    f"epoch {ep + 1}/{self.epochs}, loss={np.mean(losses):.6f}"
                )

        self._fit_score_calibration(W)

        return self

    @torch.no_grad()
    def _window_scores(self, W):
        if self.model is None:
            raise RuntimeError("Model is not fitted.")

        self.model.eval()

        clean_errs = []
        noise_energies = []

        ds = TensorDataset(torch.from_numpy(W.astype(np.float32)))
        dl = DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )

        for (xb,) in dl:
            xb = xb.to(self.device)

            out = self.model(xb)

            clean_target = out["input"] if self.ablation_main == "no_clean_target" else out["clean"]
            clean_err = ((out["clean_hat"] - clean_target) ** 2).mean(dim=-1)
            if self.score_mode == "clean_raw":
                noise_energy = (out["noise"] ** 2).mean(dim=-1)
            else:
                noise_energy = (out["white_noise"] ** 2).mean(dim=-1)

            clean_errs.append(clean_err.detach().cpu().numpy())
            noise_energies.append(noise_energy.detach().cpu().numpy())

        clean_errs = np.concatenate(clean_errs, axis=0)
        noise_energies = np.concatenate(noise_energies, axis=0)

        clean_z = (clean_errs - self.err_med_) / (1.4826 * self.err_mad_ + 1e-8)
        noise_z = (noise_energies - self.noise_med_) / (1.4826 * self.noise_mad_ + 1e-8)

        if self.score_mode == "clean":
            score = clean_z
        elif self.score_mode == "residual":
            score = noise_z
        else:
            score = clean_z + self.beta * noise_z

        return score, clean_errs, noise_energies

    def _fit_score_calibration(self, W):
        _, clean_errs, noise_energies = self._window_scores(W)

        ce = clean_errs.reshape(-1)
        ne = noise_energies.reshape(-1)

        self.err_med_ = float(np.median(ce))
        self.err_mad_ = float(np.median(np.abs(ce - self.err_med_)))

        if self.err_mad_ < 1e-8:
            self.err_mad_ = float(np.std(ce) + 1e-6)

        self.noise_med_ = float(np.median(ne))
        self.noise_mad_ = float(np.median(np.abs(ne - self.noise_med_)))

        if self.noise_mad_ < 1e-8:
            self.noise_mad_ = float(np.std(ne) + 1e-6)

    def decision_function(self, X):
        Xn = self._transform(X)

        T = Xn.shape[0]

        W, starts = self._make_windows(Xn)
        win_scores, _, _ = self._window_scores(W)

        point_scores = np.zeros(max(T, self.seq_len), dtype=np.float64)
        counts = np.zeros(max(T, self.seq_len), dtype=np.float64)

        for score, s in zip(win_scores, starts):
            point_scores[s:s + self.seq_len] += score
            counts[s:s + self.seq_len] += 1.0

        counts = np.maximum(counts, 1.0)
        point_scores = point_scores / counts
        point_scores = point_scores[:T]

        point_scores = np.nan_to_num(
            point_scores,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        point_scores = point_scores - np.min(point_scores)

        return point_scores.astype(np.float32)

    def anomaly_score(self, X):
        return self.decision_function(X)

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.decision_function(X)

    def predict(self, X):
        return self.decision_function(X)
