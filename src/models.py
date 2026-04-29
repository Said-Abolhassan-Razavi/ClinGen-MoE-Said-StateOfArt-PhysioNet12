"""
models.py
---------
Sequence models for synthetic clinical time-series generation.

Implements the Conditional GRU-VAE recommended for PhysioNet 2012:
  - Static features (Age, Gender, ICUType) condition the GRU encoder
  - Missingness mask appended at each timestep (missing ≠ zero)
  - Cyclical KL annealing prevents posterior collapse

Model comparison (from SOTA):
  Model              | Temporal | Missing | Static+TS | Recommended for
  -------------------|----------|---------|-----------|-------------------
  Vanilla VAE        |    ✗     |    ✗    |     ✗     | —
  GRU-VAE            |    ✓     | Partial |  Partial  | MIMIC-IV demo
  Conditional GRU-VAE|    ✓     | Partial |     ✓     | PhysioNet 2012 ✓
  mTAN-VAE           |    ✓     |    ✓    |     ✓     | Large scale

Author: Said Abolhassan Razavi
Project: TER, Université Paris-Saclay, Master 1 AI
"""

import torch
import torch.nn as nn


class ConditionalGRU_VAE(nn.Module):
    """
    Conditional GRU Variational Autoencoder for PhysioNet 2012.

    Architecture
    ------------
    Encoder:
        static_enc  : MLP(static_dim → 16)
        gru_enc     : GRU([ts_val * mask ; mask] → hidden)
        fc_mu/logvar: Linear(hidden + 16 → latent)

    Decoder:
        gru_dec : GRU([z ; static_emb] repeated T times → hidden)
        fc_out  : Linear(hidden → n_feat)

    The static embedding conditions both encoder and decoder, ensuring
    that synthetic vital signs are consistent with patient demographics.

    Parameters
    ----------
    n_feat     : int  — number of time-series features (default 24)
    static_dim : int  — static vector dimension (default 7)
    hidden     : int  — GRU hidden size (default 64)
    latent     : int  — latent space dimension (default 16)
    """

    def __init__(self, n_feat: int = 24, static_dim: int = 7,
                 hidden: int = 64, latent: int = 16):
        super().__init__()
        self.n_feat = n_feat
        self.latent = latent

        # Modality 1: static feature encoder
        self.static_enc = nn.Sequential(
            nn.Linear(static_dim, 32), nn.ReLU(),
            nn.Linear(32, 16),
        )

        # Modality 2: time-series encoder
        # Input: [value * mask ; mask] — appending mask so missing ≠ zero
        self.gru_enc = nn.GRU(2 * n_feat, hidden, batch_first=True)

        # Latent space — conditioned on time-series + static
        self.fc_mu     = nn.Linear(hidden + 16, latent)
        self.fc_logvar = nn.Linear(hidden + 16, latent)

        # Decoder
        self.gru_dec = nn.GRU(latent + 16, hidden, batch_first=True)
        self.fc_out  = nn.Linear(hidden, n_feat)

    def encode(self, ts_val: torch.Tensor, ts_mask: torch.Tensor,
               static_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode (ts_val, ts_mask, static) → (mu, logvar)."""
        x = torch.cat([ts_val * ts_mask, ts_mask], dim=-1)  # (B, T, 2F)
        _, h = self.gru_enc(x)                               # h: (1, B, H)
        h = h.squeeze(0)                                     # (B, H)
        cond = torch.cat([h, static_emb], dim=-1)            # (B, H+16)
        return self.fc_mu(cond), self.fc_logvar(cond)

    def decode(self, z: torch.Tensor, T: int,
               static_emb: torch.Tensor) -> torch.Tensor:
        """Decode (z, static_emb) → reconstructed time-series (B, T, F)."""
        inp = torch.cat([z, static_emb], dim=-1)             # (B, latent+16)
        inp = inp.unsqueeze(1).expand(-1, T, -1)             # (B, T, latent+16)
        out, _ = self.gru_dec(inp)
        return self.fc_out(out)                               # (B, T, F)

    def reparameterise(self, mu: torch.Tensor,
                       logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return mu

    def forward(self, ts_val: torch.Tensor, ts_mask: torch.Tensor,
                static_vec: torch.Tensor) -> tuple:
        """
        Forward pass.

        Parameters
        ----------
        ts_val     : (B, T, F) — time-series values
        ts_mask    : (B, T, F) — binary observation mask
        static_vec : (B, static_dim) — raw static feature vectors

        Returns
        -------
        recon   : (B, T, F) — reconstructed time-series
        mu      : (B, latent)
        logvar  : (B, latent)
        """
        se    = self.static_enc(static_vec)               # (B, 16)
        mu, logvar = self.encode(ts_val, ts_mask, se)
        z     = self.reparameterise(mu, logvar)
        recon = self.decode(z, ts_val.shape[1], se)
        return recon, mu, logvar

    @torch.no_grad()
    def sample(self, static_vec: torch.Tensor, T: int = 48) -> torch.Tensor:
        """
        Generate synthetic patients by sampling z ~ N(0,I),
        conditioned on given static features.

        Parameters
        ----------
        static_vec : (B, static_dim)
        T          : number of timesteps to generate

        Returns
        -------
        synth : (B, T, F) synthetic time-series
        """
        self.eval()
        se = self.static_enc(static_vec)
        z  = torch.randn(static_vec.shape[0], self.latent)
        return self.decode(z, T, se)


def masked_elbo(recon: torch.Tensor, ts_val: torch.Tensor,
                ts_mask: torch.Tensor, mu: torch.Tensor,
                logvar: torch.Tensor, beta: float = 0.5) -> torch.Tensor:
    """
    ELBO loss with missingness-aware reconstruction.

    Loss = masked_MSE + beta * KL

    Only observed timesteps (mask=1) contribute to the MSE term.
    This prevents the model from learning to reconstruct zeros
    at missing positions.
    """
    mse = ((recon - ts_val) ** 2 * ts_mask).sum() / (ts_mask.sum() + 1e-8)
    kl  = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(dim=-1).mean()
    return mse + beta * kl


def cyclical_beta(epoch: int, n_epochs: int,
                  n_cycles: int = 4, max_beta: float = 0.5) -> float:
    """
    Cyclical KL annealing schedule.

    Linearly increases beta from 0 to max_beta over each cycle.
    Resets at the start of each cycle.

    Prevents posterior collapse on moderate datasets (4,000 patients).
    Reference: Fu et al. (2019) — Cyclical Annealing Schedule.
    """
    period = n_epochs / n_cycles
    t = (epoch % period) / period
    return max_beta * min(1.0, 2.0 * t)
