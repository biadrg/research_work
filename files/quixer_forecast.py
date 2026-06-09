"""
quixer_forecast.py
------------------
Adapter that turns Transconnectome/TSQuantumTransformer's `QuixerTimeSeries`
(torchquantum) from an fMRI *whole-sequence regressor/classifier* into a standard
*multivariate forecaster*, and exposes the quantum-attention internals for analysis.

KEY FACTS established by reading the actual repo (QuixerTSModel.py):
  * QuixerTimeSeries is ALREADY generic: it takes (B, n_timesteps, feature_dim)
    via a `feature_projection: Linear(feature_dim, n_rots)`. So the "fMRI -> TS"
    conversion is NOT a rewrite of the model. The fMRI-specific parts live only in
    the *driver* (QuixerfMRI_Regress.py): phenotype CSV loading, per-subject
    sliding windows that all share ONE label, and classification/AUC metrics.
    We discard that driver and feed forecasting windows instead.
  * It returns a TUPLE: (output, mean_lcu_norm). The second value is the mean
    success probability of the linear-combination-of-unitaries block; the original
    code ignores it, but it is a meaningful regulariser -- add it to the loss so the
    model is pushed toward higher-probability (cheaper-to-realise) circuits.
  * HARD CONSTRAINT: n_ctrl_qubits = log2(n_timesteps). seq_len MUST be a power of
    two (16, 32, 64, 128, ...). 96 / 336 will crash. Use `next_pow2` or pad.
  * Its "attention" is QSVT/LCU polynomial mixing across timesteps. The trackable
    quantities are `poly_coeffs` (real) and `mix_coeffs` (complex, length
    n_timesteps) -- the latter is the closest analogue to a per-timestep attention
    weight and is what you visualise (see analysis hooks below).

This file is syntax-reviewed; run it in the torchquantum environment (see guide).
"""

from __future__ import annotations
from math import log2
import torch
import torch.nn as nn

# QuixerTimeSeries lives in the cloned repo; add it to PYTHONPATH or copy the file.
from QuixerTSModel import QuixerTimeSeries


def next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


class QuixerForecaster(nn.Module):
    """Wrap QuixerTimeSeries for H-step multivariate forecasting.

    The base model emits one vector of size `output_dim` per input window. We set
    output_dim = pred_len * n_vars and reshape to (B, pred_len, n_vars).
    """

    def __init__(self, seq_len, pred_len, n_vars,
                 n_qubits=4, degree=3, n_ansatz_layers=2,
                 dropout=0.1, device="cpu", lcu_reg_weight=0.0):
        super().__init__()
        assert seq_len == next_pow2(seq_len), (
            f"Quixer requires power-of-2 seq_len; got {seq_len}. "
            f"Use {next_pow2(seq_len)} or left-pad the window.")
        self.pred_len, self.n_vars = pred_len, n_vars
        self.lcu_reg_weight = lcu_reg_weight
        self.core = QuixerTimeSeries(
            n_qubits=n_qubits,
            n_timesteps=seq_len,
            degree=degree,
            n_ansatz_layers=n_ansatz_layers,
            feature_dim=n_vars,
            output_dim=pred_len * n_vars,
            dropout=dropout,
            device=torch.device(device),
        )
        self._last_norm = None

    def forward(self, x):                       # x: (B, seq_len, n_vars)
        out, mean_norm = self.core(x)
        self._last_norm = mean_norm
        return out.reshape(x.shape[0], self.pred_len, self.n_vars)

    def lcu_regulariser(self) -> torch.Tensor:
        """Penalise low LCU success probability (encourages realisable circuits).
        Add `model.lcu_regulariser()` to the data loss in your training step."""
        if self._last_norm is None:
            return torch.tensor(0.0)
        return self.lcu_reg_weight * (1.0 - self._last_norm)

    @torch.no_grad()
    def mixing_weights(self):
        """Per-timestep complex mixing coefficients (the Quixer analogue of
        attention weights). Returns magnitude |mix_coeffs|, length n_timesteps."""
        return self.core.mix_coeffs.detach().abs().cpu()

    @torch.no_grad()
    def polynomial_coeffs(self):
        """QSVT polynomial coefficients (degree+1,)."""
        return self.core.poly_coeffs.detach().cpu()


# Loss helper for the federated loop's `forward_loss` callback ----------------
def quixer_forward_loss(model: QuixerForecaster, batch, device):
    x, y = batch
    x, y = x.to(device), y.to(device)
    pred = model(x)
    mse = torch.nn.functional.mse_loss(pred, y)
    return mse + model.lcu_regulariser()
