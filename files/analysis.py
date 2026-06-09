"""
analysis.py
-----------
Thesis-ready analysis (Sec. V-G / V-I): forecast metrics, federated efficiency
curves, client-drift plots, and the quantum-attention visualisations that are the
evidence for "structural sources of quantum advantage".

WHAT IS TRACKABLE PER MODEL (read this before claiming "attention advantage"):
  * QSANN   -> a genuine L x L attention matrix per QSAL (model.attention_maps()).
               This is the only one of the three with classical-style attention
               weights you can read directly.
  * Quixer  -> NO L x L map. Its mixing is QSVT/LCU; the per-timestep complex
               mix coefficients |mix_coeffs| are the closest analogue
               (model.mixing_weights()), plus the LCU success probability.
  * QuLTSF  -> has NO attention at all. It is a Linear->AmplitudeEmbedding->
               StronglyEntanglingLayers->Linear hybrid (channel-independent).
               Do NOT report "attention" for QuLTSF; treat it as the quantum
               baseline. What you CAN inspect is the entangling-layer weights and
               per-qubit <Z> expectations.
"""

from __future__ import annotations
import numpy as np
import torch
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# Metrics                                                                     #
# --------------------------------------------------------------------------- #
def forecast_metrics(pred: torch.Tensor, true: torch.Tensor) -> dict[str, float]:
    p, t = pred.detach().cpu().numpy(), true.detach().cpu().numpy()
    mse = float(np.mean((p - t) ** 2))
    mae = float(np.mean(np.abs(p - t)))
    rmse = float(np.sqrt(mse))
    denom = np.where(np.abs(t) < 1e-6, 1e-6, np.abs(t))
    mape = float(np.mean(np.abs((p - t) / denom)))
    return {"MSE": mse, "MAE": mae, "RMSE": rmse, "MAPE": mape}


# --------------------------------------------------------------------------- #
# Federated convergence + communication                                       #
# --------------------------------------------------------------------------- #
def plot_convergence(log, title, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(log.rounds, log.val_mse, label="MSE")
    ax[0].plot(log.rounds, log.val_mae, label="MAE")
    ax[0].set_xlabel("communication round"); ax[0].set_ylabel("error")
    ax[0].set_title(f"{title}: convergence"); ax[0].legend(); ax[0].grid(alpha=.3)

    comm_mb = np.asarray(log.comm_bytes_cumulative) / 1e6
    ax[1].plot(comm_mb, log.val_mse)
    ax[1].set_xlabel("cumulative communication (MB)"); ax[1].set_ylabel("MSE")
    ax[1].set_title(f"{title}: accuracy vs. comm cost"); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def plot_drift(log, title, path):
    """Client<->global divergence per round -- the non-IID diagnostic (Sec. V-G)."""
    names = list(log.drift_per_client[0].keys())
    mat = np.array([[d[n] for n in names] for d in log.drift_per_client])  # (R, K)
    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.imshow(mat.T, aspect="auto", origin="lower", cmap="viridis")
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=6)
    ax.set_xlabel("round"); ax.set_title(f"{title}: client drift ||theta_k - theta_global||")
    fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


# --------------------------------------------------------------------------- #
# Quantum-attention visualisation                                             #
# --------------------------------------------------------------------------- #
def plot_qsann_attention(model, sample_x, path, layer=0):
    """QSANN L x L attention heatmap for one input window."""
    model.eval()
    with torch.no_grad():
        _ = model(sample_x.unsqueeze(0))
    A = model.attention_maps()[0, layer].cpu().numpy()   # (L, L)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(A, cmap="magma")
    ax.set_xlabel("key timestep"); ax.set_ylabel("query timestep")
    ax.set_title(f"QSANN quantum attention (QSAL {layer})")
    fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def plot_quixer_mixing(model, path):
    """Quixer per-timestep mixing magnitude + polynomial coefficients."""
    mix = model.mixing_weights().numpy()
    poly = model.polynomial_coeffs().numpy()
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].stem(range(len(mix)), mix)
    ax[0].set_title("Quixer |mix_coeffs| per timestep"); ax[0].set_xlabel("timestep")
    ax[1].stem(range(len(poly)), poly)
    ax[1].set_title("QSVT polynomial coefficients"); ax[1].set_xlabel("degree")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


# --------------------------------------------------------------------------- #
# Attention-entropy: a single scalar to compare across rounds / models        #
# --------------------------------------------------------------------------- #
def attention_entropy(attn: torch.Tensor) -> float:
    """Mean row entropy of an attention matrix. Low entropy = sharp, selective
    attention (focusing on few timesteps); high entropy = diffuse. Tracking this
    across federated rounds shows whether the quantum attention *specialises* as
    FedAvg proceeds -- a concrete, plottable 'structural' signal for your thesis."""
    a = attn.clamp_min(1e-12)
    ent = -(a * a.log()).sum(-1)        # per query row
    return float(ent.mean())
