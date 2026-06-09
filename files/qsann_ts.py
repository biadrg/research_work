"""
qsann_ts.py
-----------
Quantum Self-Attention Neural Network (Li, Zhao & Wang 2022/2024), implemented
from scratch in PennyLane and ADAPTED FROM TEXT CLASSIFICATION TO TIME-SERIES
FORECASTING, following your thesis Sec. V-B / V-C / V-E exactly.

Why from scratch rather than cloning a repo: the original QSANN code is text- and
Paddle-specific and unmaintained for forecasting. Your thesis already transcribes
the full QSAL (Eqs. 16-19), so a clean PennyLane implementation (a) matches your
written methodology one-to-one and (b) keeps QSANN on the SAME PennyLane+Torch
stack as QuLTSF, so one FedAvg loop wraps both.

ADAPTATION FROM TEXT -> TIME-SERIES (the three concrete changes):
  1. INPUT. Text QSANN embeds discrete tokens. Here each "token" is a time step
     x_t in R^V (V channels). A classical Linear projects R^V -> R^{n_qubits}
     so every step becomes an angle-encoding vector (thesis Eq. 13).
  2. ORDER. Text relies on word order implicitly; a quantum attention layer is
     permutation-invariant (it sees a *set*). We inject a LEARNABLE classical
     positional embedding before the quantum layer (thesis Sec. V-E) so temporal
     direction is preserved without spending a qubit.
  3. HEAD. Text QSANN ends in a class logit. We replace it with a forecasting
     head producing H future steps x V channels (thesis Eq. 6).

The quantum Q/K/V circuit (Eqs. 16-18) and the classical Gaussian/softmax
attention over rescaled inner products (Eq. 19) are exactly as in the thesis.
The Q/K/V circuit and the softmax assembly in this file were validated numerically
(correct (L, n_qubits) outputs, attention rows summing to 1).
"""

from __future__ import annotations
import pennylane as qml
import torch
import torch.nn as nn


def _qkv_qnode(n_qubits: int, n_layers: int, device_name: str = "default.qubit"):
    """One PQC: angle-encode a projected token, apply a strongly-entangling
    ansatz, measure <PauliZ> on every wire. Wrapped as a qml.qnn.TorchLayer so
    its weights are ordinary torch Parameters and cross the FedAvg wire."""
    dev = qml.device(device_name, wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        # inputs: (batch, n_qubits) -> AngleEmbedding broadcasts over the batch
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="X")   # Eq. 13
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))      # ansatz
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]       # M = Pauli-Z

    weight_shapes = {"weights": (n_layers, n_qubits, 3)}
    return qml.qnn.TorchLayer(circuit, weight_shapes)


class QuantumSelfAttentionLayer(nn.Module):
    """A single QSAL. Three PQCs produce per-token query/key/value vectors in
    R^{n_qubits} (Eqs. 16-18); attention scores and the weighted value sum are
    computed CLASSICALLY (Eq. 19), which avoids extracting a normalised
    distribution from quantum measurements (thesis Sec. V-C)."""

    def __init__(self, n_qubits: int, n_layers: int, device_name="default.qubit"):
        super().__init__()
        self.n_qubits = n_qubits
        self.q_pqc = _qkv_qnode(n_qubits, n_layers, device_name)
        self.k_pqc = _qkv_qnode(n_qubits, n_layers, device_name)
        self.v_pqc = _qkv_qnode(n_qubits, n_layers, device_name)
        self.scale = n_qubits ** 0.5

    def forward(self, x):                       # x: (B, L, n_qubits)
        B, L, D = x.shape
        flat = x.reshape(B * L, D)
        q = self.q_pqc(flat).reshape(B, L, D)
        k = self.k_pqc(flat).reshape(B, L, D)
        v = self.v_pqc(flat).reshape(B, L, D)
        scores = torch.matmul(q, k.transpose(1, 2)) / self.scale   # (B, L, L)
        attn = torch.softmax(scores, dim=-1)                       # Eq. 19
        out = torch.matmul(attn, v)                                # (B, L, n_qubits)
        return out + x, attn          # residual; return attn for the analysis hooks


class QSANNForecaster(nn.Module):
    """Encoder-only quantum self-attention forecaster.

    Args
    ----
    seq_len      input window length L
    pred_len     forecast horizon H
    n_vars       number of channels V (1 for univariate)
    n_qubits     PQC width = model dimension after projection
    n_qsal       number of stacked QSALs
    n_layers     strongly-entangling ansatz depth per PQC
    """

    def __init__(self, seq_len, pred_len, n_vars, n_qubits=4,
                 n_qsal=1, n_layers=2, device_name="default.qubit"):
        super().__init__()
        self.seq_len, self.pred_len, self.n_vars = seq_len, pred_len, n_vars

        self.input_proj = nn.Linear(n_vars, n_qubits)                 # R^V -> R^{nq}
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, n_qubits) * 0.02)  # Sec. V-E
        self.layers = nn.ModuleList(
            QuantumSelfAttentionLayer(n_qubits, n_layers, device_name)
            for _ in range(n_qsal))
        self.head = nn.Linear(seq_len * n_qubits, pred_len * n_vars)  # forecasting head

        self._last_attn = None    # populated each forward for the analysis hooks

    def forward(self, x):                       # x: (B, seq_len, n_vars)
        h = self.input_proj(x) + self.pos_embed
        attns = []
        for layer in self.layers:
            h, a = layer(h)
            attns.append(a)
        self._last_attn = torch.stack(attns, dim=1)   # (B, n_qsal, L, L)
        B = h.shape[0]
        out = self.head(h.reshape(B, -1))
        return out.reshape(B, self.pred_len, self.n_vars)

    @torch.no_grad()
    def attention_maps(self):
        """Return the attention weights from the most recent forward pass,
        shape (B, n_qsal, L, L). Use for the quantum-attention analysis."""
        return self._last_attn
