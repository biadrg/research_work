"""
federated.py
------------
Framework-agnostic FedAvg harness for the thesis.

WHY framework-agnostic: QuLTSF and the from-scratch QSANN run on PennyLane
(qml.qnn.TorchLayer), while QuixerTimeSeries runs on torchquantum. All three are
plain torch.nn.Module objects, so the *only* thing the federated layer touches is
model.state_dict() / load_state_dict() -- pure torch tensors. This lets a SINGLE
FedAvg loop wrap all three architectures unchanged (thesis Sec. V-D).

Implements FedAvg exactly as written in the thesis (Eq. 7): each round, every
sampled client receives the current global parameters, runs E local epochs of SGD
on its private data, and returns updated parameters; the server forms the next
global vector as a dataset-size-weighted average.

Also produces the two efficiency artifacts the thesis Sec. V-G asks for:
  * parameter count + per-round communication cost (bytes on the wire),
  * client<->global divergence (drift) per round, for the non-IID analysis.
"""

from __future__ import annotations
import copy
from dataclasses import dataclass, field
from typing import Callable, Iterable

import torch
from torch.utils.data import DataLoader


# --------------------------------------------------------------------------- #
# Parameter (de)serialisation                                                 #
# --------------------------------------------------------------------------- #
def get_trainable_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Return a CPU copy of the trainable parameters only (what crosses the wire).

    We deliberately exclude registered buffers (e.g. RevIN running stats,
    BatchNorm stats) from the *averaged* payload by default, matching the thesis
    framing that 'all trainable parameters' are the three PQCs / circuit weights.
    Buffers, if any, stay local. Switch to model.state_dict() if you want to
    average buffers too.
    """
    return {k: v.detach().cpu().clone()
            for k, v in model.named_parameters() if v.requires_grad}


def set_trainable_state(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    own = dict(model.named_parameters())
    with torch.no_grad():
        for k, v in state.items():
            own[k].copy_(v.to(own[k].device))


def fedavg(states: list[dict[str, torch.Tensor]],
           client_sizes: list[int]) -> dict[str, torch.Tensor]:
    """Dataset-size-weighted average (thesis Eq. 7)."""
    total = float(sum(client_sizes))
    agg = {k: torch.zeros_like(v) for k, v in states[0].items()}
    for sd, n in zip(states, client_sizes):
        w = n / total
        for k in agg:
            agg[k] += w * sd[k]
    return agg


def state_drift(client_state: dict[str, torch.Tensor],
                global_state: dict[str, torch.Tensor]) -> float:
    """L2 distance between a client's post-local-training params and the global
    params it started from. This is the per-round 'divergence' the thesis tracks
    to quantify the effect of non-IID client profiles (Sec. V-G)."""
    sq = 0.0
    for k in global_state:
        sq += torch.sum((client_state[k] - global_state[k]) ** 2).item()
    return sq ** 0.5


def param_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --------------------------------------------------------------------------- #
# Client                                                                      #
# --------------------------------------------------------------------------- #
@dataclass
class Client:
    """One federated node. `name` is the station / asset / sensor id."""
    name: str
    train_loader: DataLoader
    n_samples: int


# --------------------------------------------------------------------------- #
# Local training step                                                         #
# --------------------------------------------------------------------------- #
def local_train(model: torch.nn.Module,
                loader: DataLoader,
                epochs: int,
                lr: float,
                device: torch.device,
                forward_loss: Callable[[torch.nn.Module, tuple], torch.Tensor]) -> None:
    """Run E local epochs in place. `forward_loss(model, batch) -> scalar loss`
    is supplied per-architecture because the three models have different forward
    signatures (QuLTSF returns a tensor; QuixerTimeSeries returns (pred, norm))."""
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        for batch in loader:
            opt.zero_grad()
            loss = forward_loss(model, batch)
            loss.backward()
            opt.step()


# --------------------------------------------------------------------------- #
# FedAvg driver                                                               #
# --------------------------------------------------------------------------- #
@dataclass
class FedLog:
    rounds: list[int] = field(default_factory=list)
    val_mse: list[float] = field(default_factory=list)
    val_mae: list[float] = field(default_factory=list)
    drift_per_client: list[dict[str, float]] = field(default_factory=list)
    comm_bytes_cumulative: list[int] = field(default_factory=list)


def run_fedavg(global_model: torch.nn.Module,
               clients: list[Client],
               n_rounds: int,
               local_epochs: int,
               lr: float,
               device: torch.device,
               forward_loss: Callable,
               evaluate: Callable[[torch.nn.Module], tuple[float, float]],
               clients_per_round: int | None = None,
               seed: int = 0) -> FedLog:
    """Full federated training loop.

    evaluate(model) -> (mse, mae) on a held-out global/test set.
    """
    g = torch.Generator().manual_seed(seed)
    log = FedLog()
    bytes_per_param = 4  # float32 on the wire
    n_params = param_count(global_model)
    cum_bytes = 0

    for r in range(n_rounds):
        # ---- server samples a subset S_r of clients --------------------------
        if clients_per_round and clients_per_round < len(clients):
            idx = torch.randperm(len(clients), generator=g)[:clients_per_round].tolist()
            sampled = [clients[i] for i in idx]
        else:
            sampled = clients

        global_state = get_trainable_state(global_model)
        local_states, sizes, drifts = [], [], {}

        # ---- each client trains locally on its private data ------------------
        for c in sampled:
            local = copy.deepcopy(global_model).to(device)
            set_trainable_state(local, global_state)
            local_train(local, c.train_loader, local_epochs, lr, device, forward_loss)
            ls = get_trainable_state(local)
            local_states.append(ls)
            sizes.append(c.n_samples)
            drifts[c.name] = state_drift(ls, global_state)
            # download (global->client) + upload (client->global)
            cum_bytes += 2 * n_params * bytes_per_param

        # ---- server aggregation (Eq. 7) -------------------------------------
        new_global = fedavg(local_states, sizes)
        set_trainable_state(global_model, new_global)

        # ---- bookkeeping -----------------------------------------------------
        mse, mae = evaluate(global_model)
        log.rounds.append(r)
        log.val_mse.append(mse)
        log.val_mae.append(mae)
        log.drift_per_client.append(drifts)
        log.comm_bytes_cumulative.append(cum_bytes)
        print(f"[round {r:3d}] MSE={mse:.5f} MAE={mae:.5f} "
              f"mean_drift={sum(drifts.values())/len(drifts):.4f} "
              f"comm={cum_bytes/1e6:.2f} MB")

    return log
