"""
train_federated.py
------------------
End-to-end example: train the from-scratch QSANN forecaster across Beijing PRSA
station-clients with FedAvg. The same harness wraps QuLTSF or Quixer -- only the
model constructor and the `forward_loss` callback change (see guide Sec. 3).
"""

from __future__ import annotations
import os
import numpy as np
import torch

from partition_data import load_beijing, client_loaders, PRSA_FEATURE_COLS
from qsann_ts import QSANNForecaster
from federated import Client, run_fedavg

DEVICE = torch.device("cpu")          # quantum simulation is CPU-bound here
SEQ_LEN, PRED_LEN = 64, 16            # power of 2 keeps the same windows usable by Quixer
BATCH = 16


def build_clients():
    beijing = load_beijing("climate/PRSA2017_Data_20130301-20170228.zip")
    series = {name: df[PRSA_FEATURE_COLS].to_numpy(np.float32)
              for name, df in beijing.items()
              if all(c in df.columns for c in PRSA_FEATURE_COLS)}
    loaders = client_loaders(series, SEQ_LEN, PRED_LEN, BATCH)
    clients, test_sets = [], {}
    for name, (loader, Xte, Yte, n) in loaders.items():
        clients.append(Client(name=name, train_loader=loader, n_samples=n))
        test_sets[name] = (Xte, Yte)
    return clients, test_sets, len(PRSA_FEATURE_COLS)


def qsann_forward_loss(model, batch):
    x, y = batch
    x, y = x.to(DEVICE), y.to(DEVICE)
    return torch.nn.functional.mse_loss(model(x), y)


def make_evaluator(test_sets):
    # pool all clients' test windows into one global held-out set
    Xs = torch.cat([Xte for Xte, _ in test_sets.values()])
    Ys = torch.cat([Yte for _, Yte in test_sets.values()])

    @torch.no_grad()
    def evaluate(model):
        model.eval()
        preds = []
        for i in range(0, len(Xs), 64):
            preds.append(model(Xs[i:i + 64].to(DEVICE)).cpu())
        p = torch.cat(preds)
        mse = torch.mean((p - Ys) ** 2).item()
        mae = torch.mean(torch.abs(p - Ys)).item()
        return mse, mae
    return evaluate


if __name__ == "__main__":
    clients, test_sets, n_vars = build_clients()
    model = QSANNForecaster(SEQ_LEN, PRED_LEN, n_vars,
                            n_qubits=4, n_qsal=1, n_layers=2).to(DEVICE)
    evaluate = make_evaluator(test_sets)
    log = run_fedavg(
        global_model=model,
        clients=clients,
        n_rounds=30,
        local_epochs=1,
        lr=1e-3,
        device=DEVICE,
        forward_loss=qsann_forward_loss,
        evaluate=evaluate,
        clients_per_round=None,     # use all stations each round
    )
    os.makedirs("results", exist_ok=True)
    np.savez("results/qsann_fed_log.npz",
             rounds=log.rounds, mse=log.val_mse, mae=log.val_mae,
             comm=log.comm_bytes_cumulative)
    torch.save(model.state_dict(), "results/qsann_global.pt")
    print("Saved results/qsann_fed_log.npz and results/qsann_global.pt")
