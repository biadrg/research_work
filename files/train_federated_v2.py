"""
train_federated_v2.py
---------------------
Fixed version of train_federated.py. Changes from v1:
  1. Saves per-client drift as a 2D array (rounds x clients) in the npz.
  2. Saves RMSE alongside MSE and MAE.
  3. Logs attention entropy every round (your core quantum-advantage signal).
  4. Evaluates per-client MSE/MAE separately — not just the pooled average.
  5. Checkpoints the best model during training, not only the final one.
"""

from __future__ import annotations
import os, math
import numpy as np
import torch
import torch.nn.functional as F

from partition_data import load_beijing, client_loaders, PRSA_FEATURE_COLS
from qsann_ts import QSANNForecaster
from federated import Client, run_fedavg

DEVICE    = torch.device("cpu")
SEQ_LEN   = 64
PRED_LEN  = 16
BATCH     = 16
N_QUBITS  = 4
N_LAYERS  = 2
N_ROUNDS  = 60      # extended — drift was still falling at round 29
LOCAL_E   = 1
LR        = 1e-3
DATA_PATH = "climate/PRSA2017_Data_20130301-20170228.zip"


def build_clients():
    beijing = load_beijing(os.path.expanduser(DATA_PATH))
    series  = {name: df[PRSA_FEATURE_COLS].to_numpy(np.float32)
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
    return F.mse_loss(model(x.to(DEVICE)), y.to(DEVICE))


def make_evaluator(test_sets, model_ref):
    """Returns a function that evaluates both pooled and per-client metrics."""
    client_names = list(test_sets.keys())

    @torch.no_grad()
    def evaluate(model):
        model.eval()
        all_pred, all_true = [], []
        per_client = {}

        for name in client_names:
            Xte, Yte = test_sets[name]
            preds = []
            for i in range(0, len(Xte), 64):
                preds.append(model(Xte[i:i+64].to(DEVICE)).cpu())
            p = torch.cat(preds)
            mse_c = F.mse_loss(p, Yte).item()
            mae_c = torch.mean(torch.abs(p - Yte)).item()
            per_client[name] = {"mse": mse_c, "mae": mae_c,
                                 "rmse": math.sqrt(mse_c)}
            all_pred.append(p); all_true.append(Yte)

        P = torch.cat(all_pred); T = torch.cat(all_true)
        mse  = F.mse_loss(P, T).item()
        mae  = torch.mean(torch.abs(P - T)).item()

        # ---- attention entropy (QSANN-specific) ----------------------------
        sample_x = test_sets[client_names[0]][0][:1].to(DEVICE)
        _ = model(sample_x)
        attn = model.attention_maps()           # (1, n_qsal, L, L)
        a = attn[0, 0].clamp_min(1e-12)
        ent = float(-(a * a.log()).sum(-1).mean())

        return mse, mae, per_client, ent

    return evaluate


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    clients, test_sets, n_vars = build_clients()
    model = QSANNForecaster(SEQ_LEN, PRED_LEN, n_vars,
                            n_qubits=N_QUBITS, n_qsal=1,
                            n_layers=N_LAYERS).to(DEVICE)
    client_names = [c.name for c in clients]
    evaluate     = make_evaluator(test_sets, model)

    # ---- storage arrays ----------------------------------------------------
    rounds_log, mse_log, mae_log, rmse_log, comm_log, ent_log = [], [], [], [], [], []
    drift_matrix = []          # shape will be (n_rounds, n_clients) after run
    per_client_mse = {n: [] for n in client_names}
    per_client_mae = {n: [] for n in client_names}
    best_mse, best_round = float("inf"), -1

    # ---- custom round loop (wraps run_fedavg internals) --------------------
    import copy
    from federated import (get_trainable_state, set_trainable_state,
                           fedavg, state_drift, local_train, param_count)
    import torch

    g          = torch.Generator().manual_seed(0)
    n_params   = param_count(model)
    cum_bytes  = 0

    for r in range(N_ROUNDS):
        global_state = get_trainable_state(model)
        local_states, sizes, drifts = [], [], {}

        for c in clients:
            local = copy.deepcopy(model).to(DEVICE)
            set_trainable_state(local, global_state)
            local_train(local, c.train_loader, LOCAL_E, LR, DEVICE, qsann_forward_loss)
            ls = get_trainable_state(local)
            local_states.append(ls)
            sizes.append(c.n_samples)
            drifts[c.name] = state_drift(ls, global_state)
            cum_bytes += 2 * n_params * 4

        new_global = fedavg(local_states, sizes)
        set_trainable_state(model, new_global)

        mse, mae, pc, ent = evaluate(model)
        rmse = math.sqrt(mse)

        # checkpoint best
        if mse < best_mse:
            best_mse, best_round = mse, r
            torch.save(model.state_dict(), "results/qsann_best.pt")

        # log
        rounds_log.append(r); mse_log.append(mse); mae_log.append(mae)
        rmse_log.append(rmse); comm_log.append(cum_bytes); ent_log.append(ent)
        drift_matrix.append([drifts[n] for n in client_names])
        for n in client_names:
            per_client_mse[n].append(pc[n]["mse"])
            per_client_mae[n].append(pc[n]["mae"])

        mean_drift = sum(drifts.values()) / len(drifts)
        print(f"[round {r:3d}] MSE={mse:.5f} RMSE={rmse:.5f} MAE={mae:.5f} "
              f"attn_ent={ent:.4f} drift={mean_drift:.4f} comm={cum_bytes/1e6:.2f} MB")

    # ---- save everything ---------------------------------------------------
    np.savez("results/qsann_fed_log_v2.npz",
             rounds      = np.array(rounds_log),
             mse         = np.array(mse_log),
             mae         = np.array(mae_log),
             rmse        = np.array(rmse_log),
             comm        = np.array(comm_log),
             attn_entropy= np.array(ent_log),
             drift_matrix= np.array(drift_matrix),   # (R, K)
             client_names= np.array(client_names),
             **{f"pc_mse_{n}": np.array(per_client_mse[n]) for n in client_names},
             **{f"pc_mae_{n}": np.array(per_client_mae[n]) for n in client_names})

    torch.save(model.state_dict(), "results/qsann_final.pt")
    print(f"\nBest model: round {best_round}, MSE={best_mse:.5f} → results/qsann_best.pt")
    print("Full log   → results/qsann_fed_log_v2.npz")
    print("Final model→ results/qsann_final.pt")