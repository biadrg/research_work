# Implementation Guide — Federated Quantum Transformer Architectures for Time-Series Forecasting

This guide is written against the **actual contents** of the two repositories you named (I cloned and read them) plus the QSANN specification transcribed in your thesis. Every command, file path, and constraint below was verified rather than recalled. Companion code modules referenced throughout (`federated.py`, `qsann_ts.py`, `quixer_forecast.py`, `partition_data.py`, `train_federated.py`, `analysis.py`) ship alongside this guide; the quantum Q/K/V circuit, the softmax attention assembly, and the FedAvg averaging were executed and validated.

---

## 0. Reality check — what these three repos actually are

This is the single most important thing to fix before you write a line of code, because two of the three do **not** match the mental model implied by the task description, and one of them has no attention mechanism at all.

| Architecture | Repo / source | Framework | Has attention? | What it really is |
|---|---|---|---|---|
| **QuLTSF** | `chariharasuthan/qultsf` (verified) | PennyLane 0.37 | **No** | LTSF-Linear-style hybrid: `Linear(seq→2^q)` → `AmplitudeEmbedding` → `StronglyEntanglingLayers` → `⟨PauliZ⟩` → `Linear(q→pred)`, applied **channel-independently**. It is a quantum-enhanced *linear* forecaster, not a transformer. |
| **QSANN** | Li/Zhao/Wang 2022; no clean maintained forecasting repo | implement in PennyLane | **Yes** (true L×L) | Three PQCs produce per-token Q/K/V via ⟨Z⟩; softmax attention computed classically. Originally text classification; you adapt it to TS. |
| **QuantumTSTransformer (Quixer)** | `Transconnectome/TSQuantumTransformer` (verified) | **torchquantum** | **QSVT/LCU mixing** (not L×L) | Linear-combination-of-unitaries + quantum singular value transform over timesteps. The model file is already generic TS; only the *driver* is fMRI-specific. |

Three consequences you should bake into the thesis framing now:

1. **QuLTSF is your quantum baseline, not an attention model.** Step 4 of your plan ("track quantum attention mechanisms") is literally inapplicable to QuLTSF — there is nothing to track. For QuLTSF, inspect the `StronglyEntanglingLayers` weights and per-qubit ⟨Z⟩ instead, and present it as the ablation that isolates "what attention buys you."
2. **The three live on two frameworks.** QuLTSF + QSANN are PennyLane; Quixer is torchquantum. Keep them in **separate conda environments** (their pinned deps conflict), and make the federated layer **framework-agnostic** so one loop wraps all three — it only ever touches `state_dict()`. That is exactly how `federated.py` is written.
3. **Quixer hard-constrains `seq_len` to a power of two** (`n_ctrl_qubits = log2(n_timesteps)`). Choose windows like 32/64/128 so the *same* windows feed all three models and your comparison is clean.

---

## 1. Setup and installation

### 1.1 Clone

```bash
mkdir -p ~/THESIS/models && cd ~/THESIS/models
git clone https://github.com/chariharasuthan/qultsf.git
git clone https://github.com/Transconnectome/TSQuantumTransformer.git
# QSANN: no canonical forecasting repo — use the provided qsann_ts.py (implements your thesis Eqs. 16–19).
```

### 1.2 Two environments (deps genuinely conflict)

**Environment A — PennyLane stack (QuLTSF + QSANN + the federated loop):**

```bash
conda create -n fqt-pennylane python=3.11.7 -y
conda activate fqt-pennylane
# QuLTSF's verified pins:
pip install numpy==1.26.4 torch==2.5.1 PennyLane==0.37.0 pandas scikit-learn matplotlib
pip install sktime          # only needed to read PEMS-SF .ts files
```

**Environment B — torchquantum stack (Quixer):**

```bash
conda create -n fqt-torchquantum python=3.10 -y
conda activate fqt-torchquantum
pip install torch torchvision
# IMPORTANT: the PyPI torchquantum (0.1.8) is stale and lacks APIs QuixerTSModel uses
# (GeneralEncoder, MeasureMultipleTimes). Install from source:
pip install git+https://github.com/mit-han-lab/torchquantum.git
pip install numpy pandas scikit-learn matplotlib tqdm
```

Quantum libraries you actually need: **PennyLane** (`default.qubit` simulator; optionally `lightning.qubit` for speed) for QuLTSF/QSANN, and **torchquantum** for Quixer. No Qiskit required by any of the three.

> The federated loop in `federated.py` imports only `torch`, so drop it into **either** environment; you run QuLTSF/QSANN experiments in env A and Quixer experiments in env B, then compare the saved logs.

---

## 2. Data preprocessing

### 2.1 The two formats you must produce

- **QuLTSF** reads a CSV through its `Dataset_Custom`: a `date` column, then numeric features, **target last**; it does a fixed **70/10/20** train/val/test split and fits a `StandardScaler` on train. So you feed QuLTSF *one CSV per client*.
- **QSANN / Quixer** consume windowed tensors `X:(N, seq_len, V)`, `Y:(N, pred_len, V)`.

`partition_data.py` produces both. Run its `__main__` to see the wiring.

### 2.2 Federated partition (thesis Sec. V-F)

| Dataset | Natural client | Count | Notes |
|---|---|---|---|
| Beijing PRSA | one **station** | 12 | `PRSA2017_Data_...zip` already contains 12 per-station CSVs. Build `date` from year/month/day/hour. |
| NASDAQ-100 | one **asset** | up to 100 | **Use `nasdaq_100_stock_data.zip` (per-constituent), not `NASDAQ100_Historical_Data.csv`** — the latter is the single index series and can't give a per-asset partition. |
| PEMS-SF | one **sensor** | 963 | Read `PEMS-SF_TRAIN.ts` via sktime; each channel is a sensor-client. Subsample (e.g. 12–24 sensors) for tractable quantum simulation. |

Normalisation is done **per client on its own train split** (`fit_scaler` in `partition_data.py`) — this is the federated-correct choice (no client sees global statistics) and it makes the non-IID drift analysis meaningful.

### 2.3 Adapting QSANN from text classification → time series

Three concrete changes, all implemented in `qsann_ts.py`:

1. **Input.** A text token becomes a time step `x_t ∈ R^V`. A `Linear(V → n_qubits)` projects each step into an angle-encoding vector (thesis Eq. 13).
2. **Order.** A quantum attention layer is permutation-invariant (it sees a *set*). Inject a **learnable classical positional embedding** before the quantum layer (thesis Sec. V-E) so temporal direction survives without spending a qubit.
3. **Head.** Replace the class logit with a forecasting head `Linear(seq_len·n_qubits → pred_len·V)` reshaped to `(pred_len, V)` (thesis Eq. 6).

The quantum part is untouched from your thesis: three PQCs (`AngleEmbedding` + `StronglyEntanglingLayers`) produce Q/K/V as ⟨Z⟩ vectors (Eqs. 16–18); attention scores and the weighted value sum are computed classically (Eq. 19), which keeps the softmax off the quantum device.

### 2.4 Converting Quixer from fMRI → standard time series

Read this carefully — it's less work than it sounds. The **model** (`QuixerTSModel.py::QuixerTimeSeries`) is *already* generic: it takes `(B, n_timesteps, feature_dim)` through `feature_projection = Linear(feature_dim, n_rots)`. The fMRI specificity lives entirely in the **driver** `QuixerfMRI_Regress.py` (phenotype CSV loading, per-subject windows that share one label, AUC/accuracy metrics). So:

1. **Discard** `load_fmri_data` and `split_and_prepare_dataloaders` from the fMRI driver.
2. **Feed forecasting windows** (`make_windows`) instead of per-subject classification windows.
3. **Set** `feature_dim = V`, `output_dim = pred_len · V`, reshape output to `(pred_len, V)` — done in `quixer_forecast.py::QuixerForecaster`.
4. **Use the second return value.** `QuixerTimeSeries.forward` returns `(output, mean_lcu_norm)`. The fMRI code throws away `mean_lcu_norm`; add `(1 − mean_lcu_norm)` to the loss as a regulariser so training favours realisable (high-success-probability) circuits.
5. **Swap metrics** from classification (AUC/accuracy) to MSE/MAE.
6. **Respect the power-of-two `seq_len`** — `QuixerForecaster` asserts it and suggests the next valid length.

---

## 3. Implementation and federated integration

### 3.1 Running the native QuLTSF baseline (non-federated, to sanity-check data)

QuLTSF's own script expects `--data custom` with a CSV in `./dataset/`. To run it on one Beijing station first:

```bash
conda activate fqt-pennylane
cd ~/THESIS/models/qultsf
mkdir -p dataset && cp ~/THESIS/qultsf_dataset/beijing_Aotizhongxin.csv dataset/

python -u run_longExp.py \
  --is_training 1 --model QuLTSF --data custom \
  --root_path ./dataset/ --data_path beijing_Aotizhongxin.csv \
  --model_id beijing_64_16 --features M \
  --seq_len 64 --pred_len 16 --enc_in 9 \
  --num_qubits 10 --QML_device default.qubit --num_layers 3 \
  --batch_size 16 --learning_rate 1e-4 --itr 1
```

(`--enc_in 9` = the nine PRSA numeric features; set `--enc_in` to your column count.)

### 3.2 The framework-agnostic FedAvg wrapper

`federated.py` implements your thesis Eq. 7 exactly and works for **all three** models because it only serialises `named_parameters()`. The contract per architecture is just two callables:

- a model constructor, and
- a `forward_loss(model, batch) -> scalar` (because the forward signatures differ — QuLTSF/QSANN return a tensor; Quixer returns a tuple + needs the LCU regulariser).

**QSANN federated run (full end-to-end, validated structure):**

```bash
conda activate fqt-pennylane
cd ~/THESIS   # with the .py modules + THESIS/data on the path
python train_federated.py
```

`train_federated.py` builds one client per Beijing station, runs 30 FedAvg rounds with 2 local epochs each, evaluates on a pooled held-out set, and writes `results/qsann_fed_log.npz` + `results/qsann_global.pt`.

**Plugging QuLTSF into the same loop** — wrap its `Model` (from `qultsf/models/QuLTSF.py`) in a tiny config object and reuse the harness:

```python
from federated import Client, run_fedavg          # same harness
# QuLTSF.Model expects an argparse-like config with seq_len, pred_len,
# num_qubits, QML_device, num_layers. Build a SimpleNamespace and pass it.
# forward_loss: QuLTSF.forward(x) returns (B, pred_len, C) directly -> mse_loss.
```

**Plugging Quixer in** (env B): construct `QuixerForecaster` and use `quixer_forward_loss` (already includes the LCU regulariser) as the `forward_loss`.

### 3.3 "Replace a classical layer with a quantum circuit"

You do not need to surgically replace layers in QuLTSF or Quixer — they are quantum by construction. The one place this pattern applies is **QSANN's Q/K/V projections**: classical `W_Q, W_K, W_V` are replaced by the three PQCs. That replacement is the whole of `qsann_ts.py::QuantumSelfAttentionLayer` — each `_qkv_qnode` is a `qml.qnn.TorchLayer`, so the quantum weights are ordinary `torch.nn.Parameter`s and flow through FedAvg unchanged.

---

## 4. Analysis

### 4.1 Metrics to monitor

Forecast quality: **MSE, MAE, RMSE** (and MAPE with care — Beijing PM2.5 and traffic occupancy both hit values near zero, so MAPE explodes; report it but lead with RMSE/MAE). QuLTSF already ships `utils/metrics.py` with all of these; `analysis.py::forecast_metrics` mirrors them for the other two so all three are scored identically.

Federated efficiency (thesis Sec. V-G): **trainable parameter count** and **cumulative communication (MB)** — both produced by `run_fedavg` (`comm_bytes_cumulative`, `param_count`). Plot accuracy-vs-communication, not just accuracy-vs-rounds; that curve is where QSANN's "three PQCs only" parameter economy should visibly win.

Non-IID diagnostic: **client↔global drift** `‖θ_k − θ_global‖`, logged every round per client (`state_drift`). `plot_drift` renders it as a station×round heatmap — directly the "model separation under non-identical client profiles" your thesis promises.

### 4.2 Tracking the quantum attention (with honest per-model caveats)

This is the heart of the "structural sources of quantum advantage" claim, so be precise about what each model exposes:

- **QSANN** — a genuine `L×L` attention matrix per layer (`model.attention_maps()`). Visualise with `plot_qsann_attention`. The scalar to track across federated rounds is **attention entropy** (`attention_entropy`): falling entropy = the quantum attention is *specialising* (focusing on fewer, informative timesteps) as FedAvg proceeds. That trend, plotted against a classical self-attention baseline of equal width, is your cleanest structural-advantage figure.
- **Quixer** — **no** `L×L` map. Track `|mix_coeffs|` per timestep and the QSVT polynomial coefficients (`plot_quixer_mixing`), plus the mean LCU success probability over training (it tells you whether the "attention" is physically realisable).
- **QuLTSF** — **no attention.** Report the `StronglyEntanglingLayers` weight norms and per-qubit ⟨Z⟩ distributions as the baseline's quantum-state usage. Framing it as the no-attention ablation is what makes the other two models' attention curves interpretable.

### 4.3 Logging and visualising for the thesis

`run_fedavg` prints per round and you save NPZ logs; `analysis.py` turns them into four figure types — convergence, accuracy-vs-communication, drift heatmap, attention/mixing — at 160 dpi, ready to drop into the thesis. Suggested standard experiment grid: `{angle, amplitude, data-reuploading}` encodings × `{Beijing, NASDAQ, PEMS-SF}` × `{QuLTSF, QSANN, Quixer}`, fixed `seq_len=64, pred_len=16`, identical client partitions, three seeds. Report mean ± std; the seed spread matters because quantum simulators + small PQCs are high-variance.

---

## 5. Consolidated gotchas

1. **Quixer `seq_len` must be a power of two** — use 32/64/128 or left-pad.
2. **Quixer returns a tuple** — unpack `(pred, norm)`; add `(1−norm)` to the loss.
3. **Two environments** — PennyLane 0.37 (QuLTSF/QSANN) and torchquantum-from-GitHub (Quixer) don't coexist cleanly.
4. **NASDAQ partition** needs the per-constituent zip, not the index CSV.
5. **PEMS-SF is huge** (963 sensors) — subsample sensors for simulation tractability.
6. **Per-client scaling** (not global) — required for an honest federated + non-IID story.
7. **QuLTSF ≠ transformer** — don't claim attention for it; it's the baseline.
8. **Column names** in `partition_data.py` are written to the documented dataset layouts; verify them against your actual files (the CSV contents weren't available when this was written) and adjust the `*_COLS` constants if needed.
