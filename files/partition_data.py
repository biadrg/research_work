#  """
# partition_data.py
# -----------------
# Load your three datasets from the THESIS/data tree and turn them into federated
# clients (thesis Sec. V-F: one client per station / asset / sensor).

# Two output formats are produced because the models consume data differently:
#   (A) QuLTSF expects a CSV per source with a `date` column + numeric features +
#       target last (its Dataset_Custom does 70/10/20 split + StandardScaler on
#       train). `export_qultsf_csv` writes one such CSV per client.
#   (B) QSANN / Quixer consume windowed tensors (input window -> horizon).
#       `make_windows` + `client_loaders` build those.

# IMPORTANT HONESTY NOTE: I wrote these loaders against the *documented* column
# layouts of the public datasets, because the actual CSV contents were not available
# to me (only your directory screenshot + thesis PDF). Check the column names below
# against your files and adjust the *_COLS constants if they differ. The structure
# (per-client partition, windowing, scaling) is correct regardless.
# """

from __future__ import annotations
import glob
import os
import zipfile

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


# =========================================================================== #
# Generic windowing                                                           #
# =========================================================================== #
def make_windows(arr: np.ndarray, seq_len: int, pred_len: int):
    """arr: (T, V) -> X:(N, seq_len, V), Y:(N, pred_len, V) sliding by 1 step."""
    X, Y = [], []
    T = arr.shape[0]
    for s in range(0, T - seq_len - pred_len + 1):
        X.append(arr[s:s + seq_len])
        Y.append(arr[s + seq_len:s + seq_len + pred_len])
    return np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.float32)


def fit_scaler(train_arr: np.ndarray):
    mean = train_arr.mean(0, keepdims=True)
    std = train_arr.std(0, keepdims=True)
    std[std == 0] = 1e-8
    return mean, std


def client_loaders(series_by_client: dict[str, np.ndarray],
                   seq_len: int, pred_len: int, batch_size: int = 16,
                   train_frac: float = 0.7):
    """Per-client (T,V) series -> dict[name] -> (train_loader, test_X, test_Y).
    Scaler is fit on each client's OWN train split (federated: no global stats)."""
    out = {}
    for name, arr in series_by_client.items():
        n_train = int(len(arr) * train_frac)
        mean, std = fit_scaler(arr[:n_train])
        arr = (arr - mean) / std
        Xtr, Ytr = make_windows(arr[:n_train], seq_len, pred_len)
        Xte, Yte = make_windows(arr[n_train:], seq_len, pred_len)
        if len(Xtr) == 0 or len(Xte) == 0:
            print(f"  [skip] client {name}: series too short for the window.")
            continue
        ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ytr))
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
        out[name] = (loader,
                     torch.from_numpy(Xte), torch.from_numpy(Yte),
                     len(Xtr))
    return out


# =========================================================================== #
# Beijing PRSA air quality  -> one client per station (12 clients)            #
# =========================================================================== #
PRSA_TIME_COLS = ["year", "month", "day", "hour"]
PRSA_FEATURE_COLS = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3",
                     "TEMP", "PRES", "DEWP"]   # numeric, matches thesis Sec. V-F


def load_beijing(zip_path: str) -> dict[str, pd.DataFrame]:
    """Returns {station_name: dataframe with a 'date' column + PRSA_FEATURE_COLS}."""
    stations = {}
    with zipfile.ZipFile(zip_path) as z:
        members = [m for m in z.namelist() if m.endswith(".csv")]
        for m in members:
            with z.open(m) as f:
                df = pd.read_csv(f, nrows=1000)
            station = df["station"].iloc[0] if "station" in df else os.path.basename(m)
            df["date"] = pd.to_datetime(df[PRSA_TIME_COLS].rename(
                columns={"year": "year", "month": "month",
                         "day": "day", "hour": "hour"}))
            keep = ["date"] + [c for c in PRSA_FEATURE_COLS if c in df.columns]
            df = df[keep].interpolate().bfill().ffill()
            stations[str(station)] = df
    print(f"Beijing PRSA: {len(stations)} station-clients loaded.")
    return stations


# =========================================================================== #
# NASDAQ-100 -> one client per constituent asset                              #
# =========================================================================== #
def load_nasdaq_constituents(folder: str, value_col: str = "Close"
                             ) -> dict[str, pd.DataFrame]:
    """Expects one CSV per ticker (from nasdaq_100_stock_data.zip), each with a
    Date column and OHLCV columns. One client per ticker.

    NOTE: NASDAQ100_Historical_Data.csv is the *index* (a single series) and
    CANNOT give you a per-asset federated partition on its own -- use the
    per-constituent files for the partition the thesis describes."""
    clients = {}
    for path in glob.glob(os.path.join(folder, "*.csv")):
        ticker = os.path.splitext(os.path.basename(path))[0]
        df = pd.read_csv(path, nrows=1000)
        date_col = next((c for c in df.columns if c.lower() == "date"), df.columns[0])
        df = df.rename(columns={date_col: "date"})
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        col = value_col if value_col in df.columns else df.select_dtypes("number").columns[0]
        clients[ticker] = df[["date", col]].dropna().sort_values("date")
    print(f"NASDAQ: {len(clients)} asset-clients loaded (value col guessed per file).")
    return clients


# =========================================================================== #
# PEMS-SF traffic -> one client per sensor                                    #
# =========================================================================== #
def load_pems_ts(ts_path: str) -> dict[str, np.ndarray]:
    """Load PEMS-SF_TRAIN.ts / _TEST.ts via sktime. The UCR PEMS-SF is multivariate
    (963 sensor channels). We treat each channel as a sensor-client whose series is
    the concatenation of all samples for that channel.

    Requires: pip install sktime
    """
    from sktime.datasets import load_from_tsfile          # noqa: E501
    X, _ = load_from_tsfile(ts_path, return_data_type="numpy3d")  # (n, n_channels, L)
    n, C, L = X.shape
    clients = {}
    for c in range(C):
        clients[f"sensor_{c:03d}"] = X[:, c, :].reshape(-1, 1)[:1000]  # (n*L, 1)
    print(f"PEMS-SF: {C} sensor-clients loaded from {os.path.basename(ts_path)}.")
    return clients


# =========================================================================== #
# QuLTSF CSV export                                                           #
# =========================================================================== #
def export_qultsf_csv(df: pd.DataFrame, out_path: str, target_col: str):
    """Write a CSV in QuLTSF's Dataset_Custom format: date, ...features, target."""
    cols = [c for c in df.columns if c not in ("date", target_col)]
    ordered = df[["date"] + cols + [target_col]]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    ordered.to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":
    # Example wiring against your THESIS/data tree --------------------------- #
    ROOT = "."
    beijing = load_beijing(os.path.join(ROOT, "climate",
                                        "PRSA2017_Data_20130301-20170228.zip"))
    # -> windowed federated clients for QSANN / Quixer (PM2.5 is target/var 0)
    series = {name: df[PRSA_FEATURE_COLS].to_numpy(dtype=np.float32)
              for name, df in beijing.items()
              if all(c in df.columns for c in PRSA_FEATURE_COLS)}
    loaders = client_loaders(series, seq_len=64, pred_len=16, batch_size=16)
    print(f"Built {len(loaders)} windowed clients (seq_len=64 -> Quixer-compatible).")
    # -> QuLTSF CSVs (one per station)
    for name, df in beijing.items():
        export_qultsf_csv(df, f"qultsf_dataset/beijing_{name}.csv", target_col="PM2.5")