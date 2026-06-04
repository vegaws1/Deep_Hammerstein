"""Reproducibility, metrics and small I/O helpers."""

import os
import json
import csv
import random

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False

DTYPE = "float32"


def set_seed(seed: int = 1234):
    random.seed(seed)
    np.random.seed(seed)
    if _HAS_TORCH:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device():
    if _HAS_TORCH and torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ----------------------------------------------------------------------
# Numpy metrics
# ----------------------------------------------------------------------
def mse_np(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))


def mae_np(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse_np(y_true, y_pred):
    return float(np.sqrt(mse_np(y_true, y_pred)))


def r2_np(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-12))


def nrmse_np(y_true, y_pred, eps=1e-8):
    return float(rmse_np(y_true, y_pred) / (np.max(y_true) - np.min(y_true) + eps))


def fit_percent_np(y_true, y_pred, eps=1e-8):
    num = np.linalg.norm(y_true - y_pred)
    den = np.linalg.norm(y_true - np.mean(y_true)) + eps
    return float(max(0.0, 100.0 * (1.0 - num / den)))


def wmape_np(y_true, y_pred, eps=1e-6):
    return float(100.0 * np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + eps))


def relative_param_error(true_arr, est_arr, eps=1e-12):
    true_arr = np.asarray(true_arr, dtype=float)
    est_arr = np.asarray(est_arr, dtype=float)
    return float(np.linalg.norm(est_arr - true_arr) / (np.linalg.norm(true_arr) + eps))


def evaluate_metrics(Y_true, Y_pred):
    return {
        "MSE": mse_np(Y_true, Y_pred),
        "MAE": mae_np(Y_true, Y_pred),
        "RMSE": rmse_np(Y_true, Y_pred),
        "NRMSE": nrmse_np(Y_true, Y_pred),
        "R2": r2_np(Y_true, Y_pred),
        "FIT": fit_percent_np(Y_true, Y_pred),
        "wMAPE": wmape_np(Y_true, Y_pred),
    }


def evaluate_metrics_with_channels(Y_true, Y_pred):
    overall = evaluate_metrics(Y_true, Y_pred)
    per_channel = {}
    for k in range(Y_true.shape[-1]):
        per_channel[f"channel_{k + 1}"] = evaluate_metrics(Y_true[..., k], Y_pred[..., k])
    return {"overall": overall, "per_channel": per_channel}


def per_sequence_rmse(Y_true, Y_pred):
    """RMSE per sequence (used for paired significance tests)."""
    diff2 = (Y_true - Y_pred) ** 2
    return np.sqrt(diff2.reshape(diff2.shape[0], -1).mean(axis=1))


# ----------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def save_csv_dicts(rows, path):
    if len(rows) == 0:
        return
    all_keys = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in all_keys})


def flatten_metric_block(prefix, block):
    row = {}
    for k, v in block["overall"].items():
        row[f"{prefix}_{k}"] = v
    for ch_name, ch_metrics in block["per_channel"].items():
        for k, v in ch_metrics.items():
            row[f"{prefix}_{ch_name}_{k}"] = v
    return row
