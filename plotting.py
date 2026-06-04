"""Figure generation (matplotlib, Agg backend)."""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _save(path):
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_training_history(hist, path):
    tr = [h[0]["total"] for h in hist]
    va = [h[1]["total"] for h in hist]
    ep = np.arange(1, len(hist) + 1)
    plt.figure(figsize=(7, 4))
    plt.plot(ep, tr, label="Train")
    plt.plot(ep, va, label="Validation")
    plt.xlabel("Epoch"); plt.ylabel("Total loss"); plt.grid(alpha=0.3); plt.legend()
    plt.title("Training convergence (staged)")
    _save(path)


def plot_prediction(Y_true, Y_pred, path, seq=0, channel=1, title="Prediction vs target"):
    t = np.arange(Y_true.shape[1])
    plt.figure(figsize=(7, 4))
    plt.plot(t, Y_true[seq, :, channel], label="Target", lw=1.8)
    plt.plot(t, Y_pred[seq, :, channel], "--", label="Prediction", lw=1.8)
    plt.xlabel("Time"); plt.ylabel(f"Channel {channel + 1}"); plt.grid(alpha=0.3); plt.legend()
    plt.title(title)
    _save(path)


def plot_acf(acf, band, path):
    s = acf.shape[0]
    lags = np.arange(acf.shape[1])
    fig, axes = plt.subplots(1, s, figsize=(5 * s, 3.5), squeeze=False)
    for k in range(s):
        ax = axes[0, k]
        ax.stem(lags, acf[k])
        ax.axhline(band, color="r", ls="--", lw=1)
        ax.axhline(-band, color="r", ls="--", lw=1)
        ax.set_title(f"Innovation ACF - channel {k + 1}")
        ax.set_xlabel("Lag"); ax.grid(alpha=0.3)
    _save(path)


def plot_cross_corr(corr, path):
    plt.figure(figsize=(4.5, 4))
    im = plt.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    plt.colorbar(im, fraction=0.046)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            plt.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=9)
    plt.title("Cross-channel innovation correlation")
    plt.xticks(range(corr.shape[0])); plt.yticks(range(corr.shape[0]))
    _save(path)


def plot_qq(qq, path):
    plt.figure(figsize=(5, 5))
    plt.plot(qq["student_t"], qq["sample"], ".", ms=3, label="vs Student-t")
    plt.plot(qq["gaussian"], qq["sample"], ".", ms=3, alpha=0.5, label="vs Gaussian")
    lim = [min(qq["student_t"].min(), qq["sample"].min()),
           max(qq["student_t"].max(), qq["sample"].max())]
    plt.plot(lim, lim, "k--", lw=1)
    plt.xlabel("Theoretical quantiles"); plt.ylabel("Sample quantiles")
    plt.title("QQ plot of standardized innovations"); plt.grid(alpha=0.3); plt.legend()
    _save(path)


def plot_eigenvalues(true_eig, est_eig, path):
    theta = np.linspace(0, 2 * np.pi, 200)
    plt.figure(figsize=(5, 5))
    plt.plot(np.cos(theta), np.sin(theta), "k-", lw=1, label="unit circle")
    plt.scatter(true_eig.real, true_eig.imag, marker="o", s=70,
                facecolors="none", edgecolors="b", label="true")
    plt.scatter(est_eig.real, est_eig.imag, marker="x", s=70, color="r", label="estimated")
    plt.axhline(0, color="gray", lw=0.5); plt.axvline(0, color="gray", lw=0.5)
    plt.gca().set_aspect("equal"); plt.grid(alpha=0.3); plt.legend()
    plt.title("Companion-matrix eigenvalues")
    _save(path)


def plot_ablation_heatmap(rows, metrics, path):
    labels = [r["ablation"] for r in rows]
    data = np.array([[r.get(m, np.nan) for m in metrics] for r in rows], float)
    # normalize each column to [0,1] for visual comparison
    norm = (data - np.nanmin(data, 0)) / (np.nanmax(data, 0) - np.nanmin(data, 0) + 1e-9)
    plt.figure(figsize=(1.4 * len(metrics) + 3, 0.5 * len(labels) + 2))
    im = plt.imshow(norm, aspect="auto", cmap="viridis")
    plt.colorbar(im, fraction=0.046, label="column-normalized")
    plt.xticks(range(len(metrics)), metrics, rotation=30, ha="right")
    plt.yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(metrics)):
            plt.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center",
                     color="w", fontsize=7)
    plt.title("Ablation results (raw values shown)")
    _save(path)
