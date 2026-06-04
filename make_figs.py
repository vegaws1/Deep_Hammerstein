"""Draw the architecture schematic and collect result figures next to main.tex.

Run AFTER run_paper.py:  python make_figs.py
"""

import os
import shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)


def _box(ax, xy, w, h, text, fc):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.03",
                                linewidth=1.4, edgecolor="#234", facecolor=fc))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=9)


def _arrow(ax, p0, p1):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=14,
                                 linewidth=1.3, color="#234"))


def architecture():
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.set_xlim(0, 11); ax.set_ylim(0, 4.6); ax.axis("off")
    y = 2.8
    _box(ax, (0.2, y), 1.7, 1.0, "Input $u_t$\n(step/PRBS/\nmultisine/random)", "#eef3fb")
    _box(ax, (2.3, y), 2.0, 1.0, "Nonlinear latent\nstatic $g_\\theta(u_t)$ /\nrecurrent $h_\\theta(u_{1:t})$", "#dce7f7")
    _box(ax, (4.7, y), 2.0, 1.0, "Canonicalize $n_t$\n(mean 0, cov $I$,\nsign/triangular)", "#cfe8d6")
    _box(ax, (7.1, y), 2.0, 1.0, "Stable MIMO ARX\n$\\rho(A_c)<1-\\epsilon$\n$\\hat x_t$", "#f7e6cc")
    _box(ax, (9.5, y), 1.3, 1.0, "$\\hat x_t$", "#f7e6cc")
    for x0, x1 in [(1.9, 2.3), (4.3, 4.7), (6.7, 7.1), (9.1, 9.5)]:
        _arrow(ax, (x0, y + 0.5), (x1, y + 0.5))
    # observation head
    _box(ax, (7.1, 1.1), 2.0, 1.0, "Multivariate Student-$t$\nAR residual model\n$\\Sigma=LL^\\top,\\ \\nu$", "#f7d6d6")
    _arrow(ax, (8.1, 2.8), (8.1, 2.1))
    ax.text(8.45, 2.45, "$y_t-\\hat x_t$", fontsize=8)
    # training stages strip
    _box(ax, (0.2, 0.1), 10.6, 0.7,
         "Staged training:  (1) robust step-response init  $\\to$  (2) deterministic recovery on $x_t$  "
         "$\\to$  (3) residual-likelihood refine  $\\to$  (4) joint fine-tune  (x-supervised or y-only)", "#eeeeee")
    ax.text(5.5, 4.35, "Identifiability-Regularized Probabilistic Deep Hammerstein Model",
            ha="center", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "architecture.png"), dpi=160, bbox_inches="tight")
    plt.close()
    print("wrote", os.path.join(FIG_DIR, "architecture.png"))


def collect():
    pairs = [
        ("results/figures/deterministic_X_channel2.png", "deterministic_X_channel2.png"),
        ("results/figures/measured_Y_channel2.png", "measured_Y_channel2.png"),
        ("results/figures/innovation_acf.png", "innovation_acf.png"),
        ("results/figures/qq_innovations.png", "qq_innovations.png"),
        ("results/figures/cross_correlation.png", "cross_correlation.png"),
        ("results/figures/eigenvalues.png", "eigenvalues.png"),
        ("results/figures/training_history.png", "training_history.png"),
        ("results/ablation/figures_ablation.png", "ablation.png"),
    ]
    for src, dst in pairs:
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(FIG_DIR, dst))
            print("copied", dst)
        else:
            print("MISSING", src)


if __name__ == "__main__":
    architecture()
    collect()
