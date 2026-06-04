# MINO — Identifiability-Regularized Probabilistic Deep Hammerstein Modeling of Noisy MIMO Systems

Revised implementation accompanying the manuscript
*"Identifiability-Regularized Probabilistic Deep Hammerstein Modeling of Noisy MIMO Systems."*

This is the major-revision rebuild of the earlier *staged GRU* code. It turns a
structured deep predictor into a rigorously formulated, **identifiable,
stable, probabilistic** deep Hammerstein framework, and it regenerates every
table and figure in the paper **from scratch**.

## What is new relative to the original code

| Area | Original | This version |
|------|----------|--------------|
| Nonlinear block | recurrent GRU only | **static (memoryless) + recurrent** variants (`mino/models.py`) |
| Identifiability | none | latent **canonicalization** + **aligned** parameter metrics (Prop. 1–2, `mino/identify.py`) |
| Linear block | unconstrained ARX | **Schur-stable** companion (exact projection + norm penalty), BIBO (Prop. 3) |
| Noise model | channel-wise Student-t | **multivariate Student-t** AR likelihood, full Σ=LLᵀ, learnable ν>2, residual-AR stationarity (Prop. 4) |
| Losses | unit-dependent | **dimensionless** normalization |
| Stage C | ambiguous | explicit **simulator-supervised (x)** and **measurement-only (y)** variants |
| Baselines | GRU, ARX | + **LSTM, TCN, NARX-MLP**, + **matched polynomial Hammerstein–ARX** |
| Evidence | 5-seed means | **paired significance tests** (Wilcoxon, bootstrap CI, Cohen dz), **ablation matrix**, **residual diagnostics** (ACF, cross-corr, Ljung–Box, QQ/PIT, coverage) |

### Added in the second revision (responding to reviewer feedback)

| Area | Change |
|------|--------|
| Statistics | **seed-level / hierarchical bootstrap** significance (primary), alongside the pooled per-sequence test (secondary) — `mino/stats.py:hierarchical_comparison` |
| Noise model | explicit **variance floor** Σ = LLᵀ + σ_min²·I (anti-degeneracy; Theorem 4a) |
| Stress tests | **5 noise-regime sweep** (white/colored × Gaussian/Student-t, outliers, strong cross-correlation) reporting NLL / coverage / whiteness — `run_regimes` |
| Front-end | **TCN nonlinear front-end** variant → framework is front-end agnostic (GRU / MLP / TCN) |
| Complexity | **parameter-count + per-step runtime** table — `measure_complexity` |
| Theory | full **Theorems 1–4 with complete proofs** (Appendix A of the manuscript) |
| Claims | calibrated: ablation framed as **trade-offs**, no raw-accuracy-supremacy claim, coverage called "conservative" |

### Added in the final revision (prediction enhancement, kept separate from identification)

| Area | Change |
|------|--------|
| Prediction variant | **polynomial input path** `n=TCN(u)+W_p·φ(u)` (`poly_degree`), **multi-horizon rollout loss** (`use_rollout`), optional **regularized residual-correction head** `Δ(u,n,x)` for prediction only (`use_residual_head`) — structural estimate/params untouched |
| Driver | `run_prediction_variants` / `run_pred_only.py` → `results/prediction/prediction_variants.csv` (GRU / TCN / TCN+poly / TCN+poly+residual vs black-box TCN) |
| Notation | manuscript `triu_1(B_1)` = strict upper (code already used `diagonal=1`) |

Earlier main/ablation/regime/complexity results are unchanged — the new options default off.

## Install

```bash
pip install -r requirements.txt          # CPU is fine
# torch CPU wheel: pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Regenerate all results

```bash
python run_paper.py                       # default config (multi-seed), writes results/
python run_paper.py --smoke               # ~1 min end-to-end sanity check
python run_paper.py --seeds 0 1 2 3 4 --ablation-seeds 0 1 --regime-seeds 0 1 --threads 8
python run_paper.py --no-regimes --no-ablation   # main comparison only (fast)
python run_paper.py --plant recurrent --noise gaussian --no-color   # other plant/noise
```

Outputs (under `results/`):

- `main_aggregate.csv` — per-model mean±std (RMSE, R², FIT, wMAPE) on X and Y (10 models)
- `parameter_recovery.csv` — rel(A), aligned/raw rel(B), impulse-response error, ν, stability
- `significance_vs_direct_gru.csv` — pooled per-sequence ΔRMSE, Wilcoxon p, bootstrap CI, dz (secondary)
- `significance_hierarchical.csv` — **seed-level** ΔRMSE, paired-t p, hierarchical bootstrap CI (primary)
- `per_seed_rmse_X.json` — raw per-seed per-sequence RMSE for re-analysis
- `diagnostics_summary.json` — whiteness, coverage@{50,80,95}, PIT-KS
- `complexity.csv` — parameter counts and per-train-step wall-time
- `ablation/ablation.csv` — ablation matrix
- `regimes/regimes.csv` — noise-regime stress test (NLL, coverage, whiteness, cross-corr) × {Gaussian, diag-t, mvt}
- `summary.json` — everything above, machine-readable
- `figures/` — training history, prediction, innovation ACF, cross-correlation, QQ, eigenvalues

Helper scripts: `make_figs.py` (collect figures next to `main.tex`), `fill_numbers.py`
(print LaTeX-ready table rows from `results/`), `package.py` (zip the project).
Released code is tagged; see the version line below.

## Package layout

```
mino/
  data.py          static & recurrent plants, noise regimes, excitation, scalers
  models.py        static/GRU blocks, stable linear block, multivariate Student-t head, baselines
  training.py      staged trainer (A / noise-pick / B / C), baseline trainer
  identify.py      linear ARX, latent alignment, aligned-B, eigenvalues, impulse response
  diagnostics.py   ACF, Ljung–Box, cross-correlation, QQ, PIT/coverage, whiteness
  stats.py         paired significance tests
  experiment.py    datasets, staged orchestration, multi-seed + ablation drivers
  plotting.py      figures
run_paper.py       entry point (regenerates everything)
main.tex           revised manuscript
References.bib     bibliography
REVISION_RESPONSE.md  point-by-point response to the major-revision report
```

## Notes

- The default `run_paper.py` configuration is deliberately lightweight so it runs
  on a laptop CPU in minutes; it is **honest but small**. Increase `--seeds`,
  the dataset sizes and epoch counts in `mino/experiment.py:RunConfig` for
  publication-scale runs.
- `rel(A)` is reported but remains high by design: Section 4 of the manuscript
  explains that exact `A` recovery is intrinsically limited by latent/output
  collinearity; coordinate-robust impulse-response and eigenvalue errors are the
  meaningful structural metrics, and the **static** variant clearly beats the
  **recurrent** one on aligned-`B` (Proposition 2 / Theorem 2).
- On this benchmark the proposed method is **not** the strongest raw predictor (a
  TCN and a polynomial Hammerstein–ARX are competitive/better); its contribution
  is the identifiability, stability and calibrated heavy-tailed noise model the
  black-box predictors lack. The paper states this explicitly.

---
**Version:** v2.0 (second major revision). Tag this commit before submission and
record the hash here for the Data & Code Availability statement.
