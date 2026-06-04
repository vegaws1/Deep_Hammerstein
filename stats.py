"""Paired significance testing for small accuracy margins (Sec. 6.5)."""

import numpy as np
from scipy import stats


def paired_comparison(rmse_baseline, rmse_model, n_boot=2000, seed=0):
    """Paired analysis of per-sequence RMSE: delta = baseline - model (>0 favors model).

    Returns mean/median paired difference, 95% t-CI, Wilcoxon signed-rank
    p-value, a bootstrap CI over sequences, and the Cohen d_z effect size.
    """
    a = np.asarray(rmse_baseline, float)
    b = np.asarray(rmse_model, float)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    delta = a - b

    mean = float(delta.mean())
    median = float(np.median(delta))
    sd = float(delta.std(ddof=1)) if n > 1 else 0.0
    se = sd / np.sqrt(n) if n > 0 else 0.0
    tcrit = stats.t.ppf(0.975, df=max(n - 1, 1))
    ci = (float(mean - tcrit * se), float(mean + tcrit * se))

    try:
        w_p = float(stats.wilcoxon(a, b).pvalue)
    except ValueError:
        w_p = float("nan")

    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(delta, size=n, replace=True).mean() for _ in range(n_boot)])
    boot_ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))

    dz = float(mean / sd) if sd > 0 else 0.0
    return {
        "n_pairs": int(n),
        "mean_delta_rmse": mean,
        "median_delta_rmse": median,
        "ci95_t": ci,
        "wilcoxon_p": w_p,
        "bootstrap_ci95": boot_ci,
        "cohen_dz": dz,
        "significant_05": bool((w_p == w_p) and w_p < 0.05 and boot_ci[0] > 0),
    }


def hierarchical_comparison(baseline_per_seed, model_per_seed, n_boot=4000, seed=0):
    """Seed-level paired analysis with a two-level (seed, sequence) bootstrap.

    Inputs are lists (one entry per seed) of per-sequence RMSE arrays. The unit
    of analysis is the *seed*: we form the per-seed mean RMSE difference
    delta_s = mean_q RMSE_baseline[s,q] - mean_q RMSE_model[s,q], then test
    these S paired values. The hierarchical bootstrap resamples seeds with
    replacement and, within each chosen seed, sequences with replacement -- so
    the reported uncertainty does not treat sequences from one trained model as
    independent runs.
    """
    S = len(baseline_per_seed)
    base = [np.asarray(b, float) for b in baseline_per_seed]
    mod = [np.asarray(m, float) for m in model_per_seed]
    delta_s = np.array([base[s].mean() - mod[s].mean() for s in range(S)])

    mean = float(delta_s.mean())
    sd = float(delta_s.std(ddof=1)) if S > 1 else 0.0
    se = sd / np.sqrt(S) if S > 0 else 0.0
    tcrit = stats.t.ppf(0.975, df=max(S - 1, 1))
    ci = (float(mean - tcrit * se), float(mean + tcrit * se))
    # paired t-test of delta_s against 0
    if S > 1 and sd > 0:
        tstat = mean / se
        t_p = float(2.0 * stats.t.sf(abs(tstat), df=S - 1))
    else:
        t_p = float("nan")
    n_pos = int((delta_s > 0).sum())  # sign agreement across seeds

    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for k in range(n_boot):
        chosen = rng.integers(0, S, S)
        vals = np.empty(S)
        for j, si in enumerate(chosen):
            b, m = base[si], mod[si]
            bi = rng.integers(0, len(b), len(b))
            mi = rng.integers(0, len(m), len(m))
            vals[j] = b[bi].mean() - m[mi].mean()
        boots[k] = vals.mean()
    boot_ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))

    return {
        "n_seeds": int(S),
        "seed_mean_delta_rmse": mean,
        "seed_ci95_t": ci,
        "paired_t_p": t_p,
        "seeds_favoring_model": n_pos,
        "hier_bootstrap_ci95": boot_ci,
        "significant_seedlevel": bool(boot_ci[0] > 0 and (t_p == t_p) and t_p < 0.05),
    }
