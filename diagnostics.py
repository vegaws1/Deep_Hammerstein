"""Residual / innovation diagnostics for the probabilistic noise model."""

import numpy as np
from scipy import stats


def autocorrelation(z, max_lag=20):
    """Per-channel ACF of standardized innovations. z: (N, T, s)."""
    N, T, s = z.shape
    acf = np.zeros((s, max_lag + 1))
    for k in range(s):
        zk = z[:, :, k] - z[:, :, k].mean()
        var = (zk ** 2).mean() + 1e-12
        for lag in range(max_lag + 1):
            if lag == 0:
                acf[k, lag] = 1.0
            else:
                acf[k, lag] = (zk[:, :-lag] * zk[:, lag:]).mean() / var
    band = 1.96 / np.sqrt(N * T)
    return acf, float(band)


def ljung_box(z, lags=10):
    """Ljung-Box portmanteau test per channel (reported cautiously)."""
    N, T, s = z.shape
    acf, _ = autocorrelation(z, max_lag=lags)
    n = N * T
    out = {}
    for k in range(s):
        r = acf[k, 1:lags + 1]
        Q = n * (n + 2) * np.sum((r ** 2) / (n - np.arange(1, lags + 1)))
        p = float(stats.chi2.sf(Q, df=lags))
        out[f"channel_{k + 1}"] = {"Q": float(Q), "p_value": p}
    return out


def cross_correlation(z):
    """Contemporaneous cross-channel correlation of innovations (s x s)."""
    zf = z.reshape(-1, z.shape[-1])
    return np.corrcoef(zf.T)


def whiteness_score(z, lags=10):
    """Mean absolute ACF over lags 1..L across channels (lower = whiter)."""
    acf, _ = autocorrelation(z, max_lag=lags)
    return float(np.mean(np.abs(acf[:, 1:lags + 1])))


def qq_data(z, df=5.0):
    """Sample quantiles vs Student-t and Gaussian theoretical quantiles."""
    zf = np.sort(z.reshape(-1))
    n = len(zf)
    probs = (np.arange(1, n + 1) - 0.5) / n
    return {"sample": zf, "student_t": stats.t.ppf(probs, df), "gaussian": stats.norm.ppf(probs)}


def pit_coverage(z, df=5.0, levels=(0.5, 0.8, 0.95)):
    """Empirical central coverage of standardized innovations under t_df."""
    zf = z.reshape(-1)
    cov = {}
    for a in levels:
        q = stats.t.ppf(0.5 + a / 2.0, df)
        cov[f"coverage_{int(a * 100)}"] = float(np.mean(np.abs(zf) <= q))
    # PIT uniformity (KS statistic against uniform)
    u = stats.t.cdf(zf, df)
    ks = float(stats.kstest(u, "uniform").statistic)
    cov["pit_ks"] = ks
    return cov
