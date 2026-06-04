"""Identification-aware metrics: aligned parameter recovery, companion
eigenvalues and impulse-response error.

Raw ``rel_error(B)`` is meaningless because the latent block is identifiable
only up to a coordinate transform ``C`` (Proposition 1).  We therefore align
the estimated latent coordinates to the true ones before scoring ``B``.
"""

import numpy as np
from sklearn.linear_model import HuberRegressor


# ----------------------------------------------------------------------
# Linear ARX baseline (numpy)
# ----------------------------------------------------------------------
def build_arx_regressors(U, Y, na=2, nb=2):
    N, T, p = U.shape
    s = Y.shape[2]
    maxlag = max(na, nb)
    Xreg, Yreg = [], []
    for n in range(N):
        u, y = U[n], Y[n]
        for t in range(maxlag, T):
            feat = []
            for i in range(na):
                feat.extend((-y[t - i - 1]).tolist())
            for j in range(nb):
                feat.extend((u[t - j - 1]).tolist())
            Xreg.append(feat); Yreg.append(y[t].tolist())
    return np.asarray(Xreg), np.asarray(Yreg)


def fit_linear_arx(U, X, na=2, nb=2, max_iter=300):
    Xreg, Yreg = build_arx_regressors(U, X, na, nb)
    s, p = Yreg.shape[1], U.shape[2]
    coefs = []
    for k in range(s):
        m = HuberRegressor(epsilon=1.35, alpha=1e-4, fit_intercept=False, max_iter=max_iter)
        m.fit(Xreg, Yreg[:, k]); coefs.append(m.coef_.copy())
    coefs = np.stack(coefs, 0)
    A = np.zeros((na, s, s)); B = np.zeros((nb, s, p))
    idx = 0
    for i in range(na):
        A[i] = coefs[:, idx:idx + s]; idx += s
    for j in range(nb):
        B[j] = coefs[:, idx:idx + p]; idx += p
    return A, B


def poly_features(U, degree=2):
    """Static polynomial basis phi(u_t) for a classical Hammerstein baseline."""
    u1, u2 = U[..., 0], U[..., 1]
    feats = [u1, u2]
    if degree >= 2:
        feats += [u1 ** 2, u2 ** 2, u1 * u2]
    if degree >= 3:
        feats += [u1 ** 3, u2 ** 3, (u1 ** 2) * u2, u1 * (u2 ** 2)]
    return np.stack(feats, axis=-1)


def fit_poly_hammerstein(U, X, na=2, nb=2, degree=2):
    """Classical polynomial Hammerstein-ARX: x_t = -sum A_i x_{t-i} + sum B_j phi(u_{t-j})."""
    return fit_linear_arx(poly_features(U, degree), X, na, nb)


def simulate_poly_hammerstein(U, A, B, na=2, nb=2, degree=2):
    return simulate_arx(poly_features(U, degree), A, B, na, nb)


def simulate_arx(U, A, B, na=2, nb=2):
    N, T, p = U.shape
    s = A.shape[1]
    X = np.zeros((N, T, s))
    for n in range(N):
        for t in range(T):
            cur = np.zeros(s)
            for i in range(na):
                if t - i - 1 >= 0:
                    cur -= A[i] @ X[n, t - i - 1]
            for j in range(nb):
                if t - j - 1 >= 0:
                    cur += B[j] @ U[n, t - j - 1]
            X[n, t] = cur
    return X


# ----------------------------------------------------------------------
# Latent alignment and aligned B error
# ----------------------------------------------------------------------
def fit_alignment(N_true, N_hat):
    """Find C with n_true ~= C n_hat (least squares over all samples)."""
    Nt = N_true.reshape(-1, N_true.shape[-1])
    Nh = N_hat.reshape(-1, N_hat.shape[-1])
    # rows: n_true_row ~= n_hat_row @ C^T  ->  C^T = lstsq(Nh, Nt)
    CT, *_ = np.linalg.lstsq(Nh, Nt, rcond=None)
    return CT.T  # C


def aligned_B_error(B_hat, B_true, N_true, N_hat):
    C = fit_alignment(N_true, N_hat)
    try:
        Cinv = np.linalg.inv(C)
    except np.linalg.LinAlgError:
        Cinv = np.linalg.pinv(C)
    B_aligned = np.stack([B_hat[j] @ Cinv for j in range(B_hat.shape[0])], 0)
    num = np.linalg.norm(B_aligned - B_true)
    den = np.linalg.norm(B_true) + 1e-12
    raw = np.linalg.norm(B_hat - B_true) / den
    return {"rel_error_B_aligned": float(num / den), "rel_error_B_raw": float(raw),
            "C": C, "B_aligned": B_aligned}


# ----------------------------------------------------------------------
# Companion eigenvalues and impulse response
# ----------------------------------------------------------------------
def companion(A_list):
    na = len(A_list); s = A_list[0].shape[0]
    top = np.concatenate([-A_list[i] for i in range(na)], axis=1)
    if na == 1:
        return top
    eye = np.eye(s * (na - 1)); zeros = np.zeros((s * (na - 1), s))
    return np.concatenate([top, np.concatenate([eye, zeros], axis=1)], axis=0)


def eigenvalue_report(A_true_list, A_hat_list):
    et = np.sort(np.abs(np.linalg.eigvals(companion(A_true_list))))[::-1]
    eh = np.sort(np.abs(np.linalg.eigvals(companion(A_hat_list))))[::-1]
    return {"true_abs_eig": et.tolist(), "est_abs_eig": eh.tolist(),
            "true_spectral_radius": float(et[0]), "est_spectral_radius": float(eh[0]),
            "eig_abs_rel_error": float(np.linalg.norm(eh - et) / (np.linalg.norm(et) + 1e-12))}


def markov_parameters(A_list, B_list, K=20):
    """Impulse response (Markov parameters) of the latent->output linear block.

    Latent impulse n_tau = e_c at tau = 0 (zero elsewhere); the recursion
    x_t = -sum_i A_i x_{t-i} + sum_j B_j n_{t-j-1} is rolled forward K steps.
    """
    na, nb = len(A_list), len(B_list)
    s = A_list[0].shape[0]; m = B_list[0].shape[1]
    G = np.zeros((K, s, m))
    for c in range(m):
        x = []
        for t in range(K):
            cur = np.zeros(s)
            for i in range(na):
                if t - i - 1 >= 0:
                    cur -= A_list[i] @ x[t - i - 1]
            for j in range(nb):
                if t - j - 1 == 0:          # latent impulse active at index 0
                    cur += B_list[j][:, c]
            x.append(cur)
            G[t, :, c] = cur
    return G


def impulse_response_error(A_true, B_true, A_hat, B_hat_aligned, K=20):
    Gt = markov_parameters(A_true, B_true, K)
    Gh = markov_parameters(A_hat, [B_hat_aligned[j] for j in range(len(B_hat_aligned))], K)
    return float(np.linalg.norm(Gh - Gt) / (np.linalg.norm(Gt) + 1e-12))
