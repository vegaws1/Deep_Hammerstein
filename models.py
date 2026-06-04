"""Model components for the revised deep Hammerstein framework.

Includes the two nonlinear variants (static / recurrent), the
stability-constrained recursive MIMO linear block, the multivariate
Student-t colored-noise head, the composed :class:`HammersteinModel`, and the
deep / linear baselines used for comparison.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DTYPE = torch.float32


# ======================================================================
# Nonlinear blocks
# ======================================================================
class StaticNonlinearBlock(nn.Module):
    """Memoryless map g_theta(u_t): used for the identification variant."""

    def __init__(self, in_dim, latent_dim, hidden_size=64, depth=2, dropout=0.0):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden_size), nn.GELU()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            d = hidden_size
        layers += [nn.Linear(d, latent_dim)]
        self.mlp = nn.Sequential(*layers)
        self.skip = nn.Linear(in_dim, latent_dim)

    def forward(self, u):  # u: (B, T, p) -> (B, T, m), applied per time step
        return self.mlp(u) + self.skip(u)


class GRUNonlinearBlock(nn.Module):
    """Recurrent map h_theta(u_{1:t}): used for the prediction variant."""

    def __init__(self, in_dim, latent_dim, hidden_size=64, num_layers=2, dropout=0.0):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden_size, num_layers=num_layers,
                          batch_first=True, dropout=(dropout if num_layers > 1 else 0.0))
        self.fc = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.GELU(),
                                nn.Linear(hidden_size, latent_dim))
        self.skip = nn.Linear(in_dim, latent_dim)

    def forward(self, u):
        h, _ = self.gru(u)
        return self.fc(h) + self.skip(u)


class TCNNonlinearBlock(nn.Module):
    """Causal temporal-convolutional latent block h_theta(u_{1:t}).

    A memoryful front-end (like the GRU): used to show the framework is
    front-end agnostic.
    """

    def __init__(self, in_dim, latent_dim, hidden_size=48, levels=3, kernel_size=3, dropout=0.0):
        super().__init__()
        chans = [in_dim] + [hidden_size] * levels
        blocks = []
        for i in range(levels):
            dilation = 2 ** i
            pad = (kernel_size - 1) * dilation
            blocks += [nn.Conv1d(chans[i], chans[i + 1], kernel_size, padding=pad, dilation=dilation),
                       _Chomp(pad), nn.GELU()]
            if dropout > 0:
                blocks += [nn.Dropout(dropout)]
        self.net = nn.Sequential(*blocks)
        self.head = nn.Linear(hidden_size, latent_dim)
        self.skip = nn.Linear(in_dim, latent_dim)

    def forward(self, u):
        h = self.net(u.transpose(1, 2)).transpose(1, 2)
        return self.head(h) + self.skip(u)


# ======================================================================
# Stability-constrained recursive MIMO linear block
# ======================================================================
class StableLinearDynamicBlock(nn.Module):
    def __init__(self, out_dim, latent_dim, na=2, nb=2, A_init=None, B_init=None,
                 stability_margin=0.05):
        super().__init__()
        self.out_dim = out_dim
        self.latent_dim = latent_dim
        self.na, self.nb = na, nb
        self.margin = stability_margin
        if A_init is None:
            A_init = np.zeros((na, out_dim, out_dim), dtype=np.float32)
        if B_init is None:
            B_init = np.zeros((nb, out_dim, latent_dim), dtype=np.float32)
        self.A = nn.Parameter(torch.tensor(np.asarray(A_init), dtype=DTYPE))
        self.B = nn.Parameter(torch.tensor(np.asarray(B_init), dtype=DTYPE))
        self.project_stable()

    # -- recursion --
    def forward(self, n):
        B, T, _ = n.shape
        s = self.out_dim
        hist = []
        for t in range(T):
            cur = torch.zeros(B, s, device=n.device, dtype=n.dtype)
            for i in range(self.na):
                if t - i - 1 >= 0:
                    cur = cur - hist[t - i - 1] @ self.A[i].T
            for j in range(self.nb):
                if t - j - 1 >= 0:
                    cur = cur + n[:, t - j - 1, :] @ self.B[j].T
            hist.append(cur)
        return torch.stack(hist, dim=1)

    def one_step_teacher_forced(self, n, x_teacher):
        B, T, _ = n.shape
        s = self.out_dim
        out = []
        for t in range(T):
            cur = torch.zeros(B, s, device=n.device, dtype=n.dtype)
            for i in range(self.na):
                if t - i - 1 >= 0:
                    cur = cur - x_teacher[:, t - i - 1, :] @ self.A[i].T
            for j in range(self.nb):
                if t - j - 1 >= 0:
                    cur = cur + n[:, t - j - 1, :] @ self.B[j].T
            out.append(cur)
        return torch.stack(out, dim=1)

    # -- stability --
    def companion_numpy(self):
        A = [self.A[i].detach().cpu().numpy() for i in range(self.na)]
        s = self.out_dim
        top = np.concatenate([-A[i] for i in range(self.na)], axis=1)
        if self.na == 1:
            return top
        eye = np.eye(s * (self.na - 1))
        zeros = np.zeros((s * (self.na - 1), s))
        return np.concatenate([top, np.concatenate([eye, zeros], axis=1)], axis=0)

    def spectral_radius(self):
        return float(np.max(np.abs(np.linalg.eigvals(self.companion_numpy()))))

    @torch.no_grad()
    def project_stable(self):
        """Exact projection: A_i <- kappa^i A_i to force rho(A_c) <= 1-margin."""
        rho = self.spectral_radius()
        target = 1.0 - self.margin
        if rho > target and rho > 1e-8:
            kappa = target / rho
            for i in range(self.na):
                self.A[i].mul_(kappa ** (i + 1))

    def norm_penalty(self):
        """Differentiable sufficient condition: sum_i ||A_i||_2 <= 1-margin."""
        spec = sum(torch.linalg.matrix_norm(self.A[i], ord=2) for i in range(self.na))
        return F.relu(spec - (1.0 - self.margin)) ** 2


# ======================================================================
# Multivariate Student-t colored-noise head
# ======================================================================
class MultivariateStudentTNoiseHead(nn.Module):
    """Residual AR(n_eta) + multivariate Student-t innovations.

    eps_t = r_t - sum_l Phi_l r_{t-l},   eps_t ~ t_nu(0, Sigma).
    To prevent likelihood degeneracy (Sigma -> 0 while residuals are driven to
    zero), the scatter carries an explicit positive floor:
        Sigma = L L^T + sigma_min^2 I  >=  sigma_min^2 I  > 0,
    so log|Sigma| is bounded below and the likelihood is well posed.
    nu = 2 + softplus(alpha) is learnable; nu > 2 guarantees a finite covariance.
    """

    def __init__(self, out_dim, ar_order=1, Phi_init=None, mode="mvt",
                 learn_df=True, init_df=5.0, stability_margin=0.05, sigma_floor=0.02):
        super().__init__()
        self.s = out_dim
        self.ar_order = ar_order
        self.mode = mode               # "mvt" | "diag_t" | "gaussian" | "none"
        self.margin = stability_margin
        if Phi_init is None:
            Phi_init = np.zeros((ar_order, out_dim, out_dim), dtype=np.float32)
        self.Phi = nn.Parameter(torch.tensor(np.asarray(Phi_init), dtype=DTYPE))
        # Cholesky factor (lower-tri); diagonal via softplus for positivity.
        L0 = np.tril(0.1 * np.eye(out_dim)).astype(np.float32)
        self.L_raw = nn.Parameter(torch.tensor(L0, dtype=DTYPE))
        self.register_buffer("tril_mask", torch.tril(torch.ones(out_dim, out_dim)))
        self.register_buffer("eye", torch.eye(out_dim))
        self.sigma_min = float(sigma_floor)
        self.diag_idx = torch.arange(out_dim)
        self.learn_df = learn_df
        inv = math.log(math.expm1(max(init_df - 2.0, 0.1)))
        self.alpha = nn.Parameter(torch.tensor(float(inv)), requires_grad=learn_df)
        self.project_stable()

    @property
    def nu(self):
        return 2.0 + F.softplus(self.alpha)

    def factor(self):
        """Lower-triangular L with positive diagonal (diagonal-only if diag_t)."""
        L = self.L_raw * self.tril_mask
        diag = F.softplus(torch.diagonal(L)) + 1e-4
        L = L - torch.diag(torch.diagonal(L)) + torch.diag(diag)
        if self.mode == "diag_t":
            L = torch.diag(diag)
        return L

    def scatter(self):
        """Floored scatter Sigma = L L^T + sigma_min^2 I  (>= sigma_min^2 I)."""
        L = self.factor()
        return L @ L.T + (self.sigma_min ** 2) * self.eye

    def chol_sigma(self):
        return torch.linalg.cholesky(self.scatter())

    def innovations(self, residual):
        B, T, s = residual.shape
        out = []
        for t in range(T):
            cur = residual[:, t, :]
            if self.mode != "none":
                for l in range(self.ar_order):
                    if t - l - 1 >= 0:
                        cur = cur - residual[:, t - l - 1, :] @ self.Phi[l].T
            out.append(cur)
        return torch.stack(out, dim=1)

    def nll(self, innovation):
        if self.mode == "none":
            return torch.tensor(0.0, device=innovation.device, dtype=innovation.dtype)
        B, T, s = innovation.shape
        eps = innovation.reshape(-1, s)                       # (N, s)
        Lc = self.chol_sigma()                               # chol of floored Sigma
        logdet = 2.0 * torch.log(torch.diagonal(Lc)).sum()
        sol = torch.linalg.solve_triangular(Lc, eps.T, upper=False)   # (s, N)
        d = (sol ** 2).sum(dim=0)                             # Mahalanobis distances
        if self.mode == "gaussian":
            ll = -0.5 * logdet - 0.5 * d - 0.5 * s * math.log(2 * math.pi)
        else:
            nu = self.nu
            ll = (torch.lgamma((nu + s) / 2.0) - torch.lgamma(nu / 2.0)
                  - 0.5 * s * torch.log(nu * math.pi) - 0.5 * logdet
                  - ((nu + s) / 2.0) * torch.log1p(d / nu))
        return -ll.mean()

    def standardized_innovations(self, innovation):
        """z_t = chol(Sigma)^{-1} eps_t (whitened); for diagnostics."""
        s = self.s
        eps = innovation.reshape(-1, s)
        Lc = self.chol_sigma()
        z = torch.linalg.solve_triangular(Lc, eps.T, upper=False).T
        return z.reshape(innovation.shape)

    # residual-AR stationarity
    def companion_numpy(self):
        Phi = [self.Phi[i].detach().cpu().numpy() for i in range(self.ar_order)]
        s = self.s
        top = np.concatenate([Phi[i] for i in range(self.ar_order)], axis=1)
        if self.ar_order == 1:
            return top
        eye = np.eye(s * (self.ar_order - 1))
        zeros = np.zeros((s * (self.ar_order - 1), s))
        return np.concatenate([top, np.concatenate([eye, zeros], axis=1)], axis=0)

    def spectral_radius(self):
        return float(np.max(np.abs(np.linalg.eigvals(self.companion_numpy()))))

    @torch.no_grad()
    def project_stable(self):
        rho = self.spectral_radius()
        target = 1.0 - self.margin
        if rho > target and rho > 1e-8:
            kappa = target / rho
            for i in range(self.ar_order):
                self.Phi[i].mul_(kappa ** (i + 1))


# ======================================================================
# Composed Hammerstein model
# ======================================================================
class HammersteinModel(nn.Module):
    def __init__(self, in_dim, latent_dim, out_dim,
                 nonlinear="recurrent", hidden_size=64, num_layers=2, dropout=0.0,
                 na=2, nb=2, noise_ar_order=1, noise_mode="mvt",
                 A_init=None, B_init=None, Phi_init=None,
                 channel_weights=None, stability_margin=0.05,
                 x_scale=None, y_scale=None, poly_degree=0, use_residual_head=False):
        super().__init__()
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.na, self.nb = na, nb
        self.nonlinear_kind = nonlinear
        self.poly_degree = poly_degree
        self.use_residual_head = use_residual_head
        if nonlinear == "static":
            self.nonlinear = StaticNonlinearBlock(in_dim, latent_dim, hidden_size,
                                                  depth=max(num_layers, 2), dropout=dropout)
        elif nonlinear == "tcn":
            self.nonlinear = TCNNonlinearBlock(in_dim, latent_dim, hidden_size, dropout=dropout)
        else:
            self.nonlinear = GRUNonlinearBlock(in_dim, latent_dim, hidden_size, num_layers, dropout)
        # optional explicit polynomial input path  phi(u_t) -> latent
        if poly_degree > 0:
            self.poly = nn.Linear(self._poly_dim(in_dim, poly_degree), latent_dim)
        # optional residual-correction head (prediction only; not used for recovery)
        if use_residual_head:
            d_z = in_dim + latent_dim + out_dim
            self.res_head = nn.Sequential(nn.Linear(d_z, hidden_size), nn.GELU(),
                                          nn.Linear(hidden_size, out_dim))
        self.linear = StableLinearDynamicBlock(out_dim, latent_dim, na, nb, A_init, B_init,
                                               stability_margin)
        self.noise = MultivariateStudentTNoiseHead(out_dim, noise_ar_order, Phi_init,
                                                    mode=noise_mode, stability_margin=stability_margin)

        def _buf(x, d):
            return torch.tensor(np.asarray(x if x is not None else d), dtype=DTYPE)

        self.register_buffer("A_anchor", _buf(A_init, np.zeros((na, out_dim, out_dim))))
        self.register_buffer("B_anchor", _buf(B_init, np.zeros((nb, out_dim, latent_dim))))
        self.register_buffer("Phi_anchor", _buf(Phi_init, np.zeros((noise_ar_order, out_dim, out_dim))))
        self.register_buffer("channel_weights",
                             _buf(channel_weights, np.ones((out_dim,), dtype=np.float32)))
        self.register_buffer("x_scale", _buf(x_scale, np.ones((out_dim,), dtype=np.float32)))
        self.register_buffer("y_scale", _buf(y_scale, np.ones((out_dim,), dtype=np.float32)))

    @staticmethod
    def _poly_dim(p, degree):
        d = p
        if degree >= 2:
            d += p + p * (p - 1) // 2     # squares + pairwise products
        return d

    @staticmethod
    def _poly_feats(u):
        # degree-2 features for p=2: [u1, u2, u1^2, u2^2, u1 u2]
        u1, u2 = u[..., 0], u[..., 1]
        return torch.stack([u1, u2, u1 ** 2, u2 ** 2, u1 * u2], dim=-1)

    def forward(self, u):
        n = self.nonlinear(u)
        if self.poly_degree > 0:
            n = n + self.poly(self._poly_feats(u))
        x = self.linear(n)
        return x, n

    def correct(self, u, x_struct, n):
        """Optional residual-correction head: x_pred = x_struct + Delta(u, n, x_struct).

        Used for prediction only; the structural estimate x_struct (and the
        linear-block parameters) are unaffected.
        """
        if not self.use_residual_head:
            return x_struct
        z = torch.cat([u, n, x_struct], dim=-1)
        return x_struct + self.res_head(z)

    def predict_output(self, u):
        x, n = self.forward(u)
        return self.correct(u, x, n)

    # ---- dimensionless weighted losses ----
    def _w(self):
        return self.channel_weights / (self.channel_weights.sum() + 1e-8)

    def _huber(self, pred, true, scale):
        e = (pred - true) / scale.view(1, 1, -1)
        per_ch = F.smooth_l1_loss(e, torch.zeros_like(e), reduction="none").mean(dim=(0, 1))
        return torch.sum(self._w() * per_ch)

    def _mse(self, pred, true, scale):
        e = ((pred - true) / scale.view(1, 1, -1)) ** 2
        return torch.sum(self._w() * e.mean(dim=(0, 1)))

    @staticmethod
    def _peak_mask(dtrue, pct=90):
        mag = torch.abs(dtrue)
        thr = torch.quantile(mag.reshape(-1), pct / 100.0)
        return (mag > thr).float()

    def _canon_penalty(self, n):
        nf = n.reshape(-1, n.shape[-1])
        mu = nf.mean(dim=0)
        nc = nf - mu
        cov = (nc.T @ nc) / (nf.shape[0] - 1)
        m = n.shape[-1]
        loss_mu = (mu ** 2).sum()
        loss_cov = ((cov - torch.eye(m, device=n.device)) ** 2).sum()
        B1 = self.linear.B[0]
        loss_tri = (torch.triu(B1, diagonal=1) ** 2).sum()
        loss_sign = F.relu(-torch.diagonal(B1)).sum()
        return loss_mu + loss_cov + 0.5 * loss_tri + 0.5 * loss_sign

    def _rollout_loss(self, x_pred, x_true, horizons=(3, 5, 10)):
        """Multi-horizon error of the free-run prediction (prediction variant)."""
        T = x_pred.size(1)
        terms = []
        for h in horizons:
            if T > h:
                e = (x_pred[:, h:, :] - x_true[:, h:, :]) / self.x_scale.view(1, 1, -1)
                terms.append(F.smooth_l1_loss(e, torch.zeros_like(e)))
        if not terms:
            return torch.tensor(0.0, device=x_pred.device)
        return torch.stack(terms).mean()

    def losses(self, u, x_true, y_meas, weights, opts):
        """weights: dict of lambda_*; opts: dict of toggles & stage target."""
        x_hat, n = self.forward(u)
        x_pred = self.correct(u, x_hat, n)     # x_pred == x_hat unless a residual head is used
        target = x_true if opts.get("stage_c_target", "x") == "x" else None

        logs = {}
        total = torch.tensor(0.0, device=u.device)

        if target is not None:
            pred = self._huber(x_pred, x_true, self.x_scale)
            total = total + pred
            logs["pred"] = float(pred.detach())

            if opts.get("use_diff_peak", True):
                dxt = x_true[:, 1:] - x_true[:, :-1]
                dxh = x_pred[:, 1:] - x_pred[:, :-1]
                diff = self._mse(dxh, dxt, self.x_scale)
                mask = self._peak_mask(dxt, 90)
                w = self._w().view(1, 1, -1)
                wm = mask * w
                peak = torch.sum(wm * torch.abs((dxh - dxt) / self.x_scale.view(1, 1, -1))) / (wm.sum() + 1e-6)
                total = total + weights["diff"] * diff + weights["peak"] * peak
                logs["diff"], logs["peak"] = float(diff.detach()), float(peak.detach())

            if opts.get("use_one_step", True):
                x1 = self.linear.one_step_teacher_forced(n, x_true.detach())
                ml = max(self.na, self.nb)
                if x_true.size(1) > ml:
                    one = self._huber(x1[:, ml:], x_true[:, ml:], self.x_scale)
                    total = total + weights["one_step"] * one
                    logs["one_step"] = float(one.detach())

            if opts.get("use_rollout", False) and weights.get("rollout", 0.0) > 0:
                roll = self._rollout_loss(x_pred, x_true)
                total = total + weights["rollout"] * roll
                logs["rollout"] = float(roll.detach())

        # noise / observation likelihood (residual to measured output)
        if opts.get("noise_mode", "mvt") != "none" and weights.get("noise", 0.0) > 0:
            innov = self.noise.innovations(y_meas - x_pred)
            nll = self.noise.nll(innov)
            total = total + weights["noise"] * nll
            logs["noise"] = float(nll.detach())

        # y-only stage: deterministic part supervised purely through likelihood
        if target is None:
            innov = self.noise.innovations(y_meas - x_pred)
            nll = self.noise.nll(innov)
            total = total + nll
            logs["pred"] = float(nll.detach())
            if opts.get("use_diff_peak", True):
                dyt = y_meas[:, 1:] - y_meas[:, :-1]
                dyh = x_pred[:, 1:] - x_pred[:, :-1]
                diff = self._mse(dyh, dyt, self.y_scale)
                total = total + weights["diff"] * diff
                logs["diff"] = float(diff.detach())

        # keep the residual correction small so the structured block stays meaningful
        if self.use_residual_head:
            delta = (x_pred - x_hat)
            res_reg = (delta ** 2).mean()
            total = total + weights.get("res", 0.05) * res_reg
            logs["res"] = float(res_reg.detach())

        if opts.get("use_canon", True) and self.nonlinear_kind == "static":
            canon = self._canon_penalty(n)
            total = total + weights.get("canon", 0.0) * canon
            logs["canon"] = float(canon.detach())

        if opts.get("use_stability", True):
            stab = self.linear.norm_penalty()
            total = total + weights.get("stab", 0.0) * stab
            logs["stab"] = float(stab.detach())

        if opts.get("use_anchor", True):
            aA = ((self.linear.A - self.A_anchor) ** 2).mean()
            aB = ((self.linear.B - self.B_anchor) ** 2).mean()
            aP = ((self.noise.Phi - self.Phi_anchor) ** 2).mean()
            total = total + weights["anchor_A"] * aA + weights["anchor_B"] * aB + weights["anchor_Phi"] * aP

        reg = (self.linear.A ** 2).mean() + (self.linear.B ** 2).mean() + (self.noise.Phi ** 2).mean()
        total = total + weights["reg"] * reg
        logs["total"] = float(total.detach())
        return total, logs, x_pred


# ======================================================================
# Baselines
# ======================================================================
class _SeqBaseline(nn.Module):
    """Shared channel-weighted Huber+derivative loss for U->X baselines."""

    def __init__(self, out_dim, channel_weights=None, x_scale=None):
        super().__init__()
        self.out_dim = out_dim
        cw = np.ones((out_dim,), np.float32) if channel_weights is None else np.asarray(channel_weights)
        xs = np.ones((out_dim,), np.float32) if x_scale is None else np.asarray(x_scale)
        self.register_buffer("channel_weights", torch.tensor(cw, dtype=DTYPE))
        self.register_buffer("x_scale", torch.tensor(xs, dtype=DTYPE))

    def _w(self):
        return self.channel_weights / (self.channel_weights.sum() + 1e-8)

    def losses(self, u, x_true, lambda_diff=0.1, lambda_reg=1e-4):
        x_hat = self.forward(u)
        e = (x_hat - x_true) / self.x_scale.view(1, 1, -1)
        pred = torch.sum(self._w() * F.smooth_l1_loss(e, torch.zeros_like(e), reduction="none").mean(dim=(0, 1)))
        dxt = (x_true[:, 1:] - x_true[:, :-1]) / self.x_scale.view(1, 1, -1)
        dxh = (x_hat[:, 1:] - x_hat[:, :-1]) / self.x_scale.view(1, 1, -1)
        diff = torch.sum(self._w() * ((dxh - dxt) ** 2).mean(dim=(0, 1)))
        reg = sum((p ** 2).mean() for p in self.parameters())
        total = pred + lambda_diff * diff + lambda_reg * reg
        return total, {"total": float(total), "pred": float(pred)}, x_hat


class DirectGRU(_SeqBaseline):
    def __init__(self, in_dim, out_dim, hidden_size=64, num_layers=2, dropout=0.0, **kw):
        super().__init__(out_dim, **kw)
        self.gru = nn.GRU(in_dim, hidden_size, num_layers, batch_first=True,
                          dropout=(dropout if num_layers > 1 else 0.0))
        self.fc = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.GELU(),
                                nn.Linear(hidden_size, out_dim))
        self.skip = nn.Linear(in_dim, out_dim)

    def forward(self, u):
        h, _ = self.gru(u)
        return self.fc(h) + self.skip(u)


class DirectLSTM(_SeqBaseline):
    def __init__(self, in_dim, out_dim, hidden_size=64, num_layers=2, dropout=0.0, **kw):
        super().__init__(out_dim, **kw)
        self.lstm = nn.LSTM(in_dim, hidden_size, num_layers, batch_first=True,
                            dropout=(dropout if num_layers > 1 else 0.0))
        self.fc = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.GELU(),
                                nn.Linear(hidden_size, out_dim))
        self.skip = nn.Linear(in_dim, out_dim)

    def forward(self, u):
        h, _ = self.lstm(u)
        return self.fc(h) + self.skip(u)


class TCN(_SeqBaseline):
    """Small causal temporal convolutional network."""

    def __init__(self, in_dim, out_dim, hidden_size=48, levels=3, kernel_size=3, dropout=0.0, **kw):
        super().__init__(out_dim, **kw)
        chans = [in_dim] + [hidden_size] * levels
        blocks = []
        for i in range(levels):
            dilation = 2 ** i
            pad = (kernel_size - 1) * dilation
            blocks.append(nn.Conv1d(chans[i], chans[i + 1], kernel_size,
                                    padding=pad, dilation=dilation))
            blocks.append(_Chomp(pad))
            blocks.append(nn.GELU())
            if dropout > 0:
                blocks.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*blocks)
        self.head = nn.Linear(hidden_size, out_dim)
        self.skip = nn.Linear(in_dim, out_dim)

    def forward(self, u):
        h = self.net(u.transpose(1, 2)).transpose(1, 2)
        return self.head(h) + self.skip(u)


class _Chomp(nn.Module):
    def __init__(self, chomp):
        super().__init__()
        self.chomp = chomp

    def forward(self, x):
        return x[:, :, :-self.chomp] if self.chomp > 0 else x


class NARXMLP(_SeqBaseline):
    """Nonlinear ARX: MLP over a window of past inputs and outputs.

    Trained with vectorized one-step teacher forcing (fast); evaluated by
    free-run rollout (the realistic simulation regime).
    """

    def __init__(self, in_dim, out_dim, na=2, nb=2, hidden_size=64, dropout=0.0, **kw):
        super().__init__(out_dim, **kw)
        self.na, self.nb = na, nb
        self.in_dim = in_dim
        feat = na * out_dim + nb * in_dim
        self.mlp = nn.Sequential(nn.Linear(feat, hidden_size), nn.GELU(),
                                 nn.Linear(hidden_size, hidden_size), nn.GELU(),
                                 nn.Linear(hidden_size, out_dim))

    @staticmethod
    def _shift(seq, lag):
        z = torch.zeros_like(seq)
        if lag < seq.shape[1]:
            z[:, lag:, :] = seq[:, :seq.shape[1] - lag, :]
        return z

    def forward(self, u):  # free-run rollout (inference)
        B, T, p = u.shape
        s = self.out_dim
        x = torch.zeros(B, T, s, device=u.device, dtype=u.dtype)
        for t in range(T):
            feats = []
            for i in range(self.na):
                feats.append(x[:, t - i - 1, :] if t - i - 1 >= 0 else torch.zeros(B, s, device=u.device))
            for j in range(self.nb):
                feats.append(u[:, t - j - 1, :] if t - j - 1 >= 0 else torch.zeros(B, p, device=u.device))
            x = x.clone()
            x[:, t, :] = self.mlp(torch.cat(feats, dim=-1))
        return x

    def losses(self, u, x_true, lambda_diff=0.1, lambda_reg=1e-4):
        feats = ([self._shift(x_true, i + 1) for i in range(self.na)]
                 + [self._shift(u, j + 1) for j in range(self.nb)])
        x_hat = self.mlp(torch.cat(feats, dim=-1))
        e = (x_hat - x_true) / self.x_scale.view(1, 1, -1)
        pred = torch.sum(self._w() * F.smooth_l1_loss(e, torch.zeros_like(e), reduction="none").mean(dim=(0, 1)))
        reg = sum((p ** 2).mean() for p in self.parameters())
        total = pred + lambda_reg * reg
        return total, {"total": float(total), "pred": float(pred)}, x_hat
