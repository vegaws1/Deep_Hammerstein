"""Staged training for the proposed model and training for the baselines."""

import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32


class SeqDataset(Dataset):
    def __init__(self, U, X, Y):
        self.U = torch.tensor(U, dtype=DTYPE)
        self.X = torch.tensor(X, dtype=DTYPE)
        self.Y = torch.tensor(Y, dtype=DTYPE)

    def __len__(self):
        return self.U.shape[0]

    def __getitem__(self, i):
        return self.U[i], self.X[i], self.Y[i]


def make_loaders(U, X, Y, batch_size, shuffle_train=True):
    ds = SeqDataset(U, X, Y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle_train, drop_last=False)


def set_requires_grad(module, flag):
    for p in module.parameters():
        p.requires_grad = flag


def _cosine(opt, T_max, eta_min):
    return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(T_max, 1), eta_min=eta_min)


DEFAULT_WEIGHTS = dict(diff=0.10, peak=0.10, one_step=0.30, noise=0.0,
                       canon=0.02, stab=0.05, anchor_A=0.001, anchor_B=0.001,
                       anchor_Phi=0.001, reg=1e-4, rollout=0.10, res=0.05)

DEFAULT_OPTS = dict(stage_c_target="x", noise_mode="mvt", use_diff_peak=True,
                    use_one_step=True, use_canon=True, use_stability=True, use_anchor=True,
                    use_rollout=False)


# ----------------------------------------------------------------------
# Proposed model
# ----------------------------------------------------------------------
def run_epoch(model, loader, weights, opts, optimizer=None, grad_clip=1.0,
              train_parts=("nonlinear", "linear", "noise")):
    is_train = optimizer is not None
    model.train(is_train)
    for name, mod in [("nonlinear", model.nonlinear), ("linear", model.linear), ("noise", model.noise)]:
        set_requires_grad(mod, is_train and (name in train_parts))

    agg = {}
    nb = 0
    for u, x, y in loader:
        u, x, y = u.to(DEVICE), x.to(DEVICE), y.to(DEVICE)
        if is_train:
            optimizer.zero_grad()
        total, logs, _ = model.losses(u, x, y, weights, opts)
        if is_train:
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            model.linear.project_stable()
            model.noise.project_stable()
        for k, v in logs.items():
            agg[k] = agg.get(k, 0.0) + v
        nb += 1
    return {k: v / max(nb, 1) for k, v in agg.items()}


def train_model(model, train_loader, val_loader, weights, opts, epochs=60, lr=1e-3,
                grad_clip=1.0, patience=15, train_parts=("nonlinear", "linear", "noise"),
                verbose=False):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sch = _cosine(opt, epochs, lr / 100)
    best_state, best_val, best_ep, hist, bad = None, np.inf, 0, [], 0
    for ep in range(1, epochs + 1):
        tr = run_epoch(model, train_loader, weights, opts, opt, grad_clip, train_parts)
        with torch.no_grad():
            va = run_epoch(model, val_loader, weights, opts, None)
        sch.step()
        hist.append((tr, va))
        score = va.get("pred", va["total"])
        if score < best_val:
            best_val, best_state, best_ep, bad = score, copy.deepcopy(model.state_dict()), ep, 0
        else:
            bad += 1
        if verbose and (ep == 1 or ep % 10 == 0 or ep == epochs):
            print(f"    ep {ep:03d} | train {tr['total']:.4f} | val {va['total']:.4f} "
                  f"| pred {va.get('pred', float('nan')):.4f}")
        if bad >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, hist, best_ep, best_val


def predict(model, loader):
    """Returns (X_true, Y_meas, X_pred). X_pred is the corrected output if the
    model has a residual head, else the structural output."""
    model.eval()
    xs, ys, xh = [], [], []
    with torch.no_grad():
        for u, x, y in loader:
            xhat = model.predict_output(u.to(DEVICE))
            xs.append(x.numpy()); ys.append(y.numpy()); xh.append(xhat.cpu().numpy())
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(xh)


def latents(model, loader):
    model.eval()
    out = []
    with torch.no_grad():
        for u, x, y in loader:
            _, n = model(u.to(DEVICE))
            out.append(n.cpu().numpy())
    return np.concatenate(out)


def predict_structural(model, loader):
    """Structural output x_struct = forward(u) (no residual-correction head)."""
    model.eval()
    xs, ys, xh = [], [], []
    with torch.no_grad():
        for u, x, y in loader:
            xhat, _ = model(u.to(DEVICE))
            xs.append(x.numpy()); ys.append(y.numpy()); xh.append(xhat.cpu().numpy())
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(xh)


def test_nll(model, loader):
    """Mean held-out negative log-likelihood of the residual model."""
    model.eval()
    tot, nb = 0.0, 0
    with torch.no_grad():
        for u, x, y in loader:
            xhat, _ = model(u.to(DEVICE))
            innov = model.noise.innovations(y.to(DEVICE) - xhat)
            tot += float(model.noise.nll(innov)); nb += 1
    return tot / max(nb, 1)


def innovations_for_diagnostics(model, loader):
    """Return standardized innovations and raw residuals (measured-output)."""
    model.eval()
    z_all, r_all = [], []
    with torch.no_grad():
        for u, x, y in loader:
            xhat, _ = model(u.to(DEVICE))
            r = y.to(DEVICE) - xhat
            innov = model.noise.innovations(r)
            z = model.noise.standardized_innovations(innov)
            z_all.append(z.cpu().numpy()); r_all.append(r.cpu().numpy())
    return np.concatenate(z_all), np.concatenate(r_all)


# ----------------------------------------------------------------------
# Baselines
# ----------------------------------------------------------------------
def train_baseline(model, train_loader, val_loader, epochs=60, lr=1e-3,
                   lambda_diff=0.1, grad_clip=1.0, patience=12):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sch = _cosine(opt, epochs, lr / 100)
    best_state, best_val, best_ep, bad = None, np.inf, 0, 0
    for ep in range(1, epochs + 1):
        model.train()
        for u, x, y in train_loader:
            opt.zero_grad()
            total, _, _ = model.losses(u.to(DEVICE), x.to(DEVICE), lambda_diff=lambda_diff)
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
        sch.step()
        model.eval()
        with torch.no_grad():
            vtot = 0.0; nb = 0
            for u, x, y in val_loader:
                t, _, _ = model.losses(u.to(DEVICE), x.to(DEVICE), lambda_diff=lambda_diff)
                vtot += float(t); nb += 1
            v = vtot / max(nb, 1)
        if v < best_val:
            best_val, best_state, best_ep, bad = v, copy.deepcopy(model.state_dict()), ep, 0
        else:
            bad += 1
        if bad >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_ep, best_val


def predict_baseline(model, loader):
    model.eval()
    xs, ys, xh = [], [], []
    with torch.no_grad():
        for u, x, y in loader:
            xhat = model(u.to(DEVICE))
            xs.append(x.numpy()); ys.append(y.numpy()); xh.append(xhat.cpu().numpy())
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(xh)
