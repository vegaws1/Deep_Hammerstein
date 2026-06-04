"""Experiment orchestration: datasets, staged training, multi-seed comparison,
ablation matrix, diagnostics and significance tests."""

import os
import copy
from dataclasses import dataclass, field, asdict

import numpy as np
import torch

from . import data as D
from . import models as M
from . import training as T
from . import identify as I
from . import diagnostics as G
from . import stats as S
from . import plotting as P
from .utils import (set_seed, evaluate_metrics_with_channels, per_sequence_rmse,
                    relative_param_error, flatten_metric_block, ensure_dir,
                    save_json, save_csv_dicts)


# ----------------------------------------------------------------------
# Run configuration (small "paper" defaults -- runs on CPU)
# ----------------------------------------------------------------------
@dataclass
class RunConfig:
    plant_kind: str = "static"          # "static" | "recurrent"
    noise_kind: str = "studentt"
    noise_colored: bool = True
    T: int = 60
    n_train: int = 64
    n_val: int = 24
    n_test: int = 48
    n_step: int = 64
    batch_size: int = 16
    hidden_size: int = 48
    num_layers: int = 2
    epochs_A: int = 45
    epochs_B: int = 15
    epochs_C: int = 15
    epochs_base: int = 50
    lr: float = 1e-3
    stability_margin: float = 0.05
    noise_candidates: tuple = (0.005, 0.02)


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
def build_datasets(cfg: RunConfig, seed: int, noise_override=None):
    ncfg = noise_override or D.NoiseConfig(kind=cfg.noise_kind, colored=cfg.noise_colored)
    pcfg = D.PlantConfig(noise=ncfg)
    plant = D.make_plant(cfg.plant_kind, pcfg, cfg.stability_margin)

    U_s, Y_s, _, _ = D.make_dataset(plant, cfg.n_step, cfg.T, "step", seed=seed + 7)
    U_tr, Y_tr, X_tr, N_tr = D.make_dataset(plant, cfg.n_train, cfg.T, "random", seed=seed + 11)
    U_va, Y_va, X_va, N_va = D.make_dataset(plant, cfg.n_val, cfg.T, "random", seed=seed + 23)
    U_te, Y_te, X_te, N_te = D.make_dataset(plant, cfg.n_test, cfg.T, "random", seed=seed + 31)

    us, xs = D.StandardScalerSeq().fit(U_tr), D.StandardScalerSeq().fit(X_tr)
    pack = dict(plant=plant, scaler_x=xs,
                U={"step": us.transform(U_s), "tr": us.transform(U_tr),
                   "va": us.transform(U_va), "te": us.transform(U_te)},
                X={"tr": xs.transform(X_tr), "va": xs.transform(X_va), "te": xs.transform(X_te)},
                Y={"step": xs.transform(Y_s), "tr": xs.transform(Y_tr),
                   "va": xs.transform(Y_va), "te": xs.transform(Y_te)},
                Xraw={"te": X_te, "va": X_va}, Yraw={"te": Y_te, "va": Y_va},
                Ntrue={"te": N_te}, x_std=xs.std.reshape(-1).astype(np.float32))
    return pack


def _loaders(pack, cfg):
    tr = T.make_loaders(pack["U"]["tr"], pack["X"]["tr"], pack["Y"]["tr"], cfg.batch_size, True)
    va = T.make_loaders(pack["U"]["va"], pack["X"]["va"], pack["Y"]["va"], cfg.batch_size, False)
    te = T.make_loaders(pack["U"]["te"], pack["X"]["te"], pack["Y"]["te"], cfg.batch_size, False)
    return tr, va, te


# ----------------------------------------------------------------------
# Stage-1 surrogate initialization
# ----------------------------------------------------------------------
def stage1_init(pack, cfg, shrink=0.5):
    A, B = I.fit_linear_arx(pack["U"]["step"], pack["Y"]["step"], na=2, nb=2)
    # residual AR(1) init
    Xr, Yr = I.build_arx_regressors(pack["U"]["step"], pack["Y"]["step"], 2, 2)
    pred = np.zeros_like(Yr)
    A0 = A.copy(); B0 = B.copy()
    coefs = np.concatenate([np.concatenate([-A0[i] for i in range(2)], 1),
                            np.concatenate([B0[j] for j in range(2)], 1)], 1)
    pred = Xr @ coefs.T
    resid = Yr - pred
    Phi_T, *_ = np.linalg.lstsq(resid[:-1], resid[1:], rcond=None)
    Phi = (Phi_T.T)[None]
    return (A * shrink).astype(np.float32), (B * shrink).astype(np.float32), (Phi * shrink).astype(np.float32)


# ----------------------------------------------------------------------
# Coordinate transforms for parameter recovery
# ----------------------------------------------------------------------
def to_physical(A_hat, B_hat, Phi_hat, x_std):
    Dm = np.diag(x_std); Dinv = np.diag(1.0 / x_std)
    A_phys = np.stack([Dm @ A_hat[i] @ Dinv for i in range(A_hat.shape[0])], 0)
    B_phys = np.stack([Dm @ B_hat[j] for j in range(B_hat.shape[0])], 0)
    Phi_phys = np.stack([Dm @ Phi_hat[l] @ Dinv for l in range(Phi_hat.shape[0])], 0)
    return A_phys, B_phys, Phi_phys


# ----------------------------------------------------------------------
# Build / train proposed model
# ----------------------------------------------------------------------
def build_model(cfg, variant, noise_mode, A0, B0, Phi0, channel_weights,
                poly_degree=0, use_residual_head=False):
    m = M.HammersteinModel(
        in_dim=2, latent_dim=2, out_dim=2, nonlinear=variant,
        hidden_size=cfg.hidden_size, num_layers=cfg.num_layers,
        na=2, nb=2, noise_ar_order=1, noise_mode=noise_mode,
        A_init=A0, B_init=B0, Phi_init=Phi0,
        channel_weights=channel_weights, stability_margin=cfg.stability_margin,
        poly_degree=poly_degree, use_residual_head=use_residual_head,
    ).to(T.DEVICE)
    return m


def train_proposed(pack, cfg, variant="recurrent", noise_mode="mvt", toggles=None,
                   stage_c_target="x", poly_degree=0, use_residual_head=False,
                   use_rollout=False, verbose=False):
    toggles = toggles or {}
    tr, va, te = _loaders(pack, cfg)
    A0, B0, Phi0 = stage1_init(pack, cfg)
    cw = np.array([1.35, 1.0], np.float32)
    model = build_model(cfg, variant, noise_mode, A0, B0, Phi0, cw,
                        poly_degree=poly_degree, use_residual_head=use_residual_head)

    base_opts = dict(T.DEFAULT_OPTS)
    base_opts.update(noise_mode=noise_mode, use_rollout=use_rollout)
    base_opts.update({k: v for k, v in toggles.items() if k.startswith("use_")})

    W = dict(T.DEFAULT_WEIGHTS)
    if not base_opts.get("use_canon", True):
        W["canon"] = 0.0
    if not base_opts.get("use_stability", True):
        W["stab"] = 0.0
    if not base_opts.get("use_anchor", True):
        W["anchor_A"] = W["anchor_B"] = W["anchor_Phi"] = 0.0

    single_stage = toggles.get("single_stage", False)

    # ---- Stage A: deterministic recovery (no noise term) ----
    optsA = dict(base_opts); optsA.update(stage_c_target="x")
    WA = dict(W); WA["noise"] = 0.0
    model, histA, *_ = T.train_model(model, tr, va, WA, optsA, epochs=cfg.epochs_A,
                                     lr=cfg.lr, patience=12, verbose=verbose)
    if single_stage:
        return dict(model=model, hist=histA, noise_weight=0.0, loaders=(tr, va, te))

    # ---- noise-weight pick ----
    best_lam, best_score = cfg.noise_candidates[0], np.inf
    if noise_mode != "none":
        for lam in cfg.noise_candidates:
            trial = copy.deepcopy(model)
            Wt = dict(W); Wt["noise"] = lam
            ot = dict(base_opts); ot.update(stage_c_target="x")
            trial, _, _, sc = T.train_model(trial, tr, va, Wt, ot, epochs=8, lr=cfg.lr * 0.5,
                                            patience=4, train_parts=("linear", "noise"))
            if sc < best_score:
                best_score, best_lam = sc, lam

    # ---- Stage B: linear/noise refinement (freeze nonlinear) ----
    WB = dict(W); WB["noise"] = (best_lam if noise_mode != "none" else 0.0)
    optsB = dict(base_opts); optsB.update(stage_c_target="x")
    model, histB, *_ = T.train_model(model, tr, va, WB, optsB, epochs=cfg.epochs_B,
                                     lr=cfg.lr * 0.5, patience=8, train_parts=("linear", "noise"))

    # ---- Stage C: joint fine-tuning (x- or y-supervised) ----
    WC = dict(WB)
    optsC = dict(base_opts); optsC.update(stage_c_target=stage_c_target)
    model, histC, *_ = T.train_model(model, tr, va, WC, optsC, epochs=cfg.epochs_C,
                                     lr=cfg.lr * 0.25, patience=6, verbose=verbose)
    return dict(model=model, hist=histA + histB + histC, noise_weight=best_lam, loaders=(tr, va, te))


# ----------------------------------------------------------------------
# Evaluate proposed model
# ----------------------------------------------------------------------
def evaluate_proposed(res, pack, cfg, with_diagnostics=True):
    model = res["model"]; tr, va, te = res["loaders"]
    xs = pack["scaler_x"]
    Xt_n, Yt_n, Xh_n = T.predict(model, te)
    Xt = xs.inverse_transform(Xt_n); Yt = xs.inverse_transform(Yt_n); Xh = xs.inverse_transform(Xh_n)

    out = {"metrics": {
        "test_on_X": evaluate_metrics_with_channels(Xt, Xh),
        "test_vs_Y": evaluate_metrics_with_channels(Yt, Xh)},
        "rmse_seq_X": per_sequence_rmse(Xt, Xh).tolist(),
        "rmse_seq_Y": per_sequence_rmse(Yt, Xh).tolist()}

    # parameter recovery
    A_hat = model.linear.A.detach().cpu().numpy()
    B_hat = model.linear.B.detach().cpu().numpy()
    Phi_hat = model.noise.Phi.detach().cpu().numpy()
    A_phys, B_phys, Phi_phys = to_physical(A_hat, B_hat, Phi_hat, pack["x_std"])
    plant = pack["plant"]
    A_true = np.stack(plant.A, 0); B_true = np.stack(plant.B, 0); Phi_true = plant.noise.Phi[None]

    N_hat = T.latents(model, te)
    align = I.aligned_B_error(B_phys, B_true, pack["Ntrue"]["te"], N_hat)
    eig = I.eigenvalue_report([A_true[i] for i in range(2)], [A_phys[i] for i in range(2)])
    imp = I.impulse_response_error([A_true[i] for i in range(2)], [B_true[j] for j in range(2)],
                                   [A_phys[i] for i in range(2)],
                                   [align["B_aligned"][j] for j in range(2)])
    out["params"] = {
        "rel_error_A": relative_param_error(A_true, A_phys),
        "rel_error_B_aligned": align["rel_error_B_aligned"],
        "rel_error_B_raw": align["rel_error_B_raw"],
        "rel_error_Phi": relative_param_error(Phi_true, Phi_phys),
        "impulse_response_error": imp,
        **eig,
    }
    out["stability"] = {"A_spectral_radius": model.linear.spectral_radius(),
                        "Phi_spectral_radius": model.noise.spectral_radius(),
                        "nu": float(model.noise.nu.detach().cpu()),
                        "violation": int(model.linear.spectral_radius() >= 1.0)}

    if with_diagnostics and model.noise.mode != "none":
        z, r = T.innovations_for_diagnostics(model, te)
        acf, band = G.autocorrelation(z, 15)
        ref_df = 1e6 if model.noise.mode == "gaussian" else float(model.noise.nu.detach().cpu())
        out["diagnostics"] = {
            "whiteness_score": G.whiteness_score(z, 10),
            "ljung_box": G.ljung_box(z, 10),
            "cross_correlation": G.cross_correlation(z).tolist(),
            **G.pit_coverage(z, df=ref_df),
        }
        out["_diag_arrays"] = {"z": z, "acf": acf, "band": band}
    out["_arrays"] = {"Xt": Xt, "Yt": Yt, "Xh": Xh, "A_phys": A_phys, "A_true": A_true}
    return out


# ----------------------------------------------------------------------
# Baselines
# ----------------------------------------------------------------------
def train_eval_baseline(name, pack, cfg):
    tr, va, te = _loaders(pack, cfg)
    cw = np.array([1.35, 1.0], np.float32)
    xs = pack["scaler_x"]
    if name == "linear_arx":
        A, B = I.fit_linear_arx(pack["U"]["tr"], pack["X"]["tr"], 2, 2)
        Xh_n = I.simulate_arx(pack["U"]["te"], A, B, 2, 2)
        Xt = xs.inverse_transform(pack["X"]["te"]); Yt = pack["Yraw"]["te"]
        Xh = xs.inverse_transform(Xh_n)
    elif name == "poly_hammerstein":
        A, B = I.fit_poly_hammerstein(pack["U"]["tr"], pack["X"]["tr"], 2, 2, degree=3)
        Xh_n = I.simulate_poly_hammerstein(pack["U"]["te"], A, B, 2, 2, degree=3)
        Xt = xs.inverse_transform(pack["X"]["te"]); Yt = pack["Yraw"]["te"]
        Xh = xs.inverse_transform(Xh_n)
    else:
        ctor = {"direct_gru": M.DirectGRU, "lstm": M.DirectLSTM, "tcn": M.TCN, "narx_mlp": M.NARXMLP}[name]
        kw = dict(in_dim=2, out_dim=2, channel_weights=cw)
        if name in ("direct_gru", "lstm"):
            kw.update(hidden_size=cfg.hidden_size, num_layers=cfg.num_layers)
        elif name == "tcn":
            kw.update(hidden_size=cfg.hidden_size)
        else:
            kw.update(hidden_size=cfg.hidden_size, na=2, nb=2)
        model = ctor(**kw).to(T.DEVICE)
        model, _, _ = T.train_baseline(model, tr, va, epochs=cfg.epochs_base, lr=cfg.lr)
        Xt_n, Yt_n, Xh_n = T.predict_baseline(model, te)
        Xt = xs.inverse_transform(Xt_n); Yt = xs.inverse_transform(Yt_n); Xh = xs.inverse_transform(Xh_n)
    return {"test_on_X": evaluate_metrics_with_channels(Xt, Xh),
            "test_vs_Y": evaluate_metrics_with_channels(Yt, Xh),
            "rmse_seq_X": per_sequence_rmse(Xt, Xh).tolist(),
            "rmse_seq_Y": per_sequence_rmse(Yt, Xh).tolist()}


# ----------------------------------------------------------------------
# High-level drivers
# ----------------------------------------------------------------------
PROPOSED_SPECS = {
    "proposed_recurrent": dict(variant="recurrent", noise_mode="mvt", stage_c_target="x"),
    "proposed_static": dict(variant="static", noise_mode="mvt", stage_c_target="x"),
    "proposed_tcn": dict(variant="tcn", noise_mode="mvt", stage_c_target="x"),
    "proposed_static_yonly": dict(variant="static", noise_mode="mvt", stage_c_target="y"),
}
BASELINES = ["direct_gru", "lstm", "tcn", "narx_mlp", "linear_arx", "poly_hammerstein"]
HEADLINE = ["RMSE", "R2", "FIT", "wMAPE"]


def _agg(list_of_metric_blocks, split):
    out = {}
    for met in HEADLINE:
        vals = [b[split]["overall"][met] for b in list_of_metric_blocks]
        out[f"{split}_{met}_mean"] = float(np.mean(vals))
        out[f"{split}_{met}_std"] = float(np.std(vals))
    return out


def run_main(cfg: RunConfig, seeds, out_dir, verbose=True):
    ensure_dir(out_dir)
    fig_dir = ensure_dir(os.path.join(out_dir, "figures"))
    per_model_metrics = {k: [] for k in list(PROPOSED_SPECS) + BASELINES}
    per_model_seqX = {k: [] for k in per_model_metrics}
    per_model_seqY = {k: [] for k in per_model_metrics}
    proposed_params = {k: [] for k in PROPOSED_SPECS}
    diag_accum = []
    rep = {}

    for si, seed in enumerate(seeds):
        if verbose:
            print(f"\n===== SEED {seed} ({si + 1}/{len(seeds)}) =====")
        set_seed(seed)
        pack = build_datasets(cfg, seed)

        for key, spec in PROPOSED_SPECS.items():
            if verbose:
                print(f"  [train] {key}")
            res = train_proposed(pack, cfg, **spec)
            ev = evaluate_proposed(res, pack, cfg,
                                   with_diagnostics=(key == "proposed_static"))
            per_model_metrics[key].append(ev["metrics"])
            per_model_seqX[key].append(ev["rmse_seq_X"])
            per_model_seqY[key].append(ev["rmse_seq_Y"])
            proposed_params[key].append(ev["params"] | ev["stability"])
            if key == "proposed_static":
                diag_accum.append(ev.get("diagnostics", {}))
                rep = {"hist": res["hist"], "arrays": ev["_arrays"],
                       "diag": ev.get("_diag_arrays", {})}

        for key in BASELINES:
            if verbose:
                print(f"  [train] {key}")
            ev = train_eval_baseline(key, pack, cfg)
            per_model_metrics[key].append(ev)
            per_model_seqX[key].append(ev["rmse_seq_X"])
            per_model_seqY[key].append(ev["rmse_seq_Y"])

    # ---- aggregate metrics ----
    rows = []
    for key in list(PROPOSED_SPECS) + BASELINES:
        row = {"model": key}
        row.update(_agg(per_model_metrics[key], "test_on_X"))
        row.update(_agg(per_model_metrics[key], "test_vs_Y"))
        rows.append(row)
    save_csv_dicts(rows, os.path.join(out_dir, "main_aggregate.csv"))

    # ---- parameter recovery ----
    prow = []
    for key in PROPOSED_SPECS:
        agg = {"model": key}
        for field_ in ["rel_error_A", "rel_error_B_aligned", "rel_error_B_raw",
                       "rel_error_Phi", "impulse_response_error", "est_spectral_radius",
                       "A_spectral_radius", "nu", "violation"]:
            vals = [p[field_] for p in proposed_params[key] if field_ in p]
            if vals:
                agg[f"{field_}_mean"] = float(np.mean(vals))
                agg[f"{field_}_std"] = float(np.std(vals))
        prow.append(agg)
    save_csv_dicts(prow, os.path.join(out_dir, "parameter_recovery.csv"))

    # ---- save raw per-seed per-sequence RMSE (reproducibility) ----
    save_json({k: [list(map(float, a)) for a in per_model_seqX[k]] for k in per_model_seqX},
              os.path.join(out_dir, "per_seed_rmse_X.json"))

    # ---- significance vs Direct GRU: pooled (exploratory) AND seed-level (primary) ----
    stat_rows, hier_rows = [], []
    base_X = np.concatenate([np.asarray(a) for a in per_model_seqX["direct_gru"]])
    base_Y = np.concatenate([np.asarray(a) for a in per_model_seqY["direct_gru"]])
    for key in list(PROPOSED_SPECS) + ["lstm", "tcn"]:
        mX = np.concatenate([np.asarray(a) for a in per_model_seqX[key]])
        mY = np.concatenate([np.asarray(a) for a in per_model_seqY[key]])
        rX = S.paired_comparison(base_X, mX)
        rY = S.paired_comparison(base_Y, mY)
        stat_rows.append({"model_vs_direct_gru": key,
                          "X_mean_delta_rmse": rX["mean_delta_rmse"], "X_wilcoxon_p": rX["wilcoxon_p"],
                          "X_ci95_low": rX["bootstrap_ci95"][0], "X_ci95_high": rX["bootstrap_ci95"][1],
                          "X_cohen_dz": rX["cohen_dz"], "X_significant": rX["significant_05"],
                          "Y_mean_delta_rmse": rY["mean_delta_rmse"], "Y_wilcoxon_p": rY["wilcoxon_p"],
                          "Y_significant": rY["significant_05"]})
        hX = S.hierarchical_comparison(per_model_seqX["direct_gru"], per_model_seqX[key])
        hier_rows.append({"model_vs_direct_gru": key,
                          "seed_mean_delta_rmse_X": hX["seed_mean_delta_rmse"],
                          "seed_ci95_low": hX["seed_ci95_t"][0], "seed_ci95_high": hX["seed_ci95_t"][1],
                          "paired_t_p": hX["paired_t_p"],
                          "seeds_favoring_model": f"{hX['seeds_favoring_model']}/{hX['n_seeds']}",
                          "hier_boot_low": hX["hier_bootstrap_ci95"][0],
                          "hier_boot_high": hX["hier_bootstrap_ci95"][1],
                          "significant_seedlevel": hX["significant_seedlevel"]})
    save_csv_dicts(stat_rows, os.path.join(out_dir, "significance_vs_direct_gru.csv"))
    save_csv_dicts(hier_rows, os.path.join(out_dir, "significance_hierarchical.csv"))

    # ---- diagnostics aggregate ----
    diag_summary = {}
    if diag_accum and diag_accum[0]:
        for kk in ["whiteness_score", "coverage_50", "coverage_80", "coverage_95", "pit_ks"]:
            vals = [d[kk] for d in diag_accum if kk in d]
            if vals:
                diag_summary[kk] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    save_json(diag_summary, os.path.join(out_dir, "diagnostics_summary.json"))

    # ---- figures from representative seed (proposed_static) ----
    if rep:
        P.plot_training_history(rep["hist"], os.path.join(fig_dir, "training_history.png"))
        P.plot_prediction(rep["arrays"]["Xt"], rep["arrays"]["Xh"],
                          os.path.join(fig_dir, "deterministic_X_channel2.png"),
                          channel=1, title="Deterministic recovery (channel 2)")
        P.plot_prediction(rep["arrays"]["Yt"], rep["arrays"]["Xh"],
                          os.path.join(fig_dir, "measured_Y_channel2.png"),
                          channel=1, title="Measured-output tracking (channel 2)")
        if rep["diag"]:
            P.plot_acf(rep["diag"]["acf"], rep["diag"]["band"],
                       os.path.join(fig_dir, "innovation_acf.png"))
            z = rep["diag"]["z"]
            P.plot_cross_corr(G.cross_correlation(z), os.path.join(fig_dir, "cross_correlation.png"))
            P.plot_qq(G.qq_data(z), os.path.join(fig_dir, "qq_innovations.png"))
        A_true = rep["arrays"]["A_true"]; A_phys = rep["arrays"]["A_phys"]
        true_eig = np.linalg.eigvals(I.companion([A_true[i] for i in range(2)]))
        est_eig = np.linalg.eigvals(I.companion([A_phys[i] for i in range(2)]))
        P.plot_eigenvalues(true_eig, est_eig, os.path.join(fig_dir, "eigenvalues.png"))

    summary = {"config": asdict(cfg), "seeds": list(seeds),
               "aggregate": rows, "parameter_recovery": prow,
               "significance": stat_rows, "significance_hierarchical": hier_rows,
               "diagnostics": diag_summary}
    save_json(summary, os.path.join(out_dir, "summary.json"))
    if verbose:
        print(f"\n[main] results written to {out_dir}")
    return summary


def run_ablations(cfg: RunConfig, seeds, out_dir, verbose=True):
    ensure_dir(out_dir)
    ABL = {
        "full": {},
        "no_canonicalization": {"use_canon": False},
        "no_stability": {"use_stability": False},
        "diagonal_student_t": {"_noise": "diag_t"},
        "gaussian_innovations": {"_noise": "gaussian"},
        "no_residual_noise": {"_noise": "none"},
        "no_one_step": {"use_one_step": False},
        "no_diff_peak": {"use_diff_peak": False},
        "no_anchor": {"use_anchor": False},
        "single_stage": {"single_stage": True},
    }
    rows = []
    for name, tog in ABL.items():
        noise_mode = tog.pop("_noise", "mvt")
        accs = {"RMSE_X": [], "RMSE_Y": [], "relB": [], "white": [], "viol": []}
        for seed in seeds:
            set_seed(seed)
            pack = build_datasets(cfg, seed)
            res = train_proposed(pack, cfg, variant="static", noise_mode=noise_mode, toggles=tog)
            ev = evaluate_proposed(res, pack, cfg, with_diagnostics=(noise_mode != "none"))
            accs["RMSE_X"].append(ev["metrics"]["test_on_X"]["overall"]["RMSE"])
            accs["RMSE_Y"].append(ev["metrics"]["test_vs_Y"]["overall"]["RMSE"])
            accs["relB"].append(ev["params"]["rel_error_B_aligned"])
            accs["white"].append(ev.get("diagnostics", {}).get("whiteness_score", np.nan))
            accs["viol"].append(ev["stability"]["violation"])
        rows.append({"ablation": name,
                     "RMSE_X": float(np.mean(accs["RMSE_X"])),
                     "RMSE_Y": float(np.mean(accs["RMSE_Y"])),
                     "rel_error_B_aligned": float(np.mean(accs["relB"])),
                     "whiteness_score": float(np.nanmean(accs["white"])),
                     "stability_violations": int(np.sum(accs["viol"]))})
        if verbose:
            print(f"  [ablation] {name:22s} RMSE_X={rows[-1]['RMSE_X']:.4f} "
                  f"RMSE_Y={rows[-1]['RMSE_Y']:.4f} relB={rows[-1]['rel_error_B_aligned']:.4f} "
                  f"white={rows[-1]['whiteness_score']:.4f}")
    save_csv_dicts(rows, os.path.join(out_dir, "ablation.csv"))
    P.plot_ablation_heatmap(rows, ["RMSE_X", "RMSE_Y", "rel_error_B_aligned", "whiteness_score"],
                            os.path.join(out_dir, "figures_ablation.png"))
    save_json(rows, os.path.join(out_dir, "ablation.json"))
    return rows


def run_regimes(cfg: RunConfig, seeds, out_dir, verbose=True):
    """Stress-test the observation model across noise regimes.

    Trains the static variant with three residual models (Gaussian, diagonal
    Student-t, full multivariate Student-t) under four noise regimes, and reports
    held-out NLL, 95% coverage, residual whiteness and off-diagonal innovation
    correlation -- the metrics where the multivariate heavy-tailed model should win.
    """
    ensure_dir(out_dir)
    REGIMES = {
        "A_gauss_white": D.NoiseConfig(kind="gaussian", colored=False),
        "B_gauss_colored": D.NoiseConfig(kind="gaussian", colored=True),
        "C_studentt_colored": D.NoiseConfig(kind="studentt", colored=True),
        "D_studentt_outliers": D.NoiseConfig(kind="studentt", colored=True,
                                             outlier_prob=0.04, outlier_scale=6.0),
        "E_cross_correlated": D.NoiseConfig(kind="studentt", colored=True,
                                            chol=((0.12, 0.0), (0.12, 0.05))),
    }
    MODES = ["gaussian", "diag_t", "mvt"]
    rows = []
    for rname, ncfg in REGIMES.items():
        for mode in MODES:
            acc = {k: [] for k in ["rmse_y", "nll", "cov95", "white", "xcorr", "nu"]}
            for seed in seeds:
                set_seed(seed)
                pack = build_datasets(cfg, seed, noise_override=ncfg)
                res = train_proposed(pack, cfg, variant="static", noise_mode=mode)
                ev = evaluate_proposed(res, pack, cfg, with_diagnostics=True)
                _, _, te = res["loaders"]
                d = ev.get("diagnostics", {})
                cc = np.array(d.get("cross_correlation", np.eye(2)))
                off = cc[~np.eye(cc.shape[0], dtype=bool)]
                acc["rmse_y"].append(ev["metrics"]["test_vs_Y"]["overall"]["RMSE"])
                acc["nll"].append(T.test_nll(res["model"], te))
                acc["cov95"].append(d.get("coverage_95", np.nan))
                acc["white"].append(d.get("whiteness_score", np.nan))
                acc["xcorr"].append(float(np.mean(np.abs(off))))
                acc["nu"].append(ev["stability"]["nu"])
            row = {"regime": rname, "noise_model": mode,
                   "RMSE_Y": float(np.mean(acc["rmse_y"])),
                   "test_NLL": float(np.nanmean(acc["nll"])),
                   "coverage_95": float(np.nanmean(acc["cov95"])),
                   "whiteness": float(np.nanmean(acc["white"])),
                   "xcorr_offdiag": float(np.nanmean(acc["xcorr"])),
                   "nu": float(np.nanmean(acc["nu"]))}
            rows.append(row)
            if verbose:
                print(f"  [{rname:22s} | {mode:8s}] NLL={row['test_NLL']:.3f} "
                      f"cov95={row['coverage_95']:.3f} white={row['whiteness']:.3f} "
                      f"xcorr={row['xcorr_offdiag']:.3f} RMSE_Y={row['RMSE_Y']:.3f}")
    save_csv_dicts(rows, os.path.join(out_dir, "regimes.csv"))
    save_json(rows, os.path.join(out_dir, "regimes.json"))
    return rows


def run_prediction_variants(cfg: RunConfig, seeds, out_dir, verbose=True):
    """Prediction-enhanced family (all rows use the prediction protocol =
    multi-horizon rollout training). Reports BOTH the structural output
    x_struct and the corrected output x_pred, mean +/- std over seeds, and a
    seed-level paired comparison. The static variant (identification) is reported
    elsewhere."""
    ensure_dir(out_dir)
    SPECS = {  # all use the rollout (prediction) protocol
        "Proposed (GRU)": dict(variant="recurrent", use_rollout=True),
        "Proposed (TCN)": dict(variant="tcn", use_rollout=True),
        "Proposed (TCN+poly)": dict(variant="tcn", poly_degree=2, use_rollout=True),
        "Proposed (TCN+residual)": dict(variant="tcn", use_residual_head=True, use_rollout=True),
        "Proposed (TCN+poly+residual)": dict(variant="tcn", poly_degree=2,
                                             use_residual_head=True, use_rollout=True),
    }
    rows = []
    seqX = {}   # per-model list (per seed) of per-sequence corrected RMSE on X
    for name, spec in SPECS.items():
        acc = {k: [] for k in ["sx", "rx", "r2x", "ry", "r2y", "relB", "viol"]}
        seqX[name] = []
        for seed in seeds:
            set_seed(seed)
            pack = build_datasets(cfg, seed)
            res = train_proposed(pack, cfg, noise_mode="mvt", stage_c_target="x", **spec)
            xs = pack["scaler_x"]; _, _, te = res["loaders"]
            # corrected (prediction) output
            Xt_n, Yt_n, Xp_n = T.predict(res["model"], te)
            Xt = xs.inverse_transform(Xt_n); Yt = xs.inverse_transform(Yt_n); Xp = xs.inverse_transform(Xp_n)
            # structural output
            _, _, Xs_n = T.predict_structural(res["model"], te)
            Xs = xs.inverse_transform(Xs_n)
            from .utils import evaluate_metrics, per_sequence_rmse as _pseq
            acc["sx"].append(evaluate_metrics(Xt, Xs)["RMSE"])
            acc["rx"].append(evaluate_metrics(Xt, Xp)["RMSE"]); acc["r2x"].append(evaluate_metrics(Xt, Xp)["R2"])
            acc["ry"].append(evaluate_metrics(Yt, Xp)["RMSE"]); acc["r2y"].append(evaluate_metrics(Yt, Xp)["R2"])
            seqX[name].append(_pseq(Xt, Xp).tolist())
            ev = evaluate_proposed(res, pack, cfg, with_diagnostics=False)
            acc["relB"].append(ev["params"]["rel_error_B_aligned"])
            acc["viol"].append(ev["stability"]["violation"])

        def ms(key):
            return float(np.mean(acc[key])), float(np.std(acc[key]))
        rx_m, rx_s = ms("rx"); r2x_m, r2x_s = ms("r2x"); ry_m, ry_s = ms("ry"); sx_m, sx_s = ms("sx")
        rows.append({"model": name, "RMSE_X_struct_mean": sx_m, "RMSE_X_struct_std": sx_s,
                     "RMSE_X_mean": rx_m, "RMSE_X_std": rx_s, "R2_X_mean": r2x_m, "R2_X_std": r2x_s,
                     "RMSE_Y_mean": ry_m, "RMSE_Y_std": ry_s,
                     "relB_aligned": float(np.mean(acc["relB"])),
                     "stability_violations": int(np.sum(acc["viol"]))})
        if verbose:
            print(f"  [{name:30s}] X_struct={sx_m:.3f} X_pred={rx_m:.3f}+-{rx_s:.3f} "
                  f"R2_X={r2x_m:.3f} Y={ry_m:.3f} relB={np.mean(acc['relB']):.3f}")

    # black-box TCN reference
    acc = {k: [] for k in ["rx", "r2x", "ry", "r2y"]}
    seqX["TCN (black-box)"] = []
    for seed in seeds:
        set_seed(seed)
        pack = build_datasets(cfg, seed)
        ev = train_eval_baseline("tcn", pack, cfg)
        acc["rx"].append(ev["test_on_X"]["overall"]["RMSE"]); acc["r2x"].append(ev["test_on_X"]["overall"]["R2"])
        acc["ry"].append(ev["test_vs_Y"]["overall"]["RMSE"]); acc["r2y"].append(ev["test_vs_Y"]["overall"]["R2"])
        seqX["TCN (black-box)"].append(ev["rmse_seq_X"])
    rows.append({"model": "TCN (black-box)", "RMSE_X_struct_mean": float("nan"), "RMSE_X_struct_std": float("nan"),
                 "RMSE_X_mean": float(np.mean(acc["rx"])), "RMSE_X_std": float(np.std(acc["rx"])),
                 "R2_X_mean": float(np.mean(acc["r2x"])), "R2_X_std": float(np.std(acc["r2x"])),
                 "RMSE_Y_mean": float(np.mean(acc["ry"])), "RMSE_Y_std": float(np.std(acc["ry"])),
                 "relB_aligned": float("nan"), "stability_violations": -1})
    if verbose:
        print(f"  [{'TCN (black-box)':30s}] X_pred={rows[-1]['RMSE_X_mean']:.3f} R2_X={rows[-1]['R2_X_mean']:.3f}")

    # seed-level paired comparisons (data-driven best enhanced variant)
    prop_rows = [r for r in rows if r["model"] != "TCN (black-box)"]
    best = min(prop_rows, key=lambda r: r["RMSE_X_mean"])["model"]
    save_json({k: [list(map(float, a)) for a in v] for k, v in seqX.items()},
              os.path.join(out_dir, "prediction_seqX.json"))
    sig = []
    for ref in ["TCN (black-box)", "Proposed (TCN)"]:
        h = S.hierarchical_comparison(seqX[ref], seqX[best])
        sig.append({"comparison": f"{best} vs {ref}",
                    "seed_mean_delta_rmse": h["seed_mean_delta_rmse"],
                    "seeds_favoring_enhanced": f"{h['seeds_favoring_model']}/{h['n_seeds']}",
                    "paired_t_p": h["paired_t_p"],
                    "hier_boot_low": h["hier_bootstrap_ci95"][0],
                    "hier_boot_high": h["hier_bootstrap_ci95"][1]})
    save_csv_dicts(rows, os.path.join(out_dir, "prediction_variants.csv"))
    save_csv_dicts(sig, os.path.join(out_dir, "prediction_significance.csv"))
    save_json({"variants": rows, "significance": sig}, os.path.join(out_dir, "prediction_variants.json"))
    if verbose:
        for s in sig:
            print(f"  [sig] {s['comparison']}: dRMSE={s['seed_mean_delta_rmse']:+.4f} "
                  f"seeds={s['seeds_favoring_enhanced']} p={s['paired_t_p']:.3f} "
                  f"CI=[{s['hier_boot_low']:.4f},{s['hier_boot_high']:.4f}]")
    return rows


def measure_complexity(cfg: RunConfig, out_dir):
    """Parameter counts and one train-step wall-time per model."""
    import time as _time
    ensure_dir(out_dir)
    set_seed(0)
    pack = build_datasets(cfg, 0)
    tr, va, te = _loaders(pack, cfg)
    cw = np.array([1.35, 1.0], np.float32)
    A0, B0, Phi0 = stage1_init(pack, cfg)
    rows = []

    def _count(m):
        return int(sum(p.numel() for p in m.parameters()))

    def _time_step(step):
        # warmup + timed (Date.now-free environments: time.perf_counter is fine here)
        step(); t0 = _time.perf_counter()
        for _ in range(3):
            step()
        return (_time.perf_counter() - t0) / 3.0

    for variant in ["static", "recurrent", "tcn"]:
        m = build_model(cfg, variant, "mvt", A0, B0, Phi0, cw)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        batch = next(iter(tr))

        def step():
            u, x, y = (b.to(T.DEVICE) for b in batch)
            opt.zero_grad()
            tot, _, _ = m.losses(u, x, y, T.DEFAULT_WEIGHTS, T.DEFAULT_OPTS)
            tot.backward(); opt.step()
        rows.append({"model": f"proposed_{variant}", "n_params": _count(m),
                     "train_step_ms": round(1000 * _time_step(step), 1)})

    for name, ctor in [("direct_gru", M.DirectGRU), ("lstm", M.DirectLSTM),
                       ("tcn", M.TCN), ("narx_mlp", M.NARXMLP)]:
        kw = dict(in_dim=2, out_dim=2, channel_weights=cw)
        if name in ("direct_gru", "lstm"):
            kw.update(hidden_size=cfg.hidden_size, num_layers=cfg.num_layers)
        elif name == "tcn":
            kw.update(hidden_size=cfg.hidden_size)
        else:
            kw.update(hidden_size=cfg.hidden_size, na=2, nb=2)
        m = ctor(**kw).to(T.DEVICE)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        batch = next(iter(tr))

        def step():
            u, x, y = (b.to(T.DEVICE) for b in batch)
            opt.zero_grad()
            tot, _, _ = m.losses(u.to(T.DEVICE), x.to(T.DEVICE))
            tot.backward(); opt.step()
        rows.append({"model": name, "n_params": _count(m),
                     "train_step_ms": round(1000 * _time_step(step), 1)})

    save_csv_dicts(rows, os.path.join(out_dir, "complexity.csv"))
    return rows
