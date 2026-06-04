"""Print LaTeX-ready numbers from results/ for filling main.tex. Read-only."""
import json, csv, os

d = json.load(open("results/summary.json"))
agg = {r["model"]: r for r in d["aggregate"]}
par = {r["model"]: r for r in d["parameter_recovery"]}
sig = {r["model_vs_direct_gru"]: r for r in d["significance"]}
hier = {r["model_vs_direct_gru"]: r for r in d.get("significance_hierarchical", [])}
diag = d["diagnostics"]


def g(m, k):
    return agg[m][k]


LABEL = {"proposed_recurrent": "Proposed (recurrent, GRU)", "proposed_static": "Proposed (static, MLP)",
         "proposed_tcn": "Proposed (TCN front-end)", "proposed_static_yonly": "Proposed (static, $y$-only)",
         "direct_gru": "Direct GRU", "lstm": "Direct LSTM", "tcn": "TCN", "narx_mlp": "NARX-MLP",
         "linear_arx": "Linear ARX", "poly_hammerstein": "Polynomial Hammerstein--ARX"}
MAIN_ORDER = ["proposed_recurrent", "proposed_static", "proposed_tcn", "proposed_static_yonly",
              "direct_gru", "lstm", "tcn", "narx_mlp", "linear_arx", "poly_hammerstein"]

print("===== TABLE: main (RMSE_X, R2_X, RMSE_Y, R2_Y) =====")
for m in MAIN_ORDER:
    if m not in agg:
        continue
    print(f"{LABEL[m]} & ${g(m,'test_on_X_RMSE_mean'):.3f}\\pm{g(m,'test_on_X_RMSE_std'):.3f}$ "
          f"& ${g(m,'test_on_X_R2_mean'):.3f}\\pm{g(m,'test_on_X_R2_std'):.3f}$ "
          f"& ${g(m,'test_vs_Y_RMSE_mean'):.3f}\\pm{g(m,'test_vs_Y_RMSE_std'):.3f}$ "
          f"& ${g(m,'test_vs_Y_R2_mean'):.3f}\\pm{g(m,'test_vs_Y_R2_std'):.3f}$ \\\\")

print("\n===== TABLE: significance (seed-level primary | pooled secondary) =====")
for m in ["proposed_recurrent", "proposed_static", "lstm", "tcn"]:
    h = hier.get(m, {})
    r = sig[m]
    tp = h.get("paired_t_p", float("nan"))
    print(f"{LABEL.get(m,m)} & ${h.get('seed_mean_delta_rmse_X',0):.3f}$ & {h.get('seeds_favoring_model','?')} "
          f"& ${tp:.1e}$ & $[{h.get('hier_boot_low',0):.3f},{h.get('hier_boot_high',0):.3f}]$ "
          f"& ${r['X_mean_delta_rmse']:.3f}$ & ${r['X_cohen_dz']:.2f}$ \\\\")

print("\n===== TABLE: parameter recovery =====")
for m in ["proposed_static", "proposed_recurrent"]:
    r = par[m]
    print(f"{LABEL[m]} & ${r['rel_error_A_mean']:.3f}$ & ${r['rel_error_B_aligned_mean']:.3f}$ "
          f"& ${r['rel_error_B_raw_mean']:.3f}$ & ${r['impulse_response_error_mean']:.3f}$ & ${r['nu_mean']:.2f}$ \\\\")

print("\n===== INLINE =====")
print(f"ARX R2_X={g('linear_arx','test_on_X_R2_mean'):.3f}; poly_ham R2_X={g('poly_hammerstein','test_on_X_R2_mean'):.3f}; "
      f"TCN R2_X={g('tcn','test_on_X_R2_mean'):.3f}; proposed_rec R2_X={g('proposed_recurrent','test_on_X_R2_mean'):.3f}; "
      f"proposed_tcn R2_X={g('proposed_tcn','test_on_X_R2_mean'):.3f}; direct_gru R2_X={g('direct_gru','test_on_X_R2_mean'):.3f}")
print(f"nu={par['proposed_static']['nu_mean']:.2f}; coverage 50/80/95={diag['coverage_50']['mean']:.2f}/"
      f"{diag['coverage_80']['mean']:.2f}/{diag['coverage_95']['mean']:.2f}; whiteness={diag['whiteness_score']['mean']:.3f}")
if hier:
    hr = hier.get("proposed_recurrent", {})
    print(f"SEED-LEVEL recurrent: mean_delta={hr.get('seed_mean_delta_rmse_X'):.4f} seeds={hr.get('seeds_favoring_model')} "
          f"t_p={hr.get('paired_t_p'):.2e} hierCI=[{hr.get('hier_boot_low'):.4f},{hr.get('hier_boot_high'):.4f}]")

print("\n===== ABLATION =====")
if os.path.exists("results/ablation/ablation.csv"):
    rows = list(csv.DictReader(open("results/ablation/ablation.csv")))
    AblL = {"full": "Full model", "no_canonicalization": "No canonicalization", "no_stability": "No stability constraint",
            "diagonal_student_t": "Diagonal Student-$t$", "gaussian_innovations": "Gaussian innovations",
            "no_residual_noise": "No residual-AR noise", "no_one_step": "No one-step term",
            "no_diff_peak": "No derivative/peak", "no_anchor": "No anchoring", "single_stage": "Single-stage training"}
    order = list(AblL)
    rows.sort(key=lambda r: order.index(r["ablation"]) if r["ablation"] in order else 99)
    for r in rows:
        w = r["whiteness_score"]
        ws = f"{float(w):.3f}" if w not in ("", "nan") else "--"
        print(f"{AblL.get(r['ablation'],r['ablation'])} & ${float(r['RMSE_X']):.3f}$ & ${float(r['RMSE_Y']):.3f}$ "
              f"& ${float(r['rel_error_B_aligned']):.3f}$ & ${ws}$ \\\\")

print("\n===== REGIMES (NLL G/dT/mvt | cov95 G/dT/mvt) =====")
if os.path.exists("results/regimes/regimes.csv"):
    rows = list(csv.DictReader(open("results/regimes/regimes.csv")))
    reg = {}
    for r in rows:
        reg.setdefault(r["regime"], {})[r["noise_model"]] = r
    RL = {"A_gauss_white": "A (Gauss, white)", "B_gauss_colored": "B (Gauss, colored)",
          "C_studentt_colored": "C (Student-$t$, colored)", "D_studentt_outliers": "D (Student-$t$, outliers)",
          "E_cross_correlated": "E (cross-correlated)"}
    for rk in ["A_gauss_white", "B_gauss_colored", "C_studentt_colored", "D_studentt_outliers", "E_cross_correlated"]:
        if rk not in reg:
            continue
        m = reg[rk]
        def nll(k): return float(m[k]["test_NLL"])
        def cov(k): return float(m[k]["coverage_95"])
        print(f"{RL[rk]} & ${nll('gaussian'):.2f}$ & ${nll('diag_t'):.2f}$ & ${nll('mvt'):.2f}$ "
              f"& ${cov('gaussian'):.2f}$ & ${cov('diag_t'):.2f}$ & ${cov('mvt'):.2f}$ \\\\")

print("\n===== PREDICTION VARIANTS (struct | pred X+-std | pred Y+-std | relB | viol) =====")
if os.path.exists("results/prediction/prediction_variants.csv"):
    for r in csv.DictReader(open("results/prediction/prediction_variants.csv")):
        name = r["model"]
        bb = (name == "TCN (black-box)")
        resid = "residual" in name
        struct = "---" if bb else f"${float(r['RMSE_X_struct_mean']):.3f}$"
        predx = f"${float(r['RMSE_X_mean']):.3f}\\pm{float(r['RMSE_X_std']):.3f}$"
        predy = f"${float(r['RMSE_Y_mean']):.3f}\\pm{float(r['RMSE_Y_std']):.3f}$"
        relb = "---" if bb else ("diag.\\ only" if resid else f"${float(r['relB_aligned']):.3f}$")
        viol = "---" if r["stability_violations"] == "-1" else f"${int(float(r['stability_violations']))}$"
        print(f"{name} & {struct} & {predx} & {predy} & {relb} & {viol} \\\\")
if os.path.exists("results/prediction/prediction_significance.csv"):
    print("--- prediction significance ---")
    for r in csv.DictReader(open("results/prediction/prediction_significance.csv")):
        print(f"{r['comparison']}: dRMSE={float(r['seed_mean_delta_rmse']):+.4f} "
              f"seeds={r['seeds_favoring_enhanced']} t_p={float(r['paired_t_p']):.3f} "
              f"hierCI=[{float(r['hier_boot_low']):.4f},{float(r['hier_boot_high']):.4f}]")

print("\n===== COMPLEXITY =====")
if os.path.exists("results/complexity.csv"):
    rows = {r["model"]: r for r in csv.DictReader(open("results/complexity.csv"))}
    CL = {"proposed_static": "Proposed (static)", "proposed_recurrent": "Proposed (recurrent)",
          "proposed_tcn": "Proposed (TCN)", "direct_gru": "Direct GRU", "lstm": "Direct LSTM",
          "tcn": "TCN", "narx_mlp": "NARX-MLP"}
    for m in CL:
        if m in rows:
            print(f"{CL[m]} & ${int(rows[m]['n_params'])}$ & ${float(rows[m]['train_step_ms']):.1f}$ \\\\")
