"""Regenerate all paper results from zero.

Usage:
    python run_paper.py                # default "paper" config (3 seeds)
    python run_paper.py --smoke        # tiny end-to-end check (1 seed, few epochs)
    python run_paper.py --seeds 0 1 2 3 4
    python run_paper.py --plant recurrent --noise gaussian --no-color

All tables (CSV/JSON) and figures are written under ./results/.
"""

import argparse
import os
import warnings

import torch

from mino.experiment import (RunConfig, run_main, run_ablations, run_regimes,
                             measure_complexity, run_prediction_variants)

warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--ablation-seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--plant", default="static", choices=["static", "recurrent"])
    ap.add_argument("--noise", default="studentt", choices=["studentt", "gaussian"])
    ap.add_argument("--no-color", action="store_true", help="use white (uncorrelated) noise")
    ap.add_argument("--out", default="results")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--regime-seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--pred-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-ablation", action="store_true")
    ap.add_argument("--no-regimes", action="store_true")
    ap.add_argument("--no-pred", action="store_true")
    args = ap.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)

    cfg = RunConfig(plant_kind=args.plant, noise_kind=args.noise,
                    noise_colored=not args.no_color)
    seeds, abl_seeds = args.seeds, args.ablation_seeds

    if args.smoke:
        cfg = RunConfig(plant_kind=args.plant, noise_kind=args.noise,
                        noise_colored=not args.no_color,
                        T=40, n_train=24, n_val=12, n_test=16, n_step=24,
                        batch_size=8, hidden_size=24, num_layers=1,
                        epochs_A=4, epochs_B=2, epochs_C=2, epochs_base=4,
                        noise_candidates=(0.01,))
        seeds, abl_seeds = [0], [0]

    os.makedirs(args.out, exist_ok=True)
    print(f"[run_paper] device={'cuda' if torch.cuda.is_available() else 'cpu'} "
          f"plant={cfg.plant_kind} noise={cfg.noise_kind} colored={cfg.noise_colored}")
    print(f"[run_paper] seeds={seeds} ablation_seeds={abl_seeds}")

    summary = run_main(cfg, seeds, args.out)

    print("\n================ MAIN COMPARISON (test) ================")
    print(f"{'model':>24s} | {'RMSE_X':>16s} | {'R2_X':>14s} | {'RMSE_Y':>16s}")
    for r in summary["aggregate"]:
        print(f"{r['model']:>24s} | "
              f"{r['test_on_X_RMSE_mean']:.4f}±{r['test_on_X_RMSE_std']:.4f} | "
              f"{r['test_on_X_R2_mean']:.4f}±{r['test_on_X_R2_std']:.3f} | "
              f"{r['test_vs_Y_RMSE_mean']:.4f}±{r['test_vs_Y_RMSE_std']:.4f}")

    print("\n================ PARAMETER RECOVERY ===================")
    for r in summary["parameter_recovery"]:
        print(f"{r['model']:>24s} | relA={r.get('rel_error_A_mean', float('nan')):.3f} "
              f"| relB_aligned={r.get('rel_error_B_aligned_mean', float('nan')):.3f} "
              f"| relB_raw={r.get('rel_error_B_raw_mean', float('nan')):.3f} "
              f"| imp_err={r.get('impulse_response_error_mean', float('nan')):.3f}")

    print("\n================ COMPLEXITY ===========================")
    for r in measure_complexity(cfg, args.out):
        print(f"{r['model']:>24s} | params={r['n_params']:>7d} | train step {r['train_step_ms']:.1f} ms")

    if not args.no_ablation:
        print("\n================ ABLATIONS ============================")
        # ablations use the same epoch budget as the main run for consistency
        run_ablations(cfg, abl_seeds, os.path.join(args.out, "ablation"))

    if not args.no_regimes:
        print("\n================ NOISE REGIMES ========================")
        reg_seeds = [0] if args.smoke else args.regime_seeds
        run_regimes(cfg, reg_seeds, os.path.join(args.out, "regimes"))

    if not args.no_pred:
        print("\n================ PREDICTION-ENHANCED VARIANTS =========")
        pred_seeds = [0] if args.smoke else args.pred_seeds
        run_prediction_variants(cfg, pred_seeds, os.path.join(args.out, "prediction"))

    print("\n[run_paper] done.")


if __name__ == "__main__":
    main()
