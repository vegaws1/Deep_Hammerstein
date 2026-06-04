"""Re-run only the ablation matrix at more seeds (for a stable table)."""
import warnings, torch
warnings.filterwarnings("ignore")
torch.set_num_threads(8)
from mino.experiment import RunConfig, run_ablations
run_ablations(RunConfig(), [0, 1, 2, 3, 4], "results/ablation")
print("ablation-only run done")
