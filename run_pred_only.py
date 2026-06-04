"""Run only the prediction-enhanced variant comparison (5 seeds)."""
import warnings, torch
warnings.filterwarnings("ignore")
torch.set_num_threads(8)
from mino.experiment import RunConfig, run_prediction_variants
run_prediction_variants(RunConfig(), [0, 1, 2, 3, 4], "results/prediction")
print("prediction-variants run done")
