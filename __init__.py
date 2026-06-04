"""
MINO: Identifiability-regularized probabilistic deep Hammerstein modeling
for noisy MIMO systems.

This package is the revised implementation accompanying the major-revision
manuscript. Relative to the original staged-GRU code it adds:

  * a memoryless (static/canonical) nonlinear variant for identification, in
    addition to the recurrent GRU variant for prediction;
  * latent canonicalization (mean/covariance/sign-order conventions);
  * a stability-constrained recursive MIMO linear block (exact spectral-radius
    projection + a differentiable norm sufficient condition);
  * a multivariate Student-t colored-noise observation likelihood with a
    full positive-definite scatter matrix, a stationary residual AR process
    and learnable degrees of freedom;
  * dimensionless loss normalization;
  * aligned parameter-recovery metrics, companion eigenvalues and
    impulse-response error;
  * extended baselines (LSTM, TCN, NARX-MLP) on top of Direct-GRU / Linear-ARX;
  * an ablation harness, residual diagnostics and paired significance tests.
"""

__version__ = "2.0.0"
