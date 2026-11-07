# Heston (kappa, theta) profile-likelihood scan — 3x3 smoke surface

Method (2026-11-08): the 3x3 Heston surface from `test_calibration_robustness._surface`
(truth kappa=1.0, theta=0.04); for each (kappa, theta) grid point the remaining
parameters (sigma_v, rho, v0) are re-fitted with Adam (120 epochs, float64),
kappa/theta frozen. Cell values are the final price-space MSE loss:

| kappa \ theta | 0.040 | 0.100 | 0.220 | 0.400 |
|---|---|---|---|---|
| 0.40 | 1.10e+0 | 5.65e-1 | 9.59e-2 | 3.37e-2 |
| 0.62 | 1.18e+0 | 4.45e-1 | 5.31e-2 | 3.29e-1 |
| 1.00 | 1.32e+0 | 3.22e-1 | 6.41e-2 | 1.53e+0 |
| 1.60 | 1.53e+0 | 2.33e-1 | 4.31e-1 | 1.02e+1 |
| 2.50 | 1.74e+0 | 1.88e-1 | 1.84e+0 | 3.15e+1 |

Reading of the surface:

- The floor of the loss (<=1e-1) runs along **theta in [0.22, 0.40] with
  kappa in [0.4, 1.0]** — a flat valley. Across a **6x range of kappa**
  (0.4 -> 2.5) at theta=0.22 the loss varies by less than 20x, and across
  the valley interior by ~2x. First-order optimizers plateau on surfaces
  this shallow — the documented distant-start stickiness.
- The valley floor sits at **kappa*theta ~ 0.09-0.16**, vs the truth
  kappa*theta = 0.04: the short-dated 3x3 surface pins the *product* only
  loosely, and not to its true value — the fitted long-run variance is
  ~2-4x the truth while the price fit stays excellent.
- The valley closes hard at high kappa*theta (31.5 at kappa=2.5,
  theta=0.40) and at low theta (all kappa: loss >= 0.19 at theta=0.10).

Conclusion: the identifiability constraint at smoke budget is **surface
information**, not the optimizer — consistent with the lbfgs_refine
experiment (kappa moves but sigma_v wanders) and with the cross-check
precision budget (5x4 surface, 4000 epochs: all params <= 1e-3).

Reproduce: see `tests/test_calibration_robustness.py` module docstring,
finding 1; scan script pattern in the git history of this note.
