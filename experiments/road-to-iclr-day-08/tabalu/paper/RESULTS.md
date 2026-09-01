# Results

The controlled 100-task Phase-A chain-family panel passed its preregistered
gate. See `../PHASE_A_RESULTS.md`. This result supports exact execution given
successful constrained search; it does not support unconstrained
differentiable induction, which failed the initial smoke diagnostic.

A direct 30-task causal ablation then held the oracle graph and operands fixed.
Replacing protected arithmetic nodes with trained neural approximators produced
0.130 NRMSE in-distribution and 6.475 at 8× magnitude, while exact execution
remained at zero error. See `../EXACT_EXECUTION_RESULTS.md`.

In the matched-library heterogeneous synthetic panel, a sparse typed program
achieved 0.00102 future 4× NRMSE, versus 0.559 for manual preprocessing + MLP
and 0.711 for learned embeddings. Removing ordinal, datetime, or categorical
conditions increased future error by at least 163×. See
`../PHASE_F_RESULTS.md`; real-data validation remains required.

The penalized residual produced the intended IID continuum: usage correlated
0.989 with the true non-symbolic fraction and was nearly zero for a purely
symbolic target. It failed the 4× safety test, where the α=1 adaptive residual
scored 1.587 NRMSE versus 0.287 for omitting the residual. The unguarded branch
is therefore not part of the extrapolating core. See `../PHASE_G_RESULTS.md`.

The first source-pinned real temporal pilot failed. On UCI Bike Sharing, the
season router moved from 0.590 IID NRMSE to 9.779 on 2012, while XGBoost reached
0.602 future NRMSE. Unconstrained per-season elapsed-time terms caused the
failure. See `../REAL_TEMPORAL_RESULTS.md`.

A labeled post-hoc diagnostic removed all unbounded elapsed-time terms and cut
router future NRMSE to 0.782, supporting the causal diagnosis but still trailing
XGBoost and not changing the confirmatory no-go decision.

On a three-dataset numeric general pilot, sparse exact models were within 1.25×
of the best baseline on Diabetes and breast cancer and 1.46× on binary wine.
Linear/logistic regression won all three, so the result supports only
non-catastrophic compactness, not superiority. See
`../GENERAL_PILOT_RESULTS.md`.

Scaling isolates the remaining bottleneck. Oracle execution stays exact through
depth 8, while width-128 beam discovery reaches 1.027 OOD NRMSE and 6.7%
functional recovery. Known categorical routing remains predictively accurate
through eight regimes (0.00375 OOD NRMSE) but only 20% of runs meet the strict
10⁻³ whole-mixture recovery threshold. See `../SCALING_RESULTS.md`.
