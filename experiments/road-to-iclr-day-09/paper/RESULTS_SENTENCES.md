# Frozen results sentences

On the frozen PriorDial-v1.1 development suite at context size 64, increasing rho from
zero to one raised marginal-only six-way mechanism accuracy from chance-level
0.145/0.121 to 0.902/0.905 for classification/regression.

At rho=1, adding label-free marginal shape to a cross-fitted invariant context selector
reduced classification log loss by 0.0073 (95% task-bootstrap CI 0.0043–0.0103) and
standardized regression MSE by 0.2052 (0.1758–0.2370); at rho=0 neither task showed
positive evidence of marginal utility.

These are development results, not held-out confirmation. The ordinary raw linear
baseline remained worse than rank on average and is not evidence for a raw-only win.

On the frozen two-task E0 reproduction, TabICLv2 single/default had approximately 0.0187
excess total-variation disagreement on churn, versus 0.0019 for Mitra, after subtracting
the model/seed-matched identity-refit floor. On wine quality, TabICL had 0.038–0.044
normalized excess prediction disagreement while Mitra's excess was -0.0002 with a cell
bootstrap interval spanning zero. These are two-dataset development results; TabPFN-3
was unavailable and no cross-dataset confidence claim is authorized.

On the frozen exploratory PriorDial E2 panel, TabICLv2 single/default averaged
0.0896/0.0710 excess classification TV versus 0.0075 for Mitra and 0.1605/0.1356 excess
normalized regression disagreement versus 0.0113 for Mitra. All 42 paired episode
differences favored Mitra's stability in both task types; paired TabICL-minus-Mitra 95%
episode-bootstrap intervals excluded zero. This passes the scoped TabICLv2-vs-Mitra G2
screen, while TabPFN-3 remains unavailable and no universal or causal claim is made.

In the M0–M5 kill test, the final development-tuned fixed raw/rank mixture achieved
0.65040 classification log loss and 0.69178 regression MSE across 2,940 fresh test tasks
per task type, versus 0.65038 and 0.69034 for the calibrated learned gate. The paired
fixed-minus-gate difference was 0.000012 (95% task-bootstrap CI -0.000035–0.000058) for
classification and 0.001436 (0.001218–0.001659) for regression.

Despite fixed-to-oracle headroom of 0.00444 log loss and 0.00847 MSE, the learned gate
captured only 0.27% and 16.96%, respectively. Five sequential fresh-test diagnostics—
more training tasks, featurewise pooling, an auxiliary synthetic objective, and output
calibration—did not meet G3, so no dual-channel method or downstream confirmatory run was
authorized.

In an independent 8,820-episode benchmark replication at context size 96 and 12
features, marginal-only mechanism accuracy increased from 0.127/0.144 at rho=0 to
0.978/0.987 at rho=1 for classification/regression while the mechanism and warp
marginals remained balanced.

Regression shape-informed routing replicated with a rho=1 MSE gain of 0.24339 (95%
task-bootstrap CI 0.21664–0.27044), reducing loss from 0.45083 to 0.20743. Classification
contradicted development: shape-informed routing increased rho=1 log loss by 0.00402
(0.00152–0.00653), so no task-general performance claim is supported.

At rho=1, the classification selector identified 99.2% of mechanism families but still
worsened predictive loss; one-hot routing to the known matched-family expert was worse
than the stable mixture by 0.02301 (0.01933–0.02674). Task-family identification is
therefore not a sufficient training objective for predictive routing in this expert set.

In a post-hoc 2x2 regime diagnostic, increasing context size from 64 to 96 at eight
features did not materially change classification routing utility (-0.00104, 95% CI
-0.00546–0.00335), whereas increasing from 8 to 12 features at context size 64 reduced
utility by 0.00532 (0.00129–0.00937). Mechanism-selection accuracy increased as routing
utility fell; regression routing remained strongly beneficial in all four cells.

The post-hoc all-family decomposition localized the classification dimension effect:
at context size 64, moving from 8 to 12 features reduced interaction routing gain by
0.01985 (95% CI 0.00926–0.03057) and periodic gain by 0.01652 (0.00769–0.02552), while
linear routing improved by 0.00702 (0.00256–0.01190). No mechanism family was filtered
from the aggregate.

In a separately frozen 9,600-episode untouched test, routing the same six experts by
context-only three-fold predictive loss improved over a development-tuned fixed mixture
by 0.005465 classification log loss (95% equal-cell task-bootstrap CI
0.004797–0.006129) and 0.254185 standardized regression MSE (0.246516–0.262313).

Loss-aligned routing captured 29.51% (26.34–32.65%) of the fixed-to-best-individual
classification headroom and 99.66% (98.88–100.43%) in regression. At 12 features and
rho>=0.75, its classification gain remained positive at 0.004811
(0.003364–0.006253).

This contrast isolates objective alignment rather than a new ensemble algorithm:
mechanism routing was harmful despite 99.2% family identification, whereas predictive
competence routing improved held-out loss while its hard argmin matched the query-best
expert on only 42.02% of classification episodes. Cross-validated competence weighting
and dynamic ensemble selection are prior art, so the supported contribution is the
controlled metadata-identifiability-versus-predictive-usability result.

An immutable-prediction diagnostic separated weighting from selection. Replacing the
soft competence mixture by its hard context-CV argmin increased classification log loss
by 0.023797 (95% CI 0.022615–0.024993) and regression MSE by 0.024655
(0.022748–0.026604). Hard selection was itself 0.018333 worse than the fixed mixture in
classification, so the held-out gain is specifically a calibrated aggregation effect.

The soft-over-hard benefit was largest at low CV margins and on episodes where the hard
choice missed the query-best expert. Thus generator identification, individual-expert
selection, and mixture prediction are empirically distinct objectives on the same
controlled tabular contexts.

Without real-data retuning, the synthetic competence router had favorable point gains on
a seven-dataset numeric panel, but hierarchical dataset-and-split bootstrap intervals
included zero: +0.003461 log loss (-0.001813–0.011346) across three classification
datasets and +0.236997 standardized MSE (-0.009046–0.735772) across four regression
datasets. Black Friday was significantly harmed, so this small panel does not support an
external-transfer claim.

On seven previously unseen OpenML regression identities, the same synthetic-tuned router
reduced dataset-balanced standardized MSE relative to fixed by 0.29103 (95% hierarchical
dataset-and-episode bootstrap CI 0.02525–0.94647). On six unseen binary identities it was
worse by 0.00213 log loss (0.00026–0.00465), rejecting task-general transfer.

A deterministic five-dataset regression confirmation panel independently replicated the
effect: +0.10045 MSE improvement (0.00173–0.25055), with four datasets positive and
Auction Verification negative. Thus real numeric regression transfer is supported on
two disjoint panels totaling 12 identities; full synthetic-tuned binary competence
transfer is not.

An exact assignment control cyclically rotated the six CV losses while preserving each
episode's weight spectrum. Correct assignment improved over this null by 0.02574 log
loss (0.02485–0.02665) and 0.55553 MSE (0.54046–0.57051) on PriorDial, and by 0.00983
(0.00278–0.02056) / 0.56642 (0.09606–1.25180) across 9/16 real
classification/regression datasets. Dynamic concentration without expert-specific
alignment is insufficient.

On the independent regression identities, gains were already positive at 32 context
rows (+0.08257 [0.00891, 0.21011]); the gain per context doubling was +0.00665
[-0.02456, 0.04618]. Thus no 32–192-row scaling law is supported.

Two-fold competence reduced standalone expert fits by 25% relative to three-fold and
still beat fixed by 0.09089 MSE (0.00457–0.23081), but its loss relative to three-fold
was +0.00748 (-0.00295–0.02572). The upper bound failed the predeclared 0.01
noninferiority margin, so three folds remains the supported compute/performance choice.

Across all completed, identity-disjoint real panels, the retrospective dataset-balanced
synthesis gave a +0.21796 standardized-MSE gain (95% dataset-bootstrap CI
0.04168–0.45999) over 16 regression datasets. Fourteen datasets improved, the median
gain was +0.02339, the 10%-trimmed mean was +0.13463, and every leave-one-dataset-out
mean remained positive.

The corresponding nine-dataset classification synthesis was -0.00027 log loss
(-0.00298–0.00317), with a negative median and only three positive datasets. The
full-adaptation external result is therefore regression-specific rather than task-general.

The classification failure was tail-localized rather than a broad ranking collapse:
competence-minus-fixed AUC was +0.00029 (-0.00337–0.00464) over nine datasets. On the six
unseen identities, the bottom-90% pointwise-NLL difference was inconclusive, whereas the
worst-decile NLL worsened by 0.02512 (0.00850–0.04525) and the rate of NLL above two rose
by 0.00150 (0.00023–0.00293).

Regression showed the opposite tail behavior over 16 datasets: competence reduced
worst-decile squared error by 1.88738 (0.22577–5.03685), reduced bottom-90% squared error
by 0.02878 (0.01127–0.05024), and lowered the fraction of squared errors above four by
0.00483 (0.00091–0.01045). Adaptive weighting therefore suppresses regression
catastrophes while amplifying rare confident classification errors in these panels.

Within the unseen binary datasets, competence-to-fixed weight KL correlated -0.352
(-0.516–-0.192) with worst-decile-NLL gain, and high-shift episodes lost 0.07868 more
than low-shift episodes (0.03483–0.12097). Regression showed no monotone within-dataset
association (+0.012, -0.051–0.080); its high-shift benefit was concentrated in a few
large-headroom datasets, ruling out a universal adaptation-magnitude prescription.

The immutable shrinkage path showed task-specific transfer: synthetic classification
peaked at lambda=0.7, whereas the initial nine real datasets peaked at lambda=0.5 and
full adaptation reversed the point gain. Synthetic and real regression curves both
improved monotonically to full adaptation.

After freezing lambda=0.1 from that real-development curve, an independent deterministic
five-dataset CC18 panel confirmed a +0.000600 log-loss improvement (95% hierarchical CI
0.000038–0.001471) and +0.000207 Brier improvement (0.000001–0.000527), with three of
five datasets positive. This is real-development-tuned numeric-binary transfer, not
synthetic-only transfer or a novel shrinkage method.

With affine feature scaling refit on each 96-row context, the frozen 10% classification
rule again improved log loss by 0.000732 (0.000142–0.001680), now on all five datasets.
Full regression competence retained a +0.09290 point gain and four positive datasets,
but its context-rescaled interval crossed zero (-0.00146–0.23810); the strongest
regression claim therefore remains scoped to the predeclared outer-fold normalization.
