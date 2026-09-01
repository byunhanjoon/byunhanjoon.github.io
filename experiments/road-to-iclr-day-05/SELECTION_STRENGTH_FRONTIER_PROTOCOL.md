# Model-selection strength/computation frontier

Status: frozen before outcome computation on 2026-08-28.

This follow-up uses the same three panels and five-candidate model-selection
task as `ROBUST_MODEL_SELECTION_PROTOCOL.md`, but evaluates the complete
hierarchy at 512 deterministic draws per dataset:

- budget 4: one strength-1 cover, IID-4, and one complete seed block;
- budget 16: one strength-2 cover, IID-16, four strength-1 covers, and four
  seed blocks;
- budget 64: one strength-3 cover, IID-64, four strength-2 covers, sixteen
  strength-1 covers, and sixteen seed blocks;
- budget 128: the deterministic full quotient diagnostic.

All designs use common nuisance coordinates across candidate algorithms in a
draw. Selection minimizes validation Brier score or MSE and all reported
predictive outcomes use held-out test data.

Primary diagnostics are agreement with the full-quotient validation winner,
validation quotient regret, and realized held-out proper loss. The hierarchy
claim passes only if, in at least two of three panels, (i) cover agreement is
nondecreasing from strength 1 to 3, (ii) quotient regret is nonincreasing, and
(iii) each strength cover has lower mean regret than same-budget IID. Report
all competing blocked designs even if they outperform a cover.

Post-gate task-balanced addendum (frozen before its outcomes on 2026-08-28):
apply the identical 4/16/64 frontier to the eight-source task-balanced
three-candidate panel. Its monotonicity result is reported separately and does
not alter the original three-panel gate.
