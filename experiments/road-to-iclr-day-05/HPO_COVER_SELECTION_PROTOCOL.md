# Cover-stabilized hyperparameter selection

Status: frozen before this downstream outcome computation on 2026-08-28.

The existing prospective HPO tensors contain four hyperparameter candidates,
four feature orders, four category-numbering maps, binary target numbering,
and two model seeds for four datasets and two model families. Earlier analyses
tested per-schema HPO and the variance of its resulting prediction field; they
did not test designed nuisance coverage while choosing a hyperparameter.

For each of 1,024 deterministic draws per dataset/family, allocate 16 fits to
each of the four candidates, ensemble within candidate, select the lowest
validation-Brier candidate, and score its realized ensemble on held-out test
data. Compare one strength-2 cover, IID-16, four strength-1 blocks, and eight
complete two-seed blocks. Nuisance coordinates are shared across candidates.

Report quotient-winner agreement, quotient validation regret, selected
quotient test loss, and realized test Brier. The frozen gate requires
strength-2 to lower pooled realized test Brier against all three controls, beat
IID in at least 6/8 cells, and improve quotient-winner agreement over IID.
This is conditional reuse of an existing tensor, not an independent new panel.

