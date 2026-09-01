# Candidate-ranking fidelity protocol

Status: frozen before ranking outcomes on 2026-08-28. Winner agreement and
test-loss outcomes are already known; full-ranking fidelity is a new endpoint.

For every randomized action in the confirmation, changed-menu,
changed-subsample, first external OpenML, and task-balanced OpenML selection
panels, compare the complete validation candidate ranking with the full
nuisance-quotient validation ranking. Report Spearman rank correlation and the
fraction of correctly ordered candidate pairs. Use the same 1,024 draws and
all existing equal-compute controls.

The descriptive gate requires strength-2 to exceed IID-16 in both mean rank
correlation and pairwise-order accuracy in at least four of five panels. This
does not replace the prospective held-out selection gates and cannot solve
validation/test target shift; it isolates nuisance-estimation fidelity.
