# Nuisance-cover model-ranking stability

Status: frozen before outcome computation on 2026-08-28.

Winner agreement can hide reorderings among the other candidate algorithms.
On the same three exact panels, use 1,024 equal-compute draws and compare each
action's complete validation ranking of five algorithms with the exact
full-quotient validation ranking. Report pairwise ordering agreement, exact
full-ranking recovery, and top-two set recovery. Ties in the full quotient are
resolved by fixed config order, matching selection.

The ranking gate requires strength-2 to have higher panel-mean pairwise
agreement than all three controls in all three panels and higher dataset-mean
agreement than IID on at least 60% of datasets in every panel. This is a
conditional tensor analysis, not new training evidence.

