# Randomized-cover realization diagnostic

Status: post-confirmation diagnostic, frozen before realization outcomes.

On the 25 confirmation cells declared material from validation, draw 4,096
independent deployment realizations using RNG seed `2026082801`. For each cell
and draw compare one randomized budget-16 strength-2 cover with one 16-draw
iid joint ensemble, four independently randomized strength-1 covers, and four
random-schema blocks containing all four seeds. Aggregate residual across
cells and source groups.

This measures tail/reliability of the randomized action; it is not another
dataset replication and does not replace the exact expectation result.

