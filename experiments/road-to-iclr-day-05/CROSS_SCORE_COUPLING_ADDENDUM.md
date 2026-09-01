# Cross-score candidate-coupling addendum

Status: **frozen before independent-candidate outcomes**.

The primary cross-score experiment uses common randomized nuisance actions
across candidates. Repeat the strength-2 cross-score and IID-U32 comparison
with a separate deterministic random stream for every dataset×candidate×half.
The two halves remain independent within a candidate, but different candidates
no longer share any action indices.

Run 512 validation-only decisions on the same five panels and report winner
agreement, quotient validation regret, selected quotient test loss, and
realized 32-member test loss. The addendum passes if independently coupled
strength-2 has higher mean agreement and lower validation regret than IID-U32
on at least four of five panels. Test transfer is descriptive.
