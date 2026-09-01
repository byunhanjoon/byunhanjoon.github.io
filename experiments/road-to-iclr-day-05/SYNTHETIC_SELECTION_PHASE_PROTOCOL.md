# Controlled interaction-order selection phase protocol

Status: frozen before inspecting outcomes.

## Question

Does the known pure-triple adverse corner for a strength-2 prediction estimate
actually reverse the model-selection advantage of the cross-score, while
low-order and pure-four fields behave as predicted?

## Design

Construct two scalar-regression candidates on the exact `4 x 4 x 2 x 4`
nuisance product. Candidate quotient predictions have squared-loss gaps
`0.002, 0.005, 0.010, 0.020`; both receive equal-amplitude (`0.1`) centered
pure fANOVA fields of order 1, 2, 3, or 4. Use independent nuisance streams
for each candidate. Over 65,536 actions compare:

- two independent strength-2 16-fit covers with the unbiased cross-score;
- a complete IID-U32 score.

Report exact-winner inversion rate and regret. The label is scalar regression,
so no probability clipping is involved.

## Frozen gate

The phase audit passes if strength-2 has zero inversions for orders 1 and 2,
has *higher* inversion rate than IID-U for at least 3/4 pure-triple margins,
and lower inversion rate for at least 3/4 pure-four margins.

This is a controlled counterexample, not evidence about the prevalence of
interaction orders in new real datasets.
