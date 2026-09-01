# Prospective theory target: task symmetrization and metadata cost

Status: mathematical target fixed before Phase II outcomes. This is not yet a paper theorem and makes no empirical claim.

## Objects

Let a supervised task observation be `Z = (D, x*)`, where `D = ((x_i, y_i))_{i=1}^n` and `x*` is a query. A coordinate transformation `g` acts on every feature occurrence in `D` and `x*` and leaves labels unchanged. Let `G` be a finite group of measurable bijections with identity, composition, and inverse. Let `P` be a prior over latent supervised tasks, and let `g#P` denote its pushforward under the whole-task action.

The finite symmetrization is

`P_sym = |G|^{-1} sum_{g in G} g#P`.

The use of a finite group is deliberate. The full family of monotone bijections is noncompact and does not generally admit a finite invariant Haar probability measure; no such measure is assumed.

## Proposition target 1: invariant prior predictive

Assume:

1. `G` acts measurably and bijectively on task observations;
2. the label/query likelihood is equivariant under the whole-task action;
3. regular conditional probabilities exist; and
4. predictions are compared only on transformed versions of the same task, with no context/query mismatch.

Then for every `h in G`, `h#P_sym = P_sym`. Consequently, any version of the Bayes posterior predictive under `P_sym` can be chosen so that

`q_sym(y* | hD, hx*) = q_sym(y* | D, x*)`

almost surely under the symmetrized task law.

Proof skeleton: left multiplication permutes the finite group, so

`h#P_sym = |G|^{-1} sum_g (hg)#P = P_sym`.

Apply this invariance to the joint law of `(D, x*, y*)` and disintegrate with respect to `(D, x*)`. The equality is invariance, not equivariance, because feature coordinates transform while the label space does not.

## Proposition target 2: approximate finite sampling

For a sampled transformation multiset `S` that is not closed under composition, define `P_S = |S|^{-1} sum_{g in S} g#P`. Exact invariance is not guaranteed. For any `h`, posterior-predictive discrepancy is controlled by the discrepancy between the joint laws induced by `P_S` and `h#P_S`. In particular, before conditioning, total variation contracts through the prediction Markov kernel:

`TV(K P_S, K h#P_S) <= TV(P_S, h#P_S)`.

A conditional/posterior bound needs an explicit lower bound on evidence density; it must not be claimed from the unconditional contraction alone. Empirically, sampled augmentation or orbit averaging is therefore an approximation/control, not proof of invariance to the full monotone family.

## Proposition target 3: cost of deleting useful metadata

Let `S` be all retained task information after canonicalization, `M` the discarded marginal metadata, and `Y*` the query label. Under optimal log loss, the Bayes risks are conditional entropies. The exact benefit of retaining `M` is

`R_log(S) - R_log(S, M) = H(Y* | S) - H(Y* | S, M) = I(Y*; M | S) >= 0`.

Thus complete canonicalization is free only when `Y*` is conditionally independent of `M` given retained task evidence. If marginal family predicts a latent function family `F`, then `I(F; M)` alone does not prove predictive benefit; the relevant quantity is `I(Y*; M | S)`, mediated by what the finite context already reveals about `F`. This predicts:

- a larger S2–S3 benefit at small context sizes;
- decay of that benefit as context identifies the function family;
- harm under S4 when the learned association is shifted or reversed;
- no principled benefit from marginal metadata in S1, where it is independent nuisance.

For squared error, the corresponding Bayes-risk improvement is the expected reduction in conditional variance:

`E[Var(Y* | S)] - E[Var(Y* | S, M)] = E[(E[Y* | S, M] - E[Y* | S])^2]`.

## Why factorization, not blind invariance

The theory motivates separating two channels:

- a coordinate-invariant/equivariant channel for task evidence that should survive `g`;
- an explicit marginal-metadata channel whose contribution can be learned, regularized, or downweighted under conflict.

Rank canonicalization alone targets the first channel but may delete the second. Raw-plus-rank duplication preserves both but does not guarantee that the predictor will avoid a harmful raw-coordinate shortcut. A successful method must therefore demonstrate both robustness and retention of clean/meta-prior value.

## Required proof and empirical checks before paper use

1. State the task and group actions rigorously, including categorical/equality structure.
2. Prove the disintegration step under explicit standard-Borel assumptions or restrict to finite/discrete task spaces.
3. Do not extend finite-group invariance to all monotone maps without a valid approximation argument.
4. Distinguish prior-predictive invariance from a finite neural model trained approximately on sampled transformations.
5. Link the metadata-risk identity to paired S2/S3/S4 evidence and context-size curves.
6. Drop or narrow the theory if its conditional-independence or group assumptions do not match the implemented generator/method.
