# AGENT.md — Day 3: ICLR-Level Basis Geometry Experiments

## Mission

Test one paper-level claim:

> **Neural tabular models are sensitive to arbitrary information-equivalent feature bases because of optimization geometry, and this sensitivity can be reduced by canonicalization or invariant training.**

Do not add more tricks unless the core claim survives.

## 1. Reproduce Day-2 anchors
- Reproduce Adult PLE vs identity.
- Reproduce at least one additional Day-2 anchor.
- Use identical splits, seeds, preprocessing, and metrics.
- Keep prior failed/prospective datasets.

## 2. Controlled condition-number sweep — highest priority

Start from a full-rank whitened representation `Z0`.

Construct:

`Z_k = Z0 A_k`

with invertible `A_k` and target:

`kappa = 1, 3, 10, 30, 100, 300, 1000`

Match global scale across transforms.

Run identical training on each basis.

Start with MLP, then validate on ResNet and TabM.

Measure:
- realized condition number;
- train loss/convergence;
- validation/test metric;
- first-layer gradient norm;
- first-layer weight norm.

Plot:
- `log10(kappa)` vs convergence;
- `log10(kappa)` vs performance.

**Goal:** establish whether conditioning causally changes learning while information is fixed.

## 3. PLE vs identity whitening

For cases where PLE and identity span the same observed-state space:

Verify:
- rank;
- reconstruction both directions;
- principal-angle / column-space agreement.

Compare:
1. raw PLE;
2. raw identity;
3. standardized versions;
4. whitened versions;
5. aligned whitened versions if needed.

**Key test:** does whitening substantially collapse the original performance gap?

If yes, geometry becomes a strong mechanism.

## 4. Representation-invariant first-layer regularization

Implement:

`R(W) = tr(W Sigma W^T)`

where `Sigma` is the training representation covariance.

Unit-test invariance under:

`z' = A z`
`W' = W A^{-1}`
`Sigma' = A Sigma A^T`.

Compare:
1. standard weight decay;
2. no first-layer weight decay;
3. invariant first-layer regularizer.

Repeat selected `kappa` sweeps.

Report:
- std of performance across equivalent bases;
- max-min spread;
- worst-case drop.

**Goal:** show a remedy, not only a phenomenon.

## 5. Ordinal basis geometry — core extension

For true ordinal columns `c1 < ... < cK`, construct equivalent bases:

1. centered local one-hot / contrast;
2. cumulative / thermometer thresholds;
3. orthogonal ordinal contrasts;
4. whitened ordinal basis.

Verify exact rank/reconstruction equivalence.

Measure conditioning, convergence, and performance.

Then run controlled-`kappa` transforms from the whitened ordinal basis.

Do **not** impose monotonic predictions merely because the input is ordinal.

**Strong result:** cumulative ordinal bases behave like cumulative PLE, and whitening removes much of the gap.

## 6. Nominal categorical basis geometry

Compare:
1. centered one-hot / contrasts;
2. orthogonal contrasts;
3. frequency-whitened categorical basis.

For selected features, perform controlled equivalent-basis `kappa` sweeps.

Test whether categorical imbalance creates poor conditioning and whether equivalent bases alter training.

The novelty is **basis sensitivity**, not one-hot/whitening themselves.

## 7. Exact numeric–categorical block residualization

For numerical block `N` and categorical block `C`, fit on training data:

`B = argmin ||C - N B||^2`

then:

`C_perp = C - N B`.

Compare:

`[N,C]` vs `[N,C_perp]`.

Because `C = C_perp + NB`, the joint representation is span-equivalent when `N` is retained.

Measure:
- cross-block correlation;
- joint condition number;
- convergence;
- test metric.

Use Diamonds as an anchor if appropriate.

## 8. Secondary only if the core works

### Residual target encoding
Leakage-safe standard TE vs residual TE. Supporting evidence only.

### Frequency-aware categorical optimization
First measure frequency vs gradient/update statistics. Then test preconditioning if justified.

### Datetime/cyclic
Small mechanistic test only:
- centered one-hot of hour/day states;
- full real Fourier basis of the same finite cycle;
- phase-shift controls;
- controlled ill-conditioned transforms.

Full Fourier can be an exact basis test. First-harmonic `sin/cos` is not information-equivalent; treat it as separate inductive bias.

### Cross-atoms
Lowest priority. Keep only if gains are broad and survive matched controls.

## 9. Models and datasets

### Stage 1
MLP on:
- Adult;
- Black Friday;
- Miami;
- Diamonds;
- datasets with genuine ordinal columns.

### Stage 2
Repeat strongest findings on:
- ResNet;
- TabM;
- FT-Transformer if easy.

### Stage 3
Use the fixed Day-2 prospective datasets plus a broader fixed mixed-type subset.

Do not select datasets based on wins.

Use >=3 seeds for expensive runs and preferably 5 for headline results.

## 10. Required controls

Include:
- matched global feature scale;
- random orthogonal transform at `kappa=1`;
- diagonal standardization;
- whitening;
- identical model capacity;
- identical HPO budget;
- standard WD vs no first-layer WD;
- AdamW vs SGD for selected tests.

All transforms must use training data only.

## 11. ICLR success criteria

The strongest Day-3 outcome is:

1. Increasing `kappa` of an **information-equivalent** basis reproducibly worsens learning.
2. Whitening collapses much of the PLE-vs-identity gap.
3. The same phenomenon appears for ordinal and nominal categorical bases.
4. Invariant regularization substantially reduces basis sensitivity without hurting mean performance.

Then the paper becomes:

> **Phenomenon → causal intervention → mechanism → numerical/ordinal/categorical generality → remedy.**

## 12. Kill criteria

Do not rescue the thesis post hoc if:
- controlled conditioning does not affect learning;
- whitening does not reduce gaps;
- ordinal/categorical experiments do not replicate the phenomenon;
- invariant regularization does not reduce sensitivity.

## 13. Deliverable

Create `REPORT_DAY3.md` plus raw results/configs.

Minimum figures:
1. `kappa_vs_performance`
2. `kappa_vs_convergence`
3. `ple_identity_before_after_whitening`
4. `standard_vs_invariant_regularization`
5. `ordinal_local_vs_cumulative`
6. `categorical_basis_conditioning`
7. `block_residualization_diamonds`
8. `summary_geometry_vs_performance`

At the top of the report answer:

1. Did controlled conditioning causally affect learning?
2. Did whitening explain PLE vs identity?
3. Did ordinals reproduce the effect?
4. Did nominal categoricals reproduce the effect?
5. Did invariant regularization reduce sensitivity?
6. What is the strongest defensible ICLR claim now?
7. What single experiment should Day 4 run next?

## Final rule

**Do not optimize for positive results. Optimize for a decisive answer about whether information-equivalent tabular bases change neural learning through optimization geometry, and whether that dependence can be removed.**
