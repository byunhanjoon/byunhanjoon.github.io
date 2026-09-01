# Frozen protocol H2 — Precision-Delay Law

Status: **FROZEN BEFORE H2 OUTCOMES**  
Freeze date: 2026-08-28 (Asia/Seoul)

## Hypothesis

H1 established a causal binary result: float32 semantic reduction-order error
can be amplified, while a float64 interface accumulator closes the paired path.
H2 tests whether this is a quantitative dynamical law rather than a lucky
binary implementation effect.

Before nonlinear saturation, suppose aligned prediction discrepancy satisfies

`D_t^(1/2) approximately C u exp(lambda t)`,

where `u` is effective interface unit roundoff and `lambda` is a finite-time
amplification exponent shared by still-nearby trajectories.  The first epoch at
which `D_t` crosses threshold `epsilon` is then

`tau_epsilon approximately [log(sqrt(epsilon)/C) - log u] / lambda`.

Thus increasing precision should delay separation approximately linearly in
`-log u`; it need not improve the eventual saturated path or predictive loss.

## Frozen matrix

Use the same three datasets and three models as H1, seeds 7101 and 7202, two
nonidentity schema views, and 30 epochs.  Record every epoch.  Only the first
affine interface changes:

- bfloat16 operands/output (`u ~= 2^-8`),
- float16 operands/output (`u ~= 2^-11`),
- float32 ordinary interface (`u ~= 2^-24`),
- float64 accumulation cast to float32 (`u ~= 2^-53` before the cast).

Parameters, optimizer states, downstream arithmetic, minibatches, dropout
tape, and objectives remain float32 and matched.  TF32 is disabled.

## Frozen endpoints and gates

The primary hitting threshold is aligned validation prediction MSE `1e-5`.
An arm that never crosses by epoch 30 is right-censored at 31.

H2 is supported if:

1. in at least 2/3 FT-Transformer datasets, seed/view median hitting epochs are
   ordered `bfloat16 <= float16 <= float32 < float64`;
2. pooled FT Spearman correlation between `-log2(u)` and hitting epoch is at
   least 0.8 when precision medians are the four observations;
3. float64 remains exactly closed in all 9 dataset×model cells; and
4. at least 5/6 MLP/ResNet fp32 cells remain below `1e-8` at epoch 30, retaining
   a stable-architecture boundary.

If gates 1–2 fail but gate 3 passes, H1 remains confirmed and H2 is discarded.
If low precision changes loss materially, that limits intervention utility but
does not by itself falsify the delay mechanism.

## Novelty boundary

Mixed-precision training, floating-point error bounds, and extreme numerical
sensitivity of neural optimization are established.  The candidate novelty is
the precision-indexed hitting-time law for *exact semantic schema conjugacies*
with a common random tape and an interface-local precision intervention.
