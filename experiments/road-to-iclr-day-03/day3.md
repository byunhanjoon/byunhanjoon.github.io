# The same table can be an easier—or harder—learning problem

*Road to ICLR, Day 3*

On Day 2, one small change made Adult accuracy jump from **85.98% to 87.27%**.
The model, split, and training budget stayed the same. I only changed how two
columns—capital gain and capital loss—were shown to the model.

That is a **1.29 percentage-point gain**. On a mature tabular benchmark, it is
large enough to demand an explanation.

My first interpretation was that the new “identity” features gave the model
useful extra information about exact values. Day 3 revealed a more interesting
possibility:

> A representation can preserve every bit of information and still make the
> same learning problem much easier or much harder for a neural network.

This is not merely a story about Adult, or about one-hot encoding. I can produce
the effect deliberately for numerical, nominal, and ordinal features. I can
mostly remove the Adult identity gap by putting equivalent representations into
the same coordinate system. But I do not yet have a new, generally successful
optimizer or representation method. One promising method passed an internal
benchmark and then failed on seven untouched datasets.

That distinction—between a phenomenon we understand and a remedy we do
not—is the main result of Day 3.

## The one-minute version

- **The Adult result was real.** Encoding capital gain and loss by exact value
  raised mean accuracy from 85.98% to 87.27% in the original five-seed test.
- **It was not just a zero/non-zero flag.** Indicators for the dominant values
  did nothing, and the benefit grew gradually as dozens of exact values were
  retained.
- **Equivalent inputs are not equally trainable.** Invertible changes of basis,
  which preserve all information, caused large and orderly losses under a fixed
  training budget.
- **Coordinate geometry explains the controlled Adult gap.** Whitening reduced
  the mean absolute gap between exact-value identity features and PLE from
  0.210 to 0.018 points—a 91.2% reduction. Alignment made the inputs and scores
  identical.
- **Initialization explains much, but not all, of the failure.** Function-matched
  initialization removed about 95% of the high-condition-number damage, yet
  ordinary AdamW separated equivalent models after its first update.
- **The result is broad; the cure is not.** The sensitivity appears across
  numeric, nominal, and ordinal representations and across MLP and ResNet
  models. Whitening and canonicalization are strong diagnostic controls, but
  they are not a universally better encoder.
- **The next question is locality.** After equalizing scale and conditioning,
  does a basis still help because each coordinate describes a nearby, sparse,
  human-sized distinction?

## A table is not the input the optimizer sees

Imagine giving someone a map of Seoul. North can point up, right, or diagonally.
The city has not changed. Distances and routes can be recovered from any of the
maps. But some maps are easier to read.

Tabular representations have the same issue. Suppose a feature block is a row
vector `z`. We can replace it with

```text
z  →  zA
```

where `A` is invertible. Nothing has been lost: multiplying by `A⁻¹` recovers
the original input exactly. A sufficiently flexible first layer can compensate
for this change, so the set of functions the network can express is also the
same.

But gradient descent does not search over functions directly. It moves through
parameter coordinates. Stretch one input direction and squeeze another, and
the same-sized parameter update now means something different in function
space.

This gives us two notions of equivalence:

```text
same information  ≠  same training path
```

The first is an algebraic fact. The second depends on initialization,
regularization, the optimizer, and the number of updates we can afford.

## A knob for making coordinates difficult

To test this cleanly, I started from a whitened feature block and applied an
invertible transform. I varied only its **condition number**, written `κ`.

You can think of `κ` as the ratio between the most stretched and most squeezed
directions:

```text
κ = 1       all directions have equal scale
κ = 3,000   one direction is 3,000× more stretched than another
```

The geometric mean scale was held fixed. The transformed block kept the same
rank, dimension, and information. The split, model, hyperparameters, and update
budget were also fixed.

As `κ` increased, performance worsened monotonically in the main numerical MLP
experiments:

| Dataset | Easy coordinates | Hard coordinates | Change |
| --- | ---: | ---: | ---: |
| Adult | 85.755% accuracy | 84.539% | −1.216 points |
| California Housing | 0.5004 RMSE | 0.5865 | 17.2% worse |
| Diamond | 0.1482 RMSE | 0.1825 | 23.2% worse |

The endpoint also replicated with a ResNet. Adult lost 0.778 points at
`κ = 1,000`; California and Diamond RMSE worsened by 11.7% and 11.1%.

![Performance degrades as equivalent numerical coordinates become more ill-conditioned](results/day3/figures/numeric_kappa_vs_metric.png)

This experiment matters because “perhaps one encoding contains more
information” is no longer available as an explanation. The two inputs can be
converted into each other exactly.

The same pattern appeared outside numerical PLE blocks:

- For nominal categories, hard coordinates cost Adult MLP **0.968 accuracy
  points** and worsened Diamond MLP RMSE by **14.3%**.
- For true ordinal state spaces, they cost Adult MLP **0.416 points**, worsened
  Diamond MLP RMSE by **13.2%**, and worsened Diamond ResNet RMSE by **7.9%**.
- In a later 30-dataset benchmark, `κ = 1,000` harmed AdamW on **93.3% of 360
  paired cells**.

So the phenomenon is not “PLE is bad” or “one-hot is good.” It is broader:
neural tabular training can be sensitive to the coordinates used to describe
the same feature space.

## Returning to the Adult identity result

The Day 2 identity representation gave each observed capital-gain or
capital-loss value a local coordinate. PLE instead described the same values
through cumulative threshold features.

There were three plausible reasons identity could help:

1. **Point-mass awareness.** Adult has large spikes at exact values, especially
   zero. Smooth numerical encoders can treat those spikes awkwardly.
2. **Resolution allocation.** Quantile knots spend their capacity according to
   rank. Repeated values can collapse knots, while rare but useful values may
   receive little individual resolution.
3. **Coordinate geometry.** A local exact-value code may simply be easier for
   finite-budget optimization than a correlated cumulative code.

The first result already ruled out the simplest point-mass story. A flag for
`capital gain = 0`, `capital loss = 0`, and `hours = 40` changed accuracy by
essentially zero. Keeping one, two, or four common values also gave little
benefit. The gain began around eight values and continued growing through
dozens of levels. With all exact values, gains were **+1.276**, **+1.208**, and
**+1.075 points** for MLP, ResNet, and TabM.

That looks like a distributed collection of useful exceptions, not one magic
value.

Day 3 then isolated geometry. For capital gain and loss, I built identity and
PLE coordinates that span exactly the same observed-state space. Their
train/validation/test reconstruction errors were around `10⁻¹⁴`: numerically,
no information was missing from either representation.

In raw form, identity was better by **0.210 ± 0.113 accuracy points** across
five paired seeds. After ordinary per-column standardization, the signed gap
was essentially gone. After full whitening, the mean absolute gap fell from
0.210 to **0.018 points**, a **91.2% reduction**. Finally, whitening plus an
orthogonal alignment made the design matrices—and all five paired scores—the
same.

![The controlled PLE–identity gap nearly disappears after canonicalization](results/day3/figures/ple_identity_gap_before_after_whitening.png)

This is an unusually clean closure test. In this controlled construction, the
remaining PLE-versus-identity difference was not hidden predictive information.
It was the coordinate system.

There is an important numerical distinction. The exact same-space experiment
explains a **0.210-point controlled gap**, not automatically the entire original
**1.29-point gain**. The original identity view was appended to a larger model
and changed more than a single square basis. Resolution allocation, redundancy,
and regularization may still contribute. Saying “whitening explains identity”
without this qualification would be too strong.

## Where does the sensitivity enter training?

An invertible input transform can be canceled by the first layer. If one model
receives `zA`, its first-layer weight can be initialized with the corresponding
inverse transform so that both networks compute the same function at step zero.

This separates two sources of trouble:

- **Initialization mismatch:** standard random initialization represents a
  different distribution of functions after the input basis changes.
- **Update mismatch:** even if two models begin as the same function, AdamW may
  move them along different function-space paths.

The trajectory experiment found that initialization mismatch explains about
**95% of the `κ = 3,000` fixed-budget damage**. That is a major clue. Much of
what looked like an optimization failure was already present before learning
began.

But the models did not remain equivalent. With function-matched starts,
ordinary AdamW diverged after the first update. Its coordinatewise moments and
weight decay do not transform like the input basis does.

This leads to a more precise account:

```text
input basis
   ↓
initial function distribution     ← explains most of the large endpoint gap
   ↓
optimizer's coordinate system     ← breaks equivalence after one update
   ↓
finite-budget predictor
```

Both parts are real. Their relative importance depends on the model and basis.

## Canonicalization is a good control, not yet a new method

If arbitrary coordinates are the problem, a natural response is to map every
equivalent representation to one canonical coordinate system before training.
Exact anchor canonicalization does close the controlled gaps. Sketch-based and
input-natural approaches also passed broad remedy checks.

But whitening is not automatically the best practical encoding.

Why? Whitening equalizes second moments, but a representation also has
**locality** and **sparsity**. In a local code, changing one category or crossing
one interval changes only a few coordinates. A dense rotation spreads that
event across nearly every coordinate. Both bases can have `κ = 1`, yet one may
be easier for a finite neural network to use.

The natural ordinal encodings illustrate this. Their median absolute difference
after normalization was only 0.38% in the broad benchmark, and whitening did
not produce a simple universal ranking. Condition number is therefore a strong
causal lever, but not a complete theory of representation quality.

The attempted “invariant” activation penalty also failed. Aggregate basis
sensitivity was 0.02387 with the penalty versus 0.02324 under standard weight
decay; even removing first-layer weight decay only reduced it to 0.02117. A
regularizer that sounds invariant on paper is not enough unless the whole
training dynamics are invariant.

## Useful failures

Several negative results narrowed the story.

**Removing numerical–categorical dependence was not the answer.** On Diamond,
block residualization reduced normalized cross-correlation from `3.92 × 10⁻²`
to `1.97 × 10⁻⁸`, and top canonical correlation from 0.8551 to about
`5.6 × 10⁻⁷`. The geometry changed exactly as intended. RMSE still became 0.9%
worse. Simple categorical standardization did better than the sophisticated
residualization.

**Frequency alone was too weak.** Frequency features and residual target
encoding did not provide a general explanation for the identity gains.

**The cyclic result was only a synthetic check.** No released anchor dataset
had a genuine datetime feature, so the Fourier-versus-one-hot equivalence test
cannot support a real-world datetime claim.

**Early trajectory drift was descriptive, not predictive.** It did not pass the
held-out-dataset gate for deciding which representation would win.

**The proposed performance method did not survive external testing.** Selective
Measure-Orbit looked encouraging internally: it improved proper loss by 1.103%
across a frozen 21-dataset internal confirmation and raised Adult accuracy by
1.013 points in its one-fit comparison. But against an exactly update-matched
two-seed ordinary TabM ensemble on seven untouched datasets, it was worse by
**0.521%**, with a 95%
dataset-bootstrap interval of **[−0.831%, −0.195%]**. It won only 7 of 21
seed-level comparisons.

That failure is scientifically useful. A second seed is a very strong baseline,
and internal selection can mistake dataset-specific wins for a general method.
Measure-Orbit is now a diagnostic, not a performance claim.

## What we know

The evidence now supports five statements.

1. **Finite neural training is not invariant to equivalent tabular bases.** The
   effect is causal under controlled invertible transformations.
2. **The effect is not confined to one feature type.** It appears in numerical,
   nominal, and ordinal blocks, with MLP and ResNet replication.
3. **Adult's exact PLE–identity gap is mostly geometric.** Canonicalization
   nearly removes it, and exact alignment closes it.
4. **Initialization is a large part of the mechanism.** Function matching
   removes most of the extreme-conditioning loss.
5. **Existing invariance ideas can diagnose or close controlled gaps, but no
   new practical remedy here reliably beats a fair ensemble baseline.**

## What we still do not know

The open questions are now sharper than they were at the start of the day.

- How much of the original Adult +1.29-point gain comes from conditioning,
  resolution allocation, locality, or the regularization effect of appending a
  redundant view?
- Among equally conditioned bases, which properties predict learnability:
  sparsity, locality, coherence with the first layer, or something else?
- Why does function-matched initialization remove most—but not all—of the
  damage, and when does optimizer mismatch become dominant?
- Can a cheap canonicalizer preserve the helpful local structure without using
  expensive exact whitening or leaking validation choices?
- Does the core basis-sensitivity result replicate on a genuinely untouched
  benchmark suite, with the experiment frozen before any result is seen?
- Can any representation method beat the compute-matched baseline of simply
  training another seed?

## What I would try next

The most informative next experiment is an **isometric locality sweep**.

Start from the same whitened block, so every candidate has `κ = 1`. Then move
gradually from a sparse local basis to a dense random rotation while preserving
all pairwise distances and all information:

```text
local sparse basis
      ↓ progressively rotate
partly mixed basis
      ↓
dense Haar-random basis
```

Use function-matched initialization so the networks also compute the same
function at step zero. Measure:

- test performance under a fixed update budget;
- the first 100 mapped function-space updates;
- activation and gradient concentration;
- coordinate sparsity and coherence with the first layer;
- MLP, ResNet, and TabM behavior on Adult, Black Friday, and Diamond.

This experiment removes the two explanations already established—conditioning
and initial function mismatch. If performance still changes smoothly with
rotation density, locality becomes a real mechanism rather than an intuition.
If the gap vanishes, the current story becomes simpler: conditioning and
initialization were sufficient.

For Adult specifically, I would then run a small factorial experiment that
changes four ingredients separately:

| Ingredient | Controlled comparison |
| --- | --- |
| Point masses | exact zero flags vs no flags |
| Resolution | equal-rank PLE vs exact-state partition |
| Conditioning | raw vs standardized vs whitened |
| Locality | local basis vs dense orthogonal rotation |

Each cell should use matched parameter count, matched initialization where
possible, paired seeds, and the same update budget. That would turn “identity
worked” into an attribution: how many accuracy points came from each mechanism,
and which interactions matter.

Only after that would I propose another performance method. The method should
be evaluated against an update-matched seed ensemble from the beginning, not
added as a late baseline.

## A practical checklist

These results are useful even without a new algorithm. When a tabular encoding
improves performance:

1. Check whether the two representations contain the same information by
   reconstructing each from the other on train, validation, and test.
2. Report rank, per-column scale, singular values, and condition number.
3. Compare raw, standardized, and whitened versions.
4. If bases span the same space, align them and rerun paired seeds.
5. Separate “more information” from “easier coordinates.”
6. Plot performance against a controlled basis transformation, rather than
   comparing only two named encoders.
7. Match the initial function when studying optimizer behavior.
8. Treat locality and sparsity as possible mechanisms even at equal condition
   number.
9. Compare any costly representation method with the same compute spent on
   another ordinary seed.
10. Freeze one untouched external benchmark before promoting a diagnostic into
    a method claim.

The broad lesson is simple: a table does not have one natural neural
representation. Two encodings can say exactly the same thing while asking the
optimizer very different questions. Day 3 found a reliable way to expose that
gap. The next step is to learn which parts of a useful coordinate system—scale,
locality, sparsity, or initialization—we should preserve on purpose.

## Evidence and reproducibility

- [Full Day 3 report](REPORT_DAY3.md)
- [Broad benchmark report](BROAD_BENCHMARK_REPORT.md)
- [Trajectory decomposition](TRAJECTORY_DECOMPOSITION_REPORT.md)
- [Measure-Orbit internal report](MEASURE_ORBIT_REPORT.md)
- [Untouched external Measure-Orbit test](EXTERNAL_MEASURE_ORBIT_REPORT.md)
- [Concise experiment brief](day3_agent.md)
- [Raw results and figures](results/day3/)
