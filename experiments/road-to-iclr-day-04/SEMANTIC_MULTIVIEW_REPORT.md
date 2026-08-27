# Day 4 continuation: semantic multi-view alignment

## Decision first

Do not promote full-row semantic multi-view alignment as a default tabular
representation. In the frozen one-seed screen, the VICReg-style method beats
PLE in only 1/6 TabReD dataset–backbone cells and has a **-0.69% mean RMSE
gain**. Correct cyclic geometry beats a bin-permuted geometry control in 4/6
cells, but the mean gap is only **+0.03%**. The screen finds a real optimization
effect, not convincing value from the declared topology.

This is still a useful Day 4 result. It directly tests the proposed
contrastive direction without LightGBM, an anchor model, target-derived feature
types, or test-time selection. It also identifies the likely failure mode:
forcing complete row representations together is too blunt when only three to
five fields have declared cyclic semantics.

## What was tested

Each numerical field receives two deterministic, train-only charts:

1. **PLE view:** 16-bin quantile piecewise-linear encoding.
2. **Topology view:** empirical-rank cosine modes for ordinary ordered fields;
   exact Fourier modes for fields with a declared cycle.

Nominal fields use learned category embeddings and receive no invented
ordering. Binary fields receive scalar token projections. The cyclic
declarations come directly from the official TabReD preprocessing scripts:

- Weather: weekday, day-of-month, minute-of-day, hour-of-day, month;
- Cooking Time: weekday, minute-of-day, hour-of-day.

The multi-view model has separate PLE and topology tokenizers but shares the
entire prediction backbone. It applies the supervised loss to both branches,
averages their predictions, and optionally applies a VICReg-style loss to the
paired row latents. This is chart alignment, not the corruption augmentation
used by SCARF or SAINT.

The necessary controls are:

- `multiview_noalign`: same two branches and prediction average, no alignment;
- `multiview_wrong`: same VICReg objective, but cyclic phase cells are
  deterministically permuted before the Fourier map;
- `topology` and `topology_wrong`: single-view checks run on Weather.

The wrong control preserves dimension and parameter count while destroying
the declared ring adjacency. PLE and topology-only models are parameter
matched. A two-view model necessarily adds a second tokenizer, so conclusions
must be made against both PLE and `multiview_noalign`.

## Frozen screening protocol

- Official TabReD temporal train/validation/test partitions;
- deterministic subsamples of 50,000 / 15,000 / 15,000 rows;
- seed `20260827`;
- MLP, ResNet, and a field-token FT-Transformer;
- 16 basis coordinates per numerical field and 16-dimensional field tokens;
- at most 20 epochs with validation early stopping;
- no tree model, pseudo-label, anchor prediction, or target-derived topology.

This is a fixed-budget pilot, not the recovered official full-budget TabReD
configuration and not a statistical comparison. Test results were inspected
only after the method/control panel was complete. Any new variant must use
validation for decisions and must not be retroactively presented as
confirmatory on these two datasets.

## Results

Lower RMSE is better. Parentheses show gain relative to the paired PLE run.

| Dataset | Backbone | PLE | No alignment | VICReg alignment | Wrong-geometry alignment |
| --- | --- | ---: | ---: | ---: | ---: |
| Weather | MLP | 1.61870 | 1.61492 (+0.23%) | **1.60188 (+1.04%)** | 1.60272 (+0.99%) |
| Weather | ResNet | **1.60037** | 1.61106 (-0.67%) | 1.60535 (-0.31%) | 1.60740 (-0.44%) |
| Weather | FT-Transformer | **1.59903** | 1.62883 (-1.86%) | 1.61195 (-0.81%) | 1.61357 (-0.91%) |
| Cooking Time | MLP | **0.48045** | 0.48424 (-0.79%) | 0.48704 (-1.37%) | 0.48717 (-1.40%) |
| Cooking Time | ResNet | **0.48217** | 0.48637 (-0.87%) | 0.48645 (-0.89%) | 0.48643 (-0.88%) |
| Cooking Time | FT-Transformer | **0.48314** | 0.48792 (-0.99%) | 0.49184 (-1.80%) | 0.49135 (-1.70%) |

Additional Weather single-view results are negative. The topology chart loses
to PLE by 0.81%, 4.65%, and 3.72% for MLP, ResNet, and FT-Transformer. Its mean
gain is -3.06%, while the wrong topology has -3.10%.

Across all six shared cells:

- `multiview_noalign`: 1/6 wins, -0.82% mean gain;
- `multiview_vicreg`: 1/6 wins, -0.69% mean gain;
- `multiview_wrong`: 1/6 wins, -0.72% mean gain;
- correct versus wrong VICReg: 4/6 wins, +0.03% mean.

The small difference between correct and wrong alignment is the central
falsification. The Weather MLP gain cannot be assigned to cyclic semantics
when the wrong-cycle model retains nearly all of it.

## Interpretation

Three effects are entangled in a naive positive result: a second tokenizer,
prediction averaging, and representation alignment. The controls separate
them. On Weather MLP, averaging supplies +0.23%, alignment raises this to
+1.04%, but correct topology contributes only about +0.05% over the wrong
control. On Cooking Time, even the unaligned second view is harmful, and
alignment worsens it further.

The likely structural problem is granularity. The auxiliary loss aligns a
complete row latent built from 103 or 192 fields even though only five or three
fields have a declared ring. It therefore regularizes anonymous ordered fields,
nominal fields, and genuine cyclic fields through one scalar objective. That is
not the local semantic intervention proposed by “fields, not features.”

A GAN is lower priority after this result. It would introduce a discriminator
and a distribution-matching objective but would not solve the granularity
problem or make paired equivalent rows more informative. The paired views
already provide exact correspondence; adversarial distribution matching would
discard that supervision.

## The next high-value experiment

If this line continues, test **field-local chart distillation**, not another
full-row contrastive loss:

```text
PLE token_j ---------> shared downstream backbone
       \
        + zero-initialized gated semantic residual_j
       /
topology token_j -- stop-gradient/local alignment
```

Only explicitly declared fields receive the residual and local alignment.
Keep the deployed path single-view, initialize its gate at zero so it starts as
exact PLE, and compare:

1. PLE;
2. parameter-matched PLE with an ordinary residual adapter;
3. semantic residual without local alignment;
4. semantic residual with local alignment;
5. permuted-geometry residual with the same alignment.

Predeclare a validation gate: promote only if the semantic residual beats PLE,
the ordinary adapter, and wrong geometry on validation in at least five of six
Weather/Cooking architecture cells. If it clears that gate, run seeds
`20260827`, `20260828`, and `20260829` and transfer unchanged to Delivery ETA.
Otherwise stop this branch. This preserves the innovative cross-chart idea
while responding directly to the failure observed here.

### Follow-up outcome

The gated field-local experiment was run and did not pass. It cleared 3/6
validation cells rather than the required 5/6; mean gain over PLE was +0.004%,
and correct geometry was -0.009% worse than wrong geometry on average. In
accordance with the protocol, no Weather/Cooking test metrics or Delivery ETA
transfer were computed. The full validation-only record is in
[`FIELD_LOCAL_DISTILLATION_REPORT.md`](FIELD_LOCAL_DISTILLATION_REPORT.md).
The contrastive branch is therefore closed for Day 4.

## Reproduction

The runner is [`semantic_multiview_pilot.py`](semantic_multiview_pilot.py), its
focused tests are [`test_semantic_multiview.py`](test_semantic_multiview.py),
and the aggregation entry point is
[`analyze_semantic_multiview.py`](analyze_semantic_multiview.py).

From the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 \
/home/byunhanjoon/adaptive_embedding/exploratory/.venv-tabm/bin/python \
  -m pytest -q -p no:cacheprovider \
  experiments/road-to-iclr-day-04/test_semantic_multiview.py

/home/byunhanjoon/adaptive_embedding/exploratory/.venv-tabm/bin/python \
  experiments/road-to-iclr-day-04/analyze_semantic_multiview.py
```

The machine-readable conclusion is
[`results/semantic_multiview_summary.json`](results/semantic_multiview_summary.json),
and all paired cells are in
[`results/semantic_multiview_comparisons.csv`](results/semantic_multiview_comparisons.csv).

## Closest-work boundary

- [SCARF (ICLR 2022)](https://openreview.net/forum?id=CuV_qYkmKb3) forms
  positive pairs using random feature corruption.
- [SAINT](https://arxiv.org/abs/2106.01342) combines row/column attention with
  contrastive pretraining.
- [Numerical feature embeddings (NeurIPS 2022)](https://proceedings.neurips.cc/paper_files/paper/2022/hash/9e9f0ffc3d836836ca96cbf8fe14b105-Abstract-Conference.html)
  establishes PLE and periodic numerical embeddings.

The differentiated hypothesis here is not contrastive learning or Fourier
features by themselves. It is alignment between schema-equivalent,
semantics-licensed charts of the same row, with wrong-geometry controls. This
screen says the full-row version is not valuable enough; only the field-local,
PLE-preserving version remains worth one gated test.
