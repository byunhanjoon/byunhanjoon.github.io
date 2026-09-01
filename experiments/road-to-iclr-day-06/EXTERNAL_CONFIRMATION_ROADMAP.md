# Outcome-contingent external confirmation roadmap

Status: frozen before H3–H9 completion; this is a next-stage design, not Day-6
evidence and not a ranking score.

## Route A — Semantic Arithmetic survives through H7 or H9

Run a new, randomized panel rather than extending the current runtime-ordered
split:

- at least 12 independent datasets: balanced binary, imbalanced binary,
  multiclass, count regression, and continuous regression; include missingness,
  high-cardinality categoricals, and both narrow/wide tables;
- MLP, ResNet, dense-stem FT-Transformer, a native categorical tokenizer, and
  one independently implemented tabular architecture;
- two accelerator generations plus a CPU/BLAS reference stack;
- FP32, interface-only IEA64, exact canonical gather, and a lower-cost
  compensated/pairwise interface arm;
- randomized arithmetic-arm order for timing, independently randomized bundle
  order for train/test splitting, modern schedules/early stopping, and new
  seeds chosen before any run.

Primary endpoints should be dataset-level material-survival probability,
post-breach paired log-MSE ratio, canonical loss/regret, wall-time/energy cost,
and the frequency with which canonical gathering is unavailable in realistic
pipelines.  Instrument a small declared subset at every interface evaluation
to compare actual float32 rounding-cell margins with the checkpoint-level H7
survival surrogate.  The paper should lead with canonical gathering when it is
available and treat IEA64 as a causal/local fallback.

Promotion condition: the chosen H7/H9 claim replicates on at least 8/12
datasets, no material canonical-loss harm appears, and the hardware interaction
is reported rather than pooled away.

## Route B — H4/H5 Semantic Shadowing passes

Freeze a new optimizer grid and new seeds on at least 12 datasets.  Compare the
two-epoch schema shadow against equal-cost baselines:

- two ordinary pilot seeds;
- early canonical loss slope;
- gradient norm/noise-scale summaries;
- weight-space sharpness or perturbation probes; and
- a random nonsemantic input perturbation matched in initial prediction MSE.

The target should be independent-seed prediction variance and downstream
selection regret under a held-out optimizer menu.  A useful paper result must
show incremental forecast value or compute savings beyond these baselines, not
merely reproduce H4's same-source correlation.

Promotion condition: prespecified dataset-level transfer on at least 8/12
datasets, positive incremental AUROC/correlation over the strongest equal-cost
baseline, and a concrete reduction in seed-search compute or selection regret.

## Route C — neither Day-6 successor passes

Keep H1 as a numerical-semantics mechanism/boundary study, not the lead ICLR
project.  Prefer the Day-5 OrbitCover incumbent and spend new compute on its
two decisive weaknesses: independent rather than coupled gains and held-out
selection utility.  Do not invent another rule from the completed H3 matrix.

This branch structure is deliberately fixed before the final outcomes.  A
failed route is not rescued by relaxing Day-6 thresholds; it changes which
independent experiment is worth funding next.
