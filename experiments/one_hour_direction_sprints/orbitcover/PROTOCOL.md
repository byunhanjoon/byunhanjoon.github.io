# OrbitCover predictive-utility sprint protocol (frozen before analysis)

## Question

At an equal 16-fit budget, does OrbitCover's large reduction in error to its
symmetrized predictor translate into a material improvement in *actual held-out
predictive loss* over an ordinary independent-seed ensemble?

This endpoint is intentionally different from the existing OrbitCover paper's
primary quotient-residual claim.  It tests whether the direction also clears
the stronger "performant predictor" interpretation.

## Frozen evidence panel

- Reuse the audited final-closure Experiment A construction averages in
  `experiments/final_closure/summaries/experiment_a_cells.csv`.
- The panel contains 12 datasets, 3 train/validation/test splits, and 4 neural
  tabular backbones (144 cells), with 512 estimator constructions per
  method/budget cell.
- Primary comparison: `OC2-COUPLED` versus `CANONICAL-INDEPENDENT` at B=16.
- Diagnostic comparisons: `OC2-INDEPENDENT` and `SRS-JOINT` versus the same
  canonical baseline, plus every recorded budget from 4 through 64.
- `predictive_loss` is the actual task loss computed against `test_y` by the
  frozen final-closure evaluator.  No quotient residual is substituted.
- Aggregation is paired within dataset/split/backbone.  Fractional improvements
  are averaged equally by dataset; a deterministic dataset-cluster bootstrap
  supplies the interval.

## Go gate for raw predictive performance

All conditions must hold for OC2-coupled at B=16:

1. integrity and the full 144-cell panel pass;
2. equal-dataset mean fractional predictive-loss improvement is at least
   `0.005` (0.5%);
3. the dataset-clustered 95% interval has positive lower endpoint;
4. at least 60% of cells and 8 of 12 datasets improve;
5. every one of the four architecture means is nonnegative;
6. no dataset degrades by more than 1%.

The paired per-cell oracle (taking the better method after seeing test loss) is
reported only as an upper bound on gate/selection headroom and cannot make the
primary gate pass.

The analysis is capped at one wall-clock hour.  Cached predictions are reused
because retraining would answer the identical frozen 140k-fit experiment less
precisely, not because runtime is being filled artificially.
