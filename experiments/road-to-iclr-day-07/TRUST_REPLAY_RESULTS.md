# Residual trust replay — result

Status: **COMPLETE — FROZEN FEASIBILITY GATE FAILED**

## Verdict

Continuous scalar shrinkage is not the Day-7 lead. It retained positive gains,
but state-CV attenuation was worse than selecting a full-strength operator and
created an extra harmful task-split cell. The simple analytic oracle has only
small headroom over the full-strength oracle, so this is not merely a poor grid.

| Rule | Source-balanced gain | Harmful cells | Median trust |
|---|---:|---:|---:|
| selected full strength | +0.04823 | 4/45 | 1.0 |
| selected binary fallback | +0.04823 | 4/45 | 1.0 |
| selected fractional shrinkage | +0.04184 | 5/45 | 0.9 |
| test oracle, full strength | +0.05799 | 0/45 | 1.0 |
| test oracle, continuous shrinkage | +0.05846 | 0/45 | 1.0 |

The shrinkage rule was positive on all five source groups but lost `0.00708`
mean standardized-MSE gain relative to both deployable controls. Its selected
trust values were genuinely fractional—28/45 cells chose below one—so the
failure is informative rather than a disguised tie.

## Interpretation

The exact trust parabola remains correct, but estimating a scalar is not the
main statistical bottleneck. Inner state folds often attenuate operators whose
outer effects are strongly positive, especially in the string source. The
larger headroom lies in choosing the transfer relation and determining whether
the future state split resembles the validation state split. A paper should
therefore focus on a *certificate for conditional inductive-bias value* and its
shift assumptions, not sell scalar gating as the contribution.

## Frozen gates

- PASS: positive source-balanced shrinkage gain;
- FAIL: improvement over full-strength selection;
- FAIL: no more harmful cells than binary selection;
- PASS: nontrivial median trust.

Decision: do not promote scalar shrinkage. Continue with the separately frozen
neural-base transfer diagnostic and retain this result as a negative control.

