# Depth and Regime-Count Scaling

Status: **execution scales; discovery does not. Known categorical routing is
predictively stable through eight regimes but misses strict functional recovery.**

## Program depth 2/4/6/8

Six tasks per depth and five noisy discovery seeds compare oracle exact
execution, a width-128 chain beam search compiled into the exact executor, and
an MLP. All 720 records are finite.

| Depth | Oracle 4× NRMSE | Beam 4× NRMSE | Beam functional recovery | MLP 4× NRMSE |
|---:|---:|---:|---:|---:|
| 2 | 0 | 0.00044 | 0.933 | 0.721 |
| 4 | 0 | 0.598 | 0.267 | 0.869 |
| 6 | 0 | 0.746 | 0.300 | 0.787 |
| 8 | 0 | 1.027 | 0.067 | 1.037 |

Oracle execution stays exact through depth 8. Beam discovery fails both
depth-8 gates and already degrades sharply at depth 4. Syntactic recovery is
zero at depths 6 and 8 even when partial functional/operator recovery remains.
The general deep-program induction claim is rejected.

## Regime count 1/2/4/8

Four tasks per count and five seeds use known one-hot categorical regimes,
balanced training, increasingly few rows per expert, a skewed 4× test, and
short-program discovery within each expert. All 960 records are finite.

| Regimes | Hard router OOD NRMSE | Functional recovery | Operator accuracy | Neural MoE OOD |
|---:|---:|---:|---:|---:|
| 1 | 0.00109 | 0.65 | 1.000 | 2.623 |
| 2 | 0.00098 | 0.60 | 1.000 | 0.568 |
| 4 | 0.00218 | 0.25 | 0.931 | 0.867 |
| 8 | 0.00375 | 0.20 | 0.969 | 1.009 |

At eight regimes, known hard routing remains roughly 270× better than neural
MoE and single-program controls and passes four of five gates. It fails the
strict requirement that at least 70% of task-seed runs have whole-mixture NRMSE
below 10⁻³ (observed 20%). This is a predictive partial pass, not exact
functional recovery. It also assumes observed categorical labels and therefore
does not repair the failed real temporal router.
