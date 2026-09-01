# Disjoint seed and nuisance-menu repeat

Status: frozen after the initial and six-source confirmation cover outcomes,
before predictions under this repeat were computed.

Use one representative task from each confirmation source group:
Compustat direction, credit-card default, credit-card fraud, HELOC, KDD17
stock return, and Polish bankruptcy three-year. Repeat all five model families
with disjoint model seeds `505,606,707,808`, feature-permutation generator seed
`190773`, and category-map generator seed `1821571`. All data splits,
subsamples, transformations, losses, and hyperparameters are otherwise fixed.

Run the same exact strength-1 budget-four and strength-2 budget-16 analyses.
The repeat passes if each cover beats its frozen equal-budget comparators in at
least four of six source groups and in pooled material-cell mean. This is a
conditional robustness repeat, not an untouched discovery set.

