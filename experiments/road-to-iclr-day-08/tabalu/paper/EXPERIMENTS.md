# Experiments

The frozen Phase-A protocol is defined by `configs/phase_a_pilot.json`. It uses
100 independently generated tasks and five training seeds. Full details and
baseline budgets live in `README.md` and `configs/`.

The exact-execution intervention is defined by
`configs/exact_execution_ablation.json`: 30 tasks and five seeds with graph and
operands fixed, comparing exact nodes, learned neural nodes, and whole MLPs.

The heterogeneous typed panel is defined by `configs/phase_f_typed.json`: 16
tasks, five seeds, six models, and both IID and joint future/4× evaluation.

The residual continuum is defined by `configs/phase_g_residual.json`: ten tasks,
five seeds, five non-symbolic fractions, five models, and both IID and 4× tests.

The first real temporal pilot is defined by `configs/real_bike_temporal.json`.
It uses a checksum-pinned UCI archive, five seeds, IID holdouts within 2011, all
of 2012 as future data, six models, and explicit target-leakage exclusions.

The general numeric pilot is defined by `configs/general_pilot.json`: three
small real datasets, five seeds, regression and classification, and six models.

Depth and regime scaling are defined by `configs/depth_scaling.json` and
`configs/regime_scaling.json`, covering depths and regime counts 2/4/6/8 and
1/2/4/8 respectively with five seeds.
