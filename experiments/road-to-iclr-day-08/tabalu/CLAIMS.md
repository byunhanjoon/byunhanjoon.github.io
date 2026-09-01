# Claim–Evidence Matrix

| Claim | Required evidence | Status |
|---|---|---|
| Exact execution extrapolates | 100-task arithmetic pilot plus arithmetic benchmark | Partial: controlled chain-family pilot passed; external benchmark TODO |
| Exact execution itself matters | Exact-vs-neural primitive ablation | Passed: oracle-graph neural primitives grew 49.8× by 8× shift; exact stayed at zero error |
| Program remains executable | Soft, hard, compiled, and coefficient-only compiled results | Passed on clean Phase A; shifted/corrupted panels TODO |
| Program induction recovers structure | Feature/operator/exact-graph metrics under clean and corrupted conditions | Limited to short chains: depth-2 passed, depth-8 beam recovery failed |
| Operand inference handles feature noise | Measurement-noise sweep with bounded correction controls | Rejected in current setting; component excluded |
| Regime routing handles shifts | Synthetic regime panel and neural-MoE control | Rejected for current real temporal form: categorical synthetic passed, UCI season router catastrophically failed |
| Shared structure beats separate regime programs | Scarce-regime temporal coefficient panel | Not established; predictive tie, context-conditioned variant rejected |
| Typed operators help heterogeneous data | Typed ablation | Partial: matched-library synthetic panel passed; heterogeneous real data TODO |
| Residual preserves generality | Non-symbolic-fraction and general benchmark studies | Partial: IID continuum passed; unguarded residual failed 4× OOD and is excluded from extrapolating core |
| Real-world relevance | Temporal and representative general datasets | Not established: temporal pilot failed; tiny numeric general pilot was non-catastrophic but linear models won |
