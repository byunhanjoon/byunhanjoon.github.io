# Exact-support residual-token transfer pilot

Developmental because all listed tabred test partitions have appeared in earlier day 4 work.

The proposed token adds a gated learned embedding of an exact numerical level to the ordinary PLE field token. Fields are activated by the frozen target-free rule `2 <= train cardinality <= 128`; unseen values receive a zero residual. The bin control has identical tables, gates, backbone shape, and parameter count but indexes them with Q-PLE bins instead of exact levels.

| Dataset | Model | Q-support val gain | Exact vs bin-control val gain | T-support val gain | Q-support test gain | T-support test gain | Gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| cooking-time | ft_transformer | +0.246% | -0.342% | +0.368% | +0.003% | +0.279% | FAIL |
| cooking-time | mlp | +0.069% | -0.089% | -0.071% | +0.143% | +0.198% | FAIL |
| cooking-time | resnet | +0.017% | -0.066% | +0.237% | +0.241% | -0.000% | FAIL |
| weather | ft_transformer | +0.718% | +0.230% | -0.435% | +0.148% | -0.006% | FAIL |
| weather | mlp | +0.079% | -0.100% | -0.770% | +0.842% | -0.074% | FAIL |
| weather | resnet | -0.217% | +0.027% | -0.425% | -0.585% | -0.277% | FAIL |

## Frozen decision

- Architecture gate: **FAIL**.
- Development dataset gates: `{"cooking-time": false, "weather": false}`.
- Delivery ETA transfer: **not authorized** by the frozen rule.
