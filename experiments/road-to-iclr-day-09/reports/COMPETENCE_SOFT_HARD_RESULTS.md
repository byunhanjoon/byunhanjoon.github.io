# Soft competence aggregation versus hard expert selection

Status: **frozen post-result mechanism diagnostic completed**.

The same context-CV losses were used either to choose one expert (`hard_cv`) or to form
the already tuned soft mixture (`soft_cv`). No model was refit and recomputed parent
losses agreed to `2.22e-16`.

| Task | Fixed | Hard CV | Soft CV | Soft gain over hard (95% paired CI) | Hard gain over fixed (95% paired CI) |
|---|---:|---:|---:|---:|---:|
| Classification | 0.625964 | 0.644297 | 0.620500 | **0.023797 [0.022615, 0.024993]** | **-0.018333 [-0.019878, -0.016796]** |
| Regression | 0.461034 | 0.231505 | 0.206850 | **0.024655 [0.022748, 0.026604]** | **0.229530 [0.221091, 0.237805]** |

Classification hard-CV selection matches the query-best expert on 42.02% of episodes
and is materially worse than the stable fixed mixture. Soft weighting maintains an
average 3.83 effective experts and reverses that failure. Regression has stronger loss
ordering (mean CV/query Spearman 0.864 versus 0.548), 83.08% hard-selection accuracy,
and 2.18 effective experts; hard selection is already useful, but soft weighting still
improves it.

The soft advantage is largest when the two lowest CV losses are close. It falls from
0.03116 to 0.01472 across classification margin quintiles and from 0.05701 to
approximately zero across regression quintiles. When the hard choice is wrong, the soft
gain is 0.04509 classification and 0.12339 regression; when it is correct, the gains are
-0.00559 and +0.00455. This supports a noise-protection interpretation of calibrated
aggregation.

This is not independent confirmation and softmax/exponential weighting is not novel.
The scoped contribution is sharper: the PriorDial panel exposes three distinct targets
— generator-family identification, individual-expert selection, and calibrated mixture
prediction — whose success rankings can reverse even when all use the same context.
