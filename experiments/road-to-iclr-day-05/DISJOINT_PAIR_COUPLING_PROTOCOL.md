# Candidate-independent disjoint-pair protocol

Status: frozen after the primary packed-pair result and before inspecting this
coupling repeat.

Repeat the 32-fit packed-pair versus independent-pair comparison on all five
panels, but generate separate graph actions for every candidate. Use 1,024
actions and retain exact quotient losses/winners. Report agreement, validation
regret, and held-out losses; candidate RMSE/residual are unchanged marginal
targets and are checked for reproducibility only.

The gate passes if packed-pair agreement is no lower and validation regret no
higher than the independent-pair control in at least four of five panels.
Exact partition closure must remain numerical. This rules out common random
coordinates as the source of the selection result; it does not address
validation-to-test target shift.
