# Adaptive nested-cover model-selection protocol

Status: frozen before outcome computation on 2026-08-28.

Use the same confirmation, changed-menu, and changed-subsample five-candidate
panels as the frozen model-selection experiments. For each of 512 independently
randomized literal nested strength-1/2/3 schedules, make validation-only
decisions with the following fixed rule:

1. At budget 4, compute the four leave-one-member-out candidate winners. Stop
   at 4 if they are unanimous; select the full four-member winner.
2. Otherwise continue to budget 16. Compute the candidate winner separately
   in each consecutive four-row sub-block. Stop at 16 if these four winners
   are unanimous; select the full 16-member winner.
3. Otherwise continue to budget 64 and select the full 64-member winner.

Report mean fits per candidate, stopping proportions, agreement with the full
nuisance-quotient validation winner, validation quotient regret, quotient test
loss, and realized held-out test loss. Compare with the three fixed nested
prefixes from exactly the same schedules. The exploratory gate passes for a
panel if the adaptive rule uses fewer than 64 mean fits, has agreement no worse
than the fixed 16-fit prefix, and has no larger mean quotient regret.

The rule was motivated after seeing the fixed-frontier results, so it is an
exploratory follow-up rather than independent confirmation. Because stopping
depends on predictions, the stopped ensemble need not be an unbiased quotient
estimator. Claims are restricted to validation-only model-selection behavior.

## Post-failure conservative recovery

The aggressive rule failed on the menu and subsample repeats. After observing
that failure, add a deliberately conservative exploratory rule: always reach
budget 16; stop there only when the full 4-fit and 16-fit prefix winners agree,
otherwise escalate to 64. This recovery is post-outcome method development and
cannot count as confirmation. Compare it with fixed 16 and report it separately
from the failed frozen rule.
