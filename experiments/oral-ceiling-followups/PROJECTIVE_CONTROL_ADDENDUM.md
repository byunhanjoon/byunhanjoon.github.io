# PROJECTIVE FOLLOW-UP: ADVERSARIAL CONTROL ADDENDUM

Status: frozen before control outcomes are inspected. This addendum does not
change the original real-data gate or its pass result.

The large held-out-query gap motivates two post-hoc controls using the same
three datasets, seeds, windows, optimizer, 3,000 steps, and evaluation queries:

1. `QueryNetBroad` is the same direct scalar model but its training query
   distribution additionally includes differences, normalized dense queries,
   and scaled dense queries. This deliberately exposes it to the held-out
   query families, though never to test outcomes or exact evaluation queries.
2. `JointDiagNet` emits a joint mean and diagonal covariance, then projects
   analytically. It is exactly projectively consistent but cannot represent
   cross-coordinate covariance. Its parameter count is matched to the full
   low-rank-plus-diagonal ProjectiveNet.

The full ProjectiveNet survives the stronger control if it has no worse NLL
than `QueryNetBroad` in at least six of nine cells and no more than five
percentage points worse mean coverage error. A full-covariance advantage is
supported separately only if it beats `JointDiagNet` NLL in at least six of
nine cells. Failure of the second condition narrows the claim to consistency
rather than covariance modeling.
