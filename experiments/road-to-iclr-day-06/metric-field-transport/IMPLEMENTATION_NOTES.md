# E2 implementation notes

These notes resolve implementation details already constrained by the frozen
development protocol. They were written before any E1 promotion decision or E2
outcome was available.

- The first-order map sends a metric-coordinate difference through a shared
  rank-4 right factor, then through a rank-4-by-17 left factor attached to the
  warm anchor state. The 17 outputs are the rank-16 expert vector and scalar
  intercept correction.
- `transport_shuffled_metric` permutes metric identities independently within
  development-training and development-validation states. Only the transport
  graph and first-order differences are shuffled; the raw base input remains
  correct and paired.
- The reconstruction loss is the mean squared error from transported experts
  to stop-gradient warm experts for every state masked in that epoch. It is
  combined with each task minibatch at the frozen weight 0.1.
- E1b cells for the selected raw representation are reused verbatim as E2
  `raw_base` cells. They have identical architecture, optimization, partitions,
  and seeds, so retraining them would violate the no-identical-reruns rule.
