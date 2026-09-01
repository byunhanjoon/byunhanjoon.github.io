# Cheap-screen / precise-deployment allocation protocol

Status: frozen after the independent screen-then-cross failure and before
inspecting this experiment's outcomes.

## Motivation and design

The first allocation rule retained the exact winner almost perfectly but its
fresh 32-fit deployment score was too noisy to match equal 64-fit allocation.
Test the mechanism-predicted repair:

1. screen every candidate once with the ordinary loss of one randomized
   strength-2 cover (16 fits);
2. retain the two lowest pilot losses;
3. use four fresh independent strength-2 covers and their unbiased complete
   block-U score (64 fits) only on those two candidates.

All deployment blocks are independent of the screen.  For a paired control,
compute the final U64 winner over *all* candidates using the exact same blocks;
these extra scores are analysis-only and are not charged to the proposed
method.  This isolates the effect of screening from action-stream variation.
Also report the previously frozen equal-64 result descriptively.

With `M` candidates the method costs `16M+128` rather than `64M`, saving 35%
for five candidates, 25% for four, and 8.3% for three.

## Frozen gate

The post-failure gate passes if:

- savings are at least 20% in four of five panels;
- pilot top-two exact-winner inclusion is at least 98% in four of five panels;
- versus the paired all-candidate U64 control, final agreement is no lower and
  quotient-validation regret no higher in at least four of five panels.

The single-cover pilot is biased for quotient loss; only the independent
deployment U-statistic is conditionally unbiased.  Held-out test loss remains
diagnostic.
