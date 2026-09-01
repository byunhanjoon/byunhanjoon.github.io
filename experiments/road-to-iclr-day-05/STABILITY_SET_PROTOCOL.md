# Replicated unbiased-selector stability set

Status: protocol frozen before inspecting outcomes; analysis complete.

## Motivation

A single quotient winner hides whether finite nuisance-score randomness makes
the choice brittle. At the same 64-fit budget used by the block-U frontier,
run two independent unbiased 32-fit selectors and return the union of their
winners. The output is a singleton when the selectors agree and a two-model
set otherwise.

## Design

Use all five model-selection panels. For each of 512 deterministic draws:

- cover method: two disjoint pairs of independent 16-fit strength-2 covers,
  each pair scored by the cross-score;
- control: two disjoint IID-32 complete U-statistics.

Both methods use 64 fitted members per candidate. Report exact validation
quotient-winner coverage, mean set size, wrong-singleton probability, and
best/worst quotient regret within the set. Held-out winner inclusion and test
losses are descriptive only.

## Frozen gate

The stability-set gate passes if cover is at least as good as IID-U on exact
winner coverage, mean set size, and wrong-singleton probability on at least
four of five panel means, with at least one strict panel for each clause.

This is not a formal confidence set and carries no distribution-free coverage
guarantee. It is a transparent stability diagnostic related to, but weaker
than, recent stable set-valued selection methods.

## Outcome

The frozen gate **passes** every clause on all 5/5 panels; coverage and wrong
singleton rate are strict on 4/5, and size is strict on 5/5.

- Confirmation: coverage rises from 97.66% to 99.61%, mean size falls from
  1.080 to 1.037, and wrong singletons fall from 2.27% to 0.39%.
- Task-balanced: coverage rises from 98.54% to 99.83%, size falls from 1.078 to
  1.021, and wrong singletons fall from 1.46% to 0.17%.
- External: validation-winner coverage rises modestly from 96.46% to 97.17%
  with a smaller set, but test-winner inclusion falls from 26.37% to 25.00%.
  The set diagnostic therefore does not repair validation/test rank mismatch.
- Equal-source intervals are strongest for set size (exclude zero on
  confirmation, task-balanced, and subsample) and for external coverage/wrong
  singletons. Other coverage intervals touch zero because most sources are
  exact ties at ceiling; panel means should not be read as five independent
  replications.

Proposition 23 gives exact coverage, expected-size, and wrong-singleton
identities in terms of one-replicate selection probabilities.
