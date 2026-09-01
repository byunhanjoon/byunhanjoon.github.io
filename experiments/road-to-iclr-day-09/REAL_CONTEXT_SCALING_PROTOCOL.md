# Frozen real-regression context-scaling diagnostic

Frozen: 2026-09-01, after the independent regression confirmation and before any fresh
seed-155001 outcomes.

## Question

Does competence-routing gain improve as more context labels make expert loss estimation
reliable, or is the confirmed effect insensitive to context evidence?

## Design

Reuse all five confirmation dataset identities with no removal. For each dataset and 30
fresh repeats, draw one 192-row outer-train permutation and use nested prefixes of
32/64/96/192 context rows. Use one shared 96-row official-test query sample across the
four context sizes. Seed family is 155001.

Preprocessing, numeric first-32 feature cap, six experts, synthetic regression fixed
weights, temperature 0.1, and three-fold context CV are unchanged. Report competence,
fixed, hard CV, uniform, and best-individual oracle.

## Analysis

For each dataset/repeat, regress paired fixed-minus-competence gain on `log2(context
rows)`. The primary scaling statistic is the dataset-balanced mean slope with 10,000
hierarchical paired bootstrap draws over datasets then repeats. Also report the
dataset-balanced gain and interval at every context size.

- A positive slope interval supports context-evidence scaling.
- A positive gain interval at any size identifies the earliest supported context budget.
- A nonpositive slope narrows the confirmed result to a level effect at the tested scale.

This is a post-confirmation mechanism diagnostic on previously seen identities, not a
third independent external confirmation. All context sizes and datasets remain in the
report regardless of outcome.
