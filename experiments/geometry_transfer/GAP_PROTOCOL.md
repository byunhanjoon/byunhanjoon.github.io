# Hierarchy gap protocol — frozen before outcome acquisition

Frozen on 2026-08-30 after the original prospective program was completed and
before any 2023 County Business Patterns outcome row was downloaded or
inspected. This is a separately hashed confirmatory addendum. It does not
replace or modify the original prospective protocol.

## Scientific question

Does the nested, training-state-only Geometry Transfer estimate retain useful
sign and ranking information for a genuinely nonspatial hierarchy, with unseen
six-digit industries as the cold semantic states?

## Untouched source and task

- **Source:** official U.S. Census Bureau 2023 County Business Patterns state
  file, fixed URL
  `https://www2.census.gov/programs-surveys/cbp/datasets/2023/cbp23st.zip`.
- **Semantic state:** a six-digit `NAICS2017` industry. Aggregate/range codes,
  `00`, codes containing non-digits, and codes other than six digits are
  excluded.
- **Rows:** state-by-industry records with total legal form (`LFO=001`) and
  total employment-size class (`EMPSZES=001`), numeric positive annual payroll
  and employment, and numeric positive establishment count. Industries with
  fewer than 10 retained geographic states are excluded.
- **Target:** `log1p(PAYANN)`, where Census reports annual payroll in thousands
  of dollars.
- **Ordinary covariates:** geographic state identifier, `log1p(EMP)`, and
  `log1p(ESTAB)`. The base model never receives `NAICS2017` or its label.
- **Geometry:** target-independent NAICS prefix-tree path distance. Each code is
  represented by its prefixes of lengths 2, 3, 4, 5, and 6. Distance is the
  number of upward plus downward edges through the deepest common prefix.
  No payroll, employment, establishment, label text, or fitted outcome enters
  this distance.

The archive may be declared unavailable only if the frozen URL cannot be read
or the stated filters leave fewer than eight industries or 500 rows. No source,
target, covariate, hierarchy, operator, or threshold substitution is allowed.

## Frozen splits, model, and operators

- Three outer industry-held-out splits, seeds 8801, 8802, and 8803; 70% of
  industries are observed and 30% are sealed test states.
- Three inner industry folds partition the outer observed industries.
- Training residual means use three-fold row-OOF predictions within the
  currently observed industry set.
- Base model: CatBoost regression excluding semantic industry; 100 trees,
  depth 7, learning rate 0.08, L2 5, fixed seeds, and at most 75,000 fitting
  rows with at least one row retained per observed industry.
- Diagonal `Sigma` is residual sample variance divided by each observed
  industry's row count.
- Operators are fixed before outcomes: inverse-distance 3-NN, row-normalized
  Gaussian RBF at the median observed-industry pair distance, and kernel-ridge
  interpolation on the same hierarchy distance. The fallback predicts zero
  residual state effect.
- Primary loss is industry-balanced squared error on the training-standardized
  target. Source/operator aggregates are means across the three outer splits.

## Sealing and integrity

For each outer split, inner-fold predicted Delta, uncertainty, heuristics,
industry assignments, and a payload hash must be written atomically before
outer-test residual outcomes are evaluated. Outer-test outcomes may not affect
base fitting, residual means, `Sigma`, bandwidths, operators, or predicted
benefit. All split cells are retained.

## Frozen gap criteria

- **G1 integrity:** all three split seals exist, hashes validate, train/test
  industries are disjoint, and no outcome-dependent geometry is used.
- **G2 ranking:** Spearman(predicted Delta, actual Delta) is at least 0.50 over
  the nine split/operator cells.
- **G3 signs:** direct sign accuracy is at least 75% over those nine cells.
- **G4 hierarchy behavior:** at least two of the three source/operator
  aggregate signs are predicted correctly.
- **G5 combined breadth:** after appending this addendum to the three runnable
  original prospective sources, source/operator aggregate Spearman is at least
  0.60 and direct sign accuracy is at least 75%.

The gap closes only if G1–G5 all pass. A failed or mixed result remains in the
paper and changes the verdict; thresholds are not revised after acquisition.
