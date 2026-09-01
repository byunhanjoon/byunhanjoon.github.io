# Hierarchy gap audit

Status: **PASS (G1–G5)**

- The protocol and config reproduce their pre-acquisition hashes:
  `36b20009...bb2e28` and `3bdec9c0...68e7fd`.
- The official Census archive checksum is
  `9fc306ec...3b41ede`; 37,783 rows and 922 six-digit NAICS industries pass the
  frozen filters.
- The bulk-file total legal-form code (`lfo="-"`) maps to API `LFO=001`, and
  its top-level employment/payroll/establishment columns map to the API's total
  size class. This representation mapping is recorded in the source manifest;
  it does not change the frozen population.
- Three atomic prediction seals exist. Every payload hash validates, each
  training/test industry partition is disjoint, and every seal records
  `outer_test_outcomes_accessed=false`.
- All nine split/operator cells are retained. Split-level Pearson is `0.9914`,
  Spearman is `0.8333`, MAE is `0.00091`, calibration slope is `1.037`, and
  direct sign accuracy is `100%`.
- All three source/operator aggregate signs are correct; aggregate Spearman is
  `1.0`. The combined four-source prospective panel has Spearman `0.9091` and
  direct sign accuracy `100%` over 12 aggregates.
- G1 integrity, G2 ranking, G3 signs, G4 hierarchy behavior, and G5 combined
  breadth all pass exactly as frozen.

No outcome was used to alter the hierarchy, covariates, operators, thresholds,
or splits. No harmful prospective aggregate occurred, so negative-decision
validation remains outside the observed prospective support.
