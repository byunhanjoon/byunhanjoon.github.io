# Result-integrity audit

Status: final read-only audit specified after analyses, before report freeze.

For the principal complete-tensor panels, verify:

- every configured dataset×model has both NPZ tensor and JSON manifest;
- validation/test labels agree across candidates within a dataset;
- tensor factor shapes agree with manifests and validation/test tensors;
- every prediction is finite;
- classification probabilities are within tolerance and sum to one;
- every manifest declares complete status and full-product verification where
  that field exists;
- every strength-2 headline test cell was selected by its validation row, not
  by test materiality.

Parse all top-level result summaries, count non-complete statuses, and record
SHA-256 hashes for the central configs, theory, analyzers, and summary files.
The audit diagnoses artifact integrity; it does not validate the scientific
assumptions or create a new performance gate.

Final outcome: **pass**. The audit verifies 587 tensors representing 44,720
complete-product/control fits or explicitly labeled TabPFN calls, parses 156
top-level summaries, and finds zero structural or screening issues. The new
completion subpanels contribute 278 tensors and 19,584 represented fits/calls:
144 broad neural, 16 exact neural, 16 matched, 60 classical, 18 TabPFN, eight
enlarged-menu, and 16 row-order tensors. Exact/broad duplicate runs have zero
reproducibility mismatches. Classification probability-sum error is at most
`1.34e-7`. The independent algebra/construction suite passes 103 tests, and 31
figure concepts (21 retained plus ten completion figures) regenerate.
