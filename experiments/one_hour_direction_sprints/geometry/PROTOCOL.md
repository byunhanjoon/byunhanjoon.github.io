# Geometry sprint protocol (frozen before split-2 execution)

## Question

Does the previously promising per-backbone family-wise lower-confidence-bound
certificate transfer to a fresh state split across multiple neural tabular
backbones, while retaining enough gain and breadth to justify an ICLR-scale
project?

## Frozen panel

- Tasks: `acs_occupation`, `tlc_pickup_zone`, `airline_origin_airport`, and
  `medical_charges`.
- Backbones: MLP, ResNet, FT-Transformer, and TabM.
- State split: split 2 only.  Earlier neural development used splits 0 and 1.
- Construction, validation, and test states remain disjoint.
- The base learner, 40-epoch optimization, state balancing, residual adapter
  operators, and Bonferroni LCB are imported unchanged from
  `road-to-iclr-day-07/nested_backbone_certificate.py`.
- The primary selector is the pre-existing per-backbone family-wise LCB rule.
  No operator, threshold, or rule is selected using split-2 test outcomes.

## Primary metric

State-balanced standardized-MSE improvement over the neural fallback, with an
abstention contributing zero gain.  Relative gains use each cell's unadapted
test residual MSE as denominator.

## Go gate

All conditions must hold:

1. finite/integrity checks pass;
2. mean deployed absolute gain over all 16 backbone-task cells is at least
   `0.005`;
3. no selected cell has gain below `-0.002`;
4. at least 6 backbone-task cells are selected;
5. at least 2 distinct data sources and 3 distinct backbones are selected;
6. at least 3 of 4 backbones have positive mean deployed gain.

The sprint is capped at one wall-clock hour.  Missing cells at the deadline
make the gate fail rather than changing the panel.
