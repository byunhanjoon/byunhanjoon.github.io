# Final prospective mixed-schema source block C

Frozen before downloading any of the four sources or observing model outcomes.

## Panel

Use four previously unused binary OpenML sources selected for genuinely mixed
numeric/categorical schemas:

- Adult, OpenML ID 1590;
- Bank Marketing, OpenML ID 1461;
- Titanic, OpenML ID 40945;
- Churn, OpenML ID 40701.

Evaluate all five established model families—one-hot linear, ordinal forest,
native HistGB, CatBoost, and one-hot Adam MLP—over the complete
`4 x 4 x 2 x 4 = 128` nuisance product. No source or failed cell may be
replaced. Reuse the prior nuisance generators and held-out split convention.

## Frozen gates

- Strength-2 must beat IID-16 and four strength-1 blocks on at least 16/20
  cells and all 4/4 source means; it must beat all three controls on at least
  80% of material cells.
- Pair32, pack64, and unbiased pair-cross64 must each improve all 4/4 source
  means, have no adverse nondegenerate candidate, and improve at least 80% of
  nondegenerate candidates.
- Exact validation-to-test winner agreement and regret are reported without a
  gate, given the already established partition boundary.

This is the strongest final source-breadth evidence because both sources and
all outcomes are prospective relative to this protocol. It still covers only
four sources and one split each.
