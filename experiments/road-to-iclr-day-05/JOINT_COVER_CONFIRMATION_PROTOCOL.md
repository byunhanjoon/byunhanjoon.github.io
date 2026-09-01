# Conditional confirmation of the joint orthogonal cover

Status: frozen after the initial five-dataset adaptive cover result was known,
before outcomes on any dataset listed here were computed.

The confirmation uses every remaining checksum-audited task in
`/data/tokenization_icaif_grouped_v2`:

- Compustat Korea direction;
- credit-card default and credit-card fraud;
- HELOC credit risk;
- KDD17 stock direction and return;
- Polish bankruptcy horizons 1 through 5.

The five Polish horizons count as one source group, and the two KDD17 tasks
count as one source group. Together with Compustat, the two credit-card data
sources, and HELOC, this gives six confirmation source groups. Dataset/model
cells are repeated strata, not independent replications.

The frozen Tier-1 subsampling, five model families, four seeds, feature maps,
losses, and hyperparameters are unchanged. Numeric-only datasets have a
singleton category factor; regression has a singleton target-ID factor. The
budget-four cover balances every non-singleton factor and seed exactly.

Compatibility addendum (recorded after an exception and before any prediction
from the affected cell): sklearn HistGradientBoosting rejects categorical
fields with more than 255 training levels. Such fields are passed as ordinal
numeric columns for that family only; all <=255-level fields retain native
categorical treatment. Compustat/HistGB is labeled a hybrid ordinal fallback,
not silently reported as fully native categorical handling.

The source-group confirmation gate passes if the cover has lower group-mean
residual than both iid joint sampling and seed-only averaging in at least four
of six source groups, and its pooled material-cell mean is lower than both.
Materiality remains joint risk >=0.5% of member Brier/MSE. All tasks are
reported, including null-risk controls and related variants.
