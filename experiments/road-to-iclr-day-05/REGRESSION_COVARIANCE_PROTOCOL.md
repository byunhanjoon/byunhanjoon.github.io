# Multi-seed regression chart-covariance confirmation

Status: conditional confirmation. The earlier seed-17 Diamond and Black
Friday outcomes were known before this protocol; seeds 31, 47, and 59 were not
run for this transfer mechanism.

For each dataset and new seed, reuse the frozen raw-Adam chart tensor and run
the transported covariant SGD and field-vector adaptive optimizer across the
same five sample-whitened equivalent charts. Hyperparameters, early stopping,
100 SGD epochs, train/validation/test splits, and target standardization are
unchanged from `chart_regression_transfer_pilot.py`.

Primary mechanism endpoint: chart prediction risk and trajectory range.
Secondary endpoint: standardized test MSE relative to the reused raw-Adam
member and orbit-mean benchmarks. The expected closure is preregistered; no
universal accuracy improvement is expected because the known seed-17 results
were mixed. Seeds are repeated measurements and the two datasets are the
replication units.

