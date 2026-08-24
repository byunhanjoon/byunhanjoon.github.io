# Road to ICLR — Day 1

This synthetic representation probe accompanies [Day 1: When “Numerical” and “Categorical” Aren’t Types](https://byunhanjoon.github.io/blogposts/road-to-iclr-day-01.html).

It compares three preprocessing routes with the same ridge readout:

- `schema_only`: raw standardized numerical columns and one-hot categorical columns;
- `semantic_oracle`: the right route for each synthetic feature’s known behavior;
- `multi_view`: raw, piecewise-linear, and one-hot views for every discrete feature.

The experiment is intentionally small. It tests whether representation choice alone can recover signal that a binary schema hides; it is not a neural-model benchmark.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python feature_semantics.py
```

The command writes the aggregate results to `results.csv`.
