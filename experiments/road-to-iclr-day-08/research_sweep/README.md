# Reproduction guide

This directory is the isolated deliverable for the rapid feasibility sweep. The study uses controlled synthetic data only and requires no network access or external repositories.

Run from `experiments/road-to-iclr-day-08` with the available scientific Python environment:

```bash
/home/byunhanjoon/miniconda3/bin/python research_sweep/run_sweep.py
/home/byunhanjoon/miniconda3/bin/python research_sweep/run_robustness.py
```

The first command overwrites the primary metrics and figures. The second command appends the prespecified top-two robustness panel and decision record to `results.json`; therefore it must be run second. A fast end-to-end smoke mode is available as `run_sweep.py --quick`, but the checked-in artifacts are from the full run.

`results.json` contains environment versions, parameters, model sizes, seeds, runtimes, leakage checks, raw metrics, empty or populated exception logs, robustness records, and the final decision. Direction subdirectories contain flat CSV exports for independent inspection. `figures/` contains only figures generated from those records.

The experiment intentionally uses grouped splits for exact-equivalence blocks, disjoint schema names for semantic transfer, separate IID calibration data for prior-shift thresholds, and a detector trained only on unlabeled IID summaries.
