# Direction 3 artifacts

- `training_log.csv`: three DeepSets ensemble seeds, model sizes, losses, and runtimes.
- `task_metrics.csv`: per-task truth, predictions, error, ensemble variance, OOD score, coverage, and naive baseline.
- `shift_summary.csv`: metrics for six shift families at three severities plus IID.
- `model_seed_metrics.csv`: per-model-seed MAE and RMSE for every cell.

The Isolation Forest sees only unlabeled summaries from the IID training prior. Bad-error thresholds and interval scaling are frozen on a separate IID calibration set before any shifted evaluation.
