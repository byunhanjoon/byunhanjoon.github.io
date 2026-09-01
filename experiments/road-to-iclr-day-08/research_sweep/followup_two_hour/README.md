# Follow-up reproduction guide

The scientific conclusions and exact metrics are in `FOLLOWUP_RESULTS.md`; the frozen design is in `PROTOCOL.md`; machine-readable gates are in `summary.json`.

Run from `experiments/road-to-iclr-day-08` with the environment used for this study:

```bash
python research_sweep/followup_two_hour/run_direction1_extended.py --data-seeds 45 --tabpfn-estimators 16 --device cuda:0
python research_sweep/followup_two_hour/run_direction1_continuous_tabpfn.py --seeds 30 --estimators 16 --device cuda:1
python research_sweep/followup_two_hour/run_direction1_distractor_stress.py --seeds 15 --estimators 8 --device cuda:1
python research_sweep/followup_two_hour/run_direction2_extended.py --numeric-seeds 12 --device cuda:1
python research_sweep/followup_two_hour/run_direction2_crosssplit.py --device cuda:1
python research_sweep/followup_two_hour/run_direction2_real_panel.py
python research_sweep/followup_two_hour/analyze_followup.py
```

The first two commands may run concurrently on separate GPUs. Cached TabPFN and Hugging Face model weights are required. Raw records, seeds, runtimes, environment versions, and exception arrays are preserved in each direction's JSON files.
