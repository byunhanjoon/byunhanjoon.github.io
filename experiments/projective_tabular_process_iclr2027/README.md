# Projective tabular process experiment

This directory contains the frozen protocol, experiment code, complete result-generation pipeline, and ICLR-style manuscript for a projective covariance lift of TabICLv2. The final empirical conclusion is negative: the exact process construction passes every integrity test, but the predeclared performance claim fails. See `FINAL_RESULTS.md` for the concise scientific handoff.

## Artifact map

- `paper/iclr2027/projective_tfm.pdf`: compiled 16-page paper; main text ends on page 7.
- `paper/iclr2027/projective_tfm.tex`: main manuscript source.
- `paper/iclr2027/sections/appendix.tex`: proofs, complete HPO, datasets, per-dataset results, audits, deviations, and reproduction details.
- `paper/iclr2027/references.bib`: paper bibliography, favoring primary conference and official sources.
- `PROTOCOL.md`: frozen statistical protocol and gates.
- `config.json`: machine-readable benchmark configuration.
- `PROTOCOL_HASHES.txt`: frozen hashes.
- `PROTOCOL_DEVIATIONS.md`: every implementation deviation discovered during execution.
- `generate_paper_artifacts.py`: regenerates all paper tables, figures, macros, and their manifest directly from result files.
- `validate_final_artifacts.py`: validates complete caches, finite arrays, failure logs, shared marginals, projectivity, hashes, and PDF page limits.

Large caches and result files are intentionally external to the Git source tree at:

```text
/data/byunhanjoon/projective_tabular_process_iclr2027
```

## Frozen identities

```text
protocol  a19db1969070fdce89e16c8e60d976a9c49973fc1510373f30a651c48bd4ec89
config    4e74dd70c4479418944e43a89626dc90f8a27452d1677e069eec9830452fa1ea
```

Do not edit `PROTOCOL.md` or `config.json` and then interpret the resulting run as the same frozen experiment.

## Reproduce from cached model outputs

From this directory, the shortest deterministic verification path is:

```bash
/home/byunhanjoon/miniconda3/bin/python -m pytest -q test_core.py
/home/byunhanjoon/miniconda3/bin/python generate_paper_artifacts.py
cd paper/iclr2027
/data/byunhanjoon/projective_tabular_process_iclr2027/tex_env/bin/tectonic \
  -X compile projective_tfm.tex --keep-logs --keep-intermediates
cd ../..
/home/byunhanjoon/miniconda3/bin/python validate_final_artifacts.py
```

The last command writes `results/final_validation.json` under the external cache and must report `"status": "PASS"`.

## Full execution order

The core sequence used for the singleton-projective experiment is:

```bash
python extract_tabicl.py --stage dev --query-mode singleton --device cuda
python train_head.py --query-mode singleton
python extract_tabicl.py --stage eval --query-mode singleton --device cuda
python score_projective.py --query-mode singleton

python tune_classical.py
python extract_classical.py --stage eval
python extract_tabpfn.py --stage dev --device cuda
python extract_tabpfn.py --stage eval --device cuda
tabpfn3_env/bin/python extract_tabpfn3.py --stage dev --device cuda
tabpfn3_env/bin/python extract_tabpfn3.py --stage eval --device cuda
tabdpt_env/bin/python extract_tabdpt.py --stage eval --device cuda
python score_baselines.py --query-mode singleton

python audit_query_projectivity.py --query-mode singleton

python extract_tabicl.py --stage app --query-mode singleton --device cuda
python extract_classical.py --stage app
python extract_tabpfn.py --stage app --device cuda
tabpfn3_env/bin/python extract_tabpfn3.py --stage app --device cuda
tabdpt_env/bin/python extract_tabdpt.py --stage app --device cuda
python score_applications.py

python generate_paper_artifacts.py
python validate_final_artifacts.py
```

The isolated environment paths above are schematic when reproducing on a new machine. The evaluated versions and checkpoints are recorded in the paper appendix and in every episode's metadata: TabPFN-2.5 package 6.3.0, TabPFN-3 package 8.5.0, TabDPT package 1.2.0 with `tabdpt1_2.safetensors`, and TabICLv2 checkpoint `tabicl-regressor-v2-20260212.ckpt`.

Extraction scripts support `--shard-index` and `--num-shards`. Outputs are written atomically, and valid existing episode files are skipped. The same cached context/query indices and aggregate coefficients are shared across methods.

## Important implementation distinction

`--query-mode batched` is retained only to reproduce the failed diagnostic. It is PSD within a supplied batch but is not a projective stochastic process because TabICLv2 preprocessing and representations depend on the other query rows. Primary results always use `--query-mode singleton`.

`adaptive_rho.py` and `tune_adaptive_rho.py` are exploratory prototypes from before the singleton correction. They are excluded from the frozen primary experiment, all reported tables, and the manuscript's empirical claims.

