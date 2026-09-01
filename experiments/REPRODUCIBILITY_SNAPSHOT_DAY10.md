# Reproducibility snapshot through Day 10

Snapshot date: 2026-09-02

This repository snapshot preserves the reproducibility-critical experiment
layer through Road to ICLR Day 10:

- experiment implementations, tests, launch scripts, and environment files;
- frozen protocols, configurations, research logs, reports, and paper sources;
- compact CSV/JSON summaries, audit tables, and publication figures; and
- the compact Day-10 forensic-audit target slices needed to rerun that audit.

The working experiment tree also contains about 12 GB of generated material in
roughly 262,000 files.  Raw downloaded datasets, per-fit predictions, model
checkpoints, cache/quarantine directories, fit registries, large tensor files,
and oversized draw or manifest files are intentionally not stored in ordinary
Git.  GitHub rejects individual files in this tree that are hundreds of
megabytes, and committing the full generated tree would make the repository
impractical to clone.

Accordingly, this is a source-and-compact-evidence snapshot, not a bit-for-bit
backup of every local generated artifact.  Recreate the omitted artifacts with
the commands and frozen settings in each experiment's README, protocol, config,
or run script.  The omitted local files should be archived separately with Git
LFS or object storage if preservation of every prediction and checkpoint is
required.
