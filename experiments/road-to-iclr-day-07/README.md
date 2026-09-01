# Day 7 — ICLR 2027 direction search

This directory contains the explicitly exploratory Day-7 ICLR 2027 direction
search.

The search is now closed. The evidence- and collision-adjusted lead is
OrbitCover, synthesized from the prospectively frozen and audited closure in
`../final_closure/`. Start the submission package with:

- `ORBITCOVER_PAPER_DRAFT.md` — compact main-paper draft;
- `paper/main.tex` — anonymous ICLR 2027 source using the official template;
- `SUBMISSION_STATUS.md` — format/evidence audit, current deadlines, and the
  final human-only upload checklist;
- `CLAIM_EVIDENCE_MATRIX.md` — allowed claims, exact evidence, and forbidden
  overclaims;
- `REPRODUCIBILITY_MAP.md` — frozen protocols, result sources, figures, and
  regeneration order;
- `LITERATURE_AUDIT.md` — 2024–2026 tabular audit and direct novelty
  collisions;
- `DIRECTION_RANKING.md` — final portfolio decision.

The authoritative OrbitCover verdict is **PARTIALLY SUPPORTED**. The coupled
finite-target effect is real and broad at small budget; fresh-RNG schema
balance, convergence efficiency, and held-out selection improvement fail.

The first experiment asks whether externally supplied feature geometry should
be treated as a **shrinkable residual expert**, rather than as an unconditional
tokenizer. It reuses the fixed-base residual caches and public state metrics
from `geometry_transfer`, but freezes the new trust-selection rule before its
outcomes are computed.

Start with:

- `PILOT_PROTOCOL.md` — frozen replay design and limitations;
- `THEORY.md` — exact trust law and proposed theorem program;
- `shrinkage_replay.py` — state-held-out selection and outer replay;
- `neural_transfer.py` — fixed-MLP development transfer/certificate test;
- `bayes_structured_prior.py` — analytic latent-structure PFN target;
- `learned_structured_pfn.py` — three-seed neural learnability test of that
  target;
- `LEARNED_PFN_RESULTS.md` — learned mechanism result, collision, and limits;
- `results/summary.json` — machine-readable outcome after the run;
- `DIRECTION_RANKING.md` — literature-subtracted topic ranking (written after
  the evidence and literature audit).
