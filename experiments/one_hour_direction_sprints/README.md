# One-hour direction sprints

This directory contains three time-boxed falsification experiments requested on
2026-08-31.  They reuse immutable inputs from the repository but write all new
code, caches, and summaries below this directory.

The projects are evaluated independently:

1. `geometry`: fresh split-2 replay of the frozen risk-gated neural geometry
   adapter.
2. `projective`: static-tabular joint-law learning versus capacity-matched
   direct query prediction.
3. `orbitcover`: equal-budget *predictive-loss* audit of OrbitCover, distinct
   from the already-established quotient-estimation endpoint.

Each protocol is written before its outcome is inspected.  A failed gate is a
useful result and is not retuned inside the sprint.
