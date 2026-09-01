# OrbitCover submission status

Status: **submission package complete; expert novelty review recommended before
upload**.

## Format audit

- Official ICLR 2027 style and bibliography files are vendored unmodified.
- Anonymous `paper/main.tex` compiles to `paper/main.pdf`.
- PDF: 10 total pages; references begin on page 8; appendix begins on page 9.
- Main text is therefore within the official nine-page submission limit.
- Required AI-use statement, ethics statement, and reproducibility statement
  are present.
- No undefined citations, undefined references, or overfull boxes.

## Evidence audit

- Frozen final verdict: **PARTIALLY SUPPORTED**.
- 144/144 coupled small-budget wins and the 55.9% dataset-balanced reduction
  are always paired with the different-estimand caveat.
- The same-target mechanism claim uses the separate 48-cell ablation.
- The OC2-independent failure, target shift, convergence boundary,
  matched-function boundary, and held-out selection limitation appear in the
  main text.
- Every headline number maps to the authoritative closure summaries through
  `CLAIM_EVIDENCE_MATRIX.md` and `REPRODUCIBILITY_MAP.md`.
- Read-only closure audit: 140,592 fit keys, 1,128 prediction arrays,
  125,616 independent seed records, and 116/116 checks passed.

## Current ICLR 2027 deadlines

The official author guidelines list:

- abstract submission: September 18, 2026, 11:59 PM AoE;
- full paper submission: September 25, 2026, AoE.

Source: https://iclr.cc/Conferences/2027/AuthorGuidelines

## Human-only pre-upload actions

1. Register the genuine title, abstract, and complete author list in
   OpenReview before the abstract deadline.
2. Have a statistical-design or symmetry expert collision-check the narrow
   complete-pipeline composition against randomized-OA integration and
   antithetic-CV work.
3. Read the final PDF end to end and confirm that the AI-use disclosure
   accurately reflects every author's workflow.
4. Upload code/data artifacts anonymously and check every archive for author
   identity or machine-specific paths.
5. Do not enable `\iclrfinalcopy` for the review submission.
