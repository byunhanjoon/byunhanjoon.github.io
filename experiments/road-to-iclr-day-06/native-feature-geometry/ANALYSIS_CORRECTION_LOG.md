# Analysis correction log

## 2026-08-29 — H1 precision mismatch

The first complete analysis marked H1 failed with maximum error
`7.146489444664894e-07`.  The frozen H1 gate specifies a float64 construction
audit at `1e-10`, but the analyzer reconstructed the Gram matrix from the
float32 table stored for training and compared it with the float64 compiler
Gram.  The already-frozen unit test correctly exercised the float64 compiler
and passed.

The analyzer was corrected to recompute the float64 compiler output for the H1
gate and to report the float32 cast discrepancy separately.  No threshold,
hypothesis gate, training artifact, or scientific outcome was changed.  This
correction was made after all 24 bundles completed and is therefore recorded
explicitly.

