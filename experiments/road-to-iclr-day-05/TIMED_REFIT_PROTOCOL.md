# Frozen protocol: end-to-end late-panel timing

Regenerate all 12 late-source complete tensors into a separate directory with
the same configuration and one BLAS/model thread per process. Record setup,
128-fit loop, and end-to-end (including tensor serialization) wall time using
`time.perf_counter`. Run at most four cells concurrently, so per-cell timings
reflect a modest shared-machine workload rather than isolated benchmarks.

Verify every regenerated validation/test tensor and label array exactly equals
the original deterministic artifact. Report timing by model family and source,
throughput in fits/second, and output tensor bytes. This is a local-system
audit, not a portable hardware or cost claim; no comparison method requires
fewer model fits at a fixed declared budget.
