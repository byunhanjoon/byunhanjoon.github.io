# Systems-efficiency audit

Status: diagnostic, specified before running the manifest scan/overhead timing.

Scan all 177 principal tensor manifests for per-fit elapsed, duration, CPU, or
wall-clock fields. Separately benchmark construction of the full mixed cover
graph, 1,024 sequential four-pack actions, and the eight-coset resolution.

The benchmark may establish that action generation is operationally small, but
it cannot establish an end-to-end speedup without fit timings. If no complete
timing telemetry exists, the paper must use “equal fit budget” and “statistical
efficiency,” not latency, throughput, energy, or cost speedup language.
