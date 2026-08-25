"""P2 cross-atoms were pre-registered but intentionally not launched on Day 3.

The P0 matrix has priority.  This entry point exists to keep the experiment
inventory explicit and fails closed rather than silently producing an
uncontrolled feature-cross result.
"""

raise SystemExit("P2 cross-atoms not launched: fixed P0/P1 compute budget took priority")
