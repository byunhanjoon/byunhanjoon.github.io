"""Independent numerical audit of the post-hoc simplex-QP solver."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize

from aggregation_aware_followup import risk_objective, solve_weights


HERE = Path(__file__).resolve().parent


def main() -> None:
    rng = np.random.default_rng(20261107)
    rows = []
    for system in range(21):
        candidates = (16, 32, 64)[system % 3]
        outputs = 1 + system % 4
        discrepancy = rng.normal(0, 0.35, size=(candidates, outputs)).astype(np.float32)
        variance = rng.lognormal(-1.0, 0.65, size=candidates).astype(np.float32)
        tensor_d = torch.from_numpy(discrepancy)[None]
        tensor_v = torch.from_numpy(variance)[None]
        weights, diagnostics = solve_weights(tensor_d, tensor_v, "full")
        fista_w = weights[0].numpy().astype(np.float64)

        def objective(value: np.ndarray) -> float:
            aggregate = value @ discrepancy.astype(np.float64)
            return float(aggregate @ aggregate + (np.square(value) * variance).sum())

        def gradient(value: np.ndarray) -> np.ndarray:
            aggregate = value @ discrepancy.astype(np.float64)
            return 2.0 * discrepancy.astype(np.float64) @ aggregate + 2.0 * value * variance

        reference = minimize(
            objective,
            np.full(candidates, 1.0 / candidates),
            jac=gradient,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * candidates,
            constraints={"type": "eq", "fun": lambda value: value.sum() - 1.0},
            options={"ftol": 1e-13, "maxiter": 2000},
        )
        if not reference.success:
            raise AssertionError(reference.message)
        rows.append({
            "system": system,
            "candidates": candidates,
            "outputs": outputs,
            "fista_objective": objective(fista_w),
            "slsqp_objective": float(reference.fun),
            "objective_gap": objective(fista_w) - float(reference.fun),
            "simplex_error": abs(float(fista_w.sum()) - 1.0),
            "minimum_weight": float(fista_w.min()),
            "objective_nonincrease_fraction": diagnostics["objective_nonincrease_fraction"],
        })

    audit = {
        "status": "complete",
        "systems": len(rows),
        "max_objective_gap_vs_slsqp": max(row["objective_gap"] for row in rows),
        "max_simplex_error": max(row["simplex_error"] for row in rows),
        "minimum_weight": min(row["minimum_weight"] for row in rows),
        "objective_nonincrease_min": min(row["objective_nonincrease_fraction"] for row in rows),
        "passed": bool(
            max(row["objective_gap"] for row in rows) <= 1e-5
            and max(row["simplex_error"] for row in rows) <= 1e-6
            and min(row["minimum_weight"] for row in rows) >= -1e-8
        ),
        "rows": rows,
    }
    if not audit["passed"]:
        raise AssertionError(json.dumps(audit, indent=2))
    (HERE / "aggregation_solver_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({key: value for key, value in audit.items() if key != "rows"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
