"""Write exact fANOVA spectral coefficients for frozen cover shapes."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from analyze_strength2_cover import (
    component_coefficients,
    incidence_covariance,
    strength1_family,
    strength2_family,
)


HERE = Path(__file__).resolve().parent


def main() -> None:
    from analyze_strength3_cover import strength3_family

    rows = []
    payload = {}
    for category_levels, class_levels in ((4, 2), (1, 2), (4, 1), (1, 1)):
        cardinalities = (4, category_levels, class_levels, 4)
        key = "x".join(str(value) for value in cardinalities)
        payload[key] = {}
        for strength, family in (
            (1, strength1_family(category_levels, class_levels)),
            (2, strength2_family(category_levels, class_levels)),
            (3, strength3_family(category_levels, class_levels)),
        ):
            covariance = incidence_covariance(family, cardinalities)
            coefficients = component_coefficients(covariance, cardinalities)
            payload[key][f"strength_{strength}"] = {
                "design_count": len(family), "coefficients": coefficients,
            }
            for component, coefficient in coefficients.items():
                rows.append({
                    "factor_shape": key, "strength": strength,
                    "design_count": len(family), "component": component,
                    "component_order": component.count(":") + 1,
                    "coefficient": coefficient,
                })
    output = HERE / "results"
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "cover_component_coefficients.csv", index=False)
    (output / "cover_component_coefficients.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": "complete", "factor_shapes": len(payload), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
