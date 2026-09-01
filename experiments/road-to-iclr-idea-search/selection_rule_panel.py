"""Consolidate the frozen 3 x 3 selection-rule orbit pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
DATASETS = ("adult", "churn", "otto")
FAMILIES = ("ordinal_forest", "native_histgb", "catboost_native")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=HERE / "selection_rule_panel.json")
    args = parser.parse_args()
    cells = []
    for dataset in DATASETS:
        for family in FAMILIES:
            path = HERE / f"selection_rule_{dataset}_{family}.json"
            data = json.loads(path.read_text())
            frozen = data["frozen_identity_selection_orbit"]
            selected = data["representation_wise_selection_orbit"]
            bootstrap = data.get("paired_row_bootstrap")
            cells.append(
                {
                    "dataset": dataset,
                    "family": family,
                    "factor_shape": data["factor_shape"],
                    "identity_config": data["selection_protocol"][
                        "canonical_identity_selected_config"
                    ],
                    "selection_counts": data["selection_counts"],
                    "selection_entropy_bits": data["selection_entropy_bits"],
                    "fraction_different_from_identity": data[
                        "fraction_different_from_identity_config"
                    ],
                    "frozen_schema_risk": frozen["anova"]["total"],
                    "selected_schema_risk": selected["anova"]["total"],
                    "schema_risk_ratio": data["selection_to_frozen_schema_risk_ratio"],
                    "schema_risk_difference": data[
                        "selection_minus_frozen_schema_risk"
                    ],
                    "orbit_mean_brier_difference": data[
                        "selection_minus_frozen_orbit_mean_brier"
                    ],
                    "paired_row_bootstrap": bootstrap,
                    "selection_decision_anova": data.get("selection_decision_anova"),
                    "configuration_switch_decomposition": data.get(
                        "configuration_switch_decomposition"
                    ),
                }
            )
    unstable = [cell for cell in cells if cell["fraction_different_from_identity"] > 0]
    risk_increases = [cell for cell in unstable if cell["schema_risk_difference"] > 0]
    ratios = np.asarray([cell["schema_risk_ratio"] for cell in unstable])
    result = {
        "status": "exploratory_conditional_on_candidate_search_and_fits",
        "design": {
            "datasets": list(DATASETS),
            "families": list(FAMILIES),
            "candidate_count_per_family": 4,
            "validation_rule": "minimum validation Brier with first-index tie break",
            "comparator": "select on identity representation then freeze",
            "selection_uses_test_labels": False,
        },
        "summary": {
            "cells": len(cells),
            "selection_unstable_cells": len(unstable),
            "unstable_cells_with_higher_schema_risk": len(risk_increases),
            "unstable_schema_risk_ratio_range": [float(ratios.min()), float(ratios.max())],
            "exact_two_sided_sign_test_p_if_condition_on_unstable_cells": 0.25,
            "sign_test_warning": "three selected unstable cells are descriptive and underpowered",
            "clearly_nonzero_brier_improvement_cells": [
                "churn/ordinal_forest"
            ],
        },
        "cells": cells,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
