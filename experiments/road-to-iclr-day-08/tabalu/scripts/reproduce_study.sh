#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="/home/byunhanjoon/miniconda3/bin/python"

"${PYTHON_BIN}" "${SCRIPT_DIR}/run_phase_a.py" --config "${PROJECT_DIR}/configs/phase_a_pilot.json"
"${PYTHON_BIN}" "${SCRIPT_DIR}/run_phase_b_recovery.py" --config "${PROJECT_DIR}/configs/phase_b_recovery.json"
"${PYTHON_BIN}" "${SCRIPT_DIR}/run_phase_c_operand.py" --config "${PROJECT_DIR}/configs/phase_c_operand.json"
"${PYTHON_BIN}" "${SCRIPT_DIR}/run_phase_d_regime.py" --config "${PROJECT_DIR}/configs/phase_d_regime.json"
"${PYTHON_BIN}" "${SCRIPT_DIR}/run_phase_e_temporal.py" --config "${PROJECT_DIR}/configs/phase_e_temporal.json"
"${PYTHON_BIN}" "${SCRIPT_DIR}/run_exact_execution_ablation.py" --config "${PROJECT_DIR}/configs/exact_execution_ablation.json"
"${PYTHON_BIN}" "${SCRIPT_DIR}/run_phase_f_typed.py" --config "${PROJECT_DIR}/configs/phase_f_typed.json"
"${PYTHON_BIN}" "${SCRIPT_DIR}/run_phase_g_residual.py" --config "${PROJECT_DIR}/configs/phase_g_residual.json"
"${PYTHON_BIN}" "${SCRIPT_DIR}/run_real_bike_temporal.py" --config "${PROJECT_DIR}/configs/real_bike_temporal.json"
"${PYTHON_BIN}" "${SCRIPT_DIR}/run_real_bike_bounded_diagnostic.py" --config "${PROJECT_DIR}/configs/real_bike_bounded_diagnostic.json"
"${PYTHON_BIN}" "${SCRIPT_DIR}/run_general_pilot.py" --config "${PROJECT_DIR}/configs/general_pilot.json"
"${PYTHON_BIN}" "${SCRIPT_DIR}/run_depth_scaling.py" --config "${PROJECT_DIR}/configs/depth_scaling.json"
"${PYTHON_BIN}" "${SCRIPT_DIR}/run_regime_scaling.py" --config "${PROJECT_DIR}/configs/regime_scaling.json"
"${PYTHON_BIN}" "${SCRIPT_DIR}/audit_study.py"
