#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
/home/byunhanjoon/miniconda3/bin/python "${SCRIPT_DIR}/run_phase_a.py" --config "${PROJECT_DIR}/configs/phase_a_pilot.json"
