#!/usr/bin/env bash
set -euo pipefail

mitra_prefix="${REPARAM_MITRA_ENV:-/data/byunhanjoon/reparam_mitra_env}"
conda_executable="${CONDA_EXE:-/home/byunhanjoon/miniconda3/bin/conda}"
pip_cache="${REPARAM_PIP_CACHE:-/data/byunhanjoon/pip-cache}"

if [[ ! -x "${mitra_prefix}/bin/python" ]]; then
  "${conda_executable}" create --yes --prefix "${mitra_prefix}" python=3.11
fi

PIP_CACHE_DIR="${pip_cache}" "${mitra_prefix}/bin/python" -m pip install "autogluon.tabular[mitra]==1.6.1"
PIP_CACHE_DIR="${pip_cache}" "${mitra_prefix}/bin/python" -m pip install --force-reinstall \
  "torch==2.10.0" --index-url https://download.pytorch.org/whl/cu128

"${mitra_prefix}/bin/python" - <<'PY'
import importlib.metadata
import torch
from autogluon.tabular.models.mitra.mitra_model import MitraModel

assert torch.cuda.is_available(), "Mitra environment cannot see CUDA"
print("autogluon.tabular", importlib.metadata.version("autogluon.tabular"))
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("device", torch.cuda.get_device_name(0))
print("adapter", MitraModel.__module__)
PY
