"""Keep this experiment's deliberately local ``src`` package ahead of sibling projects."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
while str(PROJECT) in sys.path:
    sys.path.remove(str(PROJECT))
sys.path.insert(0, str(PROJECT))
loaded = sys.modules.get("src")
if loaded is not None and Path(getattr(loaded, "__file__", "")).resolve().parent != PROJECT / "src":
    for name in list(sys.modules):
        if name == "src" or name.startswith("src."):
            del sys.modules[name]
