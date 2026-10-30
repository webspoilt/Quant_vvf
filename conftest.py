"""Ensure the repository root is importable so `libs.*` resolves everywhere.

 pytest/bias-checks/mypy all import `libs.core.*` and `libs.data.*` directly
 from the repo root without an editable install; this conftest pins sys.path
 so notebooks, sim, and actors share identical import resolution.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
