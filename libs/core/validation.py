"""Fail-Closed Numerical Integrity Validation.

Central validation helpers used by every risk/execution gate. Motivated by the
IEEE 754 comparison hazard: ``NaN > x``, ``NaN < x`` and ``NaN == x`` are all
``False``, so any risk limit implemented with ordinary comparison operators is
silently *fail-open* on non-finite (NaN/±Inf) values. During a market shock —
exactly when the risk layer matters most — a corrupted data feed produces NaN
that must abort the check, never pass it.

Contract:
- :func:`is_finite_number` answers "is this a finite real number?" (bool is
  rejected: treating a boolean as a numeric metric is itself a corruption).
- :func:`non_finite_fields` returns the names of corrupt fields in a mapping so
  risk gates can fail with a precise, machine-auditable breach report.
- Gates consuming these helpers MUST reject on any non-finite input
  (fail-closed) rather than proceeding on the malignant comparison semantics
  above.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

__all__ = [
    "NumericalIntegrityError",
    "is_finite_number",
    "non_finite_fields",
    "assert_finite_scalar",
    "check_finite_mapping",
]


class NumericalIntegrityError(ValueError):
    """Raised when calculation inputs or state contain non-finite numbers."""


def is_finite_number(val: Any) -> bool:
    """True iff ``val`` is a finite ``int``/``float`` (no NaN, ±Inf, bool, None).

    Note: ``bool`` is a subclass of ``int``; it is rejected here because a
    boolean occupying a numeric metric slot is a type corruption, not data.
    """
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        return False
    return math.isfinite(val)


def assert_finite_scalar(val: Any, name: str) -> float:
    """Ensures a value is a finite float (not NaN, Inf, or non-numeric)."""
    if not is_finite_number(val):
        raise NumericalIntegrityError(f"Corrupt input: {name}={val!r} must be finite")
    return float(val)


def check_finite_mapping(metrics: Mapping[str, Any]) -> tuple[bool, list[str]]:
    """Non-raising check returning pass status and list of corrupt keys."""
    corrupt = non_finite_fields(metrics)
    return (len(corrupt) == 0, corrupt)


def non_finite_fields(fields: Mapping[str, Any]) -> list[str]:
    """Return ``name=value!r`` descriptors for every non-finite entry.

    The returned list is empty iff the whole mapping is numerically clean, and
    is designed to be dropped directly into breach/reason tuples so a corrupt
    input is named field-by-field in risk logs.
    """
    return [
        f"{name}={value!r}"
        for name, value in fields.items()
        if not is_finite_number(value)
    ]
