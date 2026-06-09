"""core/labels_tr.py - whitelist + Turkish display strings.

Single source of truth for:
  * which v3 class names this engine is allowed to act on (WHITELIST), and
  * the Turkish text shown to humans (TR_DISPLAY / tr()).

Code and comments are English; only the *values* in TR_DISPLAY are Turkish.
"""
from __future__ import annotations

import warnings

# v3 class NAMES this 3-class engine accepts. Every other v3 class
# (Mask, NO-Hardhat, NO-Mask, NO-Safety Vest, Safety Cone, machinery,
# vehicle) is dropped everywhere in the system.
WHITELIST: frozenset[str] = frozenset({"Person", "Hardhat", "Safety Vest"})

# Human-facing Turkish strings, keyed by internal state / label keys.
TR_DISPLAY: dict[str, str] = {
    # worker compliance states
    "worker_ok": "Kurallara uygun",
    "helmet_missing": "Baret takılmamış",
    "vest_missing": "Güvenlik yeleği yok",
    # class display names
    "person": "İşçi",
    "hardhat": "Baret",
    "vest": "Güvenlik yeleği",
}


def tr(key: str) -> str:
    """Return the Turkish display string for `key`; fall back to the key itself."""
    if key in TR_DISPLAY:
        return TR_DISPLAY[key]
    warnings.warn(f"tr(): no Turkish string for key {key!r}; returning the key.", stacklevel=2)
    return key


def is_allowed(class_name: str) -> bool:
    """True if a v3 class name is within the 3-class scope."""
    return class_name in WHITELIST


__all__ = ["WHITELIST", "TR_DISPLAY", "tr", "is_allowed"]
