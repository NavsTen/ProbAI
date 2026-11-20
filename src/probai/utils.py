"""Small helpers: validation, number parsing and discrete-probability maths."""

from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, Optional

import numpy as np

from .core import MissingParametersError

NUMBER_WORDS: Dict[str, int] = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "twenty": 20,
}
# Regex fragment that matches a digit string or a number word (one capture group).
NUM = r"(\d+|" + "|".join(NUMBER_WORDS) + r")"

COMPARISON_SYMBOLS = {"eq": "=", "ge": ">=", "gt": ">", "le": "<=", "lt": "<"}

_COMPARISON_ALIASES = {
    "eq": {"eq", "==", "=", "exactly", "equal", "equals"},
    "ge": {"ge", ">=", "≥", "at_least", "atleast", "no_less_than"},
    "gt": {"gt", ">", "more_than", "greater_than", "above", "over"},
    "le": {"le", "<=", "≤", "at_most", "atmost", "no_more_than"},
    "lt": {"lt", "<", "less_than", "fewer_than", "below", "under"},
}

# Order matters: "at least" must be tested before "more than", and so on.
_COMPARISON_TEXT_PATTERNS = [
    ("ge", r"\bat least\b|\bor more\b|\bor higher\b|\bor greater\b|\bno less than\b|>=|≥"),
    ("le", r"\bat most\b|\bor less\b|\bor fewer\b|\bor lower\b|\bno more than\b|<=|≤"),
    ("gt", r"\bmore than\b|\bgreater than\b|\bhigher than\b|\bexceeds?\b|\babove\b|\bover\b|>"),
    ("lt", r"\bless than\b|\bfewer than\b|\blower than\b|\bbelow\b|\bunder\b|<"),
    ("eq", r"\bexactly\b|\bequals?\b|\bequal to\b|=="),
]


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def require(params: Dict[str, Any], keys: Iterable[str], engine: str, hint: str = "") -> None:
    missing = [k for k in keys if params.get(k) is None]
    if missing:
        raise MissingParametersError(engine, missing, hint)


def as_float(value: Any, name: str) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Parameter '{name}' must be a number, got {value!r}.") from None
    if math.isnan(x) or math.isinf(x):
        raise ValueError(f"Parameter '{name}' must be finite, got {value!r}.")
    return x


def as_probability(value: Any, name: str) -> float:
    x = as_float(value, name)
    if not 0.0 <= x <= 1.0:
        raise ValueError(f"Parameter '{name}' must be between 0 and 1, got {x}.")
    return x


def as_positive(value: Any, name: str, allow_zero: bool = False) -> float:
    x = as_float(value, name)
    if x < 0 or (x == 0 and not allow_zero):
        bound = "non-negative" if allow_zero else "positive"
        raise ValueError(f"Parameter '{name}' must be {bound}, got {x}.")
    return x


def as_int(value: Any, name: str, minimum: int = 0) -> int:
    x = as_float(value, name)
    if x != int(x):
        raise ValueError(f"Parameter '{name}' must be a whole number, got {value!r}.")
    if int(x) < minimum:
        raise ValueError(f"Parameter '{name}' must be at least {minimum}, got {int(x)}.")
    return int(x)


def first_not_none(*values: Any) -> Any:
    for v in values:
        if v is not None:
            return v
    return None


# --------------------------------------------------------------------------- #
# Text parsing
# --------------------------------------------------------------------------- #
def parse_int_token(token: str) -> int:
    token = token.lower()
    return int(token) if token.isdigit() else NUMBER_WORDS[token]


def normalise_comparison(value: Any, default: str = "eq") -> str:
    if value is None:
        return default
    v = str(value).strip().lower().replace(" ", "_")
    for key, aliases in _COMPARISON_ALIASES.items():
        if v in aliases:
            return key
    raise ValueError(
        f"Unknown comparison {value!r}. Use one of: eq, ge, gt, le, lt "
        f"(or 'exactly', 'at_least', 'more_than', 'at_most', 'less_than')."
    )


def infer_comparison(text: str) -> Optional[str]:
    t = text.lower()
    for key, pattern in _COMPARISON_TEXT_PATTERNS:
        if re.search(pattern, t):
            return key
    return None


def infer_count_after_comparison(text: str) -> Optional[int]:
    """Find 'at least 3', 'exactly five', 'more than 10' ... and return the number."""
    m = re.search(
        rf"\b(?:at least|at most|exactly|more than|less than|fewer than|greater than|"
        rf"no more than|no less than)\s+{NUM}\b",
        text.lower(),
    )
    return parse_int_token(m.group(1)) if m else None


# --------------------------------------------------------------------------- #
# Discrete probability helpers
# --------------------------------------------------------------------------- #
def clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def prob_from_frozen(dist: Any, k: int, comparison: str) -> float:
    """P(X <comparison> k) for a frozen scipy discrete distribution."""
    if comparison == "eq":
        p = dist.pmf(k)
    elif comparison == "le":
        p = dist.cdf(k)
    elif comparison == "lt":
        p = dist.cdf(k - 1)
    elif comparison == "ge":
        p = dist.sf(k - 1)
    elif comparison == "gt":
        p = dist.sf(k)
    else:
        raise ValueError(f"Unknown comparison {comparison!r}.")
    return clip01(p)


def prob_from_pmf(pmf: np.ndarray, offset: int, k: int, comparison: str) -> float:
    """P(X <comparison> k) where ``pmf[i]`` is P(X = offset + i)."""
    values = offset + np.arange(len(pmf))
    masks = {
        "eq": values == k,
        "ge": values >= k,
        "gt": values > k,
        "le": values <= k,
        "lt": values < k,
    }
    if comparison not in masks:
        raise ValueError(f"Unknown comparison {comparison!r}.")
    return clip01(float(pmf[masks[comparison]].sum()))
