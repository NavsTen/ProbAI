"""Reliability engineering: failure and survival of components over time."""

from __future__ import annotations

import math
import re
from typing import Any, Dict

from ..core import Engine, EngineResult, Extraction, MissingParametersError
from ..extraction import extract_reliability
from ..utils import as_positive, require

_SURVIVAL_WORDS = (
    r"\b(surviv\w*|still (?:work\w*|run\w*|function\w*|operat\w*)|keeps? (?:working|running)|"
    r"last\w*|longer than|without (?:fail\w*|break\w*))\b"
)


class ReliabilityEngine(Engine):
    name = "reliability"
    description = (
        "Lifetimes and failures of machines, devices and components: exponential model "
        "(MTBF) and Weibull model (shape, scale)."
    )
    param_keys = frozenset({"mtbf", "shape", "scale"})
    calibratable = True

    def extract(self, text: str) -> Extraction:
        return extract_reliability(text)

    def estimate(self, text: str, params: Dict[str, Any]) -> EngineResult:
        has_exp = params.get("mtbf") is not None
        has_weibull = params.get("shape") is not None and params.get("scale") is not None
        missing = []
        if params.get("time") is None:
            missing.append("time")
        if not (has_exp or has_weibull):
            missing.append("mtbf  (or shape and scale for Weibull)")
        if missing:
            raise MissingParametersError(
                self.name,
                missing,
                "Example: params={'mtbf': 10000, 'time': 2000} (same time unit for both).",
            )
        time = as_positive(params["time"], "time", allow_zero=True)
        outcome = str(
            params.get("outcome")
            or ("survival" if re.search(_SURVIVAL_WORDS, text.lower()) else "failure")
        ).lower()
        if outcome not in ("failure", "survival"):
            raise ValueError("'outcome' must be 'failure' or 'survival'.")

        if has_exp:
            mtbf = as_positive(params["mtbf"], "mtbf")
            hazard = time / mtbf
            method = "Exponential lifetime (constant failure rate)"
            detail = {"mtbf": mtbf}
        else:
            shape = as_positive(params["shape"], "shape")
            scale = as_positive(params["scale"], "scale")
            try:
                hazard = (time / scale) ** shape
            except OverflowError:
                hazard = math.inf
            method = "Weibull lifetime"
            detail = {"shape": shape, "scale": scale}

        survival = math.exp(-hazard)
        prob = survival if outcome == "survival" else -math.expm1(-hazard)
        return EngineResult(
            probability=prob,
            method=method,
            explanation=f"Probability that the component {'survives past' if outcome == 'survival' else 'fails before'} time {time:g}.",
            details={"outcome": outcome, "time": time, "survival": survival, **detail},
        )
