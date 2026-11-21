"""Finance and credit risk: default over a horizon, and price-threshold probabilities.

Both models are textbook simplifications (constant hazard rate; geometric Brownian
motion). Real markets have fat tails and regime changes. Nothing here is financial advice.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict

from scipy.stats import norm

from ..core import Engine, EngineResult, Extraction, MissingParametersError
from ..extraction import extract_finance
from ..utils import as_float, as_positive, as_probability, first_not_none, require

_PRICE_WORDS = (
    r"\b(stock|share|shares|price|index|bitcoin|btc|s&p|nasdaq|dow|etf|currency|euro|"
    r"exceeds?|above|below|portfolio|drops?|falls?|rises?)\b"
)
_ABOVE_WORDS = r"\b(above|exceeds?|over|rises?|higher|greater|reaches|surpass\w*)\b"


class FinanceEngine(Engine):
    name = "finance"
    description = (
        "Banking and markets: probability of default within a horizon (constant hazard) and "
        "probability that a price ends above/below a level (lognormal model)."
    )
    param_keys = frozenset({"hazard_rate", "annual_pd", "spot", "strike", "volatility"})
    calibratable = True

    def extract(self, text: str) -> Extraction:
        return extract_finance(text)

    def estimate(self, text: str, params: Dict[str, Any]) -> EngineResult:
        task = params.get("task") or self._infer_task(text, params)
        if task == "default":
            return self._default(params)
        if task == "price_threshold":
            return self._price_threshold(text, params)
        raise ValueError(f"Unknown task {task!r}; use 'default' or 'price_threshold'.")

    @staticmethod
    def _infer_task(text: str, params: Dict[str, Any]) -> str:
        if params.get("hazard_rate") is not None or params.get("annual_pd") is not None:
            return "default"
        if any(params.get(k) is not None for k in ("spot", "strike", "volatility")):
            return "price_threshold"
        return "price_threshold" if re.search(_PRICE_WORDS, text.lower()) else "default"

    def _default(self, params: Dict[str, Any]) -> EngineResult:
        if params.get("hazard_rate") is None and params.get("annual_pd") is None:
            raise MissingParametersError(
                self.name,
                ["hazard_rate (or annual_pd)", "years"],
                "Example: params={'annual_pd': 0.02, 'years': 3}.",
            )
        require(
            params,
            ["years"],
            self.name,
            "State the time horizon in the question, e.g. 'within 3 years' or 'over 6 months'.",
        )
        years = as_positive(params["years"], "years")
        if params.get("hazard_rate") is not None:
            lam = as_positive(params["hazard_rate"], "hazard_rate", allow_zero=True)
            prob = -math.expm1(-lam * years)
            method = "Constant hazard rate: P(default by t) = 1 - exp(-lambda * t)"
            detail = {"hazard_rate": lam}
        else:
            pd = as_probability(params["annual_pd"], "annual_pd")
            prob = 1.0 if pd == 1.0 else -math.expm1(years * math.log1p(-pd))
            method = "Independent yearly defaults: P(default by t) = 1 - (1 - PD)^t"
            detail = {"annual_pd": pd}
        return EngineResult(
            probability=prob,
            method=method,
            explanation=f"Probability of at least one default within {years:g} year(s).",
            details={"task": "default", "years": years, **detail},
        )

    def _price_threshold(self, text: str, params: Dict[str, Any]) -> EngineResult:
        require(
            params,
            ["spot", "strike", "volatility", "years"],
            self.name,
            "Example: params={'spot': 100, 'strike': 90, 'volatility': 0.2, 'years': 1}.",
        )
        spot = as_positive(params["spot"], "spot")
        strike = as_positive(params["strike"], "strike")
        vol = as_positive(params["volatility"], "volatility")
        years = as_positive(params["years"], "years")
        drift = as_float(first_not_none(params.get("drift"), 0.0), "drift")
        direction = str(
            first_not_none(
                params.get("direction"),
                "above" if re.search(_ABOVE_WORDS, text.lower()) else "below",
            )
        ).lower()
        if direction not in ("above", "below"):
            raise ValueError("'direction' must be 'above' or 'below'.")

        z = ((math.log(strike) - math.log(spot)) - (drift - 0.5 * vol**2) * years) / (vol * math.sqrt(years))
        prob = float(norm.cdf(z) if direction == "below" else norm.sf(z))
        return EngineResult(
            probability=prob,
            method="Geometric Brownian motion (lognormal price at the horizon)",
            explanation=(
                f"Price starts at {spot:g}, annual drift {drift:g}, volatility {vol:g}. "
                f"Probability it ends {direction} {strike:g} after {years:g} year(s). "
                f"Real markets have fatter tails than this model."
            ),
            details={"task": "price_threshold", "direction": direction, "z": z, "drift": drift},
        )
