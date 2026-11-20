"""Counting events: Poisson (arrivals at a rate) and binomial (successes in n trials)."""

from __future__ import annotations

from typing import Any, Dict

from scipy.stats import binom, poisson

from ..core import Engine, EngineResult, Extraction, MissingParametersError
from ..extraction import extract_counting
from ..utils import (
    COMPARISON_SYMBOLS,
    as_int,
    as_positive,
    as_probability,
    first_not_none,
    infer_comparison,
    infer_count_after_comparison,
    normalise_comparison,
    prob_from_frozen,
)


class CountingEngine(Engine):
    name = "counting"
    description = (
        "How many times does an event happen? Poisson for arrivals at an average rate "
        "(calls, customers, errors), binomial for successes among n independent trials."
    )
    param_keys = frozenset({"rate", "duration"})
    calibratable = False

    def extract(self, text: str) -> Extraction:
        return extract_counting(text)

    def evidence(self, found: Dict[str, Any]) -> float:
        return 1.0 if ("rate" in found or ("n" in found and "p" in found)) else 0.0

    def estimate(self, text: str, params: Dict[str, Any]) -> EngineResult:
        k_raw = first_not_none(params.get("k"), infer_count_after_comparison(text))
        has_poisson = params.get("rate") is not None
        has_binomial = params.get("n") is not None and params.get("p") is not None

        missing = []
        if k_raw is None:
            missing.append("k")
        if not (has_poisson or has_binomial):
            missing.append("rate (Poisson)  or  n and p (binomial)")
        if missing:
            raise MissingParametersError(
                self.name,
                missing,
                "Example: params={'rate': 12, 'k': 15} or params={'n': 50, 'p': 0.02, 'k': 3}.",
            )

        k = as_int(k_raw, "k", 0)
        cmp = normalise_comparison(params.get("comparison"), infer_comparison(text) or "eq")

        if has_poisson:
            rate = as_positive(params["rate"], "rate", allow_zero=True)
            duration = as_positive(first_not_none(params.get("duration"), 1.0), "duration")
            mu = rate * duration
            prob = prob_from_frozen(poisson(mu), k, cmp)
            return EngineResult(
                probability=prob,
                method="Poisson distribution",
                explanation=(
                    f"Events arrive independently at {rate:.6g} per unit time; over {duration:.6g} "
                    f"unit(s) the expected count is {mu:.6g}. P(count {COMPARISON_SYMBOLS[cmp]} {k})."
                ),
                details={"rate": rate, "duration": duration, "mean": mu, "k": k, "comparison": cmp},
            )

        n = as_int(params["n"], "n", 1)
        p = as_probability(params["p"], "p")
        prob = prob_from_frozen(binom(n, p), k, cmp)
        return EngineResult(
            probability=prob,
            method="Binomial distribution",
            explanation=(
                f"{n} independent trials, each succeeding with probability {p:.6g}. "
                f"P(successes {COMPARISON_SYMBOLS[cmp]} {k})."
            ),
            details={"n": n, "p": p, "k": k, "comparison": cmp},
        )
