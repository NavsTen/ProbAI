"""Data-driven estimate from observed outcomes (Beta-Binomial posterior)."""

from __future__ import annotations

from typing import Any, Dict

from scipy.stats import beta

from ..core import Engine, EngineResult, Extraction, MissingParametersError
from ..extraction import extract_empirical
from ..utils import as_int, as_positive, as_probability, first_not_none


class EmpiricalEngine(Engine):
    name = "empirical"
    description = (
        "When you have past data (k successes in n trials): the probability of the next "
        "outcome with a credible interval, via a Beta-Binomial posterior."
    )
    param_keys = frozenset({"successes", "trials", "outcomes"})
    calibratable = False

    def extract(self, text: str) -> Extraction:
        return extract_empirical(text)

    def evidence(self, found: Dict[str, Any]) -> float:
        return 1.0 if ("successes" in found and "trials" in found) else 0.0

    def estimate(self, text: str, params: Dict[str, Any]) -> EngineResult:
        outcomes = params.get("outcomes")
        if outcomes is not None:
            values = list(outcomes)
            if any(v not in (0, 1) for v in values):
                raise ValueError("'outcomes' must contain only 0/1 (or False/True).")
            successes, trials = sum(int(v) for v in values), len(values)
        elif params.get("successes") is not None and params.get("trials") is not None:
            successes = as_int(params["successes"], "successes", 0)
            trials = as_int(params["trials"], "trials", 0)
            if successes > trials:
                raise ValueError("'successes' cannot exceed 'trials'.")
        else:
            raise MissingParametersError(
                self.name,
                ["successes and trials  (or outcomes)"],
                "Example: params={'successes': 37, 'trials': 400}.",
            )

        a = as_positive(first_not_none(params.get("alpha"), 1.0), "alpha")
        b = as_positive(first_not_none(params.get("beta"), 1.0), "beta")
        level = as_probability(first_not_none(params.get("level"), 0.95), "level")
        post_a, post_b = a + successes, b + (trials - successes)
        mean = post_a / (post_a + post_b)
        low = float(beta.ppf((1.0 - level) / 2.0, post_a, post_b))
        high = float(beta.ppf(1.0 - (1.0 - level) / 2.0, post_a, post_b))
        return EngineResult(
            probability=mean,
            method="Beta-Binomial posterior mean",
            explanation=(
                f"{successes} successes in {trials} trials with a Beta({a:g}, {b:g}) prior "
                f"(Beta(1, 1) = Laplace smoothing). {level:.0%} credible interval: "
                f"[{low:.4f}, {high:.4f}]. Assumes trials are independent and identically distributed."
            ),
            details={
                "successes": successes,
                "trials": trials,
                "credible_interval": [low, high],
                "level": level,
                "observed_frequency": (successes / trials) if trials else None,
            },
        )
