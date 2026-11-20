"""Bayesian updating: how likely is a hypothesis after seeing evidence?"""

from __future__ import annotations

import re
from typing import Any, Dict

from ..core import Engine, EngineResult, Extraction, MissingParametersError
from ..extraction import extract_bayes
from ..utils import as_float, as_probability, first_not_none, require


class BayesEngine(Engine):
    name = "bayes"
    description = (
        "Bayes' rule: probability of a hypothesis given evidence (medical tests, spam "
        "filters, fraud alerts, forensic matches)."
    )
    param_keys = frozenset({"prior", "sensitivity", "specificity", "false_positive_rate", "likelihood_ratio"})
    calibratable = False

    def extract(self, text: str) -> Extraction:
        return extract_bayes(text)

    def estimate(self, text: str, params: Dict[str, Any]) -> EngineResult:
        require(
            params,
            ["prior"],
            self.name,
            "prior is the base rate of the hypothesis before the evidence, e.g. params={'prior': 0.01, "
            "'sensitivity': 0.99, 'specificity': 0.95}.",
        )
        prior = as_probability(params["prior"], "prior")

        if params.get("likelihood_ratio") is not None:
            lr = as_float(params["likelihood_ratio"], "likelihood_ratio")
            if lr < 0:
                raise ValueError("'likelihood_ratio' must be non-negative.")
            if prior >= 1.0:
                posterior = 1.0
            else:
                odds = prior / (1.0 - prior) * lr
                posterior = odds / (1.0 + odds)
            return EngineResult(
                probability=posterior,
                method="Bayes' rule in odds form (posterior odds = prior odds x likelihood ratio)",
                explanation=f"Prior {prior:g} updated with a likelihood ratio of {lr:g}.",
                details={"prior": prior, "likelihood_ratio": lr},
            )

        if params.get("sensitivity") is None or (
            params.get("specificity") is None and params.get("false_positive_rate") is None
        ):
            raise MissingParametersError(
                self.name,
                ["sensitivity", "specificity (or false_positive_rate)"],
                "Alternatively pass likelihood_ratio.",
            )
        sensitivity = as_probability(params["sensitivity"], "sensitivity")
        if params.get("specificity") is not None:
            specificity = as_probability(params["specificity"], "specificity")
        else:
            specificity = 1.0 - as_probability(params["false_positive_rate"], "false_positive_rate")

        evidence = str(
            first_not_none(params.get("evidence"), "negative" if re.search(r"\bnegative\b", text.lower()) else "positive")
        ).lower()
        if evidence not in ("positive", "negative"):
            raise ValueError("'evidence' must be 'positive' or 'negative'.")

        if evidence == "positive":
            p_e_given_h, p_e_given_not_h = sensitivity, 1.0 - specificity
        else:
            p_e_given_h, p_e_given_not_h = 1.0 - sensitivity, specificity

        numerator = prior * p_e_given_h
        denominator = numerator + (1.0 - prior) * p_e_given_not_h
        if denominator == 0.0:
            raise ValueError("This evidence is impossible under the given numbers (division by zero).")
        posterior = numerator / denominator
        return EngineResult(
            probability=posterior,
            method="Bayes' theorem",
            explanation=(
                f"P(H) = {prior:g}, P(E|H) = {p_e_given_h:g}, P(E|not H) = {p_e_given_not_h:g} "
                f"({evidence} evidence). P(H|E) = P(E|H)P(H) / P(E)."
            ),
            details={
                "prior": prior,
                "evidence": evidence,
                "p_evidence": denominator,
                "p_e_given_h": p_e_given_h,
                "p_e_given_not_h": p_e_given_not_h,
            },
        )
