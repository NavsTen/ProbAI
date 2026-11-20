"""Core types shared by every part of ProbAI."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple


class ProbAIError(Exception):
    """Base class for all ProbAI errors."""


class MissingParametersError(ProbAIError):
    """Raised when an engine needs input data that was not provided."""

    def __init__(self, engine: str, missing: Sequence[str], hint: str = "") -> None:
        self.engine = engine
        self.missing = list(missing)
        message = f"Engine '{engine}' needs parameter(s): {', '.join(self.missing)}."
        if hint:
            message += f" {hint}"
        super().__init__(message)


class LowConfidenceError(ProbAIError):
    """Raised when the router cannot decide which engine fits the question."""

    def __init__(self, candidates: Sequence[Tuple[str, float]], threshold: float) -> None:
        self.candidates = list(candidates)
        self.threshold = threshold
        top = ", ".join(f"{name} ({p:.0%})" for name, p in self.candidates)
        super().__init__(
            f"Could not decide which probability model fits this question "
            f"(best guesses: {top}; needed at least {threshold:.0%}). "
            f"Rephrase the question, add parameters, or choose an engine explicitly."
        )


@dataclass
class EngineResult:
    """What an engine returns: a probability plus a human-readable justification."""

    probability: float
    method: str
    explanation: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Extraction:
    """Parameters an engine found in the question text, plus notes on how they were read."""

    params: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


@dataclass
class Estimate:
    """The final answer returned to the user."""

    probability: float
    engine: str
    method: str
    explanation: str
    raw_probability: float
    calibrated: bool
    routing_confidence: float
    routing: List[Tuple[str, float]]
    details: Dict[str, Any] = field(default_factory=dict)
    extracted: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    language: str = "en"
    params_used: Dict[str, Any] = field(default_factory=dict)
    normalized_text: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Engine(ABC):
    """A probability model for one family of problems.

    Subclasses must set ``name`` and ``description`` and implement ``estimate``.

    ``param_keys`` lists parameter names that are distinctive for this engine;
    if the caller passes one of them, the router gives this engine a boost.
    ``calibratable`` marks engines whose output rests on real-world assumptions
    (and can therefore be corrected with observed outcomes). Exact mathematical
    engines (dice, coins, Bayes' rule) must leave it ``False``.
    """

    name: str = ""
    description: str = ""
    param_keys: FrozenSet[str] = frozenset()
    calibratable: bool = False

    def evidence(self, found: Dict[str, Any]) -> float:
        """How strongly the parameters found in the text point at this engine (0 to 1).

        Used by the router as a second opinion next to the text classifier: a question that
        contains "12 calls per hour" is evidence for the counting engine whatever its wording.
        """
        return 1.0 if any(k in found for k in self.param_keys) else 0.0

    def extract(self, text: str) -> Extraction:
        """Read this engine's input data out of the question text.

        Override it so users can write numbers in plain English instead of passing
        ``params``. Return only what the text clearly states; explicit ``params``
        given by the caller always win over extracted ones.
        """
        return Extraction()

    @abstractmethod
    def estimate(self, text: str, params: Dict[str, Any]) -> EngineResult:
        """Compute the probability for ``text`` using ``params`` as input data."""
