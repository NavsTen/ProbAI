"""Learning from reality: correct model-based probabilities with observed outcomes."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Union

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression

from .core import ProbAIError

PathLike = Union[str, Path]
_FORMAT_VERSION = 1


class Calibrator:
    """Per-engine isotonic calibration.

    Engines such as ``signal`` or ``finance`` give the probability *under a model's
    assumptions*. If you record what actually happened (``add_feedback``), the
    calibrator learns a monotone correction: when the engine says 20% but the event
    really occurs 35% of the time, future 20% answers are mapped to ~35%.

    Until an engine has at least ``min_samples`` feedback points, nothing changes.
    """

    def __init__(self, min_samples: int = 30) -> None:
        if min_samples < 2:
            raise ProbAIError("min_samples must be at least 2.")
        self.min_samples = min_samples
        self._data: Dict[str, List[Tuple[float, int]]] = {}
        self._models: Dict[str, IsotonicRegression] = {}

    def add_feedback(self, engine: str, predicted: float, outcome: bool) -> None:
        if not 0.0 <= predicted <= 1.0:
            raise ProbAIError("predicted must be between 0 and 1.")
        self._data.setdefault(engine, []).append((float(predicted), int(bool(outcome))))

    def n_samples(self, engine: str) -> int:
        return len(self._data.get(engine, []))

    def fit(self) -> List[str]:
        """(Re)fit every engine that has enough feedback; return the engines fitted."""
        fitted = []
        for engine, rows in self._data.items():
            if len(rows) < self.min_samples:
                continue
            x = np.array([r[0] for r in rows])
            y = np.array([r[1] for r in rows])
            model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            model.fit(x, y)
            self._models[engine] = model
            fitted.append(engine)
        return fitted

    def is_fitted(self, engine: str) -> bool:
        return engine in self._models

    def apply(self, engine: str, probability: float) -> float:
        model = self._models.get(engine)
        if model is None:
            return probability
        return float(model.predict([probability])[0])

    def save(self, path: PathLike) -> None:
        joblib.dump(
            {"format": _FORMAT_VERSION, "min_samples": self.min_samples, "data": self._data, "models": self._models},
            path,
        )

    @classmethod
    def load(cls, path: PathLike) -> "Calibrator":
        """Load a saved calibrator. Only load files you trust: joblib uses pickle."""
        blob = joblib.load(path)
        if not isinstance(blob, dict) or blob.get("format") != _FORMAT_VERSION:
            raise ProbAIError("This file is not a ProbAI calibrator (or has an unsupported version).")
        cal = cls(min_samples=blob["min_samples"])
        cal._data = blob["data"]
        cal._models = blob["models"]
        return cal
