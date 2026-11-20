"""Built-in probability engines.

To add your own engine, subclass :class:`probai.core.Engine`, add training
examples labelled with its ``name`` to the router data, and register it with
``ProbAI.register_engine``. See CONTRIBUTING.md.
"""

from __future__ import annotations

from typing import Dict

from ..core import Engine
from .bayes import BayesEngine
from .classical import ClassicalEngine
from .counting import CountingEngine
from .empirical import EmpiricalEngine
from .finance import FinanceEngine
from .reliability import ReliabilityEngine
from .signal import SignalEngine

__all__ = [
    "BayesEngine",
    "ClassicalEngine",
    "CountingEngine",
    "EmpiricalEngine",
    "FinanceEngine",
    "ReliabilityEngine",
    "SignalEngine",
    "default_engines",
]


def default_engines() -> Dict[str, Engine]:
    """Return a fresh registry containing every built-in engine."""
    engines = [
        ClassicalEngine(),
        CountingEngine(),
        SignalEngine(),
        BayesEngine(),
        FinanceEngine(),
        ReliabilityEngine(),
        EmpiricalEngine(),
    ]
    return {e.name: e for e in engines}
