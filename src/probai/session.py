"""Conversation context: refine the previous question with a follow-up.

    s = Session()
    s.ask("Probability of bit error on a BPSK radio link at 8 dB")
    s.ask("and at 10 dB?")        # same model, same question, new SNR
    s.ask("а при 12 дБ?")          # the same in Russian
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from .core import Estimate, LowConfidenceError
from .language import detect_language, normalize_text
from .model import ProbAI, _t
from .utils import infer_comparison

_FOLLOW_UP = re.compile(
    r"^\s*(?:and\b|what if\b|what about\b|how about\b|and if\b|then\b|now\b|same\b|but with\b|instead\b|"
    r"а\b|и\b|теперь\b|тогда\b|что если\b|как насчет\b|как насчёт\b|то же самое\b|но при\b)",
    re.I,
)


class Session:
    """Remembers the last answer so that short follow-up questions work.

    A follow-up keeps the previous model and every number it used, and changes only what the
    new text states: "and at 10 dB?", "what if the rate is 20?", "а если 5 лет?".
    A text is treated as a follow-up when it starts like one *and* contains a new number for the
    previous model, or when it cannot be routed on its own but does contain such a number.
    """

    def __init__(self, ai: Optional[ProbAI] = None, **kwargs: Any) -> None:
        self.ai = ai or ProbAI(**kwargs)
        self.history: List[Tuple[str, Estimate]] = []

    @property
    def last(self) -> Optional[Estimate]:
        return self.history[-1][1] if self.history else None

    def reset(self) -> None:
        self.history.clear()

    # ----------------------------------------------------------------- ask
    def ask(self, text: str, params: Optional[Dict[str, Any]] = None, engine: Optional[str] = None) -> Estimate:
        explicit = dict(params or {})
        if self.history and engine is None and _FOLLOW_UP.match(text):
            est = self._follow_up(text, explicit)
            if est is not None:
                return self._remember(self.history[-1][0], est)
        try:
            est = self.ai.estimate(text, explicit, engine)
        except LowConfidenceError:
            if self.history and engine is None:
                est = self._follow_up(text, explicit)
                if est is not None:
                    return self._remember(self.history[-1][0], est)
            raise
        return self._remember(text, est)

    def _remember(self, question: str, est: Estimate) -> Estimate:
        self.history.append((question, est))
        return est

    def _follow_up(self, text: str, explicit: Dict[str, Any]) -> Optional[Estimate]:
        prev_question, prev = self.history[-1]
        language = detect_language(text)
        normalized = normalize_text(text)
        found = self.ai.engines[prev.engine].extract(normalized).params
        if not found and not explicit:
            return None  # nothing new for the previous model: this is a new question
        merged = {**prev.params_used, **found, **explicit}
        comparison = infer_comparison(normalized)
        if comparison and any(k in found for k in ("k", "target")):
            merged["comparison"] = comparison
        est = self.ai.estimate(prev_question, merged, engine=prev.engine, language=language)
        changed = {k: v for k, v in {**found, **explicit}.items() if prev.params_used.get(k) != v}
        shown = ", ".join(f"{k}={v}" for k, v in changed.items()) or "-"
        est.notes.append(
            _t(
                language,
                f"Follow-up: kept the previous question and model '{prev.engine}'; changed: {shown}.",
                f"Уточнение: сохранён предыдущий вопрос и модель «{prev.engine}»; изменено: {shown}.",
            )
        )
        return est
