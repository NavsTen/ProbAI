"""The public entry point: :class:`ProbAI`."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from . import ru_output
from .calibration import Calibrator
from .core import Engine, Estimate, LowConfidenceError, MissingParametersError, ProbAIError
from .engines import default_engines
from .language import detect_language, normalize_text
from .router import Router


def _t(language: str, en: str, ru: str) -> str:
    return ru if language == "ru" else en


class ProbAI:
    """Routes a question to the right probability model and computes the answer.

    Parameters
    ----------
    router:
        A trained :class:`Router`. By default one is trained on the bundled seed data.
    calibrator:
        Optional :class:`Calibrator` that corrects model-based engines with feedback.
    min_confidence:
        If the router's best guess is below this, ``LowConfidenceError`` is raised
        instead of guessing. Pass ``engine=...`` to ``estimate`` to bypass routing.
    param_boost:
        Score added to an engine when the caller passes one of its distinctive
        parameters (e.g. ``snr_db`` points to the ``signal`` engine).
    evidence_boost:
        Score added to an engine for numbers *found in the question text* that only that
        kind of problem has ("12 calls per hour", "8 dB", "99% sensitivity"). It acts as a
        second opinion next to the text classifier. Off by default (0): in my tests the
        classifier alone was already right and this signal added noise, because phrases such as
        "3 per hour" or "above 15" appear in many kinds of questions. Try 0.1-0.3 on your data.
    auto_extract:
        Read numbers (rates, percentages, dB, durations...) out of the question text.
        Parameters passed explicitly in ``params`` always override extracted ones.
    fallback:
        When the best engine lacks data, try the next most likely engine if the text
        contains everything *it* needs, and say so in ``Estimate.notes``.

    Questions may be written in English or Russian; the answer is in the question's language.
    """

    def __init__(
        self,
        router: Optional[Router] = None,
        calibrator: Optional[Calibrator] = None,
        min_confidence: float = 0.30,
        param_boost: float = 0.5,
        auto_extract: bool = True,
        evidence_boost: float = 0.0,
        fallback: bool = True,
    ) -> None:
        self.router = router or Router.from_seed()
        self.calibrator = calibrator
        self.min_confidence = min_confidence
        self.param_boost = param_boost
        self.auto_extract = auto_extract
        self.evidence_boost = evidence_boost
        self.fallback = fallback
        self.engines: Dict[str, Engine] = default_engines()

    # ------------------------------------------------------------- registry
    def register_engine(self, engine: Engine) -> None:
        """Add or replace an engine. Retrain the router so it learns to pick it."""
        if not engine.name:
            raise ProbAIError("An engine must have a non-empty 'name'.")
        self.engines[engine.name] = engine

    # -------------------------------------------------------------- routing
    def route(self, text: str, params: Optional[Dict[str, Any]] = None) -> List[Tuple[str, float]]:
        """Return engines ranked by how well they fit the question and the given data."""
        params = params or {}
        scores = {name: 0.0 for name in self.engines}
        for name, p in self.router.predict_proba(text):
            if name in scores:
                scores[name] = p

        claimed = [n for n, e in self.engines.items() if e.param_keys & set(params)]
        if len(claimed) == 1:
            scores[claimed[0]] += self.param_boost

        if self.auto_extract and self.evidence_boost > 0:
            normalized = normalize_text(text)
            for name, engine in self.engines.items():
                scores[name] += self.evidence_boost * engine.evidence(engine.extract(normalized).params)

        total = sum(scores.values())
        if total <= 0:
            raise ProbAIError("The router does not know any registered engine; retrain it.")
        return sorted(((n, s / total) for n, s in scores.items()), key=lambda x: x[1], reverse=True)

    # ------------------------------------------------------------- estimate
    def _run(self, name: str, text: str, params: Dict[str, Any]):
        """Run one engine. Returns (result, extracted, notes, used_params)."""
        chosen = self.engines[name]
        extracted: Dict[str, Any] = {}
        notes: List[str] = []
        if self.auto_extract:
            found = chosen.extract(text)
            extracted = {k: v for k, v in found.params.items() if k not in params}
            notes = list(found.notes) if extracted else []
        used = {**extracted, **params}
        return chosen.estimate(text, used), extracted, notes, used

    def estimate(
        self,
        text: str,
        params: Optional[Dict[str, Any]] = None,
        engine: Optional[str] = None,
        language: Optional[str] = None,
    ) -> Estimate:
        """Estimate the probability described by ``text`` (English or Russian).

        ``params`` is optional: numbers written in ``text`` ("12 calls per hour", "8 dB",
        "2% default rate", "12 звонков в час") are understood automatically. Anything in
        ``params`` overrides what was read from the text. ``language`` forces the language
        of the answer ("en" or "ru"); by default it follows the question.
        """
        params = dict(params or {})
        language = language or detect_language(text)
        original_text = text
        text = normalize_text(text)  # Russian -> English terms; English is unchanged

        extra_notes: List[str] = []
        if engine is not None:
            if engine not in self.engines:
                raise ProbAIError(f"Unknown engine {engine!r}. Available: {', '.join(sorted(self.engines))}.")
            name, confidence, ranking = engine, 1.0, [(engine, 1.0)]
        else:
            ranking = self.route(original_text, params)
            name, confidence = ranking[0]
            if confidence < self.min_confidence:
                err = LowConfidenceError(ranking[:3], self.min_confidence)
                if language == "ru":
                    err.message_ru = ru_output.error_message(err)  # type: ignore[attr-defined]
                raise err

        try:
            result, extracted, notes, used = self._run(name, text, params)
        except MissingParametersError as first_error:
            outcome = self._fallback(ranking, name, confidence, text, params, language) if engine is None else None
            if outcome is None:
                if language == "ru":
                    first_error.message_ru = ru_output.error_message(first_error)  # type: ignore[attr-defined]
                raise
            name, confidence, (result, extracted, notes, used), why = outcome
            extra_notes.append(why)
        else:
            if engine is None and len(ranking) > 1:
                second, p2 = ranking[1]
                if p2 >= 0.2 and p2 >= 0.5 * confidence:
                    extra_notes.append(
                        _t(
                            language,
                            f"The question could also be about '{second}' ({p2:.0%}). "
                            f"If you meant that, pass engine='{second}'.",
                            f"Вопрос мог относиться и к модели «{second}» ({p2:.0%}). "
                            f"Если имелось в виду это, укажите engine='{second}'.",
                        )
                    )

        chosen = self.engines[name]
        raw = float(result.probability)
        final, calibrated = raw, False
        if self.calibrator is not None and chosen.calibratable and self.calibrator.is_fitted(name):
            final, calibrated = self.calibrator.apply(name, raw), True

        method, explanation, details = result.method, result.explanation, dict(result.details)
        if language == "ru":
            ru_method, ru_explanation = ru_output.describe(name, result.details, used)
            if ru_method:
                details["method_en"], details["explanation_en"] = method, explanation
                method, explanation = ru_method, ru_explanation
            notes = [ru_output.translate_note(n) for n in notes]

        return Estimate(
            probability=final,
            engine=name,
            method=method,
            explanation=explanation,
            raw_probability=raw,
            calibrated=calibrated,
            routing_confidence=confidence,
            routing=ranking[:3],
            details=details,
            extracted=extracted,
            notes=notes + extra_notes,
            language=language,
            params_used=used,
            normalized_text=text if detect_language(original_text) == "ru" else None,
        )

    def _fallback(self, ranking, first, first_score, text, params, language):
        """If the best engine lacks data, use the runner-up when the text fully supplies it."""
        if not self.fallback:
            return None
        for name, score in ranking[1:3]:
            if score < 0.2 or score < 0.4 * first_score:
                continue
            engine = self.engines[name]
            if self.auto_extract and engine.evidence(engine.extract(text).params) <= 0.0:
                continue  # the text says nothing this engine could use
            try:
                outcome = self._run(name, text, params)
            except (MissingParametersError, ValueError):
                continue
            why = _t(
                language,
                f"'{first}' lacked data, so the next most likely model '{name}' ({score:.0%}) was used, "
                f"because the question contains everything it needs.",
                f"Для модели «{first}» не хватало данных, поэтому использована следующая по вероятности "
                f"модель «{name}» ({score:.0%}): в вопросе есть всё, что ей нужно.",
            )
            return name, score, outcome, why
        return None

    # ------------------------------------------------------------- feedback
    def add_feedback(self, estimate: Estimate, outcome: bool, refit: bool = True) -> None:
        """Tell ProbAI what actually happened, so model-based engines can be calibrated."""
        if self.calibrator is None:
            self.calibrator = Calibrator()
        self.calibrator.add_feedback(estimate.engine, estimate.raw_probability, outcome)
        if refit:
            self.calibrator.fit()
