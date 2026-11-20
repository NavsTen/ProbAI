"""The learned part of ProbAI: a text classifier that picks the right probability engine."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import FeatureUnion, Pipeline

from .core import ProbAIError
from .language import has_cyrillic, normalize_text

PathLike = Union[str, Path]
_FORMAT_VERSION = 1


def load_examples(path: Optional[PathLike] = None) -> Tuple[List[str], List[str]]:
    """Read ``{"text": ..., "engine": ...}`` lines from a JSONL file.

    Without ``path`` the bundled seed dataset is used.
    """
    if path is None:
        from importlib.resources import files

        raw = files("probai").joinpath("data/router_seed.jsonl").read_text(encoding="utf-8")
    else:
        raw = Path(path).read_text(encoding="utf-8")

    texts: List[str] = []
    labels: List[str] = []
    for number, line in enumerate(raw.splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            row = json.loads(line)
            texts.append(str(row["text"]))
            labels.append(str(row["engine"]))
        except (json.JSONDecodeError, KeyError, TypeError):
            raise ProbAIError(
                f'Line {number}: expected JSON like {{"text": "...", "engine": "..."}}.'
            ) from None
    if not texts:
        raise ProbAIError("No training examples found.")
    return texts, labels


def prepare(text: str) -> str:
    """Text the classifier sees. Russian is kept as written and followed by its English
    rewrite, so both the Russian word forms and the English domain vocabulary are features."""
    if has_cyrillic(text):
        return f"{text.lower()} || {normalize_text(text)}"
    return text


class Router:
    """Maps a free-text question to a probability distribution over engines.

    Features: word 1-2-grams plus character 3-5-grams (TF-IDF), which makes the
    router tolerant to typos and unseen phrasing. Classifier: logistic regression.
    """

    def __init__(self) -> None:
        self._pipeline: Optional[Pipeline] = None

    @staticmethod
    def _make_pipeline() -> Pipeline:
        features = FeatureUnion(
            [
                ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
                ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)),
            ]
        )
        return Pipeline([("features", features), ("clf", LogisticRegression(C=20.0, max_iter=2000, class_weight="balanced"))])

    # ------------------------------------------------------------- training
    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> "Router":
        if len(texts) != len(labels):
            raise ProbAIError("texts and labels must have the same length.")
        if len(set(labels)) < 2:
            raise ProbAIError("Need examples for at least two different engines to train a router.")
        self._pipeline = self._make_pipeline().fit([prepare(t) for t in texts], list(labels))
        return self

    @classmethod
    def from_seed(cls) -> "Router":
        texts, labels = load_examples()
        return cls().fit(texts, labels)

    @staticmethod
    def cross_validate(texts: Sequence[str], labels: Sequence[str], folds: int = 5) -> float:
        """Mean accuracy over stratified folds (reduced automatically for tiny classes)."""
        counts = {}
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
        folds = min(folds, min(counts.values()))
        if folds < 2:
            raise ProbAIError("Need at least 2 examples per engine to cross-validate.")
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
        scores = cross_val_score(Router._make_pipeline(), [prepare(t) for t in texts], list(labels), cv=cv)
        return float(np.mean(scores))

    # ------------------------------------------------------------ inference
    @property
    def classes(self) -> List[str]:
        self._check_fitted()
        return [str(c) for c in self._pipeline.classes_]  # type: ignore[union-attr]

    def predict_proba(self, text: str) -> List[Tuple[str, float]]:
        """Return ``[(engine, probability), ...]`` sorted from most to least likely."""
        self._check_fitted()
        prepared = prepare(text)
        word_vec = self._pipeline.named_steps["features"].transformer_list[0][1]  # type: ignore[union-attr]
        if word_vec.transform([prepared]).nnz == 0:  # not a single known word: no evidence, no opinion
            uniform = 1.0 / len(self.classes)
            return [(name, uniform) for name in self.classes]
        probs = self._pipeline.predict_proba([prepared])[0]  # type: ignore[union-attr]
        pairs = list(zip(self.classes, (float(p) for p in probs)))
        return sorted(pairs, key=lambda item: item[1], reverse=True)

    def _check_fitted(self) -> None:
        if self._pipeline is None:
            raise ProbAIError("The router is not trained yet. Call fit() or Router.from_seed().")

    # ---------------------------------------------------------- persistence
    def save(self, path: PathLike) -> None:
        self._check_fitted()
        joblib.dump({"format": _FORMAT_VERSION, "pipeline": self._pipeline}, path)

    @classmethod
    def load(cls, path: PathLike) -> "Router":
        """Load a saved router. Only load files you trust: joblib uses pickle."""
        data = joblib.load(path)
        if not isinstance(data, dict) or data.get("format") != _FORMAT_VERSION:
            raise ProbAIError("This file is not a ProbAI router (or has an unsupported version).")
        router = cls()
        router._pipeline = data["pipeline"]
        return router
