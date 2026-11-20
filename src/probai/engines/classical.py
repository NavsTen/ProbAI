"""Classical probability: dice, coins, playing cards and coloured balls in an urn (exact combinatorics)."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

import numpy as np
from scipy.stats import binom, hypergeom

from ..core import Engine, EngineResult, Extraction, MissingParametersError
from ..extraction import extract_classical
from ..utils import (
    COMPARISON_SYMBOLS,
    NUM,
    as_int,
    as_probability,
    first_not_none,
    infer_comparison,
    infer_count_after_comparison,
    normalise_comparison,
    parse_int_token,
    prob_from_frozen,
    prob_from_pmf,
)

_MAX_DICE = 100
_MAX_SIDES = 1000
_MAX_PMF_SIZE = 20000


def _cut(text: str, match: "re.Match[str]") -> str:
    return text[: match.start()] + " " + text[match.end():]


_COLORS = r"(?:red|blue|green|white|black|yellow|orange|purple|pink|brown|grey|gray|golden|silver)"
_URN_ITEMS = r"(?:balls?|marbles?|beads?|chips?|candies|candy|sweets?|stones?|tokens?)"


def _parse_urn_text(text: str) -> Dict[str, Any]:
    """Colour counts, the colour asked about and the number of draws, e.g. '3 red, 6 white and 1 blue'."""
    t = text.lower()
    sentences = [x for x in re.split(r"(?<=[?.!])\s+", t) if x.strip()]
    ask = next((x for x in reversed(sentences) if re.search(r"probab|chance|likel|odds|what", x)), t)
    # Count colours only outside the question sentence ("draw 2 red balls" is not a count of balls in the box).
    body = " ".join(x for x in sentences if x is not ask) or t
    if ask is not t and not re.search(rf"\b{NUM}\s+(?:(?:are|is)\s+)?{_COLORS}\b", body):
        body = t
    counts: Dict[str, int] = {}
    for m in re.finditer(rf"\b{NUM}\s+(?:(?:are|is)\s+)?({_COLORS})\b", body):
        counts[m.group(2)] = counts.get(m.group(2), 0) + parse_int_token(m.group(1))
    total = None
    m = re.search(rf"\b(\d+)\s+{_URN_ITEMS}\b", body)
    if m:
        total = int(m.group(1))
    target = None
    m = re.search(rf"\b({_COLORS})\b", ask)
    if m:
        target = m.group(1)
    draws = None
    m = re.search(rf"\b(?:draw\w*|pick\w*|tak\w+|pull\w*|select\w*|get\w*|grab\w*)\s+{NUM}\s+(?:{_COLORS}\s+)?{_URN_ITEMS}\b", t) or re.search(
        rf"\b{NUM}\s+(?:attempts?|draws?|tries)\b", t
    )
    if m:
        draws = parse_int_token(m.group(1))
    k = infer_count_after_comparison(ask)
    if k is None and draws and draws > 1 and re.search(r"\bin a row\b|\ball\b|\bboth\b", ask):
        k = draws
    return {
        "counts": counts,
        "total": total,
        "target": target,
        "draws": draws,
        "k": k,
        "comparison": infer_comparison(ask),
        "replacement": bool(re.search(r"\bwith replacement\b|\bput(?:s|ting)? (?:it |them )?back\b|\breplac", t)),
    }


def _detect_kind(text: str) -> str:
    t = text.lower()
    if re.search(rf"\b{_URN_ITEMS}\b|\burn\b", t) and re.search(rf"\b{_COLORS}\b", t):
        return "urn"
    if re.search(r"\b(coins?|flips?|flipping|tosses|toss|tossing|heads|tails)\b", t):
        return "coin"
    if re.search(r"\b(dice|die|d\d+|\d+d\d+|rolls?|rolled|rolling|sixes|sixth)\b", t):
        return "dice"
    if re.search(r"\b(cards?|deck|aces?|hearts?|spades?|clubs?|diamonds?)\b", t):
        return "cards"
    raise ValueError(
        "Could not tell whether this is about dice, coins or cards. "
        "Pass params={'kind': 'dice' | 'coin' | 'cards'}."
    )


# --------------------------------------------------------------------------- #
# Dice
# --------------------------------------------------------------------------- #
def _parse_dice_text(text: str) -> Dict[str, Any]:
    t = text.lower()
    out: Dict[str, Any] = {
        "sides": None,
        "n_dice": None,
        "target": None,
        "comparison": infer_comparison(t),
    }
    m = re.search(r"\b(\d+)d(\d+)\b", t)  # 2d6
    if m:
        out["n_dice"], out["sides"] = int(m.group(1)), int(m.group(2))
        t = _cut(t, m)
    m = re.search(rf"\b{NUM}-sided\b", t)  # six-sided, 20-sided
    if m:
        out["sides"] = parse_int_token(m.group(1))
        t = _cut(t, m)
    m = re.search(r"\bd(\d+)\b", t)  # d20
    if m:
        out["sides"] = out["sides"] or int(m.group(1))
        t = _cut(t, m)
    m = re.search(rf"\b(?:{NUM}|an?)\s+(?:(?:fair|standard|regular|loaded)\s+)*(?:dice|die)\b", t)
    if m:
        out["n_dice"] = parse_int_token(m.group(1)) if m.group(1) else 1
        t = _cut(t, m)
    elif re.search(r"\b(?:dice|die)\b", t) or out["sides"]:
        out["n_dice"] = out["n_dice"] or 1

    numbers = re.findall(r"\b\d+\b", t)
    if numbers:
        out["target"] = int(numbers[0])
    else:
        m = re.search(r"\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twenty)\b", t)
        if m:
            out["target"] = parse_int_token(m.group(1))
    return out


def _sum_pmf(n_dice: int, sides: int) -> np.ndarray:
    """Exact distribution of the sum of ``n_dice`` fair dice (index 0 = minimum sum)."""
    single = np.full(sides, 1.0 / sides)
    pmf = np.array([1.0])
    for _ in range(n_dice):
        pmf = np.convolve(pmf, single)
    return pmf


# --------------------------------------------------------------------------- #
# Coins
# --------------------------------------------------------------------------- #
def _parse_coin_text(text: str) -> Dict[str, Any]:
    t = text.lower()
    out: Dict[str, Any] = {"n": None, "k": None, "comparison": infer_comparison(t), "side": "heads"}
    m = re.search(rf"\b{NUM}\s+(?:coin\s+)?(?:flips|tosses|throws|times)\b", t)
    if not m:
        m = re.search(rf"\b(?:flip|toss|flipping|tossing|throw)\s+(?:a\s+)?{NUM}\s+coins\b", t)
    if not m:
        m = re.search(rf"\b{NUM}\s+coins\b", t)
    if m:
        out["n"] = parse_int_token(m.group(1))
    m = re.search(rf"\b{NUM}\s+(heads?|tails?)\b", t)
    if m:
        out["k"] = parse_int_token(m.group(1))
    if re.search(r"\btails?\b", t) and not re.search(r"\bheads?\b", t):
        out["side"] = "tails"
    if re.search(r"\ball\b", t):
        out["k"] = -1  # means "all n"
    return out


# --------------------------------------------------------------------------- #
# Cards
# --------------------------------------------------------------------------- #
_NAMED_RANKS = r"(?:aces?|kings?|queens?|jacks?)"
# Number words are ranks only in the plural ("twos") or before "of" ("the two of hearts");
# otherwise "two hearts" means a count of two.
_NUMERIC_RANKS = r"(?:(?:twos|threes|fours|fives|sixes|sevens|eights|nines|tens)|(?:two|three|four|five|six|seven|eight|nine|ten)(?=\s+of\b))"
_RANKS = rf"(?:{_NAMED_RANKS}|{_NUMERIC_RANKS})"
_SUITS = r"(?:hearts?|spades?|clubs?|diamonds?)"
_PLURAL_GROUPS = r"(?:aces|kings|queens|jacks|hearts|spades|clubs|diamonds|face cards|red cards|black cards)"


def _parse_cards_text(text: str) -> Dict[str, Any]:
    t = text.lower()
    out: Dict[str, Any] = {"draws": None, "successes": None, "k": None, "comparison": infer_comparison(t)}
    t = re.sub(r"\b(?:deck|pack)\s+of\s+\d+\s+cards?\b|\b\d+[- ]cards?\s+(?:deck|pack)\b", " deck ", t)
    m = re.search(rf"\b{NUM}\s+cards?\b", t)
    if m:
        out["draws"] = parse_int_token(m.group(1))
        t = _cut(t, m)
    if re.search(r"\bface cards?\b", t):
        out["successes"] = 12
    else:
        has_rank = re.search(rf"\b{_RANKS}\b", t)
        has_suit = re.search(rf"\b{_SUITS}\b", t)
        if has_rank and has_suit:
            out["successes"] = 1
        elif has_rank:
            out["successes"] = 4
        elif has_suit:
            out["successes"] = 13
        elif re.search(r"\b(red|black)\b", t):
            out["successes"] = 26
    out["k"] = infer_count_after_comparison(t)
    if out["draws"] is None:  # "drawing two hearts" = two cards, both of that kind
        m = re.search(rf"\b{NUM}\s+{_PLURAL_GROUPS}\b", t)
        if m and not re.search(r"\b(?:at least|at most|exactly|more than|less than|fewer than)\s+" + NUM + r"\s+" + _PLURAL_GROUPS, t):
            out["draws"] = parse_int_token(m.group(1))
            if out["k"] is None:
                out["k"] = out["draws"]
    return out


# --------------------------------------------------------------------------- #
class ClassicalEngine(Engine):
    name = "classical"
    description = "Dice, coins and playing cards: exact combinatorial probability."
    param_keys = frozenset({"sides", "n_dice", "rolls", "draws", "population", "kind"})
    calibratable = False

    def extract(self, text: str) -> Extraction:
        return extract_classical(text)

    def estimate(self, text: str, params: Dict[str, Any]) -> EngineResult:
        kind = params.get("kind") or _detect_kind(text)
        if kind == "dice":
            return self._dice(text, params)
        if kind == "coin":
            return self._coin(text, params)
        if kind == "cards":
            return self._cards(text, params)
        if kind == "urn":
            return self._urn(text, params)
        raise ValueError(f"Unknown kind {kind!r}; use 'dice', 'coin', 'cards' or 'urn'.")

    # ----------------------------------------------------------------- urn
    def _urn(self, text: str, params: Dict[str, Any]) -> EngineResult:
        parsed = _parse_urn_text(text)
        counts = parsed["counts"]
        total = first_not_none(params.get("population"), parsed["total"])
        if total is None or int(total) < sum(counts.values()):
            total = sum(counts.values())
        total = as_int(total, "population", 1)
        successes = params.get("successes")
        if successes is None:
            if parsed["target"] is None or parsed["target"] not in counts:
                raise MissingParametersError(
                    self.name,
                    ["successes"],
                    "I could not tell which colour to draw; e.g. params={'successes': 1} for one blue ball.",
                )
            successes = counts[parsed["target"]]
        successes = as_int(successes, "successes", 0)
        draws = as_int(first_not_none(params.get("draws"), parsed["draws"], 1), "draws", 1)
        replacement = bool(params.get("replacement", parsed["replacement"]))
        if successes > total or (draws > total and not replacement):
            raise ValueError("'successes' and 'draws' cannot exceed the population size.")
        k = as_int(first_not_none(params.get("k"), parsed["k"], 1), "k", 0)
        default_cmp = "ge" if draws > 1 and parsed["k"] is None else "eq"
        cmp = normalise_comparison(params.get("comparison"), parsed["comparison"] or default_cmp)
        if replacement:
            dist, method = binom(draws, successes / total), "Binomial distribution (drawing with replacement)"
        else:
            dist, method = hypergeom(total, successes, draws), "Hypergeometric distribution (drawing without replacement)"
        prob = prob_from_frozen(dist, k, cmp)
        return EngineResult(
            probability=prob,
            method=method,
            explanation=(
                f"Draw {draws} item(s) from {total}, of which {successes} match; "
                f"P(matches {COMPARISON_SYMBOLS[cmp]} {k})."
            ),
            details={"kind": "urn", "population": total, "successes": successes, "draws": draws,
                     "k": k, "comparison": cmp, "replacement": replacement, "counts": counts},
        )

    # ----------------------------------------------------------------- dice
    def _dice(self, text: str, params: Dict[str, Any]) -> EngineResult:
        parsed = _parse_dice_text(text)
        sides = as_int(first_not_none(params.get("sides"), parsed["sides"], 6), "sides", 2)
        if sides > _MAX_SIDES:
            raise ValueError(f"'sides' is limited to {_MAX_SIDES}.")

        if params.get("rolls") is not None:  # repeated rolls: count a specific face
            rolls = as_int(params["rolls"], "rolls", 1)
            p = as_probability(first_not_none(params.get("p"), 1.0 / sides), "p")
            k = as_int(first_not_none(params.get("k"), 1), "k", 0)
            cmp = normalise_comparison(params.get("comparison"), parsed["comparison"] or "ge")
            prob = prob_from_frozen(binom(rolls, p), k, cmp)
            return EngineResult(
                probability=prob,
                method="Binomial distribution over independent rolls",
                explanation=(
                    f"{rolls} independent rolls of a {sides}-sided die, each success with "
                    f"probability {p:.6g}; P(successes {COMPARISON_SYMBOLS[cmp]} {k})."
                ),
                details={"kind": "dice", "rolls": rolls, "p": p, "k": k, "comparison": cmp},
            )

        n_dice = as_int(first_not_none(params.get("n_dice"), parsed["n_dice"], 1), "n_dice", 1)
        if n_dice > _MAX_DICE or n_dice * sides > _MAX_PMF_SIZE:
            raise ValueError(
                f"Too many dice: limit is {_MAX_DICE} dice and {_MAX_PMF_SIZE} total faces."
            )
        target = first_not_none(params.get("target"), parsed["target"])
        if target is None:
            raise MissingParametersError(
                self.name,
                ["target"],
                "I could not find the number to evaluate in the text, e.g. params={'target': 7}.",
            )
        target = as_int(target, "target", -10**9)
        cmp = normalise_comparison(params.get("comparison"), parsed["comparison"] or "eq")
        prob = prob_from_pmf(_sum_pmf(n_dice, sides), n_dice, target, cmp)
        what = "the die shows" if n_dice == 1 else f"the sum of {n_dice} dice is"
        return EngineResult(
            probability=prob,
            method="Exact enumeration of the sum of fair dice (convolution)",
            explanation=(
                f"Fair {sides}-sided dice, {n_dice} thrown; P({what} "
                f"{COMPARISON_SYMBOLS[cmp]} {target})."
            ),
            details={"kind": "dice", "n_dice": n_dice, "sides": sides, "target": target, "comparison": cmp},
        )

    # ----------------------------------------------------------------- coin
    def _coin(self, text: str, params: Dict[str, Any]) -> EngineResult:
        parsed = _parse_coin_text(text)
        n = as_int(first_not_none(params.get("n"), parsed["n"], 1), "n", 1)
        k_raw = first_not_none(params.get("k"), parsed["k"], 1)
        k = n if k_raw == -1 else as_int(k_raw, "k", 0)
        p_heads = as_probability(first_not_none(params.get("p"), 0.5), "p")
        p = 1.0 - p_heads if parsed["side"] == "tails" else p_heads
        default_cmp = "eq" if (parsed["k"] is not None or n == 1) else "ge"
        cmp = normalise_comparison(params.get("comparison"), parsed["comparison"] or default_cmp)
        prob = prob_from_frozen(binom(n, p), k, cmp)
        return EngineResult(
            probability=prob,
            method="Binomial distribution of coin flips",
            explanation=(
                f"{n} independent flip(s), P({parsed['side']}) = {p:.6g} per flip; "
                f"P(number of {parsed['side']} {COMPARISON_SYMBOLS[cmp]} {k})."
            ),
            details={"kind": "coin", "n": n, "k": k, "p": p, "comparison": cmp},
        )

    # ---------------------------------------------------------------- cards
    def _cards(self, text: str, params: Dict[str, Any]) -> EngineResult:
        parsed = _parse_cards_text(text)
        population = as_int(first_not_none(params.get("population"), 52), "population", 1)
        successes = first_not_none(params.get("successes"), parsed["successes"])
        if successes is None:
            raise MissingParametersError(
                self.name,
                ["successes"],
                "I could not tell which cards count as a success; e.g. params={'successes': 4} for aces.",
            )
        successes = as_int(successes, "successes", 0)
        draws = as_int(first_not_none(params.get("draws"), parsed["draws"], 1), "draws", 1)
        if successes > population or draws > population:
            raise ValueError("'successes' and 'draws' cannot exceed the population size.")
        k = as_int(first_not_none(params.get("k"), parsed["k"], 1), "k", 0)
        default_cmp = "ge" if draws > 1 and parsed["k"] is None else "eq"
        cmp = normalise_comparison(params.get("comparison"), parsed["comparison"] or default_cmp)
        prob = prob_from_frozen(hypergeom(population, successes, draws), k, cmp)
        return EngineResult(
            probability=prob,
            method="Hypergeometric distribution (drawing without replacement)",
            explanation=(
                f"Draw {draws} card(s) without replacement from {population} cards of which "
                f"{successes} count as a success; P(successes {COMPARISON_SYMBOLS[cmp]} {k})."
            ),
            details={
                "kind": "cards",
                "population": population,
                "successes": successes,
                "draws": draws,
                "k": k,
                "comparison": cmp,
            },
        )
