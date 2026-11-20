"""Understand the numbers in a question: turn plain English into engine parameters.

Every ``extract_*`` function takes the question text and returns an
:class:`~probai.core.Extraction` with the parameters it is confident about and
notes for anything that was an interpretation (e.g. "SNR 0.8 has no unit, so it
was read as a plain ratio"). Nothing is guessed silently, and parameters that are
not in the text are simply left out so the engine can ask for them.
"""

from __future__ import annotations

import datetime
import math
import re
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

from .core import Extraction
from .utils import NUMBER_WORDS

Span = Tuple[int, int]

_WORDS = "|".join(sorted(NUMBER_WORDS, key=len, reverse=True))
_NUMBER = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
_NUMW = rf"(?:{_NUMBER}|{_WORDS})"
_QTY = rf"(?:{_NUMBER}|{_WORDS}|an?)"
_UNIT = r"(?:seconds?|secs?|minutes?|mins?|hours?|hrs?|days?|weeks?|months?|years?|yrs?|quarters?)"
_PCT = r"(?:%|percent|per cent)"

_UNIT_SECONDS = {
    "second": 1.0, "sec": 1.0, "minute": 60.0, "min": 60.0, "hour": 3600.0, "hr": 3600.0,
    "day": 86400.0, "week": 604800.0, "month": 2629800.0, "year": 31557600.0,
    "yr": 31557600.0, "quarter": 7889400.0,
}
_SECONDS_PER_YEAR = 31557600.0


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _num(token: str) -> float:
    t = token.lower().replace(",", "")
    if t in NUMBER_WORDS:
        return float(NUMBER_WORDS[t])
    if t in ("a", "an"):
        return 1.0
    return float(t)


def _unit_seconds(unit: str) -> float:
    return _UNIT_SECONDS[unit.lower().rstrip("s")]


def _overlaps(span: Span, taken: Sequence[Span]) -> bool:
    return any(span[0] < t[1] and t[0] < span[1] for t in taken)


def _fmt(x: float) -> str:
    return f"{x:.6g}"


class _Dur(NamedTuple):
    value: float
    unit: str
    seconds: float
    span: Span


def _durations(text: str) -> List[_Dur]:
    """Every 'N unit' phrase ('10 minutes', 'an hour', 'five years') plus 'next year', 'tomorrow'."""
    found: List[_Dur] = []
    for m in re.finditer(rf"\b(?P<n>{_QTY})[\s-]*(?P<u>{_UNIT})\b", text, re.I):
        before = text[: m.start()]
        if re.search(r"\b(?:per|every|each)\s*$", before, re.I):
            continue
        if re.search(_PCT + r"\s*$", before, re.I):  # "2% a year" is a rate, not a horizon
            continue
        v = _num(m.group("n"))
        found.append(_Dur(v, m.group("u").lower(), v * _unit_seconds(m.group("u")), m.span()))
    for m in re.finditer(r"\bnext\s+(?P<u>year|quarter|month|week|day|hour)\b", text, re.I):
        found.append(_Dur(1.0, m.group("u").lower(), _unit_seconds(m.group("u")), m.span()))
    for m in re.finditer(r"\b(?:tomorrow|today|tonight)\b", text, re.I):
        found.append(_Dur(1.0, "day", 86400.0, m.span()))
    return sorted(found, key=lambda d: d.span[0])


def _first_free_duration(text: str, taken: Sequence[Span]) -> Optional[_Dur]:
    for d in _durations(text):
        if not _overlaps(d.span, taken):
            return d
    return None


def _prob_value(raw: str, pct: Optional[str]) -> Optional[float]:
    v = _num(raw)
    if pct:
        v /= 100.0
    return v if 0.0 <= v <= 1.0 else None


def _prob_for(text: str, keywords: str) -> Optional[Tuple[float, Span]]:
    """A probability/percentage stated next to a keyword: '99% sensitivity' or 'sensitivity of 0.99'."""
    stop = r"(?!and\b|or\b|with\b|but\b|of\b|is\b|the\b|to\b)"
    before = (
        rf"(?<![\w.])(?P<v>{_NUMBER})\s*(?P<p>{_PCT})\s*(?:{stop}[a-z-]+\s+){{0,2}}(?:{keywords})\b"
    )
    after = (
        rf"(?:{keywords})(?:(?!\band\b|\bwith\b|\bbut\b|\bor\b)[^0-9.;,%\n]){{0,25}}?"
        rf"(?<![\w.])(?P<v>{_NUMBER})\s*(?P<p>{_PCT})?"
    )
    for pattern in (before, after):
        m = re.search(pattern, text, re.I)
        if m:
            p = _prob_value(m.group("v"), m.group("p"))
            if p is not None:
                return p, m.span()
    return None


def _free_integers(text: str, taken: Sequence[Span]) -> List[int]:
    out = []
    for m in re.finditer(rf"(?<![\w.])(\d+)(?![\w]|\.\d|\s*{_PCT})", text):
        if not _overlaps(m.span(), taken):
            out.append(int(m.group(1)))
    return out


# --------------------------------------------------------------------------- #
# classical: dice, coins, cards
# --------------------------------------------------------------------------- #
def extract_classical(text: str) -> Extraction:
    params: Dict[str, Any] = {}
    notes: List[str] = []
    t = text.lower()

    # Biased coin: "70% chance of heads", "lands tails 60% of the time"
    if re.search(r"\b(coin|heads|tails|flip\w*|toss\w*)\b", t):
        side, value = None, None
        m = re.search(rf"(?<![\w.])({_NUMBER})\s*{_PCT}\s*(?:(?!\d)[a-z-]+\s+){{0,4}}?(heads|tails)\b", t)
        if m:
            side, value = m.group(2), _num(m.group(1)) / 100.0
        else:
            m = re.search(rf"\b(heads|tails)\b(?:(?!\d)[^;.]){{0,30}}?(?<![\w.])({_NUMBER})\s*{_PCT}", t)
            if m:
                side, value = m.group(1), _num(m.group(2)) / 100.0
        if side and value is not None and 0.0 <= value <= 1.0 and value != 0.5:
            params["p"] = value if side == "heads" else 1.0 - value
            notes.append(f"Coin is biased: P({side}) = {_fmt(value)}.")

    # Repeated rolls of one die: "at least one six in four rolls", "roll a die 3 times"
    if re.search(r"\b(dice|die|d\d+|rolls?|rolled|rolling|sixes)\b", t) and not re.search(r"\bsum\b|\btotal\b", t):
        m = re.search(
            rf"\b(?:in|within|after|over)\s+({_NUMW})\s+(?:rolls?|throws?|tosses|times|tries|attempts)\b"
            rf"|\b({_NUMW})\s+times\b",
            t,
        )
        if m:
            params["kind"] = "dice"
            params["rolls"] = int(_num(m.group(1) or m.group(2)))
            km = re.search(
                rf"\b(?:at least|at most|exactly|more than|less than|fewer than)\s+({_NUMW})\b", t
            )
            if km:
                params["k"] = int(_num(km.group(1)))
            elif re.search(r"\ball\b|\bevery\b|\beach\b|\balways\b|\bin a row\b|\bconsecutive\b|\bstraight\b", t):
                params["k"] = params["rolls"]
                params["comparison"] = "eq"
                notes.append("'all' / 'every time' / 'in a row' means every one of the rolls must succeed.")
            if re.search(r"\bdouble[- ]?(?:six|sixes|6|1|ones?)\b|\bsnake eyes\b|\bboxcars\b", t):
                params["p"] = 1.0 / 36.0
                notes.append("A 'double' with two dice has probability 1/36 per roll.")

    m = re.search(r"\b(?:deck|pack) of (\d+)\b|\b(\d+)[- ]card (?:deck|pack)\b", t)
    if m:
        params["population"] = int(m.group(1) or m.group(2))
    return Extraction(params, notes)


# --------------------------------------------------------------------------- #
# counting: Poisson arrivals and binomial successes
# --------------------------------------------------------------------------- #
_K_PATTERNS = [
    rf"\b(?:at least|at most|exactly|more than|less than|fewer than|greater than|no more than|no less than)\s+(?P<k>{_WORDS}|\d+|once|twice)\b",
    rf"\b(?P<k>{_WORDS}|\d+)\s+or\s+(?:more|fewer|less|higher|lower|greater)\b",
]


def _find_k(text: str) -> Tuple[Optional[int], Optional[Span], Optional[int], Optional[Span]]:
    """Return (k, k_span, n, n_span); n is set only for 'k out of n'."""
    for pattern in _K_PATTERNS:
        m = re.search(pattern, text, re.I)
        if m:
            word = m.group("k").lower()
            return {"once": 1, "twice": 2}.get(word) or int(_num(word)), m.span(), None, None
    m = re.search(rf"\b(?P<k>{_WORDS}|\d+)\s+out\s+of\s+(?P<n>\d+)\b", text, re.I)
    if m:
        return int(_num(m.group("k"))), m.span("k"), int(m.group("n")), m.span("n")
    m = re.search(r"\b(?:no|zero|none)\b(?!\s+(?:more|less|fewer)\b)", text, re.I)
    if m and re.search(r"\b(?:no|zero)\s+\w+|\bnone\b", text, re.I):
        return 0, m.span(), None, None
    return None, None, None, None


def extract_counting(text: str) -> Extraction:
    params: Dict[str, Any] = {}
    notes: List[str] = []
    taken: List[Span] = []
    rate_unit: Optional[str] = None

    filler = r"(?:(?!(?:in|within|during|over|for|next|after|than|least|most|exactly|and|if|when|that)\b)[a-z-]+\s+)"
    m = re.search(
        rf"\b(?P<v>{_NUMW})\s+{filler}{{0,3}}?(?:(?:per|an?|each|every)\s+|/\s*)(?P<u>{_UNIT})\b",
        text,
        re.I,
    )
    if m:
        params["rate"] = _num(m.group("v"))
        rate_unit = m.group("u")
        taken.append(m.span())
    else:
        for pattern in (
            rf"\b(?:averag(?:e|es|ing)|mean|rate)\s+(?:of\s+|is\s+|at\s+|about\s+|around\s+)?(?P<v>{_NUMBER})(?!\s*{_PCT})",
            rf"(?<![\w.])(?P<v>{_NUMBER})\s+(?:[a-z-]+\s+){{0,2}}?on\s+average",
        ):
            m = re.search(pattern, text, re.I)
            if m:
                params["rate"] = _num(m.group("v"))
                taken.append(m.span())
                break

    if "rate" in params:
        d = _first_free_duration(text, taken)
        if d is not None:
            taken.append(d.span)
            if rate_unit:
                params["duration"] = d.seconds / _unit_seconds(rate_unit)
            else:
                params["duration"] = d.value
                notes.append("No time unit was given for the rate, so the duration is taken in the same unit.")

    k, k_span, n_from_k, n_span = _find_k(text)
    if k is not None:
        params["k"] = k
        taken.append(k_span)  # type: ignore[arg-type]
    if n_from_k is not None:
        params["n"] = n_from_k
        taken.append(n_span)  # type: ignore[arg-type]

    if "rate" not in params:
        # Binomial: successes among n independent trials, each with probability p.
        if "n" not in params:
            m = re.search(
                rf"\b(?:out of|among|batch of|sample of|lot of|total of|population of|group of)\s+(?P<n>\d+)\b"
                rf"|\b(?P<n2>\d+)\s+(?:independent\s+)?(?:trials|attempts|tests|samples|shots|tries)\b",
                text,
                re.I,
            )
            if m:
                params["n"] = int(m.group("n") or m.group("n2"))
                taken.append(m.span())

        m = re.search(rf"(?:\bprobability|\bchance|\blikelihood|\bp)\s*(?:of|is|=|:)?\s*(?P<v>0?\.\d+)\b", text, re.I)
        if m:
            params["p"] = float(m.group("v"))
            taken.append(m.span())
        else:
            pcts = [pm for pm in re.finditer(rf"(?<![\w.])(?P<v>{_NUMBER})\s*{_PCT}", text, re.I)]
            if len(pcts) == 1:
                p = _prob_value(pcts[0].group("v"), "%")
                if p is not None:
                    params["p"] = p
                    taken.append(pcts[0].span())

        free = _free_integers(text, taken)
        if "p" in params and len(free) == 1:
            if "n" not in params:
                params["n"] = free[0]
                notes.append(f"Took {free[0]} as the number of trials.")
            elif "k" not in params:
                params["k"] = free[0]
                notes.append(f"Took {free[0]} as the number of successes.")
        if "n" in params and "p" not in params:
            params.pop("n")  # a lone number is not enough to claim a binomial setup
    else:
        free = _free_integers(text, taken)
        if "k" not in params and len(free) == 1:
            params["k"] = free[0]
            notes.append(f"Took {free[0]} as the number of events to evaluate.")

    if "n" in params and "p" not in params:
        params.pop("n")
    return Extraction(params, notes)


# --------------------------------------------------------------------------- #
# signal
# --------------------------------------------------------------------------- #
_MODULATIONS = [
    (r"\bnon[- ]?coherent\s+b?fsk\b|\bb?fsk\s+non[- ]?coherent", "bfsk_noncoherent", ""),
    (r"\bdpsk\b|\bdifferential\s+(?:phase|psk)", "dpsk", ""),
    (r"\bqpsk\b|\b4[- ]?psk\b|\bquadrature\s+phase", "qpsk", ""),
    (r"\bbpsk\b|\b2[- ]?psk\b|\bbinary\s+phase", "bpsk", ""),
    (r"\bb?fsk\b|\bfrequency[- ]shift", "bfsk_coherent", "FSK was assumed to be coherent."),
]


def extract_signal(text: str) -> Extraction:
    params: Dict[str, Any] = {}
    notes: List[str] = []

    m = re.search(r"(?<![\w.])(?P<v>-?\d+(?:\.\d+)?)\s*-?\s*dB\b", text, re.I)
    if m:
        params["snr_db"] = float(m.group("v"))
    else:
        m = re.search(
            r"(?:\bsnr\b|signal[- ]to[- ]noise(?:\s+ratio)?|\beb\s*/?\s*n0\b|\bes\s*/?\s*n0\b|\bcnr\b)"
            r"(?:(?!\band\b|\bwith\b)[^0-9;,%.]){0,30}?(?P<v>-?\d+(?:\.\d+)?)",
            text,
            re.I,
        )
        if m and float(m.group("v")) > 0:
            ratio = float(m.group("v"))
            params["snr_db"] = 10.0 * math.log10(ratio)
            notes.append(
                f"SNR {_fmt(ratio)} has no unit, so it was read as a plain power ratio "
                f"(not decibels) and converted to {params['snr_db']:.3g} dB. "
                f"Write '{_fmt(ratio)} dB' if you meant decibels."
            )

    for pattern, name, note in _MODULATIONS:
        if re.search(pattern, text, re.I):
            params["modulation"] = name
            if note:
                notes.append(note)
            break

    if re.search(r"\b(?:packet|frame|message|block|payload|datagram|codeword|transmission|file)s?\b", text, re.I):
        m = re.search(r"(?<![\w.])(?P<v>\d[\d,]*)[\s-]*bits?\b(?!\s+error)", text, re.I)
        if m:
            params["packet_bits"] = int(m.group("v").replace(",", ""))
        else:
            m = re.search(r"(?<![\w.])(?P<v>\d[\d,]*)[\s-]*bytes?\b", text, re.I)
            if m:
                params["packet_bits"] = int(m.group("v").replace(",", "")) * 8
                notes.append("Bytes were converted to bits (x 8).")

    m = re.search(
        r"(?:false[- ]alarm(?:\s+(?:probability|rate))?|\bpfa\b|\bp_fa\b)"
        r"(?:(?!\band\b)[^0-9;,%]){0,25}?(?P<v>\d+(?:\.\d+)?(?:[eE]-?\d+)?)\s*(?P<p>%|percent)?",
        text,
        re.I,
    )
    if m:
        p = _prob_value(m.group("v"), m.group("p"))
        if p is not None and 0.0 < p < 1.0:
            params["pfa"] = p
    return Extraction(params, notes)


# --------------------------------------------------------------------------- #
# bayes
# --------------------------------------------------------------------------- #
def extract_bayes(text: str) -> Extraction:
    params: Dict[str, Any] = {}
    notes: List[str] = []

    r = _prob_for(text, r"prevalence|base[- ]rate|prior(?:\s+probability)?|incidence|baseline(?:\s+rate)?")
    if r:
        params["prior"] = r[0]
    else:
        m = re.search(
            rf"(?<![\w.])(?P<v>{_NUMBER})\s*{_PCT}\s+of\s+(?:the\s+|all\s+|our\s+)?(?:[a-z-]+\s+){{0,3}}?"
            r"(?:have|has|had|are|is|carry|carries|suffer|suffers|get|got|contain|contains)\b",
            text,
            re.I,
        )
        if m:
            p = _prob_value(m.group("v"), "%")
            if p is not None:
                params["prior"] = p

    r = _prob_for(text, r"sensitivity|sensitive|true[- ]positive(?:\s+rate)?|recall")
    if r:
        params["sensitivity"] = r[0]
    r = _prob_for(text, r"specificity|specific|true[- ]negative(?:\s+rate)?")
    if r:
        params["specificity"] = r[0]
    r = _prob_for(text, r"false[- ]positive(?:\s+(?:rate|probability))?")
    if r:
        params["false_positive_rate"] = r[0]

    if "sensitivity" not in params and "specificity" not in params and "false_positive_rate" not in params:
        r = _prob_for(text, r"accurate|accuracy")
        if r:
            params["sensitivity"] = params["specificity"] = r[0]
            notes.append("'Accurate' was read as sensitivity = specificity = that value.")

    m = re.search(r"(?:likelihood ratio|bayes factor)(?:(?!\band\b)[^0-9;,%.]){0,20}?(?P<v>\d+(?:\.\d+)?)", text, re.I)
    if m:
        params["likelihood_ratio"] = float(m.group("v"))
    return Extraction(params, notes)


# --------------------------------------------------------------------------- #
# finance
# --------------------------------------------------------------------------- #
def _money(m: "re.Match[str]") -> float:
    value = _num(m.group("v"))
    suffix = (m.group("s") or "").lower()
    return value * 1000.0 if suffix == "k" else value


def extract_finance(text: str) -> Extraction:
    params: Dict[str, Any] = {}
    notes: List[str] = []
    taken: List[Span] = []

    r = _prob_for(text, r"hazard\s+rate|default\s+intensity")
    if r:
        params["hazard_rate"] = r[0]
        taken.append(r[1])
    else:
        r = _prob_for(
            text,
            r"(?:(?:annual|yearly)\s+)?(?:default\s+(?:probability|rate|risk|likelihood)|probability\s+of\s+default|pd)",
        )
        if r:
            params["annual_pd"] = r[0]
            taken.append(r[1])

    # A rate quoted per month/quarter must not be read as a yearly one.
    for key in ("hazard_rate", "annual_pd"):
        if key in params and taken:
            tail = text[taken[-1][0] : taken[-1][1] + 20]
            per = re.search(r"(?:per|a|each|every)\s+(month|quarter)|\b(monthly|quarterly)\b", tail, re.I)
            if per:
                n = 12 if (per.group(1) or per.group(2) or "").lower().startswith(("month",)) else 4
                p0 = params[key]
                params[key] = p0 * n if key == "hazard_rate" else 1 - (1 - p0) ** n
                notes.append(f"{key} was given per {'month' if n == 12 else 'quarter'}; converted to a yearly value.")
            break

    strike_re = (
        r"(?:below|under|above|over|exceeds?|exceeding|reach(?:es)?|hits?|surpass\w*|"
        r"(?:falls?|drops?|declines?|rises?|goes?)\s+(?:to|below|under|above|over)|"
        r"(?:less|more|greater|higher|lower)\s+than)\s+(?:\$|usd\s*)?"
        rf"(?P<v>{_NUMBER})(?P<s>k\b)?(?!\s*{_PCT})(?![\s-]*{_UNIT}\b)"
    )
    m = re.search(strike_re, text, re.I)
    if m:
        params["strike"] = _money(m)
        taken.append(m.span())

    m = re.search(
        r"(?:current(?:ly)?(?:\s+price)?|spot(?:\s+price)?|(?:is\s+)?(?:trading|trades|stands|priced|valued|sits|sitting|starts|opens)"
        r"(?:\s+at)?|now\s+at)\s*(?:of\s+|is\s+|at\s+|=\s*)?(?:\$|usd\s*)?"
        rf"(?P<v>{_NUMBER})(?P<s>k\b)?(?!\s*{_PCT})",
        text,
        re.I,
    )
    if m:
        params["spot"] = _money(m)
        taken.append(m.span())

    r = _prob_for(text, r"vol(?:atility)?")
    if r:
        params["volatility"] = r[0]
        taken.append(r[1])
    r = _prob_for(text, r"drift|expected\s+(?:annual\s+)?return|expected\s+growth")
    if r:
        params["drift"] = r[0]
        taken.append(r[1])

    d = _first_free_duration(text, taken)
    if d is not None:
        params["years"] = d.seconds / _SECONDS_PER_YEAR
    elif re.search(r"\b(?:by|until|before|at)\s+(?:the\s+)?(?:end\s+of\s+(?:the\s+|this\s+)?year|year[- ]end)\b", text, re.I):
        today = datetime.date.today()
        days = max((datetime.date(today.year, 12, 31) - today).days, 1)
        params["years"] = days / 365.0
        notes.append(f"'Year end' was read as the {days} days left until 31 December {today.year}.")
    return Extraction(params, notes)


# --------------------------------------------------------------------------- #
# reliability
# --------------------------------------------------------------------------- #
def _hours(value: float, unit: Optional[str]) -> float:
    return value * _unit_seconds(unit) / 3600.0 if unit else value


def extract_reliability(text: str) -> Extraction:
    params: Dict[str, Any] = {}
    notes: List[str] = []
    taken: List[Span] = []
    gap = r"(?:(?!\band\b)[^0-9;,%.]){0,25}?"

    m = re.search(
        r"(?:\bmtbf\b|mean\s+time\s+(?:between|to)\s+failures?|mean\s+(?:life|lifetime|time\s+to\s+failure)|"
        r"average\s+(?:life|lifetime)|life\s+expectancy|expected\s+(?:life|lifetime))"
        rf"{gap}(?P<v>{_NUMBER})\s*(?P<u>{_UNIT})?\b",
        text,
        re.I,
    )
    if m:
        params["mtbf"] = _hours(_num(m.group("v")), m.group("u"))
        taken.append(m.span())
    else:
        m = re.search(
            rf"failure\s+rate{gap}(?P<v>{_NUMBER})\s*(?:failures?\s*)?(?:per|/)\s*(?P<u>{_UNIT})\b", text, re.I
        )
        if m and _num(m.group("v")) > 0:
            params["mtbf"] = _hours(1.0 / _num(m.group("v")), m.group("u"))
            taken.append(m.span())
            notes.append("MTBF was computed as 1 / failure rate.")

    sm = re.search(rf"\bshape(?:\s+parameter)?{gap}(?P<v>{_NUMBER})", text, re.I)
    cm = re.search(rf"\bscale(?:\s+parameter)?{gap}(?P<v>{_NUMBER})\s*(?P<u>{_UNIT})?\b", text, re.I)
    if sm and cm:
        params["shape"] = _num(sm.group("v"))
        params["scale"] = _hours(_num(cm.group("v")), cm.group("u"))
        taken += [sm.span(), cm.span()]

    d = _first_free_duration(text, taken)
    if d is not None:
        params["time"] = d.seconds / 3600.0
        if params:
            notes.append("All times were converted to hours.") if ("mtbf" in params or "scale" in params) else None
    return Extraction(params, [n for n in notes if n])


# --------------------------------------------------------------------------- #
# empirical
# --------------------------------------------------------------------------- #
def extract_empirical(text: str) -> Extraction:
    params: Dict[str, Any] = {}
    notes: List[str] = []

    m = re.search(
        rf"(?<![\w.])(?P<a>\d+)\s+(?:wins?|successes?|hits?|yes(?:es)?|positives?)\s+(?:and|,)\s+(?P<b>\d+)\s+"
        r"(?:losses|loss|failures?|misses|no(?:es)?|negatives?|defeats?)\b",
        text,
        re.I,
    )
    if m:
        a, b = int(m.group("a")), int(m.group("b"))
        params.update(successes=a, trials=a + b)
        return Extraction(params, notes)

    patterns = [
        # "We tested 500 units and saw 9 rejects"
        (r"\b(?:tested|observed|sampled|inspected|checked|surveyed)\s+(?P<n>\d+)\b[^.;?]*?\b(?:saw|found|had|observed|recorded|got)\s+(?P<k>\d+)\b", False),
        # "37 conversions out of 400 visitors", "40 of the last 100 days", "14 recoveries among 30"
        (r"(?<![\w.])(?P<k>\d+)\s+(?:[a-z-]+\s+){0,3}?(?:out of|of the (?:last|past)|among|from)\s+(?:the\s+)?(?:last\s+|past\s+|total\s+of\s+)?(?P<n>\d+)\b", False),
        # "12 successes in 80 trials"
        (r"(?<![\w.])(?P<k>\d+)\s+(?:[a-z-]+\s+){0,2}?in\s+(?P<n>\d+)\s+(?:trials|attempts|tests|tries|games|matches|visits|days|samples|cases|patients|customers|visitors|users|experiments|runs|rounds)\b", False),
    ]
    for pattern, _ in patterns:
        m = re.search(pattern, text, re.I)
        if m and int(m.group("k")) <= int(m.group("n")):
            params.update(successes=int(m.group("k")), trials=int(m.group("n")))
            break
    return Extraction(params, notes)
