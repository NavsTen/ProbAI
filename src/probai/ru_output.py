"""Russian wording for answers, notes and errors.

Used when the question was written in Russian. The English text is always kept in
``Estimate.details['method_en']`` / ``['explanation_en']``.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

_CMP_SYMBOL = {"eq": "=", "ge": "≥", "gt": ">", "le": "≤", "lt": "<"}


def num(x: Any, fmt: str = ".6g") -> str:
    """Format a number the Russian way (decimal comma)."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return str(x)
    return format(x, fmt).replace(".", ",")


def pct(p: float) -> str:
    return f"{p * 100:.2f}".replace(".", ",") + "%"


_MOD_RU = {
    "bpsk": "BPSK", "qpsk": "QPSK", "dpsk": "DPSK", "bfsk_coherent": "когерентной FSK",
    "bfsk_noncoherent": "некогерентной FSK",
}


def _cmp(d: Dict[str, Any]) -> str:
    return _CMP_SYMBOL.get(d.get("comparison", "eq"), "=")


# --------------------------------------------------------------------------- #
# Method and explanation
# --------------------------------------------------------------------------- #
def describe(engine: str, details: Dict[str, Any], params: Dict[str, Any]) -> Tuple[str, str]:
    """Return (method, explanation) in Russian, or ("", "") when the case is not covered."""
    d, p = details, params
    try:
        if engine == "classical":
            return _classical(d)
        if engine == "counting":
            return _counting(d)
        if engine == "signal":
            return _signal(d)
        if engine == "bayes":
            return _bayes(d)
        if engine == "finance":
            return _finance(d, p)
        if engine == "reliability":
            return _reliability(d)
        if engine == "empirical":
            return _empirical(d)
    except (KeyError, TypeError):
        pass
    return "", ""


def _classical(d: Dict[str, Any]) -> Tuple[str, str]:
    kind = d.get("kind")
    c = _cmp(d)
    if kind == "dice" and "rolls" in d:
        return (
            "Биномиальное распределение независимых бросков",
            f"{d['rolls']} независимых бросков кубика, вероятность успеха в одном броске {num(d['p'])}; "
            f"P(число успехов {c} {d['k']}).",
        )
    if kind == "dice":
        what = "на кубике выпало" if d["n_dice"] == 1 else f"сумма {d['n_dice']} кубиков равна"
        return (
            "Точный перебор суммы честных кубиков (свёртка)",
            f"Честные кубики с {d['sides']} гранями, брошено {d['n_dice']}; P({what} {c} {d['target']}).",
        )
    if kind == "coin":
        return (
            "Биномиальное распределение подбрасываний монеты",
            f"{d['n']} независимых подбрасываний, вероятность успеха в одном {num(d['p'])}; "
            f"P(число успехов {c} {d['k']}).",
        )
    if kind == "cards":
        return (
            "Гипергеометрическое распределение (выбор без возвращения)",
            f"Из {d['population']} карт вытягивают {d['draws']} без возвращения; подходящих карт {d['successes']}; "
            f"P(число подходящих {c} {d['k']}).",
        )
    return "", ""


def _counting(d: Dict[str, Any]) -> Tuple[str, str]:
    c = _cmp(d)
    if "rate" in d:
        return (
            "Распределение Пуассона",
            f"События происходят независимо со средней интенсивностью {num(d['rate'])} за единицу времени; за "
            f"{num(d['duration'])} ед. времени ожидается в среднем {num(d['mean'])}. P(число событий {c} {d['k']}).",
        )
    return (
        "Биномиальное распределение",
        f"{d['n']} независимых испытаний, вероятность успеха в каждом {num(d['p'])}. P(успехов {c} {d['k']}).",
    )


def _signal(d: Dict[str, Any]) -> Tuple[str, str]:
    task = d.get("task")
    snr = num(d["snr_db"], ".4g")
    if task == "detection":
        return (
            "Обнаружение известного сигнала в AWGN по Нейману–Пирсону (согласованный фильтр)",
            f"Отношение энергии к шуму {snr} дБ, вероятность ложной тревоги задана {num(d['pfa'])}. "
            f"Вероятность того, что приёмник верно обнаружит присутствующий сигнал.",
        )
    mod = _MOD_RU.get(d.get("modulation", "bpsk"), d.get("modulation", "").upper())
    ber = num(d["ber"], ".3e")
    if task == "bit_error" and d.get("outcome") == "correct":
        return (
            f"1 − BER для {mod} в канале с белым гауссовым шумом",
            f"Eb/N0 = {snr} дБ. Вероятность того, что бит принят верно ({mod}, идеальный канал AWGN); "
            f"вероятность битовой ошибки {ber}. Для целого пакета укажите его размер в битах.",
        )
    if task == "bit_error":
        return (
            f"Теоретическая вероятность битовой ошибки {mod} в AWGN",
            f"Eb/N0 = {snr} дБ. Вероятность того, что принятый бит ошибочен при {mod} в идеальном канале AWGN.",
        )
    if task == "packet_success":
        return (
            f"(1 − BER)^число_бит для {mod} в AWGN, без помехоустойчивого кодирования",
            f"BER = {ber} при Eb/N0 = {snr} дБ; пакет из {d['bits']} бит принят без ошибок, только если верны все биты "
            f"(ошибки независимы, кодирования нет).",
        )
    return "", ""


def _bayes(d: Dict[str, Any]) -> Tuple[str, str]:
    if "likelihood_ratio" in d:
        return (
            "Формула Байеса в форме шансов (апостериорные шансы = априорные шансы × отношение правдоподобия)",
            f"Априорная вероятность {num(d['prior'])} обновлена с отношением правдоподобия {num(d['likelihood_ratio'])}.",
        )
    ev = "положительном" if d["evidence"] == "positive" else "отрицательном"
    return (
        "Теорема Байеса",
        f"P(H) = {num(d['prior'])}, P(E|H) = {num(d['p_e_given_h'])}, P(E|не H) = {num(d['p_e_given_not_h'])} "
        f"(при {ev} результате). P(H|E) = P(E|H)·P(H) / P(E).",
    )


def _finance(d: Dict[str, Any], p: Dict[str, Any]) -> Tuple[str, str]:
    if d.get("task") == "default":
        years = num(d["years"])
        if "hazard_rate" in d:
            return (
                "Постоянная интенсивность дефолта: P(дефолт к моменту t) = 1 − exp(−λt)",
                f"Вероятность хотя бы одного дефолта в течение {years} лет (интенсивность {num(d['hazard_rate'])} в год).",
            )
        return (
            "Независимые годовые дефолты: P(дефолт к моменту t) = 1 − (1 − PD)^t",
            f"Вероятность хотя бы одного дефолта в течение {years} лет (годовая вероятность {num(d['annual_pd'])}).",
        )
    direction = "выше" if d["direction"] == "above" else "ниже"
    return (
        "Геометрическое броуновское движение (логнормальная цена на горизонте)",
        f"Цена сейчас {num(p['spot'])}, годовой снос {num(d['drift'])}, волатильность {num(p['volatility'])}. "
        f"Вероятность, что через {num(p['years'])} лет она окажется {direction} {num(p['strike'])}. "
        f"Реальные рынки имеют более тяжёлые хвосты, чем эта модель.",
    )


def _reliability(d: Dict[str, Any]) -> Tuple[str, str]:
    model = (
        "Экспоненциальный срок службы (постоянная интенсивность отказов)"
        if "mtbf" in d
        else "Распределение Вейбулла"
    )
    what = "проработает дольше" if d["outcome"] == "survival" else "выйдет из строя раньше"
    return model, f"Вероятность того, что компонент {what} момента времени {num(d['time'])} (в тех же единицах, что и наработка)."


def _empirical(d: Dict[str, Any]) -> Tuple[str, str]:
    lo, hi = d["credible_interval"]
    level = num(d["level"] * 100, ".0f")
    return (
        "Среднее апостериорного бета-биномиального распределения",
        f"{d['successes']} успехов в {d['trials']} испытаниях, априорное Beta(1, 1) (сглаживание Лапласа). "
        f"{level}%-й доверительный интервал: [{num(lo, '.4f')}; {num(hi, '.4f')}]. "
        f"Предполагается, что испытания независимы и одинаково распределены.",
    )


# --------------------------------------------------------------------------- #
# Notes
# --------------------------------------------------------------------------- #
_NOTE_RULES: List[Tuple[str, Any]] = [
    (
        r"SNR (\S+) has no unit, so it was read as a plain power ratio \(not decibels\) and converted to (\S+) dB\. Write '(\S+) dB' if you meant decibels\.",
        lambda m: f"ОСШ {m.group(1).replace('.', ',')} дано без единицы, поэтому оно прочитано как обычное отношение "
        f"мощностей (не в децибелах) и переведено в {m.group(2).replace('.', ',')} дБ. "
        f"Напишите «{m.group(3).replace('.', ',')} дБ», если имелись в виду децибелы.",
    ),
    (r"FSK was assumed to be coherent\.", lambda m: "FSK принята когерентной."),
    (
        r"'all' / 'every time' / 'in a row' means every one of the rolls must succeed\.",
        lambda m: "«все» / «каждый раз» / «подряд» означает, что успехом должен закончиться каждый из бросков.",
    ),
    (
        r"(hazard_rate|annual_pd) was given per (month|quarter); converted to a yearly value\.",
        lambda m: f"{m.group(1)} задан в {'месяц' if m.group(2) == 'month' else 'квартал'}; пересчитан в годовое значение.",
    ),
    (r"Bytes were converted to bits \(x 8\)\.", lambda m: "Байты переведены в биты (× 8)."),
    (
        r"'Accurate' was read as sensitivity = specificity = that value\.",
        lambda m: "«Точность» прочитана как чувствительность = специфичность = это значение.",
    ),
    (r"MTBF was computed as 1 / failure rate\.", lambda m: "Наработка на отказ вычислена как 1 / интенсивность отказов."),
    (r"All times were converted to hours\.", lambda m: "Все сроки переведены в часы."),
    (
        r"No time unit was given for the rate, so the duration is taken in the same unit\.",
        lambda m: "Для интенсивности не указана единица времени, поэтому интервал взят в тех же единицах.",
    ),
    (r"Took (\d+) as the number of trials\.", lambda m: f"Число {m.group(1)} принято за число испытаний."),
    (r"Took (\d+) as the number of successes\.", lambda m: f"Число {m.group(1)} принято за число успехов."),
    (
        r"Took (\d+) as the number of events to evaluate\.",
        lambda m: f"Число {m.group(1)} принято за число событий, вероятность которого оценивается.",
    ),
    (
        r"'Year end' was read as the (\d+) days left until 31 December (\d+)\.",
        lambda m: f"«Конец года» прочитан как {m.group(1)} дн. до 31 декабря {m.group(2)}.",
    ),
    (
        r"Coin is biased: P\((heads|tails)\) = (\S+)\.",
        lambda m: f"Монета несимметрична: P({'орёл' if m.group(1) == 'heads' else 'решка'}) = {m.group(2).replace('.', ',')}.",
    ),
    (
        r"A 'double' with two dice has probability 1/36 per roll\.",
        lambda m: "«Дубль» на двух кубиках имеет вероятность 1/36 за бросок.",
    ),
]


def translate_note(note: str) -> str:
    for pattern, fn in _NOTE_RULES:
        m = re.fullmatch(pattern, note)
        if m:
            return fn(m)
    return note


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
_PARAM_RU = {
    "snr_db": "отношение сигнал/шум", "years": "срок", "prior": "распространённость или априорная вероятность",
    "sensitivity": "чувствительность", "specificity": "специфичность", "rate": "интенсивность",
    "k": "искомое число", "n": "число испытаний", "p": "вероятность успеха", "time": "время",
    "mtbf": "наработка на отказ", "successes": "число успехов", "trials": "число испытаний",
    "target": "целевое число", "spot": "текущая цена", "strike": "уровень цены", "volatility": "волатильность",
    "packet_bits": "размер пакета в битах",
}
_HINT_RU = {
    "classical": "Укажите в вопросе число, которое нужно оценить, например «выпадет 7».",
    "counting": "Например: «в среднем 12 звонков в час, ровно 15 звонков за час» или «из 50 изделий 2% бракованных, не менее 3».",
    "signal": "Укажите отношение сигнал/шум, например «при 8 дБ». Для пакета добавьте его размер: «1000-битного пакета».",
    "bayes": "Укажите распространённость, чувствительность и специфичность: «распространённость 1%, чувствительность 99%, специфичность 95%».",
    "finance": "Укажите срок («в течение 3 лет») и годовую вероятность дефолта, либо цену, уровень, волатильность и срок.",
    "reliability": "Укажите срок и наработку на отказ: «наработка на отказ 10000 часов, 2000 часов».",
    "empirical": "Укажите число успехов и испытаний: «37 из 400».",
}


def error_message(exc: Exception) -> Optional[str]:
    """Russian text for MissingParametersError / LowConfidenceError, else None."""
    from .core import LowConfidenceError, MissingParametersError

    if isinstance(exc, MissingParametersError):
        names = []
        for item in exc.missing:
            key = re.split(r"[ (]", item, maxsplit=1)[0]
            ru = _PARAM_RU.get(key)
            names.append(f"{item} ({ru})" if ru and ru != item else item)
        hint = _HINT_RU.get(exc.engine, "")
        return f"В вопросе не хватает данных для модели «{exc.engine}»: {', '.join(names)}. {hint}".strip()
    if isinstance(exc, LowConfidenceError):
        top = ", ".join(f"{n} ({p:.0%})" for n, p in exc.candidates)
        return (
            f"Не удалось понять, к какой области относится вопрос (лучшие варианты: {top}; нужно не менее "
            f"{exc.threshold:.0%}). Переформулируйте вопрос, добавьте числа или укажите модель явно."
        )
    return None
