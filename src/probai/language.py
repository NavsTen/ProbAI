"""Language support: detect Russian and rewrite a Russian question into the English form
that the router, the extractors and the engines understand.

The rewrite is rule-based and deliberately conservative:

* English text is returned untouched.
* Numbers written in words ("двенадцать"), decimal commas ("0,8"), thousands ("10 тыс."),
  percentages, dB, time units, comparisons ("не менее", "ровно") and the vocabulary of every
  engine (кубик, монета, чувствительность, дефолт, наработка на отказ ...) are mapped to English.
* Any Cyrillic word that is not known is transliterated to Latin letters, so sentence
  structure (words between a number and its unit, for example) is preserved.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

_CYR = re.compile(r"[а-яё]", re.I)


def has_cyrillic(text: str) -> bool:
    return bool(_CYR.search(text))


def detect_language(text: str) -> str:
    """Return ``"ru"`` when the text contains Cyrillic letters, otherwise ``"en"``."""
    return "ru" if has_cyrillic(text) else "en"


# --------------------------------------------------------------------------- #
# Numbers written in words
# --------------------------------------------------------------------------- #
_UNITS: Dict[str, int] = {
    "ноль": 0, "нуль": 0, "ноля": 0, "нуля": 0,
    "один": 1, "одна": 1, "одно": 1, "одного": 1, "одной": 1, "одну": 1, "одним": 1,
    "два": 2, "две": 2, "двух": 2, "двум": 2, "двумя": 2,
    "три": 3, "трех": 3, "трем": 3, "тремя": 3,
    "четыре": 4, "четырех": 4, "четырем": 4,
    "пять": 5, "пяти": 5, "шесть": 6, "шести": 6, "семь": 7, "семи": 7,
    "восемь": 8, "восьми": 8, "девять": 9, "девяти": 9,
    "десять": 10, "десяти": 10, "одиннадцать": 11, "двенадцать": 12, "двенадцати": 12,
    "тринадцать": 13, "четырнадцать": 14, "пятнадцать": 15, "пятнадцати": 15,
    "шестнадцать": 16, "семнадцать": 17, "восемнадцать": 18, "девятнадцать": 19,
}
_TENS: Dict[str, int] = {
    "двадцать": 20, "двадцати": 20, "тридцать": 30, "тридцати": 30, "сорок": 40, "сорока": 40,
    "пятьдесят": 50, "пятидесяти": 50, "шестьдесят": 60, "семьдесят": 70, "восемьдесят": 80, "девяносто": 90,
}
_HUNDREDS: Dict[str, int] = {
    "сто": 100, "ста": 100, "двести": 200, "триста": 300, "четыреста": 400, "пятьсот": 500,
    "шестьсот": 600, "семьсот": 700, "восемьсот": 800, "девятьсот": 900,
}
_ALL_NUMBER_WORDS = {**_UNITS, **_TENS, **_HUNDREDS}
_NUMWORD_RE = re.compile(
    r"(?<![а-яё])(" + "|".join(sorted(_ALL_NUMBER_WORDS, key=len, reverse=True)) + r")(?![а-яё])"
    r"(?:\s+(" + "|".join(sorted(_UNITS, key=len, reverse=True)) + r")(?![а-яё]))?"
)


def _numword_sub(m: "re.Match[str]") -> str:
    first = _ALL_NUMBER_WORDS[m.group(1)]
    second = m.group(2)
    if second is not None:
        s = _UNITS[second]
        if (first >= 20 and first % 10 == 0 and 1 <= s <= 9) or (first >= 100 and first % 100 == 0 and s >= 1):
            return str(first + s)
        return f"{first} {s}"  # e.g. "три пять": two separate numbers
    return str(first)


def _numbers(s: str) -> str:
    s = _NUMWORD_RE.sub(_numword_sub, s)
    # "100 000" -> "100000" (Russian thousands separator is a space)
    s = re.sub(r"(?<![\d.])(\d{1,3})((?:[  ]\d{3})+)(?![\d])", lambda m: m.group(1) + re.sub(r"\D", "", m.group(2)), s)

    def scale(m: "re.Match[str]") -> str:
        v = float(m.group(1))
        word = m.group(2)
        mult = 1e9 if word.startswith("млрд") or word.startswith("миллиард") else 1e6 if word.startswith(("млн", "миллион")) else 1e3
        return f"{v * mult:.15g}"

    s = re.sub(r"(\d+(?:\.\d+)?)\s*(тыс(?:яч[а-яё]*)?|млн|миллион[а-яё]*|млрд|миллиард[а-яё]*)\.?(?![а-яё])", scale, s)
    return s


# --------------------------------------------------------------------------- #
# Phrase rules: (regex, replacement). Order matters: specific before generic.
# --------------------------------------------------------------------------- #
def _w(stem: str) -> str:
    """A word that starts with ``stem`` (any Russian ending)."""
    return rf"(?<![а-яё]){stem}[а-яё]*"


_END = r"(?![а-яё])"
_START = r"(?<![а-яё])"

_RULES_SRC: List[Tuple[str, str]] = [
    # ---- units that follow numbers
    (r"(\d)\s*(?:дб|db)" + _END, r"\1 dB"),
    (_w("децибел"), "dB"),
    (r"(\d)\s*ч" + _END, r"\1 hours"),
    (r"(\d)\s*(?:мин)\.?" + _END, r"\1 minutes"),
    (r"(\d)\s*сек\.?" + _END, r"\1 seconds"),
    (r"(\d)\s*(?:кб|kb)" + _END, r"\1 kilobytes"),
    (_START + r"полгода" + _END, "6 months"),
    (_START + r"полчаса" + _END, "30 minutes"),
    # ---- question openers
    (_w("какова") + r"\s+вероятность(?:,?\s+(?:того,?\s+)?что)?", "what is the probability that"),
    (_w("какая") + r"\s+вероятность(?:,?\s+(?:того,?\s+)?что)?", "what is the probability that"),
    (r"чему\s+равна\s+вероятность(?:,?\s+(?:того,?\s+)?что)?", "what is the probability that"),
    (r"с\s+какой\s+вероятностью(?:,?\s+что)?", "what is the probability that"),
    (r"насколько\s+вероятно(?:,?\s+что)?", "how likely is it that"),
    (r"какие\s+шансы(?:,?\s+что)?", "what are the chances that"),
    (_w("найдите") + r"\s+вероятность(?:,?\s+(?:того,?\s+)?что)?", "find the probability that"),
    (_w("вычислите") + r"\s+вероятность(?:,?\s+(?:того,?\s+)?что)?", "calculate the probability that"),
    (_w("оцените") + r"\s+вероятность(?:,?\s+(?:того,?\s+)?что)?", "estimate the probability that"),
    (_w("рассчитайте") + r"\s+вероятность(?:,?\s+(?:того,?\s+)?что)?", "calculate the probability that"),
    (r"вероятност[а-яё]*,?\s+(?:того,?\s+)?что", "probability that"),
    (r"а\s+(?:что\s+)?если|а\s+как\s+насчет|а\s+что\s+будет\s+если", "what if"),
    (_START + r"дважды" + _END, "twice"),
    (_START + r"трижды" + _END, "3 times"),
    (r"хотя\s+бы\s+раз" + _END, "at least once"),
    (r"ни\s+разу", "no"),
    # ---- comparisons (before numbers get glued to other words)
    (r"(\d+)\s+(?:или|и)\s+более" + _END, r"\1 or more"),
    (r"(\d+)\s+(?:или|и)\s+(?:менее|меньше)" + _END, r"\1 or fewer"),
    (r"(?:не\s+менее|не\s+меньше|как\s+минимум|по\s+меньшей\s+мере|хотя\s+бы)" + _END + r"(?:\s+чем)?", "at least"),
    (_START + r"минимум" + _END, "at least"),
    (r"(?:не\s+более|не\s+больше|самое\s+большее|не\s+превы[а-яё]*)" + _END, "at most"),
    (_START + r"максимум" + _END, "at most"),
    (_START + r"(?:ровно|в\s+точности)" + _END, "exactly"),
    (_START + r"точно(?=\s+\d)", "exactly"),
    (_START + r"(?:более|больше|свыше)" + _END + r"(?:\s+чем)?", "more than"),
    (_START + r"(?:менее|меньше)" + _END + r"(?:\s+чем)?", "less than"),
    (_w("превы"), "exceeds"),
    (_START + r"ниже" + _END, "below"),
    (_START + r"выше" + _END, "above"),
    (r"ни\s+одн[а-яё]*|ни\s+единого|ни\s+единой", "no"),
    (r"без\s+(?:единой\s+)?ошибок", "without errors"),
    (_w("безошибочн"), "error-free"),
    # ---- rates, windows and horizons
    (r"(?:в|каждый|каждую)\s+час" + _END + r"|ежечасно", "per hour"),
    (r"(?:в|каждую)\s+минуту|ежеминутно", "per minute"),
    (r"в\s+секунду", "per second"),
    (r"в\s+(?:день|сутки)|ежедневно", "per day"),
    (r"в\s+неделю|еженедельно", "per week"),
    (r"в\s+месяц|ежемесячно", "per month"),
    (r"в\s+год" + _END + r"|ежегодно", "per year"),
    (r"(?:за|в\s+течение|на\s+протяжении)\s+(?:одн[а-яё]*\s+|1\s+)?час(?:а|у)?" + _END, "in an hour"),
    (r"(?:за|в\s+течение)\s+(?:одн[а-яё]*\s+|1\s+)?минут[а-яё]*" + _END, "in a minute"),
    (r"(?:за|в\s+течение)\s+(?:одн[а-яё]*\s+|1\s+)?(?:день|дня|сутки|суток)" + _END, "in a day"),
    (r"(?:за|в\s+течение)\s+(?:одн[а-яё]*\s+|1\s+)?недел[а-яё]*" + _END, "in a week"),
    (r"(?:за|в\s+течение)\s+(?:одн[а-яё]*\s+|1\s+)?месяц[а-яё]*" + _END, "in a month"),
    (r"(?:за|в\s+течение|на\s+протяжении)\s+(?:одн[а-яё]*\s+|1\s+)?(?:год|года)" + _END + r"|в\s+ближайший\s+год|в\s+следующем\s+году|через\s+год", "in 1 year"),
    (r"к\s+концу\s+(?:этого\s+)?года|до\s+конца\s+(?:этого\s+)?года|на\s+конец\s+года", "by year end"),
    (r"за\s+последни[а-яё]*\s+(?=\d)", "in the last "),
    (r"в\s+течение\s+следующи[а-яё]*\s+(?=\d)|в\s+ближайши[а-яё]*\s+(?=\d)|в\s+следующи[а-яё]*\s+(?=\d)", "in the next "),
    (r"в\s+течение\s+(?=\d)", "within "),
    (r"через\s+(?=\d)", "in "),
    (_START + r"за\s+(?=\d)", "in "),
    (_START + r"завтра" + _END, "tomorrow"),
    (_START + r"сегодня" + _END, "today"),
    (r"в\s+среднем\s+(?=\d)", "average of "),
    (r"в\s+среднем", "on average"),
    (_w("пуассон"), "poisson"),
    (r"интенсивност[а-яё]*\s+дефолт[а-яё]*", "hazard rate"),
    (r"интенсивност[а-яё]*\s+отказ[а-яё]*", "failure rate"),
    (r"интенсивност[а-яё]*", "rate"),
    # ---- time units (after windows/rates are handled)
    (_START + r"час(?:а|ов|ы|у|е)?" + _END, "hours"),
    (_w("минут"), "minutes"),
    (_w("секунд"), "seconds"),
    (_START + r"(?:день|дня|дней|дни|дне|сутки|суток|сутками)" + _END, "days"),
    (_w("недел"), "weeks"),
    (_w("месяц"), "months"),
    (_w("квартал"), "quarter"),
    (_START + r"(?:год|года|году|годы|лет|годам|годах)" + _END, "years"),
    # ---- generic probability words
    (r"отношени[а-яё]*\s+правдоподоби[а-яё]*", "likelihood ratio"),
    (r"вероятност[а-яё]*\s+дефолт[а-яё]*", "probability of default"),
    (r"вероятност[а-яё]*\s+(?:потер[а-яё]+|утрат[а-яё]+)\s+(?:работ[а-яё]*|доход[а-яё]*)", "probability of default"),
    (r"вероятност[а-яё]*\s+отказ[а-яё]*", "probability of failure"),
    (r"вероятност[а-яё]*\s+безотказн[а-яё]*\s+работ[а-яё]*", "probability of surviving"),
    (_w("вероятност"), "probability"),
    (_w("шанс"), "chance"),
    (_w("правдоподоби") + r"\s*", "likelihood "),
    # ---- classical: dice, coins, cards
    (r"(\d+)[- ]?гранн[а-яё]*", r"\1-sided"),
    (_START + r"(?:четырех|четырёх)гранн[а-яё]*", "4-sided"),
    (_START + r"шестигранн[а-яё]*", "6-sided"),
    (_START + r"восьмигранн[а-яё]*", "8-sided"),
    (_START + r"десятигранн[а-яё]*", "10-sided"),
    (_START + r"двенадцатигранн[а-яё]*", "12-sided"),
    (_START + r"двадцатигранн[а-яё]*", "20-sided"),
    (r"(\d+)\s+(?:игральн[а-яё]*\s+)?(?:кубик|куб|кост)[а-яё]*", r"\1 dice"),
    (r"(?:игральн[а-яё]*\s+)?(?:кубики|кубиков|кости|костей)" + _END, "dice"),
    (r"(?:игральн[а-яё]*\s+)?(?:кубик|кост)[а-яё]*", "die"),
    (_w("честн") + r"|" + _w("симметричн"), "fair"),
    (_w("стандартн"), "standard"),
    (r"сумм[а-яё]*(?:\s+очков)?", "sum"),
    (_w("выпад"), "shows"),
    (r"(?:очков|очки|очка|очко)" + _END, ""),
    (_w("числ"), "number"),
    (r"(\d+)\s+раз(?:а|ов)?" + _END, r"\1 times"),
    (r"(\d+)\s+(?:под)?брос[а-яё]*", r"\1 tosses"),
    (r"(\d+)\s+подбрасыван[а-яё]*", r"\1 tosses"),
    (_w("подбрас"), "toss"),
    (_w("брос"), "roll"),
    (r"(\d+)\s+монет[а-яё]*", r"\1 coins"),
    (_w("монет"), "coin"),
    (r"все(?=\s+(?:\d+\s+)?(?:times|раз|броск|орл|орёл|решк))|(?<![а-яё])всегда(?![а-яё])", "all"),
    (r"каждый\s+раз|каждом\s+броске", "every time"),
    (r"красн[а-яё]*", "red"),
    (r"син(?:ий|его|ие|их|им|ей|яя|ее|юю|ем)(?![а-яё])", "blue"),
    (r"бел(?:ый|ого|ые|ых|ым|ая|ое|ую)(?![а-яё])", "white"),
    (r"зел[её]н[а-яё]*", "green"),
    (r"ч[её]рн(?:ый|ого|ые|ых|ым|ая|ое|ую)(?![а-яё])", "black"),
    (r"ж[её]лт[а-яё]*", "yellow"),
    (r"(?<![а-яё])шар(?:ов|а|ы|ами|ом|ик[а-яё]*)?(?![а-яё])", "balls"),
    (r"(?<![а-яё])(?:урн[а-яё]*|мешк[а-яё]*|ящик[а-яё]*|коробк[а-яё]*)", "box"),
    (r"подряд", "in a row"),
    (r"(?:орл[а-яё]*|орел|орёл)" + _END, "heads"),
    (r"решк[а-яё]*|решек" + _END, "tails"),
    (r"(?:колод[а-яё]*)\s+из", "deck of"),
    (_w("колод"), "deck"),
    (r"(?:фигурн[а-яё]*\s+карт[а-яё]*|картинк[а-яё]*)", "face card"),
    (r"красн[а-яё]*\s+(?:карт[а-яё]*|масти?)", "red card"),
    (r"(?:черн|чёрн)[а-яё]*\s+(?:карт[а-яё]*|масти?)", "black card"),
    (r"(\d+)\s+карт[а-яё]*", r"\1 cards"),
    (_w("карт"), "card"),
    (r"(?:тузов|тузы)" + _END, "aces"),
    (_w("туз"), "ace"),
    (r"(?:королей|короли)" + _END, "kings"),
    (_w("король") + r"|" + _w("короля"), "king"),
    (r"(?:дамы|дам)" + _END, "queens"),
    (_START + r"дам(?:а|у|е|ой)?" + _END, "queen"),
    (r"(?:валетов|валеты)" + _END, "jacks"),
    (_w("валет"), "jack"),
    (_w("черв"), "hearts"),
    (_START + r"пик(?:а|и|у|ой|ам|ами|ах|ов)?" + _END, "spades"),
    (_w("треф"), "clubs"),
    (_w("бубн"), "diamonds"),
    # ---- counting
    (r"(?:браковн[а-яё]*\s+)?(?:процент|доля|доли)\s+брака", "defect rate"),
    (_w("дефектн") + r"|" + _w("бракован"), "defective"),
    (_w("брак"), "reject"),
    (_w("партии") + r"|" + _w("партия"), "batch"),
    # ---- signal
    (r"отношени[а-яё]*\s+сигнал\s*[/-]\s*шум", "signal-to-noise ratio"),
    (r"сигнал\s*[/-]\s*шум", "signal-to-noise ratio"),
    (_START + r"осш" + _END, "signal-to-noise ratio"),
    (r"двоичн[а-яё]*\s+фазов[а-яё]*\s+манипуляци[а-яё]*", "binary phase shift keying"),
    (r"квадратурн[а-яё]*\s+фазов[а-яё]*\s+манипуляци[а-яё]*", "quadrature phase shift keying"),
    (r"частотн[а-яё]*\s+манипуляци[а-яё]*|" + _START + r"чмн" + _END, "fsk"),
    (r"битов[а-яё]*\s+ошибк[а-яё]*|ошибк[а-яё]*\s+на\s+бит[а-яё]*", "bit error"),
    (r"(\d+)[- ]?битн[а-яё]*", r"\1-bit"),
    (r"(\d+)\s+бит[а-яё]*", r"\1 bits"),
    (r"(\d+)\s+байт[а-яё]*", r"\1 bytes"),
    (_START + r"бит(?:а|ов|ы|у|ом|е)?" + _END, "bit"),
    (_w("ошибк") + r"|" + _w("ошибок"), "error"),
    (_w("пакет"), "packet"),
    (_w("кадр"), "frame"),
    (_w("радиолокатор") + r"|" + _w("радар") + r"|" + _w("локатор"), "radar"),
    (_w("гидролокатор") + r"|" + _w("сонар"), "sonar"),
    (r"ложн[а-яё]*\s+тревог[а-яё]*", "false alarm"),
    (_w("обнаруж"), "detect"),
    (_w("правильн") + r"|" + _w("корректн") + r"|" + _w("верно"), "correct"),
    (_w("успешн"), "successful"),
    (_w("прием") + r"|" + _w("приём"), "reception"),
    (_w("принят") + r"|" + _w("принять") + r"|" + _w("принима"), "receive"),
    (_w("радио"), "radio"),
    (_w("канал"), "channel"),
    (_w("гауссов"), "gaussian"),
    (_w("белый") + r"|" + _w("белого"), "white"),
    (_w("шум"), "noise"),
    (_w("помех"), "interference"),
    (_w("спутник"), "satellite"),
    (_w("приемник") + r"|" + _w("приёмник"), "receiver"),
    # ---- bayes
    (r"отношени[а-яё]*\s+правдоподоби[а-яё]*", "likelihood ratio"),
    (r"фактор[а-яё]*\s+байеса|байесовск[а-яё]*\s+фактор[а-яё]*", "bayes factor"),
    (r"априорн[а-яё]*\s+вероятност[а-яё]*", "prior probability"),
    (_w("априорн"), "prior"),
    (_w("апостериорн"), "posterior"),
    (_w("чувствительн"), "sensitivity"),
    (_w("специфичн"), "specificity"),
    (r"распространенност[а-яё]*|распространённост[а-яё]*|превалентност[а-яё]*|встречаемост[а-яё]*|доля\s+больных|доля\s+заболевших", "prevalence"),
    (r"базов[а-яё]*\s+(?:частот[а-яё]*|уровен[а-яё]*|доля)", "base rate"),
    (r"доля\s+ложн[а-яё]*\s+срабатываний|ложноположительн[а-яё]*\s+(?:результат[а-яё]*|срабатыван[а-яё]*)", "false positive rate"),
    (_w("ложноположительн"), "false positive"),
    (r"положительн[а-яё]*|положителен" + _END, "positive"),
    (r"отрицательн[а-яё]*|отрицателен" + _END, "negative"),
    (_w("точност"), "accuracy"),
    (_w("точн") + r"|" + _START + r"точен" + _END, "accurate"),
    (r"(\d+(?:\.\d+)?)\s*%\s+(?:людей|населения|пациентов|взрослых|жителей|человек|заболевших)\s+(?:\w+\s+){0,2}?(?:болеют|больны|заражены|инфицированы|страдают|имеют|носители)", r"\1% of people have"),
    (r"у\s+(\d+(?:\.\d+)?)\s*(?:%|процент[а-яё]*)\s+(?:людей|населения|пациентов|взрослых|жителей|человек)\s+(?:есть|имеется|обнаружен[а-яё]*)", r"\1% of people have"),
    (_w("болезн") + r"|" + _w("заболеван"), "disease"),
    (_w("инфицирован") + r"|" + _w("заражен") + r"|" + _w("заражён"), "infected"),
    (_w("больн") + r"|" + _START + r"болен" + _END, "sick"),
    (_w("тест"), "test"),
    (_w("анализ") + _END, "test"),
    (_w("спам"), "spam"),
    (_w("мошеннич") + r"|" + _START + r"фрод[а-яё]*", "fraud"),
    (_w("улик") + r"|" + _w("доказательств"), "evidence"),
    (_w("виновн"), "guilty"),
    # ---- finance
    (_w("годов"), "annual"),
    (_w("дефолт") + r"|" + _w("невозврат"), "default"),
    (_w("обанкрот") + r"|" + _w("банкрот"), "bankrupt"),
    (_w("заемщик") + r"|" + _w("заёмщик"), "borrower"),
    (_w("кредит"), "loan"),
    (_w("ипотек"), "mortgage"),
    (_w("облигаци"), "bond"),
    (_w("компани"), "company"),
    (_w("банк"), "bank"),
    (_w("акци") + _END, "stock"),
    (_w("акции") + r"|" + _w("акций"), "stock"),
    (_w("биткоин") + r"|" + _w("биткойн"), "bitcoin"),
    (_w("нефт"), "oil"),
    (_w("золот"), "gold"),
    (_w("доллар"), "dollar"),
    (_START + r"евро" + _END, "euro"),
    (_w("индекс"), "index"),
    (_w("портфел"), "portfolio"),
    (_w("волатильност"), "volatility"),
    (_w("доходност"), "return"),
    (r"текущ[а-яё]*\s+цен[а-яё]*", "current price"),
    (r"(?:сейчас|в\s+данный\s+момент|в\s+настоящее\s+время)\s+(?:стоит|торгуется|находится|составляет)(?:\s+(?:на|по|около)(?:\s+уровне)?)?", "currently at"),
    (r"торгуется\s+(?:на\s+уровне|по)", "trading at"),
    (r"(?:упадет|упадёт|опустится|снизится|окажется|закроется|будет)\s+(?=below)", "falls "),
    (_START + r"цен(?:а|ы|у|е|ой)?" + _END, "price"),
    (_w("рыночн"), "market"),
    # ---- reliability
    (r"средн[а-яё]*\s+(?:время\s+)?(?:наработк[а-яё]*\s+)?(?:на\s+отказ|между\s+отказами)|наработк[а-яё]*\s+на\s+отказ|" + _START + r"сно" + _END, "mean time between failures"),
    (r"средн[а-яё]*\s+срок[а-яё]*\s+службы|средн[а-яё]*\s+время\s+жизни|средн[а-яё]*\s+ресурс[а-яё]*", "mean lifetime"),
    (_w("срок") + r"\s+службы", "lifetime"),
    (r"гарантийн[а-яё]*\s+срок[а-яё]*", "warranty period"),
    (_w("гаранти"), "warranty"),
    (r"выйд[а-яё]*\s+из\s+строя|выход[а-яё]*\s+из\s+строя", "fails"),
    (r"не\s+(?:откажет|сломается)|безотказн[а-яё]*|без\s+отказов", "without failing"),
    (r"(?:проработает|прослужит|будет\s+работать|продолжит\s+работать|продолжает\s+работать|останется\s+работоспособн[а-яё]*|сохранит\s+работоспособност[а-яё]*)", "keeps working"),
    (_w("выжив"), "survives"),
    (r"(?:откажет|сломается|перестанет\s+работать|ломается)", "fails"),
    (_w("отказ"), "failure"),
    (_w("поломк"), "failure"),
    (_w("вейбулл"), "weibull"),
    (r"параметр[а-яё]*\s+формы", "shape parameter"),
    (r"параметр[а-яё]*\s+масштаба", "scale parameter"),
    (r"жестк[а-яё]*\s+диск[а-яё]*", "hard drive"),
    (r"лампочк[а-яё]*|лампа|лампы|лампу", "light bulb"),
    (_w("двигател"), "engine"),
    (_w("насос"), "pump"),
    (_w("сервер"), "server"),
    (r"аккумулятор[а-яё]*|батаре[а-яё]*", "battery"),
    (_w("подшипник"), "bearing"),
    (_w("вентилятор"), "fan"),
    (_w("ноутбук"), "laptop"),
    (_w("компонент") + r"|" + _w("деталь") + r"|" + _w("детали") + r"|" + _w("узел"), "component"),
    (_w("устройств"), "device"),
    # ---- empirical
    (r"на\s+основе\s+(?:исторических\s+|прошлых\s+|имеющихся\s+)?данных|по\s+(?:историческим|прошлым|имеющимся)\s+данным", "based on historical data"),
    (r"по\s+нашим\s+записям|по\s+моим\s+записям|по\s+(?:нашей|моей)\s+статистике", "from our records"),
    (_w("наблюдал") + r"|" + _w("наблюдени"), "observed"),
    (r"(?:протестировал|проверил|испытал|опросил)[а-яё]*", "tested"),
    (r"(?:увидел|обнаружил|нашл|нашли|зафиксировал|получил)[а-яё]*", "saw"),
    (_w("побед") + r"|" + _w("выигрыш") + r"|" + _w("выигра"), "wins"),
    (_w("поражени") + r"|" + _w("проигрыш") + r"|" + _w("проигра"), "losses"),
    (_w("успех"), "successes"),
    (_w("неудач"), "failures"),
    (r"штук[а-яё]*|единиц[а-яё]*|изделий|изделия", "units"),
    (_w("посетител"), "visitors"),
    (_w("клиент") + r"|" + _w("покупател"), "customers"),
    (_w("пациент"), "patients"),
    (_w("конверси"), "conversions"),
    (r"доверительн[а-яё]*\s+интервал[а-яё]*", "credible interval"),
    # ---- glue words
    (r"из\s+последних", "of the last"),
    (_START + r"из" + _END, "out of"),
    (_START + r"среди" + _END, "among"),
    (_START + r"но" + _END, "but"),
    (_START + r"и" + _END, "and"),
    (_START + r"или" + _END, "or"),
    (_START + r"если" + _END, "if"),
    (_START + r"что" + _END, "that"),
    (_START + r"при" + _END, "at"),
    (_START + r"с" + _END, "with"),
    (_START + r"в" + _END, "in"),
    (_START + r"(?:а|то)" + _END, ""),
    (_w("процент"), "percent"),
    (r"\bна\s+(?=\d+\s*(?:%|percent))", "of "),
]
_RULES = [(re.compile(p), r) for p, r in _RULES_SRC]

_TRANSLIT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e", "ж": "zh", "з": "z", "и": "i",
    "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t",
    "у": "u", "ф": "f", "х": "kh", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "shch", "ъ": "", "ы": "y",
    "ь": "", "э": "e", "ю": "yu", "я": "ya",
}


def _translit(s: str) -> str:
    return "".join(_TRANSLIT.get(ch, ch) for ch in s)


_STOP = r"(?!(?:exactly|at|least|most|more|less|than|fewer|or)\b)"
_NOT_NUM_AFTER = r"(?!\d|\.\d|\s*(?:%|percent))"


def _reorder_n_k(s: str) -> str:
    """'из 400 посетителей 37 купили' -> '37 out of 400 посетителей купили'.

    Russian often puts the total first. Only applied when the second number is not
    introduced by a comparison word and is not larger than the total.
    """

    def fix(m: "re.Match[str]") -> str:
        n, words, k = int(m.group(1)), m.group(2), int(m.group(3))
        if k > n:
            return m.group(0)
        return f"{k} out of {n} {words.strip()}".strip()

    s = re.sub(
        rf"(?<![\w.])out of (\d+)((?:\s+{_STOP}[a-zа-яё-]+){{0,3}}?)\s+(\d+){_NOT_NUM_AFTER}", fix, s
    )

    def fix_last(m: "re.Match[str]") -> str:
        n, unit, k = int(m.group(1)), m.group(2), int(m.group(4))
        if k > n:
            return m.group(0)
        return f"{k} of the last {n} {unit}{m.group(3)}"

    return re.sub(
        rf"in the last (\d+) (hours|days|weeks|months|years|[a-zа-яё]+)((?:\s+{_STOP}[a-zа-яё-]+){{0,3}}?)\s+in (\d+){_NOT_NUM_AFTER}",
        fix_last,
        s,
    )


def normalize_text(text: str) -> str:
    """Rewrite a Russian question in English terms; return other text unchanged."""
    if not has_cyrillic(text):
        return text
    s = text.lower().replace("ё", "е").replace("−", "-").replace(" ", " ")
    s = re.sub(r"(?<=\d),(?=\d)", ".", s)  # decimal comma
    s = re.sub(r"(?<![а-яё])ни\s+(?:одн[а-яё]*|единого|единой)(?![а-яё])", "no", s)
    s = _numbers(s)
    for pattern, repl in _RULES:
        s = pattern.sub(repl, s)
    s = _reorder_n_k(s)
    s = _translit(s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s+([?.,;:!])", r"\1", s)
    return s.strip()
