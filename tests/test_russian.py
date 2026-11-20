"""Russian-language support: normalizer, end-to-end answers, output and errors."""

import json
from math import comb, erfc, exp, factorial, log, sqrt

import pytest

from probai import MissingParametersError, ProbAI
from probai.cli import main
from probai.language import detect_language, normalize_text

AI = ProbAI()


def qf(x):
    return 0.5 * erfc(x / sqrt(2))


def cdf(x):
    return 0.5 * erfc(-x / sqrt(2))


def pois(mu, k):
    return exp(-mu) * mu**k / factorial(k)


def binom(n, p, k):
    return comb(n, k) * p**k * (1 - p) ** (n - k)


# (question, expected engine, probability from an independent formula)
CASES = [
    ("Какова вероятность выпадения 6 на честном кубике?","classical",1/6),
    ("Чему равна вероятность того, что сумма очков на двух кубиках равна 7?","classical",6/36),
    ("Бросаем три кубика, какова вероятность, что сумма не менее 12?","classical",sum(1 for a in range(1,7) for b in range(1,7) for c in range(1,7) if a+b+c>=12)/216),
    ("Какова вероятность, что 20-гранный кубик покажет число больше 10?","classical",0.5),
    ("Вероятность ровно 3 орлов при 5 подбрасываниях монеты","classical",10/32),
    ("Какова вероятность хотя бы одной решки в 3 бросках монеты?","classical",7/8),
    ("Вероятность вытащить туза из стандартной колоды из 52 карт","classical",4/52),
    ("Какова вероятность вытянуть две червы из колоды карт?","classical",comb(13,2)/comb(52,2)),
    ("Кубик бросают 4 раза. Какова вероятность выпадения хотя бы одной шестёрки?","classical",1-(5/6)**4),
    ("Колл-центр получает в среднем 12 звонков в час. Какова вероятность, что ровно 15 звонков за час?","counting",pois(12,15)),
    ("В среднем 6 автобусов в час, какова вероятность, что ровно 2 приедут за 10 минут?","counting",pois(1,2)),
    ("Сайт получает в среднем 4 ошибки в день. Какова вероятность не получить ни одной ошибки завтра?","counting",exp(-4)),
    ("Вероятность 5 или более землетрясений за год при интенсивности 3 в год","counting",1-sum(pois(3,i) for i in range(5))),
    ("В партии из 50 изделий процент брака 2%. Какова вероятность не менее 3 бракованных?","counting",1-sum(binom(50,.02,i) for i in range(3))),
    ("Из 20 пациентов каждый выздоравливает с вероятностью 30%. Какова вероятность, что ровно 8 выздоровеют?","counting",binom(20,.3,8)),
    ("Какова вероятность правильно принять сигнал, если отношение сигнал/шум равно 0,8?","signal",1-qf(sqrt(2*0.8))),
    ("Какова вероятность битовой ошибки для BPSK при 8 дБ?","signal",qf(sqrt(2*10**0.8))),
    ("Какова вероятность успешного приёма 1000-битного пакета по радиоканалу при 7 дБ?","signal",(1-qf(sqrt(2*10**0.7)))**1000),
    ("Радар обнаруживает цель при 12 дБ и вероятности ложной тревоги 0,001. Какова вероятность обнаружения?","signal",qf(3.0902323061678132-sqrt(2*10**1.2))),
    ("Чувствительность теста 99%, специфичность 95%, распространённость болезни 1%. Тест положителен. Какова вероятность, что я болен?","bayes",0.0099/(0.0099+0.0495)),
    ("Тест точен на 90%, и у 3% людей есть заболевание. Результат положительный. Какова вероятность болезни?","bayes",0.027/(0.027+0.097)),
    ("Априорная вероятность 2%, отношение правдоподобия 10. Какова апостериорная вероятность?","bayes",(0.02/0.98*10)/(1+0.02/0.98*10)),
    ("Какова вероятность дефолта заёмщика в течение 3 лет при годовой вероятности дефолта 2%?","finance",1-0.98**3),
    ("Какова вероятность дефолта в течение пяти лет при интенсивности дефолтов 2% в год?","finance",1-exp(-0.1)),
    ("Акция сейчас стоит 100, волатильность 20%. Какова вероятность, что через год она будет ниже 90?","finance",cdf((log(0.9)+0.02)/0.2)),
    ("Наработка на отказ лампы 10000 часов. Какова вероятность, что она проработает 2000 часов?","reliability",exp(-0.2)),
    ("Жёсткий диск со средним сроком службы 10 лет. Какова вероятность, что он выйдет из строя в течение 5 лет?","reliability",1-exp(-0.5)),
    ("Интенсивность отказов сервера 0,0002 в час. Какова вероятность, что он проработает 3 дня?","reliability",exp(-0.0002*72)),
    ("Из 400 посетителей 37 совершили покупку. Какова вероятность, что следующий посетитель купит?","empirical",38/402),
    ("За последние 100 дней дождь шёл в 40. Какой шанс дождя завтра?","empirical",41/102),
    ("Мы протестировали 500 штук и увидели 9 бракованных. Оцените вероятность брака.","empirical",10/502),
    ("У команды 7 побед и 3 поражения. Какова вероятность победы в следующей игре?","empirical",8/12),
]


@pytest.mark.parametrize("question,engine,expected", CASES)
def test_russian_question_end_to_end(question, engine, expected):
    est = AI.estimate(question)
    assert est.engine == engine
    assert est.language == "ru"
    assert est.probability == pytest.approx(expected, rel=1e-6, abs=1e-9)


def test_detect_language():
    assert detect_language("Какова вероятность?") == "ru"
    assert detect_language("What is the probability?") == "en"


def test_english_text_is_not_changed_by_normalizer():
    text = "Probability of rolling a 6 on a die?"
    assert normalize_text(text) == text


def test_normalizer_produces_english_terms():
    out = normalize_text("Вероятность ровно 3 орлов при 5 подбрасываниях монеты")
    assert "exactly 3" in out and "coin" in out


def test_russian_output_is_russian():
    est = AI.estimate("Какова вероятность выпадения 6 на честном кубике?")
    assert "кубик" in est.explanation.lower() or "вероятн" in est.explanation.lower()


def test_russian_missing_parameter_message():
    with pytest.raises(MissingParametersError) as info:
        AI.estimate("Какова вероятность дефолта заёмщика при интенсивности дефолтов 2% в год?")
    assert "лет" in info.value.message_ru or "срок" in info.value.message_ru or "горизонт" in info.value.message_ru


def test_cli_russian_output(capsys):
    assert main(["estimate", "Какова вероятность выпадения 6 на честном кубике?"]) == 0
    assert "Вероятность:" in capsys.readouterr().out


def test_cli_russian_json(capsys):
    main(["estimate", "Вероятность ровно 3 орлов при 5 подбрасываниях монеты", "--json"])
    data = json.loads(capsys.readouterr().out)
    assert data["probability"] == pytest.approx(10 / 32)



def test_russian_all_rolls_must_succeed():
    est = AI.estimate("Я кидаю 6-гранный кубик 10 раз. Какова вероятность, что все 10 раз выпадет 6")
    assert est.probability == pytest.approx(6**-10)


def test_russian_urn():
    est = AI.estimate("В коробке 10 шаров: 3 красных, 6 белых и 1 синий. Какова вероятность вытащить синий шар?")
    assert est.probability == pytest.approx(0.1)
