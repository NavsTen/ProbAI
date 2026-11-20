"""Questions written in plain English, with NO params. Expected values use independent maths."""

import math

import pytest

from probai import MissingParametersError, ProbAI
from probai.extraction import extract_counting, extract_signal


@pytest.fixture(scope="module")
def ai():
    return ProbAI()


def q_func(x):
    return 0.5 * math.erfc(x / math.sqrt(2.0))


def norm_cdf(x):
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def poisson_pmf(mu, k):
    return math.exp(-mu) * mu**k / math.factorial(k)


def binom_pmf(n, p, k):
    return math.comb(n, k) * p**k * (1 - p) ** (n - k)


# ------------------------------------------------------------------ counting
def test_poisson_rate_duration_and_k_from_text(ai):
    est = ai.estimate("A call center receives 12 calls per hour on average. Probability of exactly 15 calls in an hour?")
    assert est.engine == "counting"
    assert est.probability == pytest.approx(poisson_pmf(12, 15))
    assert est.extracted["rate"] == 12 and est.extracted["k"] == 15


def test_poisson_converts_time_units(ai):
    # 6 per hour, window of 10 minutes -> mean 1
    est = ai.estimate("Buses arrive at an average rate of 6 per hour, probability that exactly 2 arrive in 10 minutes")
    assert est.probability == pytest.approx(poisson_pmf(1, 2))
    est = ai.estimate("A call center gets 12 calls per hour. Probability of more than 20 calls in 30 minutes?")
    assert est.probability == pytest.approx(1 - sum(poisson_pmf(6, i) for i in range(21)))


def test_poisson_no_events_and_or_more(ai):
    assert ai.estimate("A website gets 4 errors per day on average, chance of no errors tomorrow").probability == pytest.approx(math.exp(-4))
    est = ai.estimate("Earthquakes follow a Poisson process with rate 3 per year, probability of 5 or more in a year")
    assert est.probability == pytest.approx(1 - sum(poisson_pmf(3, i) for i in range(5)))


def test_binomial_from_text(ai):
    est = ai.estimate("Probability of at least 3 defective items in a batch of 50 with a 2% defect rate")
    assert est.probability == pytest.approx(1 - sum(binom_pmf(50, 0.02, i) for i in range(3)))
    est = ai.estimate("Out of 20 patients each with a 30% chance to recover, probability that exactly 8 recover")
    assert est.probability == pytest.approx(binom_pmf(20, 0.3, 8))
    est = ai.estimate("Probability that 7 out of 10 free throws go in if the shooter hits 70%")
    assert est.probability == pytest.approx(binom_pmf(10, 0.7, 7))


def test_binomial_infers_n_from_the_remaining_number(ai):
    est = ai.estimate("If 15% of customers click an ad and 200 see it, probability that at least 40 click")
    assert est.probability == pytest.approx(1 - sum(binom_pmf(200, 0.15, i) for i in range(40)))
    assert est.extracted["n"] == 200 and any("number of trials" in n for n in est.notes)


# -------------------------------------------------------------------- signal
def test_snr_in_db_modulation_and_ber(ai):
    est = ai.estimate("Probability of bit error for a BPSK radio link at 8 dB")
    assert est.engine == "signal"
    assert est.probability == pytest.approx(q_func(math.sqrt(2 * 10**0.8)))
    est = ai.estimate("Bit error rate of a QPSK receiver with Eb/N0 of 6 dB")
    assert est.extracted["modulation"] == "qpsk"


def test_plain_ratio_snr_is_converted_and_explained(ai):
    est = ai.estimate("What is the probability of correctly receiving a signal, given that the signal-to-noise ratio is 0.8?")
    assert est.engine == "signal"
    assert est.extracted["snr_db"] == pytest.approx(10 * math.log10(0.8))
    assert est.probability == pytest.approx(1 - q_func(math.sqrt(2 * 0.8)))
    assert any("plain power ratio" in n for n in est.notes)


def test_explicit_db_is_not_treated_as_ratio():
    out = extract_signal("signal-to-noise ratio of 3 dB")
    assert out.params["snr_db"] == 3.0 and not out.notes


def test_packet_and_detection_from_text(ai):
    est = ai.estimate("How likely is a 1000-bit packet to be received without errors on a radio channel at 7 dB?")
    ber = q_func(math.sqrt(2 * 10**0.7))
    assert est.probability == pytest.approx((1 - ber) ** 1000)
    est = ai.estimate("Probability that a radar detects a target at 12 dB with a false alarm probability of 0.001")
    thr = 3.0902323061678132  # Q^-1(0.001)
    assert est.probability == pytest.approx(q_func(thr - math.sqrt(2 * 10**1.2)), rel=1e-6)


# --------------------------------------------------------------------- bayes
def test_bayes_numbers_from_text(ai):
    est = ai.estimate(
        "A medical test has 99% sensitivity and 95% specificity and the disease prevalence is 1%. "
        "If I test positive, what is the probability I have the disease?"
    )
    assert est.probability == pytest.approx(0.0099 / (0.0099 + 0.0495))


def test_bayes_adjective_forms_and_negative_result(ai):
    est = ai.estimate(
        "If a test is 95% sensitive, 90% specific and 2% of people are infected, "
        "what is the chance someone who tests negative is infected?"
    )
    assert est.probability == pytest.approx(0.02 * 0.05 / (0.02 * 0.05 + 0.98 * 0.9))


def test_bayes_accuracy_and_likelihood_ratio(ai):
    est = ai.estimate("The test is 90% accurate and 3% of people have the disease. Given a positive test, probability of illness?")
    assert est.probability == pytest.approx(0.027 / (0.027 + 0.97 * 0.1))
    assert any("Accurate" in n for n in est.notes)
    est = ai.estimate("Prior probability of 2% and a likelihood ratio of 10, what is the posterior?")
    odds = 0.02 / 0.98 * 10
    assert est.probability == pytest.approx(odds / (1 + odds))


# ------------------------------------------------------------------- finance
def test_default_probability_from_text(ai):
    est = ai.estimate("Probability that a borrower defaults within 3 years with an annual default rate of 2%")
    assert est.probability == pytest.approx(1 - 0.98**3)
    est = ai.estimate("Bank loan default likelihood over 5 years with a hazard rate of 2% per year")
    assert est.probability == pytest.approx(1 - math.exp(-0.1))


def test_price_threshold_from_text(ai):
    est = ai.estimate("The stock is currently at 100 with 20% volatility. Probability that it ends below 90 in one year?")
    z = (math.log(0.9) + 0.5 * 0.04) / 0.2
    assert est.probability == pytest.approx(norm_cdf(z))
    est = ai.estimate("Chance that the Nasdaq, now at 16000 with 25% volatility, drops below 15000 in 6 months")
    z = (math.log(15000 / 16000) + 0.5 * 0.0625 * 0.5) / (0.25 * math.sqrt(0.5))
    assert est.probability == pytest.approx(norm_cdf(z), rel=1e-6)


def test_year_end_is_explained(ai):
    est = ai.estimate("Probability that bitcoin price exceeds 100000 dollars at the end of the year, currently at 90000 with volatility of 60%")
    assert est.extracted["strike"] == 100000 and est.extracted["spot"] == 90000
    assert any("Year end" in n for n in est.notes)


# --------------------------------------------------------------- reliability
def test_reliability_units_are_converted(ai):
    est = ai.estimate("A lightbulb has an MTBF of 10000 hours, chance it survives 2000 hours")
    assert est.probability == pytest.approx(math.exp(-0.2))
    est = ai.estimate("What is the probability that a hard drive with a mean lifetime of 10 years fails within 5 years?")
    assert est.probability == pytest.approx(1 - math.exp(-0.5))
    est = ai.estimate("Chance that a server with a failure rate of 0.0002 per hour survives 3 days")
    assert est.probability == pytest.approx(math.exp(-0.0002 * 72))


def test_weibull_from_text(ai):
    est = ai.estimate("Weibull pump with shape 3 and scale 1000 hours, probability of failure at 500 hours")
    assert est.probability == pytest.approx(1 - math.exp(-(0.5**3)))


# ----------------------------------------------------------------- empirical
def test_empirical_phrasings(ai):
    for text, k, n in [
        ("We observed 37 conversions out of 400 visitors, probability that the next visitor converts?", 37, 400),
        ("From my records, it rained on 40 of the last 100 days. What is the chance of rain tomorrow?", 40, 100),
        ("We tested 500 units and saw 9 rejects, estimate the reject probability", 9, 500),
    ]:
        est = ai.estimate(text)
        assert est.engine == "empirical"
        assert est.probability == pytest.approx((k + 1) / (n + 2))


# ----------------------------------------------------------------- classical
def test_dice_coin_and_cards_from_text(ai):
    assert ai.estimate("Roll a die 4 times, probability of at least one six").probability == pytest.approx(1 - (5 / 6) ** 4)
    assert ai.estimate("A coin lands heads 70% of the time. Probability of exactly 3 heads in 5 flips?").probability == pytest.approx(
        binom_pmf(5, 0.7, 3)
    )
    assert ai.estimate("What is the probability that a 20-sided die will show a number greater than 10?").probability == pytest.approx(0.5)


def test_cards_count_words_are_not_ranks(ai):
    """Regression: 'two hearts' used to be read as the two of hearts (1/52)."""
    assert ai.estimate("What is the chance of drawing two hearts from a deck of cards?").probability == pytest.approx(
        math.comb(13, 2) / math.comb(52, 2)
    )
    assert ai.estimate("Probability of drawing the two of hearts from a deck").probability == pytest.approx(1 / 52)


# ----------------------------------------------------------- behaviour rules
def test_explicit_params_override_the_text(ai):
    est = ai.estimate("A call center receives 12 calls per hour. Probability of exactly 15 calls in an hour?", {"rate": 20})
    assert est.probability == pytest.approx(poisson_pmf(20, 15))
    assert "rate" not in est.extracted


def test_auto_extract_can_be_disabled():
    off = ProbAI(auto_extract=False)
    with pytest.raises(MissingParametersError):
        off.estimate("Probability of bit error for a BPSK radio link at 8 dB")


def test_missing_data_is_still_reported(ai):
    with pytest.raises(MissingParametersError) as info:
        ai.estimate("Probability of bit error on a radio link")
    assert "snr_db" in str(info.value)
    with pytest.raises(MissingParametersError):
        ai.estimate("How likely is it that a hard drive fails?")


def test_extraction_does_not_invent_numbers():
    assert extract_counting("How many customers will arrive?").params == {}
    assert extract_signal("Probability of bit error on a radio link").params == {}


def test_cli_understands_plain_questions(capsys):
    from probai.cli import main

    code = main(["estimate", "A call center receives 12 calls per hour on average. Probability of exactly 15 calls in an hour?"])
    out = capsys.readouterr().out
    assert code == 0 and "0.0723911" in out and "Understood from your question" in out and "rate=12" in out


def test_default_question_without_horizon_asks_for_it(ai):
    with pytest.raises(MissingParametersError) as info:
        ai.estimate("Bank loan default likelihood with a hazard rate of 2% per year")
    assert "years" in str(info.value) and "time horizon" in str(info.value)
    est = ai.estimate("Bank loan default likelihood with a hazard rate of 2% per year over 5 years")
    assert est.probability == pytest.approx(1 - math.exp(-0.1))


def test_monthly_default_rate_is_converted_to_yearly():
    from probai import ProbAI

    est = ProbAI().estimate("Probability of default is 2% per month; chance of default within 2 years?")
    assert est.probability == pytest.approx(1 - 0.98**24)


def test_russian_loss_of_income_as_default_hazard():
    from probai import ProbAI

    est = ProbAI().estimate("Вероятность потерять работу 2% в месяц. Какова вероятность дефолта за 2 года?")
    assert est.engine == "finance"
    assert est.probability == pytest.approx(1 - 0.98**24)
    assert any("месяц" in n for n in est.notes)
