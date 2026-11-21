"""Engine tests. Expected values come from brute force or independent formulas."""

import itertools
import math

import pytest

from probai.core import MissingParametersError
from probai.engines import (
    BayesEngine,
    ClassicalEngine,
    CountingEngine,
    EmpiricalEngine,
    FinanceEngine,
    ReliabilityEngine,
    SignalEngine,
)


def q_func(x):
    """Gaussian tail probability, implemented independently with erfc."""
    return 0.5 * math.erfc(x / math.sqrt(2.0))


def norm_cdf(x):
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


# ------------------------------------------------------------------ classical
class TestClassical:
    engine = ClassicalEngine()

    def p(self, text, **params):
        return self.engine.estimate(text, params).probability

    def test_single_die_face(self):
        assert self.p("What is the probability of rolling a 6 on a fair die?") == pytest.approx(1 / 6)

    def test_sum_of_two_dice(self):
        assert self.p("Probability that the sum of two dice equals 7") == pytest.approx(6 / 36)

    @pytest.mark.parametrize("n,threshold", [(3, 12), (2, 10), (4, 20)])
    def test_sum_at_least_matches_brute_force(self, n, threshold):
        outcomes = list(itertools.product(range(1, 7), repeat=n))
        expected = sum(1 for o in outcomes if sum(o) >= threshold) / len(outcomes)
        text = f"Throw {n} dice, probability the total is at least {threshold}"
        assert self.p(text) == pytest.approx(expected)

    def test_dice_notation_and_sides(self):
        outcomes = list(itertools.product(range(1, 9), repeat=2))
        expected = sum(1 for o in outcomes if sum(o) == 9) / len(outcomes)
        assert self.p("2d8 sum equals 9") == pytest.approx(expected)
        assert self.p("Roll a twenty-sided die, probability of getting 20") == pytest.approx(1 / 20)

    def test_repeated_rolls_de_mere(self):
        # at least one six in four rolls
        assert self.p("six in four rolls", kind="dice", rolls=4) == pytest.approx(1 - (5 / 6) ** 4)

    def test_coin_exact_and_at_least(self):
        assert self.p("Probability of exactly 3 heads in 5 coin flips") == pytest.approx(10 / 32)
        assert self.p("at least one head in 3 tosses") == pytest.approx(7 / 8)
        assert self.p("Chance of getting heads when I flip a coin") == pytest.approx(0.5)

    def test_biased_coin_tails(self):
        # p(heads) = 0.7, so p(tails) = 0.3
        assert self.p("probability of tails when I flip a coin", p=0.7) == pytest.approx(0.3)

    def test_cards(self):
        assert self.p("What are the odds of drawing an ace from a standard deck of cards?") == pytest.approx(4 / 52)
        assert self.p("Probability of drawing a heart from a deck") == pytest.approx(13 / 52)
        assert self.p("draw 2 cards, probability of at least one ace") == pytest.approx(1 - (48 * 47) / (52 * 51))

    def test_missing_target_gives_helpful_error(self):
        with pytest.raises(MissingParametersError):
            self.p("what happens when I roll a die")

    def test_unknown_topic_is_rejected(self):
        with pytest.raises(ValueError):
            self.p("something unrelated")

    def test_size_limits(self):
        with pytest.raises(ValueError):
            self.p("sum equals 5", kind="dice", n_dice=500, sides=6, target=5)


# ------------------------------------------------------------------- counting
class TestCounting:
    engine = CountingEngine()

    def test_poisson_exact(self):
        lam, k = 12, 15
        expected = math.exp(-lam) * lam**k / math.factorial(k)
        got = self.engine.estimate("exactly 15 calls", {"rate": 12}).probability
        assert got == pytest.approx(expected)

    def test_poisson_duration_scales_mean(self):
        # 6 per hour for 10 minutes -> mean 1
        got = self.engine.estimate("exactly 2 arrive", {"rate": 6, "duration": 1 / 6}).probability
        assert got == pytest.approx(math.exp(-1) / 2)

    def test_binomial_at_least(self):
        n, p = 50, 0.02
        expected = 1 - sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(3))
        got = self.engine.estimate("at least 3 defective", {"n": n, "p": p}).probability
        assert got == pytest.approx(expected)

    def test_comparison_override(self):
        less = self.engine.estimate("", {"n": 10, "p": 0.5, "k": 5, "comparison": "lt"}).probability
        assert less == pytest.approx(sum(math.comb(10, i) for i in range(5)) / 1024)

    def test_missing_parameters(self):
        with pytest.raises(MissingParametersError) as info:
            self.engine.estimate("how many calls", {})
        assert "k" in info.value.missing[0]


# --------------------------------------------------------------------- signal
class TestSignal:
    engine = SignalEngine()

    def test_bpsk_ber(self):
        snr_db = 8.0
        g = 10 ** (snr_db / 10)
        got = self.engine.estimate("bit error", {"snr_db": snr_db}).probability
        assert got == pytest.approx(q_func(math.sqrt(2 * g)), rel=1e-9)

    def test_ber_decreases_with_snr(self):
        low = self.engine.estimate("bit error", {"snr_db": 2}).probability
        high = self.engine.estimate("bit error", {"snr_db": 12}).probability
        assert high < low

    def test_noncoherent_fsk(self):
        g = 10 ** 0.6
        got = self.engine.estimate("", {"snr_db": 6, "modulation": "bfsk_noncoherent"}).probability
        assert got == pytest.approx(0.5 * math.exp(-g / 2))

    def test_packet_success(self):
        g = 10 ** 0.7
        ber = q_func(math.sqrt(2 * g))
        got = self.engine.estimate("packet", {"snr_db": 7, "packet_bits": 1000}).probability
        assert got == pytest.approx((1 - ber) ** 1000, rel=1e-9)

    def test_detection_beats_false_alarm_and_grows_with_snr(self):
        weak = self.engine.estimate("detect a radar target", {"snr_db": 3, "pfa": 1e-3}).probability
        strong = self.engine.estimate("detect a radar target", {"snr_db": 12, "pfa": 1e-3}).probability
        assert 1e-3 < weak < strong <= 1.0

    def test_detection_formula(self):
        g = 10 ** 0.5
        pfa = 0.01
        thr = 2.3263478740408408  # Q^-1(0.01)
        got = self.engine.estimate("detect", {"snr_db": 5, "pfa": pfa}).probability
        assert got == pytest.approx(q_func(thr - math.sqrt(2 * g)), rel=1e-6)

    def test_requires_snr(self):
        with pytest.raises(MissingParametersError):
            self.engine.estimate("bit error", {})

    def test_unknown_modulation(self):
        with pytest.raises(ValueError):
            self.engine.estimate("", {"snr_db": 5, "modulation": "magic"})


# ---------------------------------------------------------------------- bayes
class TestBayes:
    engine = BayesEngine()

    def test_classic_medical_test(self):
        got = self.engine.estimate(
            "positive test", {"prior": 0.01, "sensitivity": 0.99, "specificity": 0.95}
        ).probability
        assert got == pytest.approx(0.0099 / (0.0099 + 0.0495))

    def test_negative_result_lowers_probability(self):
        pos = self.engine.estimate("", {"prior": 0.1, "sensitivity": 0.9, "specificity": 0.9}).probability
        neg = self.engine.estimate(
            "negative screening", {"prior": 0.1, "sensitivity": 0.9, "specificity": 0.9}
        ).probability
        assert neg < 0.1 < pos
        assert neg == pytest.approx(0.1 * 0.1 / (0.1 * 0.1 + 0.9 * 0.9))

    def test_likelihood_ratio(self):
        odds = 0.02 / 0.98 * 10
        got = self.engine.estimate("", {"prior": 0.02, "likelihood_ratio": 10}).probability
        assert got == pytest.approx(odds / (1 + odds))

    def test_false_positive_rate_alias(self):
        a = self.engine.estimate("", {"prior": 0.2, "sensitivity": 0.8, "false_positive_rate": 0.1}).probability
        b = self.engine.estimate("", {"prior": 0.2, "sensitivity": 0.8, "specificity": 0.9}).probability
        assert a == pytest.approx(b)

    def test_validation(self):
        with pytest.raises(ValueError):
            self.engine.estimate("", {"prior": 1.5, "sensitivity": 0.9, "specificity": 0.9})
        with pytest.raises(MissingParametersError):
            self.engine.estimate("", {"prior": 0.1})
        with pytest.raises(MissingParametersError):
            self.engine.estimate("", {})


# -------------------------------------------------------------------- finance
class TestFinance:
    engine = FinanceEngine()

    def test_constant_hazard(self):
        got = self.engine.estimate("default within 3 years", {"hazard_rate": 0.02, "years": 3}).probability
        assert got == pytest.approx(1 - math.exp(-0.06))

    def test_annual_pd(self):
        got = self.engine.estimate("", {"annual_pd": 0.05, "years": 2}).probability
        assert got == pytest.approx(1 - 0.95**2)

    def test_price_below_matches_lognormal_formula(self):
        spot, strike, vol, t, mu = 100.0, 90.0, 0.2, 1.0, 0.05
        z = (math.log(strike / spot) - (mu - 0.5 * vol**2) * t) / (vol * math.sqrt(t))
        params = dict(spot=spot, strike=strike, volatility=vol, years=t, drift=mu)
        got = self.engine.estimate("stock falls below 90", params).probability
        assert got == pytest.approx(norm_cdf(z), rel=1e-9)

    def test_above_is_complement_of_below(self):
        params = dict(spot=100, strike=110, volatility=0.3, years=2)
        below = self.engine.estimate("falls below", params).probability
        above = self.engine.estimate("price exceeds the level", params).probability
        assert below + above == pytest.approx(1.0)

    def test_text_infers_task(self):
        got = self.engine.estimate(
            "Chance that the Nasdaq drops below 15000", dict(spot=16000, strike=15000, volatility=0.25, years=1)
        )
        assert got.details["task"] == "price_threshold"

    def test_missing(self):
        with pytest.raises(MissingParametersError):
            self.engine.estimate("borrower defaults", {})
        with pytest.raises(MissingParametersError):
            self.engine.estimate("stock price below", {"spot": 100})


# ---------------------------------------------------------------- reliability
class TestReliability:
    engine = ReliabilityEngine()

    def test_exponential_failure_and_survival(self):
        fail = self.engine.estimate("fails before", {"mtbf": 10000, "time": 2000}).probability
        surv = self.engine.estimate("chance it survives", {"mtbf": 10000, "time": 2000}).probability
        assert fail == pytest.approx(1 - math.exp(-0.2))
        assert surv == pytest.approx(math.exp(-0.2))

    def test_weibull_shape_one_is_exponential(self):
        w = self.engine.estimate("", {"shape": 1, "scale": 10000, "time": 2000}).probability
        e = self.engine.estimate("", {"mtbf": 10000, "time": 2000}).probability
        assert w == pytest.approx(e)

    def test_weibull_wear_out(self):
        got = self.engine.estimate("", {"shape": 3, "scale": 1000, "time": 500}).probability
        assert got == pytest.approx(1 - math.exp(-((0.5) ** 3)))

    def test_missing(self):
        with pytest.raises(MissingParametersError):
            self.engine.estimate("", {"time": 10})


# ------------------------------------------------------------------ empirical
class TestEmpirical:
    engine = EmpiricalEngine()

    def test_laplace_smoothing(self):
        res = self.engine.estimate("", {"successes": 37, "trials": 400})
        assert res.probability == pytest.approx(38 / 402)
        low, high = res.details["credible_interval"]
        assert low < res.probability < high

    def test_outcomes_list(self):
        res = self.engine.estimate("", {"outcomes": [1, 0, 1, 1, 0, 0, 1, 1]})
        assert res.details["successes"] == 5 and res.details["trials"] == 8
        assert res.probability == pytest.approx(6 / 10)

    def test_no_data_returns_prior(self):
        assert self.engine.estimate("", {"successes": 0, "trials": 0}).probability == pytest.approx(0.5)

    def test_more_data_narrows_interval(self):
        small = self.engine.estimate("", {"successes": 5, "trials": 10}).details["credible_interval"]
        big = self.engine.estimate("", {"successes": 500, "trials": 1000}).details["credible_interval"]
        assert (big[1] - big[0]) < (small[1] - small[0])

    def test_validation(self):
        with pytest.raises(ValueError):
            self.engine.estimate("", {"successes": 5, "trials": 3})
        with pytest.raises(ValueError):
            self.engine.estimate("", {"outcomes": [0, 1, 2]})
        with pytest.raises(MissingParametersError):
            self.engine.estimate("", {})


@pytest.mark.parametrize(
    "question,expected",
    [
        ("I roll a 6-sided die 10 times. What is the probability that all 10 times it shows 6?", 6**-10),
        ("Probability of getting a six every time in 4 rolls of a die", 6**-4),
        ("Probability of rolling a six 3 times in a row", 6**-3),
        ("Probability of at least one six in 4 rolls of a die", 1 - (5 / 6) ** 4),
    ],
)
def test_repeated_rolls_all_vs_at_least_one(question, expected):
    from probai import ProbAI

    assert ProbAI().estimate(question).probability == pytest.approx(expected)


def test_urn_single_draw():
    from probai import ProbAI

    q = "There are 10 balls in a box: 3 are red, 6 are white, and 1 is blue. What is the probability of drawing a blue ball? You are given 1 attempt."
    est = ProbAI().estimate(q)
    assert est.engine == "classical"
    assert est.probability == pytest.approx(0.1)


def test_urn_multiple_draws():
    from math import comb

    from probai import ProbAI

    ai = ProbAI()
    assert ai.estimate("A box holds 5 red and 7 blue balls. Probability of drawing 2 red balls in a row?").probability == pytest.approx(20 / 132)
    assert ai.estimate("An urn has 4 red, 6 green marbles. You draw 3 marbles. Probability of at least one red?").probability == pytest.approx(1 - comb(6, 3) / comb(10, 3))
    assert ai.estimate("A bag contains 4 red and 6 green marbles. Draw 3 marbles with replacement; probability exactly 2 are red?").probability == pytest.approx(3 * 0.4**2 * 0.6)


@pytest.mark.parametrize("mode,expected", [("with", 0.25), ("without", 1 / 6)])
def test_urn_replacement_mode(mode, expected):
    question = f"A bag has 2 red and 2 blue balls. Draw 2 {mode} replacement. Probability that both are red?"
    assert ClassicalEngine().estimate(question, {}).probability == pytest.approx(expected)


@pytest.mark.parametrize("parameter", ["annual_pd", "hazard_rate"])
def test_rare_default(parameter):
    result = FinanceEngine().estimate("", {parameter: 1e-20, "years": 2})
    assert result.probability == pytest.approx(2e-20, rel=1e-12, abs=0)


def test_rare_price_exceedance():
    result = FinanceEngine().estimate("", {
        "spot": 1, "strike": math.exp(9), "volatility": 1, "years": 1,
        "drift": 0.5, "direction": "above",
    })
    assert result.probability == pytest.approx(q_func(9), rel=1e-12, abs=0)


def test_rare_failure():
    result = ReliabilityEngine().estimate("", {"mtbf": 1e20, "time": 1})
    assert result.probability == pytest.approx(1e-20, rel=1e-12, abs=0)


def test_large_weibull_hazard():
    result = ReliabilityEngine().estimate("", {"shape": 1000, "scale": 1, "time": 10})
    assert result.probability == 1


def test_rare_false_positive():
    result = BayesEngine().estimate("", {
        "prior": 1e-20, "sensitivity": 1, "false_positive_rate": 1e-20,
    })
    assert result.probability == pytest.approx(0.5)
