"""Tests for the learned router, the calibrator, the ProbAI facade and the CLI."""

import json

import pytest

from probai import (
    Calibrator,
    Engine,
    EngineResult,
    LowConfidenceError,
    MissingParametersError,
    ProbAI,
    ProbAIError,
    Router,
)
from probai.cli import main
from probai.router import load_examples

# Questions that are NOT in the seed data. The router must generalise to them.
HELD_OUT = [
    ("What is the chance of rolling a 3 on a die?", "classical"),
    ("Probability of getting exactly 2 heads in 4 coin tosses", "classical"),
    ("A shop averages 3 customers per minute. Probability that exactly 5 arrive in a minute?", "counting"),
    ("Probability of 2 or fewer defects among 30 products if each is defective with probability 0.05", "counting"),
    ("What is the bit error rate of a QPSK radio link at 6 dB?", "signal"),
    ("Likelihood of a radar detecting a target in noise", "signal"),
    ("The test is 90% accurate and 3% of people have the disease. Given a positive test, what is the probability of illness?", "bayes"),
    ("Probability the email is phishing given that it contains a suspicious link", "bayes"),
    ("Probability that the company defaults on its bonds within 2 years", "finance"),
    ("Chance that the Nasdaq drops below 15000 by year end", "finance"),
    ("How likely is the motor to fail in the next 2000 hours?", "reliability"),
    ("Probability that the fan keeps working for 5 years", "reliability"),
    ("We saw 25 signups from 300 visits, how likely is the next visitor to sign up?", "empirical"),
    ("Based on past matches, what is the chance our team wins tonight?", "empirical"),
]


@pytest.fixture(scope="module")
def ai():
    return ProbAI()


# --------------------------------------------------------------------- router
def test_seed_data_covers_every_engine():
    _, labels = load_examples()
    assert set(labels) == set(ProbAI().engines)
    assert min(labels.count(l) for l in set(labels)) >= 10


def test_router_generalises_to_unseen_questions():
    router = Router.from_seed()
    hits = sum(router.predict_proba(q)[0][0] == label for q, label in HELD_OUT)
    accuracy = hits / len(HELD_OUT)
    assert accuracy >= 0.85, f"router accuracy on held-out questions: {accuracy:.0%}"


def test_cross_validated_accuracy_is_reasonable():
    texts, labels = load_examples()
    assert Router.cross_validate(texts, labels) >= 0.75


def test_router_can_learn_a_new_label(tmp_path):
    texts, labels = load_examples()
    texts += ["Chance of a penalty kick being saved", "Probability the goalkeeper saves the shot"] * 3
    labels += ["sports"] * 6
    router = Router().fit(texts, labels)
    assert "sports" in router.classes
    assert router.predict_proba("What is the probability that the goalie saves the penalty?")[0][0] == "sports"


def test_router_save_and_load_roundtrip(tmp_path):
    router = Router.from_seed()
    path = tmp_path / "router.joblib"
    router.save(path)
    loaded = Router.load(path)
    q = "bit error rate at 8 dB"
    assert router.predict_proba(q) == loaded.predict_proba(q)


def test_router_rejects_bad_files_and_bad_training(tmp_path):
    bad = tmp_path / "bad.joblib"
    import joblib

    joblib.dump({"not": "a router"}, bad)
    with pytest.raises(ProbAIError):
        Router.load(bad)
    with pytest.raises(ProbAIError):
        Router().fit(["a", "b"], ["x", "x"])  # only one class
    with pytest.raises(ProbAIError):
        Router().predict_proba("untrained")


def test_load_examples_reports_bad_lines(tmp_path):
    f = tmp_path / "data.jsonl"
    f.write_text('{"text": "ok", "engine": "classical"}\nnot json\n', encoding="utf-8")
    with pytest.raises(ProbAIError, match="Line 2"):
        load_examples(f)


# ----------------------------------------------------------------- end to end
def test_dice_end_to_end(ai):
    est = ai.estimate("What is the probability of rolling a 6 on a fair die?")
    assert est.engine == "classical"
    assert est.probability == pytest.approx(1 / 6)
    assert not est.calibrated


def test_radio_question_uses_signal_engine(ai):
    est = ai.estimate("Probability of bit error for a BPSK radio link", params={"snr_db": 8})
    assert est.engine == "signal"
    assert 0 < est.probability < 1e-3


def test_bank_question_uses_finance_engine(ai):
    est = ai.estimate("Probability that a borrower defaults on a loan", params={"annual_pd": 0.02, "years": 3})
    assert est.engine == "finance"
    assert est.probability == pytest.approx(1 - 0.98**3)


def test_parameters_steer_ambiguous_text(ai):
    est = ai.estimate("How likely is it?", params={"mtbf": 1000, "time": 100})
    assert est.engine == "reliability"


def test_engine_override_skips_routing(ai):
    est = ai.estimate("anything at all", params={"successes": 3, "trials": 10}, engine="empirical")
    assert est.engine == "empirical" and est.routing_confidence == 1.0


def test_unknown_engine_and_gibberish(ai):
    with pytest.raises(ProbAIError):
        ai.estimate("x", engine="nope")
    with pytest.raises(LowConfidenceError):
        ai.estimate("blah blah xyz qwerty")


def test_missing_data_is_reported_clearly(ai):
    with pytest.raises(MissingParametersError) as info:
        ai.estimate("Probability of bit error for a BPSK radio link")
    assert "snr_db" in str(info.value)


def test_custom_engine_can_be_registered():
    class AlwaysHalf(Engine):
        name = "half"
        description = "demo"

        def estimate(self, text, params):
            return EngineResult(0.5, "constant", "always one half")

    system = ProbAI()
    system.register_engine(AlwaysHalf())
    assert system.estimate("anything", engine="half").probability == 0.5


# ----------------------------------------------------------------- calibrator
def test_calibrator_corrects_overconfident_engine():
    cal = Calibrator(min_samples=30)
    for i in range(20):
        cal.add_feedback("signal", 0.2, i < 10)  # engine says 20%, reality 50%
    for i in range(20):
        cal.add_feedback("signal", 0.8, i < 18)  # engine says 80%, reality 90%
    assert not cal.is_fitted("signal")
    assert cal.apply("signal", 0.2) == 0.2  # no change before fitting
    assert cal.fit() == ["signal"]
    assert cal.apply("signal", 0.2) == pytest.approx(0.5)
    assert cal.apply("signal", 0.8) == pytest.approx(0.9)


def test_calibration_is_applied_only_to_calibratable_engines():
    cal = Calibrator(min_samples=4)
    for engine in ("signal", "classical"):
        for outcome in (1, 1, 1, 1):
            cal.add_feedback(engine, 0.1, outcome)
    cal.fit()
    system = ProbAI(calibrator=cal)

    radio = system.estimate("bit error", params={"snr_db": 10}, engine="signal")
    assert radio.calibrated and radio.probability == pytest.approx(1.0)
    assert radio.raw_probability < 1e-3

    dice = system.estimate("rolling a 6 on a die", engine="classical")
    assert not dice.calibrated and dice.probability == pytest.approx(1 / 6)


def test_feedback_loop_and_persistence(tmp_path):
    system = ProbAI(calibrator=Calibrator(min_samples=5))
    est = system.estimate("bit error", params={"snr_db": 4}, engine="signal")
    for _ in range(5):
        system.add_feedback(est, outcome=True)
    again = system.estimate("bit error", params={"snr_db": 4}, engine="signal")
    assert again.calibrated and again.probability == pytest.approx(1.0)

    path = tmp_path / "cal.joblib"
    system.calibrator.save(path)
    restored = Calibrator.load(path)
    assert restored.apply("signal", est.raw_probability) == pytest.approx(1.0)


# ------------------------------------------------------------------------ CLI
def test_cli_estimate_text(capsys):
    assert main(["estimate", "probability of rolling a 6 on a die"]) == 0
    out = capsys.readouterr().out
    assert "0.166667" in out and "classical" in out


def test_cli_estimate_json_with_params(capsys):
    code = main(["estimate", "bit error rate of a BPSK radio link", "-p", "snr_db=8", "--json"])
    data = json.loads(capsys.readouterr().out)
    assert code == 0 and data["engine"] == "signal"


def test_cli_reports_errors(capsys):
    assert main(["estimate", "bit error rate of a BPSK radio link"]) == 2
    assert "snr_db" in capsys.readouterr().err
    assert main(["estimate", "x", "-p", "oops"]) == 2


def test_cli_train_and_use_model(tmp_path, capsys):
    out = tmp_path / "router.joblib"
    assert main(["train", "--out", str(out)]) == 0
    assert out.exists()
    code = main(["estimate", "probability of rolling a 6 on a die", "-m", str(out)])
    assert code == 0


def test_cli_engines_lists_everything(capsys):
    assert main(["engines"]) == 0
    out = capsys.readouterr().out
    for name in ("classical", "counting", "signal", "bayes", "finance", "reliability", "empirical"):
        assert name in out
