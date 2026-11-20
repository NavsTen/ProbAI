"""Context handling: follow-ups in a Session, chat loop, and regressions."""

import pytest

from probai import ProbAI, Session
from probai.cli import main

AI = ProbAI()


def test_follow_up_changes_only_snr():
    s = Session(AI)
    first = s.ask("Probability of bit error on a BPSK radio link at 8 dB")
    second = s.ask("and at 10 dB?")
    assert second.engine == "signal"
    assert second.probability < first.probability
    assert any("Follow-up" in n for n in second.notes)


def test_follow_up_in_russian():
    s = Session(AI)
    s.ask("Probability of bit error on a BPSK radio link at 8 dB")
    est = s.ask("а при 12 дБ?")
    assert est.engine == "signal"
    assert est.language == "ru"
    assert any("Уточнение" in n for n in est.notes)


def test_new_question_is_not_a_follow_up():
    s = Session(AI)
    s.ask("Probability of bit error on a BPSK radio link at 8 dB")
    est = s.ask("What is the probability of rolling a 6 on a fair die?")
    assert est.engine == "classical"
    assert est.probability == pytest.approx(1 / 6)


def test_session_reset_and_history():
    s = Session(AI)
    s.ask("What is the probability of rolling a 6 on a fair die?")
    assert len(s.history) == 1 and s.last is not None
    s.reset()
    assert s.last is None


def test_deck_of_52_cards_regression():
    est = AI.estimate("Probability of drawing an ace from a deck of 52 cards")
    assert est.probability == pytest.approx(4 / 52)


def test_chat_command(monkeypatch, capsys):
    lines = iter(["What is the probability of rolling a 6 on a fair die?", "выход"])
    monkeypatch.setattr("builtins.input", lambda _="": next(lines))
    assert main(["chat"]) == 0
    assert "0.166667" in capsys.readouterr().out


def test_chat_survives_errors(monkeypatch, capsys):
    lines = iter(["Bank loan default likelihood with a hazard rate of 2% per year", "exit"])
    monkeypatch.setattr("builtins.input", lambda _="": next(lines))
    assert main(["chat"]) == 0
    assert "years" in capsys.readouterr().err
