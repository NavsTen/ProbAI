# ProbAI

**Ask for the probability of an event in plain English. ProbAI picks the right probability model and computes the answer.**

```python
from probai import ProbAI

ai = ProbAI()

ai.estimate("What is the probability of rolling a 7 with two dice?")
# -> 0.1667  (classical engine: exact enumeration)

ai.estimate("Probability of bit error on a BPSK radio link at 8 dB")
# -> 1.9e-04 (signal engine: Q(sqrt(2 Eb/N0)) in Gaussian noise)

ai.estimate("Probability that a borrower defaults within 3 years with an annual default rate of 2%")
# -> 0.0588  (finance engine: cumulative default probability)
```

**You do not pass parameters separately.** The numbers written in the question ("12 calls per hour", "8 dB", "2% default rate", "10 years") are understood automatically, converted to the right units, and reported back so you can check them.

Every answer comes with the engine that was used, the method, and a plain-language explanation, so you can check *why* you got that number.

## What it is (and what it is not)

ProbAI is a **router plus a library of well-known probability models**:

1. A small **machine-learning model** (TF-IDF features + logistic regression) reads your question and decides which domain it belongs to: games of chance, counting events, radio signals, medical/diagnostic reasoning, banking and markets, equipment reliability, or "I have data".
2. The matching **engine** reads the numbers out of your question (rates, percentages, dB, durations, counts), converts units, and computes the probability with the right mathematics for that domain. You can still pass `params` to override or add anything.
3. Optionally, a **calibrator** learns from what actually happened and corrects engines that rely on real-world assumptions.

What it is **not**: it does not predict the future from nothing. A probability always depends on assumptions and data. If you ask "will my startup succeed?" with no data, ProbAI cannot make up a number: it tells you what is missing or that it cannot decide. Results are only as good as the inputs and the model assumptions stated in each answer. Nothing here is financial, medical or safety advice.

## Installation

```bash
git clone https://github.com/NavsTen/ProbAI.git
cd ProbAI
pip install -e .
```

Requires Python 3.10+ (numpy, scipy, scikit-learn, joblib).

## Usage

### Python

```python
from probai import ProbAI

ai = ProbAI()
est = ai.estimate("Probability of exactly 3 heads in 5 coin flips")

print(est.probability)   # 0.3125
print(est.engine)        # classical
print(est.explanation)   # human-readable justification
print(est.routing)       # top engine candidates with scores
```

Write the numbers into the question. ProbAI shows what it understood, and states every interpretation it had to make:

```python
est = ai.estimate("Probability of correctly receiving a signal, given that the signal-to-noise ratio is 0.8")
est.probability   # 0.897
est.extracted     # {'snr_db': -0.969}
est.notes         # ["SNR 0.8 has no unit, so it was read as a plain power ratio (not decibels) ..."]
```

If a number is missing from the question you get a clear error that tells you what to add; ProbAI never invents data:

```python
ai.estimate("Probability of bit error on a radio link")
# MissingParametersError: Engine 'signal' needs parameter(s): snr_db. ...
```

Parameters passed in `params` always override what was read from the text, and `ProbAI(auto_extract=False)` turns reading off.

Skip routing when you already know the model:

```python
ai.estimate("anything", params={"successes": 37, "trials": 400}, engine="empirical")
```

### Russian and follow-ups

Questions may be written in Russian; the answer comes back in Russian too:

```bash
probai estimate "Какова вероятность выпадения 6 на честном кубике?"
probai chat      # interactive: ask, then refine with "а при 12 дБ?" or "and at 10 dB?"
```

```python
from probai import Session
s = Session()
s.ask("Probability of bit error on a BPSK radio link at 8 dB")
s.ask("а при 12 дБ?")   # same model, new SNR
```

Russian support is rule-based (a vocabulary of probability terms is translated before parsing), so
unusual wording or domains outside the seven engines may fail. Add phrases to
`src/probai/language.py` and examples to the seed file to extend it.

### Command line

```bash
probai estimate "probability of rolling a 6 on a die"
probai estimate "bit error rate of a BPSK radio link at 8 dB"
probai estimate "A call center gets 12 calls per hour. Probability of exactly 15 calls in an hour?"
probai estimate "chance a drive with an MTBF of 100000 hours fails within 5 years" --json
probai estimate "bit error rate of a radio link" -p snr_db=8     # optional override
probai engines
```

## What you can write in a question

| Engine | Understood from the text | Example |
|---|---|---|
| `classical` | dice, sides, targets, "N times", biased coins, cards | *Roll a die 4 times, probability of at least one six* |
| `counting` | rates with units, windows ("in 10 minutes"), k ("exactly", "at least", "5 or more", "no"), n and p | *6 buses per hour; exactly 2 in 10 minutes* |
| `signal` | dB or plain SNR, modulation, packet size in bits or bytes, false-alarm probability | *1000-bit packet at 7 dB* |
| `bayes` | prevalence, sensitivity, specificity, false-positive rate, "N% accurate", likelihood ratio | *99% sensitivity, 95% specificity, prevalence 1%* |
| `finance` | default rate or hazard rate, horizon, current price, level, volatility, drift | *now at 16000, 25% volatility, below 15000 in 6 months* |
| `reliability` | MTBF, mean lifetime, failure rate, Weibull shape and scale, time (units converted to hours) | *mean lifetime of 10 years, fails within 5 years* |
| `empirical` | "k out of n", "k of the last n", "tested n and saw k", "k wins and m losses" | *37 conversions out of 400 visitors* |

Wording tips: say the unit ("per hour", "dB", "years"), write percentages with `%`, and use words such as *exactly*, *at least*, *more than*, *at most*.

## Engines

| Engine | Use it for | Typical `params` | Model |
|---|---|---|---|
| `classical` | dice, coins, playing cards | optional (parsed from the text): `n_dice`, `sides`, `target`, `comparison`, `rolls`, `successes`, `draws` | exact enumeration, binomial, hypergeometric |
| `counting` | arrivals at a rate, successes in n trials | `rate`, `duration`, `k` or `n`, `p`, `k` | Poisson, binomial |
| `signal` | radio and telecom links, radar detection | `snr_db` (Eb/N0), `modulation`, `packet_bits`, `pfa` | AWGN bit error rates, matched-filter detection |
| `bayes` | tests, alerts, evidence | `prior`, `sensitivity`, `specificity` (or `likelihood_ratio`) | Bayes' theorem |
| `finance` | default risk, price thresholds | `annual_pd` or `hazard_rate`, `years`; or `spot`, `strike`, `volatility`, `years` | constant hazard, geometric Brownian motion |
| `reliability` | lifetime of components | `time`, `mtbf` (or `shape` and `scale`) | exponential, Weibull |
| `empirical` | you have past data | `successes`, `trials` (or `outcomes`) | Beta-Binomial posterior with credible interval |

Comparisons are understood from the text ("at least", "exactly", "more than", "at most", ...) or can be set with `comparison="ge" | "gt" | "le" | "lt" | "eq"`.

## How it learns

**1. The router learns your domains.** The bundled seed data (`src/probai/data/router_seed.jsonl`) teaches the router which questions belong to which engine. Add your own labelled examples and retrain:

```bash
# my_examples.jsonl: one {"text": "...", "engine": "..."} per line
probai train --data my_examples.jsonl --out my_router.joblib
probai estimate "..." --model my_router.joblib
```

Training prints a cross-validated accuracy so you know whether the new examples help.

**2. The calibrator learns from outcomes.** Model-based engines (`signal`, `finance`, `reliability`) report the probability *under their assumptions*. Tell ProbAI what really happened and it learns a monotone correction (isotonic regression):

```python
est = ai.estimate("bit error", params={"snr_db": 4}, engine="signal")
ai.add_feedback(est, outcome=True)   # after >= 30 observations per engine the correction activates
```

Exact engines (`classical`, `counting`, `bayes`) are never recalibrated: their math is not an assumption about the world.

## Extending ProbAI

Add a new domain in three steps (details in [CONTRIBUTING.md](CONTRIBUTING.md)):

1. Subclass `probai.Engine` and implement `estimate(text, params)`.
2. Add 10+ labelled example questions to the router data.
3. Register it with `ai.register_engine(MyEngine())` and retrain the router.

## Project layout

```
src/probai/
  core.py         shared types: Engine, Estimate, errors
  router.py       the learned text classifier
  calibration.py  learning from observed outcomes
  extraction.py   reads numbers and units out of the question text
  model.py        ProbAI: routing + extraction + estimation + feedback
  cli.py          command-line interface
  engines/        one file per probability model
  data/           bundled router training examples
tests/            engines are verified against brute force and independent formulas
```

## Limitations

- The router is trained on a small seed set. It is good at the phrasing it has seen and degrades on unusual wording; when unsure it raises `LowConfidenceError` instead of guessing. Contribute examples to improve it.
- Reading numbers from the text is rule-based, in English only, and covers common phrasings. When a number is ambiguous it is either reported in `est.notes` or left out (and the engine asks for it); use `params` for anything unusual. Always check `est.extracted`.
- Engines make textbook assumptions (independence, Gaussian noise, lognormal prices, constant failure rate). Each answer states them.
- `joblib` files are pickles: only load router or calibrator files you created or trust.

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT, see [LICENSE](LICENSE).
