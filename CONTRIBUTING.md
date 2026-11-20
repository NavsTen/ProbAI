# Contributing to ProbAI

Thanks for helping! ProbAI grows by adding **domains** (engines) and **training examples**. Both are easy to contribute.

## Development setup

```bash
git clone https://github.com/YOUR_USERNAME/probai.git
cd probai
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

All tests must pass before you open a pull request.

## Ways to contribute

### 1. Add training examples (easiest)

Open `src/probai/data/router_seed.jsonl` and add lines like:

```json
{"text": "Probability that a package arrives late", "engine": "empirical"}
```

Guidelines:

- Write realistic, varied phrasing, the way real people ask. Do not paste near-duplicates.
- Label with the engine whose mathematics is correct for the question, not just the topic.
- Run `pytest tests/test_router_and_model.py` and check that accuracy does not drop.

### 2. Add a new engine

An engine is a class with a name and an `estimate` method:

```python
# src/probai/engines/queueing.py
from ..core import Engine, EngineResult, Extraction
from ..utils import require, as_positive

class QueueingEngine(Engine):
    name = "queueing"
    description = "Waiting times in queues (M/M/1)."
    param_keys = frozenset({"arrival_rate", "service_rate"})   # distinctive input names
    calibratable = True   # True only if the result rests on real-world assumptions

    def extract(self, text):
        # Optional but recommended: read the numbers out of the question.
        # Return only what the text clearly states; add notes for any interpretation.
        return Extraction({"arrival_rate": 2.0}, ["..."])

    def estimate(self, text, params):
        require(params, ["arrival_rate", "service_rate"], self.name)
        ...
        return EngineResult(probability=p, method="M/M/1 queue", explanation="...", details={})
```

Then:

1. Register it in `src/probai/engines/__init__.py` (`default_engines`).
2. Add **at least 10 diverse labelled questions** for it to `router_seed.jsonl`.
3. Add tests in `tests/test_engines.py`.

Rules for engines:

- **Verify against an independent source**: brute force, a closed-form formula implemented separately in the test, or a published reference value. Never test a formula against itself.
- **Never invent numbers** in `extract`: if the text does not state a value, leave it out so the engine can ask for it. Test extraction with plain-English questions and no `params`.
- **Fail loudly and helpfully**: if input data is missing, raise `MissingParametersError` naming the parameters and showing an example. Validate ranges (probabilities in [0, 1], positive rates, ...).
- **State assumptions** in `EngineResult.explanation` (independence, Gaussian noise, constant hazard, ...).
- Keep `calibratable = False` for exact mathematics.
- Guard against huge inputs that could hang or exhaust memory.

### Russian vocabulary

Missing a Russian word or phrase? Add an ordered `(regex, replacement)` rule to `_RULES_SRC` in
`src/probai/language.py`, plus a case in `tests/test_russian.py` checked against an independent formula.

### 3. Improve text parsing, docs, or report bugs

Use the issue templates. For a wrong answer, include the exact question, the `params`, the result and what you expected (with a source if possible).

## Pull request checklist

- [ ] Tests added or updated, `pytest` passes
- [ ] New behaviour documented in `README.md` and `CHANGELOG.md`
- [ ] Code and comments in English
- [ ] No trained model files (`*.joblib`) committed

## Code style

Plain, readable Python 3.10+. Type hints on public functions. Prefer small functions and explicit error messages over cleverness.

## Conduct

Be respectful. See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
