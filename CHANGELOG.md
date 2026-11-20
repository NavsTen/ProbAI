# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.3.1] - 2026-11-17

### Added
- Urn problems in the classical engine: coloured balls/marbles in a box, with or without replacement
  ("3 red, 6 white and 1 blue; probability of drawing a blue ball"), in English and Russian.
- ~60 training examples for urns and "all N times" questions.

### Fixed
- "all 10 times", "every time", "in a row" for repeated rolls were read as "at least one"
  (P=0.838 instead of 1.65e-08 for ten sixes in ten rolls).
- A rate quoted per month/quarter is converted to a yearly one instead of being read as yearly.
- "Loss of job/income" is read as the default event in Russian questions.
- Text with no known word now gets no routing opinion (low confidence) instead of the majority class.
- Router uses balanced class weights.

## [0.3.0] - 2026-11-17

### Added
- Russian language support: questions are normalised to English terms (`probai.language`), so every
  engine works unchanged; answers, notes and errors are returned in Russian (`probai.ru_output`).
- ~700 Russian training examples in `router_seed.jsonl` (now 2034 lines).
- `Session` and `probai chat`: follow-ups such as "and at 10 dB?" / "а при 12 дБ?" reuse the previous
  model and parameters.
- Context: runner-up engine fallback when the top engine lacks data, and an ambiguity note.
- Optional `evidence_boost` (default 0: it did not help in measurements).

### Fixed
- "deck of 52 cards" was parsed as drawing 52 cards (P=1.0); "two hearts" no longer reads as two of hearts.
- "once" / "twice" are understood as counts.

## [0.2.0] - 2026-11-16

### Added
- Natural-language input: numbers in the question (rates, percentages, dB, durations, counts,
  units) are read automatically by `probai.extraction`; no separate `params` needed.
  Explicit `params` still override. `ProbAI(auto_extract=False)` disables it.
- `Estimate.extracted` and `Estimate.notes` report what was understood and every interpretation
  made (e.g. a unitless SNR read as a plain ratio). The CLI prints them.
- `Engine.extract()` hook so custom engines can read their own parameters from text.
- `signal`: "correctly receiving" / "without errors" now returns 1 - BER.

### Fixed
- `classical`: "two hearts" was read as the two of hearts; it now means two cards of that suit.

## [0.1.0] - 2026-11-16

### Added
- `ProbAI` facade: routes a question to an engine, computes the probability, explains the method.
- Learned `Router` (TF-IDF word and character n-grams + logistic regression) with bundled seed data,
  cross-validation, save/load, and a `probai train` command.
- Seven engines: `classical`, `counting`, `signal`, `bayes`, `finance`, `reliability`, `empirical`.
- `Calibrator` (per-engine isotonic regression) that learns from observed outcomes.
- Command-line interface: `probai estimate`, `probai train`, `probai engines`.
- Test suite that verifies engines against brute force and independent formulas.
