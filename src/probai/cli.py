"""Command-line interface: ``probai estimate``, ``probai chat``, ``probai train``, ``probai engines``."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional

from . import __version__, ru_output
from .core import Estimate, ProbAIError
from .engines import default_engines
from .model import ProbAI
from .router import Router, load_examples
from .session import Session


def _parse_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _parse_params(items: List[str]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ProbAIError(f"Invalid --param {item!r}; expected key=value.")
        key, value = item.split("=", 1)
        params[key.strip()] = _parse_value(value.strip())
    return params


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="probai", description="Estimate the probability of an event.")
    parser.add_argument("--version", action="version", version=f"probai {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    est = sub.add_parser("estimate", help="estimate a probability from a question (English or Russian)")
    est.add_argument("question", help="the event, in plain English or Russian")
    est.add_argument("-p", "--param", action="append", default=[], metavar="KEY=VALUE",
                     help="optional: override a value read from the question; repeatable (values may be JSON)")
    est.add_argument("-e", "--engine", help="skip routing and use this engine")
    est.add_argument("-m", "--model", help="path to a trained router (from `probai train`)")
    est.add_argument("--json", action="store_true", help="print the full result as JSON")

    chat = sub.add_parser("chat", help="interactive session: ask, then refine with follow-ups ('and at 10 dB?')")
    chat.add_argument("-m", "--model", help="path to a trained router (from `probai train`)")

    train = sub.add_parser("train", help="train the engine router on your own labelled examples")
    train.add_argument("--data", help="JSONL file with {\"text\", \"engine\"} per line (default: bundled seed)")
    train.add_argument("--out", required=True, help="where to save the trained router")

    sub.add_parser("engines", help="list the available engines")
    return parser


def _print_estimate(est: Estimate) -> None:
    if est.language == "ru":
        print(f"Вероятность: {ru_output.num(est.probability)}  ({ru_output.pct(est.probability)})")
        print(f"Модель:      {est.engine}  (уверенность выбора {est.routing_confidence:.0%})")
        print(f"Метод:       {est.method}")
        print(f"Пояснение:   {est.explanation}")
        if est.extracted:
            shown = ", ".join(f"{k}={ru_output.num(v)}" for k, v in est.extracted.items())
            print(f"Понято из вопроса: {shown}")
        for note in est.notes:
            print(f"Примечание:  {note}")
        return
    print(f"P = {est.probability:.6g}  ({est.probability:.2%})")
    print(f"Engine: {est.engine}  (routing confidence {est.routing_confidence:.0%})")
    print(f"Method: {est.method}")
    print(f"Why:    {est.explanation}")
    if est.extracted:
        shown = ", ".join(f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v}" for k, v in est.extracted.items())
        print(f"Understood from your question: {shown}")
    for note in est.notes:
        print(f"Note:   {note}")


def _error_text(exc: Exception) -> str:
    ru = getattr(exc, "message_ru", None)
    return f"ошибка: {ru}" if ru else f"error: {exc}"


def _cmd_estimate(args: argparse.Namespace) -> int:
    router = Router.load(args.model) if args.model else None
    system = ProbAI(router=router)
    est = system.estimate(args.question, params=_parse_params(args.param), engine=args.engine)
    if args.json:
        print(json.dumps(est.to_dict(), indent=2, default=str, ensure_ascii=False))
    else:
        _print_estimate(est)
    return 0


def _cmd_chat(args: argparse.Namespace) -> int:
    router = Router.load(args.model) if args.model else None
    session = Session(ProbAI(router=router))
    print("ProbAI chat. Ask a question (English or Russian), then refine it: 'and at 10 dB?' / 'а если 5 лет?'.")
    print("Empty line, 'exit' or 'выход' quits.")
    while True:
        try:
            line = input("> ")
        except EOFError:
            break
        if not line.strip() or line.strip().lower() in {"exit", "quit", "выход"}:
            break
        try:
            _print_estimate(session.ask(line))
        except (ProbAIError, ValueError) as exc:
            print(_error_text(exc), file=sys.stderr)
        print()
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    texts, labels = load_examples(args.data)
    try:
        accuracy = Router.cross_validate(texts, labels)
        print(f"Cross-validated accuracy: {accuracy:.1%} on {len(texts)} examples")
    except ProbAIError as exc:
        print(f"Skipping cross-validation: {exc}")
    router = Router().fit(texts, labels)
    router.save(args.out)
    print(f"Router saved to {args.out}")
    return 0


def _cmd_engines(_: argparse.Namespace) -> int:
    for name, engine in default_engines().items():
        print(f"{name:12s} {engine.description}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    handlers = {"estimate": _cmd_estimate, "chat": _cmd_chat, "train": _cmd_train, "engines": _cmd_engines}
    try:
        return handlers[args.command](args)
    except (ProbAIError, ValueError, OSError) as exc:
        print(_error_text(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
