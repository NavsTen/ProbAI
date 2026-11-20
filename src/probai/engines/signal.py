"""Radio and communication signals: bit errors, packet success, detection in noise.

All formulas assume an additive white Gaussian noise (AWGN) channel and an ideal
receiver. Real channels (fading, interference) are worse; see ``calibratable``.
"""

from __future__ import annotations

import math
import re
from typing import Any, Callable, Dict

from scipy.stats import norm

from ..core import Engine, EngineResult, Extraction
from ..extraction import extract_signal
from ..utils import as_float, as_int, as_probability, first_not_none, require


def _q(x: float) -> float:
    """Gaussian tail probability Q(x)."""
    return float(norm.sf(x))


# Bit error probability as a function of linear Eb/N0 (gamma).
_BER: Dict[str, Callable[[float], float]] = {
    "bpsk": lambda g: _q(math.sqrt(2.0 * g)),
    "qpsk": lambda g: _q(math.sqrt(2.0 * g)),  # per bit, Gray coding
    "bfsk_coherent": lambda g: _q(math.sqrt(g)),
    "bfsk_noncoherent": lambda g: 0.5 * math.exp(-g / 2.0),
    "dpsk": lambda g: 0.5 * math.exp(-g),
}

_MODULATION_ALIASES = {
    "fsk": "bfsk_coherent",
    "bfsk": "bfsk_coherent",
    "2psk": "bpsk",
    "4psk": "qpsk",
}


class SignalEngine(Engine):
    name = "signal"
    description = (
        "Radio / telecom signals in noise (AWGN): bit error probability, packet success, "
        "and detection probability of a known signal."
    )
    param_keys = frozenset({"snr_db", "modulation", "packet_bits", "pfa"})
    calibratable = True

    def extract(self, text: str) -> Extraction:
        return extract_signal(text)

    def estimate(self, text: str, params: Dict[str, Any]) -> EngineResult:
        require(params, ["snr_db"], self.name, "snr_db is Eb/N0 in decibels, e.g. params={'snr_db': 8}.")
        snr_db = as_float(params["snr_db"], "snr_db")
        if not -60.0 <= snr_db <= 100.0:
            raise ValueError("'snr_db' must be between -60 and 100 dB.")
        gamma = 10.0 ** (snr_db / 10.0)

        task = params.get("task") or self._infer_task(text, params)
        if task == "detection":
            return self._detection(params, snr_db, gamma)

        modulation = str(params.get("modulation", "bpsk")).lower()
        modulation = _MODULATION_ALIASES.get(modulation, modulation)
        if modulation not in _BER:
            raise ValueError(f"Unknown modulation {modulation!r}. Supported: {', '.join(sorted(_BER))}.")
        ber = _BER[modulation](gamma)

        if task == "bit_error":
            outcome = params.get("outcome") or (
                "correct"
                if re.search(r"\b(correct\w*|success\w*|without (?:any )?errors?|error[- ]free|no errors?)\b", text.lower())
                else "error"
            )
            if outcome == "correct":
                return EngineResult(
                    probability=1.0 - ber,
                    method=f"1 - BER of {modulation.upper()} in AWGN",
                    explanation=(
                        f"Eb/N0 = {snr_db:g} dB ({gamma:.4g} linear). Probability that a received bit "
                        f"is correct with {modulation.upper()} over an ideal AWGN channel "
                        f"(bit error probability {ber:.3e}). For a whole packet, give its size in bits."
                    ),
                    details={"task": task, "outcome": outcome, "modulation": modulation, "snr_db": snr_db, "ber": ber},
                )
            return EngineResult(
                probability=ber,
                method=f"Theoretical BER of {modulation.upper()} in AWGN",
                explanation=(
                    f"Eb/N0 = {snr_db:g} dB ({gamma:.4g} linear). Probability that a received bit "
                    f"is wrong with {modulation.upper()} over an ideal AWGN channel."
                ),
                details={"task": task, "modulation": modulation, "snr_db": snr_db, "ber": ber},
            )
        if task == "packet_success":
            require(params, ["packet_bits"], self.name, "Number of bits in the packet, e.g. 1000.")
            bits = as_int(params["packet_bits"], "packet_bits", 1)
            success = math.exp(bits * math.log1p(-ber)) if ber < 1.0 else 0.0
            return EngineResult(
                probability=success,
                method=f"(1 - BER)^bits with {modulation.upper()} in AWGN, no error correction",
                explanation=(
                    f"BER = {ber:.3e} at Eb/N0 = {snr_db:g} dB; a {bits}-bit packet arrives "
                    f"error-free only if all bits are correct (independent bit errors, no coding)."
                ),
                details={"task": task, "modulation": modulation, "snr_db": snr_db, "ber": ber, "bits": bits},
            )
        raise ValueError(f"Unknown task {task!r}; use 'bit_error', 'packet_success' or 'detection'.")

    @staticmethod
    def _infer_task(text: str, params: Dict[str, Any]) -> str:
        t = text.lower()
        if params.get("pfa") is not None or re.search(r"\b(detect\w*|radar|sonar|false alarm)\b", t):
            return "detection"
        if params.get("packet_bits") is not None or re.search(r"\b(packet|frame)s?\b", t):
            return "packet_success"
        return "bit_error"

    def _detection(self, params: Dict[str, Any], snr_db: float, gamma: float) -> EngineResult:
        pfa = as_probability(first_not_none(params.get("pfa"), 1e-3), "pfa")
        if not 0.0 < pfa < 1.0:
            raise ValueError("'pfa' (false-alarm probability) must be strictly between 0 and 1.")
        threshold = float(norm.isf(pfa))
        pd = _q(threshold - math.sqrt(2.0 * gamma))
        return EngineResult(
            probability=pd,
            method="Neyman-Pearson detection of a known signal in AWGN (matched filter)",
            explanation=(
                f"Energy-to-noise ratio {snr_db:g} dB, false-alarm probability fixed at {pfa:g}. "
                f"Probability that the receiver correctly detects the signal when it is present."
            ),
            details={"task": "detection", "snr_db": snr_db, "pfa": pfa, "threshold": threshold},
        )
