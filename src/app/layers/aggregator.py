"""Layer 4 — Weighted risk aggregator.

Combines signals from all layers into a single calibrated risk score and
maps it to a final decision (allow / escalate / block).

Design decisions:
- Weights reflect empirical signal quality. Taint and trajectory are cheap and
  precise; goal coherence is slower but catches semantic attacks.
- Two thresholds: ESCALATE_THRESHOLD (human review) and BLOCK_THRESHOLD (hard stop).
  Tune these against the BTU vs ASR trade-off in your ablation study.
- All layer signals and weights are emitted in metadata for the observability layer.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# Decision thresholds — tune via environment variables for ablation experiments.
# Calibrated against the empirical score distribution from the published scenarios:
#   - A single adversary-only taint signal (risk=0.90, conf=0.85) produces ~0.64
#   - Two moderate signals (taint=0.55 + trajectory=0.40) produce ~0.42
BLOCK_THRESHOLD = float(os.getenv("SENTINEL_BLOCK_THRESHOLD", "0.60"))
ESCALATE_THRESHOLD = float(os.getenv("SENTINEL_ESCALATE_THRESHOLD", "0.38"))

# Layer weights — must sum to 1.0
_WEIGHTS = {
    "taint": 0.45,         # Data-flow taint: precise, catches indirect injection
    "trajectory": 0.35,    # Sequence anomaly: catches skipped steps and forbidden jumps
    "memory": 0.20,        # Memory trust propagation: catches multi-turn poisoning
    "coherence": 0.00,     # Goal coherence: catches semantic hijacking
}

assert abs(sum(_WEIGHTS.values()) - 1.0) < 1e-6, "weights must sum to 1.0"


@dataclass
class LayerSignal:
    name: str
    risk: float
    confidence: float
    reason_codes: list[str] = field(default_factory=list)


def aggregate(signals: list[LayerSignal]) -> tuple[float, float, list[str], dict]:
    """
    Combine layer signals into (final_risk, final_confidence, reason_codes, metadata).

    Aggregation strategy:
    - Weighted average of risk scores, weighted by both layer weight and signal confidence.
    - Max risk of any single layer can escalate the score (catches sharp single signals).
    - Reason codes from all layers are merged.
    """
    if not signals:
        return 0.0, 0.5, [], {}

    weighted_risk = 0.0
    total_weight = 0.0
    max_risk = 0.0
    all_reason_codes: list[str] = []
    layer_breakdown: dict[str, dict] = {}

    for sig in signals:
        w = _WEIGHTS.get(sig.name, 0.0)
        # Effective weight: layer weight × confidence (low-confidence signals count less)
        effective_w = w * sig.confidence
        weighted_risk += sig.risk * effective_w
        total_weight += effective_w
        max_risk = max(max_risk, sig.risk)
        all_reason_codes.extend(sig.reason_codes)
        layer_breakdown[sig.name] = {
            "risk": round(sig.risk, 3),
            "confidence": round(sig.confidence, 3),
            "weight": w,
            "reason_codes": sig.reason_codes,
        }

    base_risk = weighted_risk / total_weight if total_weight > 0 else 0.0

    # Blend: 55% weighted average + 45% max-risk boost.
    # A higher max-risk weight ensures a single very-high-confidence signal
    # (e.g. ARG_TAINTED_ADVERSARY_ONLY at 0.90) can reach the block threshold
    # even when other layers are silent (risk=0).
    final_risk = min(1.0, 0.55 * base_risk + 0.45 * max_risk)

    # Final confidence: weighted average of layer confidences
    final_confidence = (
        sum(_WEIGHTS.get(s.name, 0.0) * s.confidence for s in signals)
        / sum(_WEIGHTS.get(s.name, 0.0) for s in signals if _WEIGHTS.get(s.name, 0.0) > 0)
        if signals else 0.5
    )

    # Deduplicate reason codes while preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for code in all_reason_codes:
        if code not in seen:
            seen.add(code)
            deduped.append(code)

    metadata = {
        "layers": layer_breakdown,
        "base_risk": round(base_risk, 3),
        "max_signal_risk": round(max_risk, 3),
        "thresholds": {
            "block": BLOCK_THRESHOLD,
            "escalate": ESCALATE_THRESHOLD,
        },
    }

    return round(final_risk, 3), round(final_confidence, 3), deduped, metadata


def decide_from_risk(
    risk: float,
    confidence: float,
    reason_codes: list[str],
    metadata: dict,
    is_consequential: bool = False,
) -> tuple[str, str]:
    """
    Map a risk score to (decision, explanation).

    consequential actions have a lower effective escalation threshold.
    """
    effective_escalate = ESCALATE_THRESHOLD * (0.75 if is_consequential else 1.0)

    if risk >= BLOCK_THRESHOLD:
        explanation = (
            f"Risk score {risk:.2f} exceeds block threshold {BLOCK_THRESHOLD}. "
            f"Signals: {', '.join(reason_codes[:4]) or 'none'}."
        )
        return "block", explanation

    if risk >= effective_escalate:
        explanation = (
            f"Risk score {risk:.2f} exceeds escalation threshold "
            f"({'consequential ' if is_consequential else ''}{effective_escalate:.2f}). "
            f"Signals: {', '.join(reason_codes[:4]) or 'none'}."
        )
        return "escalate", explanation

    explanation = f"Risk score {risk:.2f} is below thresholds. Action appears goal-aligned."
    return "allow", explanation
