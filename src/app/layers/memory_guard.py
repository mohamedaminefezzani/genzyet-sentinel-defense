"""Layer 2b — Memory trust propagation guard.

Tracks the trust contamination of agent memory across turns.

Key insight: the kit already tags memory entries with the trust level of the
context in which they were written. This layer enforces two rules:

1. WRITE GUARD: if the current observation contains adversary-controlled content,
   any memory_write action is rewritten to include an [UNTRUSTED] tag so the
   evaluator/future defense calls can recognize it.

2. RECALL GUARD: if the candidate action's arguments can be traced back to a
   recalled memory entry that was written in an untrusted context, raise the
   risk score accordingly.

This closes the multi-turn memory poisoning loop end-to-end.
"""

from __future__ import annotations

import re

from app.models import CandidateAction, DefenseDecision, DefenseRequest

UNTRUSTED_LEVELS = {"untrusted_internal", "untrusted_external", "adversary_controlled"}
ADVERSARY_LEVELS = {"adversary_controlled"}

_UNTRUSTED_TAG = "[UNTRUSTED SOURCE]"


def _observation_trust(request: DefenseRequest) -> str | None:
    """Return the trust level of the current observation, or None if no observation."""
    obs = request.observation
    if obs is None:
        return None
    trust_map = {r.id: r.provenance.trust_level for r in request.provenance}
    levels = {trust_map.get(pid, "trusted_internal") for pid in obs.provenance_ids}
    if levels & ADVERSARY_LEVELS:
        return "adversary_controlled"
    if levels & UNTRUSTED_LEVELS:
        return "untrusted_external"
    return None


def _recalled_memory_items(request: DefenseRequest) -> list[tuple[str, str]]:
    """Return list of (content, trust_level) for memory items in conversation."""
    trust_map = {r.id: r.provenance.trust_level for r in request.provenance}
    memory_items = []
    for item in request.conversation:
        if item.role == "memory" or item.kind == "memory_recall":
            levels = [trust_map.get(pid, "trusted_internal") for pid in item.provenance_ids]
            worst = max(levels, key=lambda l: _trust_rank(l)) if levels else "trusted_internal"
            memory_items.append((item.content, worst))
    return memory_items


def _trust_rank(level: str) -> int:
    order = [
        "system_policy", "authenticated_user", "trusted_internal",
        "untrusted_internal", "untrusted_external", "adversary_controlled",
    ]
    try:
        return order.index(level)
    except ValueError:
        return 2


def _target_action(request: DefenseRequest) -> CandidateAction:
    cand = request.candidate_action
    if cand.type == "request_confirmation" and cand.confirmation_for:
        return cand.confirmation_for
    return cand


def check(request: DefenseRequest) -> tuple[float, float, list[str], DefenseDecision | None]:
    """
    Returns (risk_score, confidence, reason_codes, optional_rewrite_decision).
    If a rewrite decision is returned, it takes precedence over risk scoring.
    """
    candidate = request.candidate_action
    action = _target_action(request)

    # --- WRITE GUARD: rewrite memory_write if observation is tainted ---
    if action.type == "memory_write" and action.content:
        obs_trust = _observation_trust(request)
        if obs_trust in ADVERSARY_LEVELS or obs_trust in UNTRUSTED_LEVELS:
            # Tag the memory content as untrusted so future turns know
            original = action.content
            if _UNTRUSTED_TAG not in original:
                tagged_content = f"{_UNTRUSTED_TAG} {original}"

                # Build the rewritten action preserving the type
                from app.models import CandidateAction as CA
                rewritten = CA(type="memory_write", content=tagged_content)

                is_adversary = obs_trust in ADVERSARY_LEVELS
                risk = 0.75 if is_adversary else 0.45
                return (
                    risk,
                    0.80,
                    ["MEMORY_WRITE_TAINTED_CONTEXT"],
                    DefenseDecision(
                        decision="rewrite",
                        risk_score=risk,
                        confidence=0.80,
                        reason_codes=["MEMORY_WRITE_TAINTED_CONTEXT"],
                        explanation=(
                            f"Memory write occurred in {obs_trust} context; "
                            "content tagged as untrusted to prevent future poisoning."
                        ),
                        rewritten_action=rewritten,
                    ),
                )

    # --- RECALL GUARD: check if tool args match untrusted recalled memory ---
    if action.type != "tool_call" or not action.arguments:
        return 0.0, 0.5, [], None

    memory_items = _recalled_memory_items(request)
    if not memory_items:
        return 0.0, 0.5, [], None

    # Build corpus of untrusted memory content
    untrusted_memory_texts = [
        content.lower()
        for content, trust in memory_items
        if trust in UNTRUSTED_LEVELS
    ]
    adversary_memory_texts = [
        content.lower()
        for content, trust in memory_items
        if trust in ADVERSARY_LEVELS
    ]

    if not untrusted_memory_texts and not adversary_memory_texts:
        return 0.0, 0.6, [], None

    max_risk = 0.0
    reason_codes: list[str] = []

    for arg_name, arg_val in action.arguments.items():
        if arg_val is None:
            continue
        val_str = str(arg_val).lower()
        if len(val_str) < 4:
            continue

        in_adversary_mem = any(val_str in text for text in adversary_memory_texts)
        in_untrusted_mem = any(val_str in text for text in untrusted_memory_texts)

        if in_adversary_mem:
            max_risk = max(max_risk, 0.80)
            reason_codes.append("ARG_FROM_ADVERSARY_MEMORY")
        elif in_untrusted_mem:
            max_risk = max(max_risk, 0.55)
            reason_codes.append("ARG_FROM_UNTRUSTED_MEMORY")

    reason_codes = list(dict.fromkeys(reason_codes))
    return max_risk, 0.70, reason_codes, None
