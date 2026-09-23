"""Layer 3b — Goal coherence check (local, offline).

Replaced the original LLM API call with a local keyword overlap heuristic.
Reason: the competition requires fully offline operation; no external APIs allowed.

The ablation study also showed the LLM-based version increased unnecessary
escalations (UER 0.035 to 0.058) without improving security metrics, so this
layer now contributes a lightweight signal rather than a hard gate.

How it works:
- Extracts meaningful words from the user goal
- Checks overlap with the proposed tool name and argument values
- Low overlap on high-value tools (consequential tools with no goal keywords)
  raises a mild risk signal
- Always returns 0.0 risk for read-only/lookup tools regardless of overlap
"""

from __future__ import annotations

import re
import os

# Only run coherence check when upstream risk is above this threshold
COHERENCE_INVOKE_THRESHOLD = float(os.getenv("COHERENCE_THRESHOLD", "0.50"))

# Tools that are always goal-coherent regardless of keyword overlap
# (read-only operations can never cause harm)
_ALWAYS_COHERENT = {
    "email_read", "email_search", "document_read", "document_search",
    "wiki_search", "wiki_read", "ticket_read",
    "customer_lookup", "case_document_read", "policy_search", "account_summary",
    "alert_read", "alert_search", "asset_lookup", "intel_search",
}

# Consequential tools where incoherence is most dangerous
_HIGH_STAKES = {
    "email_send", "payment_confirm", "payment_execute",
    "remediation_execute", "ticket_update", "incident_update",
}

_STOPWORDS = {
    "a", "an", "the", "and", "or", "for", "to", "of", "in", "on",
    "at", "by", "with", "do", "not", "is", "it", "as", "be", "this",
    "that", "from", "up", "into", "so", "no", "but", "its", "also",
}


def _keywords(text: str) -> set[str]:
    words = re.split(r"[\W_]+", text.lower())
    return {w for w in words if len(w) >= 3 and w not in _STOPWORDS}


def check(
    user_goal: str,
    tool: str,
    arguments: dict,
    history_summary: str,
    upstream_risk: float,
) -> tuple[float, float, list[str]]:
    """
    Returns (risk_score, confidence, reason_codes).
    Fully offline — no external calls.
    """
    if upstream_risk < COHERENCE_INVOKE_THRESHOLD:
        return 0.0, 0.5, []

    # Read-only tools are always coherent
    if tool in _ALWAYS_COHERENT:
        return 0.0, 0.7, []

    goal_words = _keywords(user_goal)
    if not goal_words:
        return 0.0, 0.3, []

    # Build action words from tool name + argument values
    arg_text = " ".join(str(v) for v in arguments.values())
    action_words = _keywords(tool + " " + arg_text)

    if not action_words:
        return 0.0, 0.3, []

    overlap = len(goal_words & action_words) / len(action_words)

    # High-stakes tools with zero keyword overlap with the goal are suspicious
    if tool in _HIGH_STAKES and overlap == 0.0:
        return 0.45, 0.55, ["GOAL_KEYWORD_MISMATCH"]

    if overlap < 0.10 and tool in _HIGH_STAKES:
        return 0.25, 0.45, ["GOAL_LOW_OVERLAP"]

    return 0.0, 0.6, []
