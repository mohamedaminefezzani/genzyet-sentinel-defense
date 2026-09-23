"""Layer 1 — Policy gate.

Fast, hard-rule checks that run before any analysis:
- Tool not in allowed_tools → block immediately
- Forbidden tool (out-of-policy side-effect tools) → block
- Consequential tool without recorded confirmation → escalate

Returns (decision, risk, confidence, reason_codes) or None to pass through.
"""

from __future__ import annotations

import hashlib
import json

from app.models import CandidateAction, DefenseDecision, DefenseRequest

# Tools whose *status argument* makes them consequential even if not in policy's consequential_tools
_STATUS_CONSEQUENTIAL: dict[str, set[str]] = {
    "ticket_update": {"closed", "resolved"},
    "incident_update": {"closed"},
}


def _action_digest(action: CandidateAction) -> str:
    """Canonical digest matching sentinel.core.actions.CandidateAction.digest()."""

    def canonical(v: object) -> object:
        if isinstance(v, str):
            return " ".join(v.split())
        if isinstance(v, float) and v.is_integer():
            return int(v)
        return v

    payload: dict[str, object] = {
        "type": action.type,
        "tool": action.tool,
        "arguments": {k: canonical(v) for k, v in sorted(action.arguments.items())},
        "content": action.content if action.type != "tool_call" else None,
    }
    if action.confirmation_for is not None:
        payload["confirmation_for"] = _action_digest(action.confirmation_for)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:24]


def check(request: DefenseRequest) -> DefenseDecision | None:
    """Return a hard decision or None to pass through to the next layer."""
    candidate = request.candidate_action

    # Unwrap confirmation wrappers
    action = candidate.confirmation_for if candidate.type == "request_confirmation" else candidate

    # Non-tool actions (respond, memory_write) pass through here;
    # memory_write is handled separately in the taint layer.
    if action is None or action.type != "tool_call" or action.tool is None:
        return None

    tool = action.tool
    allowed: list[str] = request.policy_context.get("allowed_tools", [])

    # Hard block: tool outside the task's allowed set
    if tool not in allowed:
        return DefenseDecision(
            decision="block",
            risk_score=0.95,
            confidence=0.95,
            reason_codes=["TOOL_NOT_IN_ALLOWED_SET"],
            explanation=f"Tool '{tool}' is not in the task's allowed_tools list.",
        )

    # Consequential check: requires recorded human confirmation
    consequential_tools: list[str] = request.policy_context.get("consequential_tools", [])
    status = action.arguments.get("status")
    is_consequential = tool in consequential_tools or (
        isinstance(status, str) and status in _STATUS_CONSEQUENTIAL.get(tool, set())
    )
    already_confirmed = _action_digest(action) in request.history_digest.confirmations_granted

    if is_consequential and not already_confirmed and candidate.type != "request_confirmation":
        return DefenseDecision(
            decision="escalate",
            risk_score=0.70,
            confidence=0.85,
            reason_codes=["CONSEQUENTIAL_UNCONFIRMED"],
            explanation=f"Tool '{tool}' is consequential and has no recorded human confirmation.",
        )

    return None  # pass through
