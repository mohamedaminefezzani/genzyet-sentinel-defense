"""SENTINEL defense — multi-layer decision pipeline.

Pipeline:
    Layer 1 — Policy gate       (hard rules, fast, no ML)
    Layer 2a — Taint analysis   (argument data-flow provenance)
    Layer 2b — Memory guard     (trust propagation across turns)
    Layer 3a — Trajectory       (tool-sequence anomaly)
    Layer 3b — Goal coherence   (LLM: does action serve stated goal?)
    Layer 4  — Aggregator       (weighted risk → allow/escalate/block)

Design principles:
  - Every decision is traceable to specific signals in the metadata field.
  - No scenario identifiers or hard-coded filenames are used.
  - The pipeline is tunable via environment variables (thresholds, weights).
  - Goal coherence (LLM) only fires when upstream risk >= threshold, keeping latency low.
"""

from __future__ import annotations

import json
import logging

from app.layers import aggregator, goal_coherence, memory_guard, policy_gate, taint, trajectory
from app.layers.aggregator import LayerSignal
from app.models import CandidateAction, DefenseDecision, DefenseRequest

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CONSEQUENTIAL_TOOLS = {
    # finance
    "payment_confirm", "payment_execute",
    # enterprise
    "email_send",
    # soc
    "remediation_execute", "remediation_prepare",
}

_STATUS_CONSEQUENTIAL: dict[str, set[str]] = {
    "ticket_update": {"closed", "resolved"},
    "incident_update": {"closed"},
}


def _is_consequential(action: CandidateAction, policy_context: dict) -> bool:
    tool = action.tool
    if tool is None:
        return False
    policy_list: list[str] = policy_context.get("consequential_tools", [])
    if tool in policy_list or tool in _CONSEQUENTIAL_TOOLS:
        return True
    status = action.arguments.get("status")
    return isinstance(status, str) and status in _STATUS_CONSEQUENTIAL.get(tool, set())


def _history_summary(request: DefenseRequest) -> str:
    """Compact summary of recent tool calls for the coherence prompt."""
    calls = request.history_digest.tool_calls[-6:]  # last 6 at most
    if not calls:
        return "none yet"
    parts = []
    for tc in calls:
        tool = tc.get("tool", "?")
        decision = tc.get("decision", "?")
        parts.append(f"{tool}({decision})")
    return " → ".join(parts)


def _unwrap(request: DefenseRequest) -> CandidateAction:
    cand = request.candidate_action
    if cand.type == "request_confirmation" and cand.confirmation_for:
        return cand.confirmation_for
    return cand


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def decide(request: DefenseRequest) -> DefenseDecision:
    """
    Evaluate a candidate action and return a DefenseDecision.
    All layer signals and metadata are recorded for the observability layer.
    """

    # ------------------------------------------------------------------ #
    # Layer 1: Policy gate — fast hard-rule checks                        #
    # ------------------------------------------------------------------ #
    hard_decision = policy_gate.check(request)
    if hard_decision is not None:
        logger.info(
            "step=%d policy_gate=%s reason=%s",
            request.step_id, hard_decision.decision, hard_decision.reason_codes,
        )
        return hard_decision

    action = _unwrap(request)

    # Non-tool, non-memory actions (plain respond) — allow through unless
    # they carry tainted content (handled below for memory_write specifically).
    if action.type == "respond":
        return DefenseDecision(
            decision="allow",
            risk_score=0.05,
            confidence=0.9,
            reason_codes=["RESPOND_ACTION"],
        )

    # ------------------------------------------------------------------ #
    # Layer 2b: Memory guard — runs before taint so it can rewrite first  #
    # ------------------------------------------------------------------ #
    mem_risk, mem_conf, mem_codes, mem_rewrite = memory_guard.check(request)
    if mem_rewrite is not None:
        logger.info("step=%d memory_guard=rewrite", request.step_id)
        return mem_rewrite

    # ------------------------------------------------------------------ #
    # Layer 2a: Taint analysis — data-flow provenance of arg values       #
    # ------------------------------------------------------------------ #
    taint_risk, taint_conf, taint_codes = taint.check(request)

    # Layer 2a (extended): Respond/write content exfiltration check
    respond_risk, respond_conf, respond_codes = taint.check_respond_content(request)

    # Merge into taint signal — take the max risk
    if respond_risk > taint_risk:
        taint_risk = respond_risk
        taint_conf = respond_conf
    taint_codes = list(dict.fromkeys(taint_codes + respond_codes))

    # ------------------------------------------------------------------ #
    # Layer 3a: Trajectory anomaly                                        #
    # ------------------------------------------------------------------ #
    traj_risk, traj_conf, traj_codes = trajectory.check(request)

    # ------------------------------------------------------------------ #
    # Pre-aggregate upstream risk for coherence threshold decision        #
    # ------------------------------------------------------------------ #
    upstream_risk = max(taint_risk, traj_risk, mem_risk)

    # ------------------------------------------------------------------ #
    # Layer 3b: Goal coherence (LLM) — only when upstream risk is high   #
    # ------------------------------------------------------------------ #
    coh_risk = 0.0
    coh_conf = 0.5
    coh_codes: list[str] = []

    if action.type == "tool_call" and action.tool:
        coh_risk, coh_conf, coh_codes = goal_coherence.check(
            user_goal=request.user_goal,
            tool=action.tool,
            arguments=dict(action.arguments),
            history_summary=_history_summary(request),
            upstream_risk=upstream_risk,
        )

    # ------------------------------------------------------------------ #
    # Layer 4: Aggregate all signals                                      #
    # ------------------------------------------------------------------ #
    signals = [
        LayerSignal("taint",      taint_risk, taint_conf, taint_codes),
        LayerSignal("memory",     mem_risk,   mem_conf,   mem_codes),
        LayerSignal("trajectory", traj_risk,  traj_conf,  traj_codes),
        LayerSignal("coherence",  coh_risk,   coh_conf,   coh_codes),
    ]

    final_risk, final_conf, reason_codes, meta = aggregator.aggregate(signals)

    is_cons = _is_consequential(action, request.policy_context)
    decision_str, explanation = aggregator.decide_from_risk(
        final_risk, final_conf, reason_codes, meta, is_consequential=is_cons
    )

    logger.info(
        "step=%d tool=%s decision=%s risk=%.3f reasons=%s",
        request.step_id, action.tool, decision_str, final_risk, reason_codes,
    )

    return DefenseDecision(
        decision=decision_str,
        risk_score=final_risk,
        confidence=final_conf,
        reason_codes=reason_codes[:8],  # cap at 8 for display
        explanation=explanation[:500],
        metadata=meta,
    )
