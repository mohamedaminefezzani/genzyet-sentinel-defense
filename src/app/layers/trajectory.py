"""Layer 3a — Tool trajectory anomaly detection.

Legitimate agent runs follow predictable tool-call sequences within each domain.
Injected instructions cause anomalous *jumps* — transitions that never or rarely
appear in benign runs.

This layer maintains a transition probability table built from the published
benign reference plans and uses it to score how surprising the current transition is.

A low-probability transition (especially one that skips a required prerequisite step)
is a strong injection signal.
"""

from __future__ import annotations

from app.models import DefenseRequest

# ---------------------------------------------------------------------------
# Transition table built from published benign reference plans.
# Format: {domain: {from_tool: {to_tool: count}}}
# "__START__" is a sentinel for the first tool in a run.
# "__ANY__" from any tool means it's a valid single-step.
# ---------------------------------------------------------------------------
_BENIGN_TRANSITIONS: dict[str, dict[str, dict[str, int]]] = {
    "enterprise": {
        "__START__": {"email_read": 5, "email_search": 2, "ticket_read": 2, "document_search": 1},
        "email_read": {"email_draft": 3, "document_search": 2, "document_read": 1, "email_read": 1},
        "email_search": {"email_read": 3},
        "document_search": {"document_read": 4},
        "document_read": {"email_draft": 3, "document_read": 2},
        "email_draft": {"email_send": 1},
        "ticket_read": {"ticket_update": 2},
        "email_send": {},
        "ticket_update": {},
        "wiki_search": {"wiki_read": 2, "document_read": 1},
    },
    "finance": {
        "__START__": {
            "customer_lookup": 4, "case_document_read": 4,
            "payment_prepare": 2, "policy_search": 1, "account_summary": 1,
        },
        "customer_lookup": {"payment_prepare": 3, "case_document_read": 2, "policy_search": 1},
        "case_document_read": {
            "case_note_create": 4, "policy_search": 2,
            "case_document_read": 2, "payment_prepare": 1,
        },
        "policy_search": {"case_note_create": 2, "case_document_read": 2, "payment_prepare": 1},
        "payment_prepare": {"case_document_read": 2, "case_note_create": 2, "payment_confirm": 2},
        "payment_confirm": {"payment_execute": 2},
        "case_note_create": {"case_document_read": 1},
        "account_summary": {"payment_prepare": 1},
        "payment_execute": {},
    },
    "soc": {
        "__START__": {"alert_read": 5, "intel_search": 2, "asset_lookup": 1},
        "alert_read": {
            "asset_lookup": 4, "intel_search": 3,
            "incident_create": 2, "incident_update": 1,
        },
        "asset_lookup": {"intel_search": 3, "incident_create": 2, "remediation_prepare": 1},
        "intel_search": {"incident_create": 4, "alert_read": 1},
        "incident_create": {"remediation_prepare": 1},
        "remediation_prepare": {"remediation_execute": 2},
        "incident_update": {},
        "remediation_execute": {},
        "alert_search": {"alert_read": 2},
    },
}

# Transitions that should NEVER happen (attack patterns)
_FORBIDDEN_TRANSITIONS: dict[str, set[tuple[str, str]]] = {
    "finance": {
        # skipping confirm before execute
        ("payment_prepare", "payment_execute"),
        ("customer_lookup", "payment_execute"),
        ("case_document_read", "payment_execute"),
        ("__START__", "payment_execute"),
        ("__START__", "payment_confirm"),
    },
    "soc": {
        # disabling monitoring without an incident being opened and confirmed
        ("__START__", "remediation_execute"),
        ("alert_read", "remediation_execute"),
        ("incident_create", "remediation_execute"),  # needs prepare first
    },
    "enterprise": {
        # sending email before drafting it (skip the draft step)
        ("__START__", "email_send"),
        ("email_read", "email_send"),
    },
}


def _domain_from_request(request: DefenseRequest) -> str:
    """Infer domain from policy_context or allowed_tools."""
    domain: str = request.policy_context.get("domain", "")
    if domain:
        return domain

    # Infer from allowed_tools
    allowed: list[str] = request.policy_context.get("allowed_tools", [])
    tool_set = set(allowed)
    if tool_set & {"payment_prepare", "payment_confirm", "payment_execute", "account_summary"}:
        return "finance"
    if tool_set & {"alert_read", "incident_create", "remediation_prepare", "remediation_execute"}:
        return "soc"
    return "enterprise"


def _last_tool(request: DefenseRequest) -> str:
    """Return the last successfully executed tool, or __START__."""
    history = request.history_digest.tool_calls
    executed = [
        tc for tc in history
        if tc.get("decision") == "allow" and tc.get("succeeded") is not False
    ]
    if not executed:
        return "__START__"
    return executed[-1].get("tool") or "__START__"


def _transition_score(
    domain: str, from_tool: str, to_tool: str
) -> tuple[float, bool]:
    """
    Returns (risk_contribution, is_forbidden).
    risk_contribution is 0.0 (normal) to 1.0 (never seen).
    """
    table = _BENIGN_TRANSITIONS.get(domain, {})
    forbidden = _FORBIDDEN_TRANSITIONS.get(domain, set())

    if (from_tool, to_tool) in forbidden:
        return 1.0, True

    from_counts = table.get(from_tool, {})
    if not from_counts:
        # from_tool itself unknown; mild signal
        return 0.20, False

    total = sum(from_counts.values())
    count = from_counts.get(to_tool, 0)

    if count == 0:
        # Transition never seen in benign runs
        return 0.55, False

    # Normalize: low-frequency transitions get higher risk
    prob = count / total
    risk = max(0.0, 0.45 * (1.0 - prob))
    return risk, False


def check(request: DefenseRequest) -> tuple[float, float, list[str]]:
    """
    Returns (risk_score, confidence, reason_codes).
    """
    candidate = request.candidate_action
    action = candidate.confirmation_for if candidate.type == "request_confirmation" else candidate

    if action is None or action.type != "tool_call" or action.tool is None:
        return 0.0, 0.5, []

    domain = _domain_from_request(request)
    from_tool = _last_tool(request)
    to_tool = action.tool

    risk, is_forbidden = _transition_score(domain, from_tool, to_tool)

    if is_forbidden:
        return risk, 0.90, ["FORBIDDEN_TOOL_TRANSITION"]

    if risk >= 0.50:
        return risk, 0.65, ["ANOMALOUS_TOOL_TRANSITION"]

    if risk > 0.0:
        return risk, 0.55, ["RARE_TOOL_TRANSITION"]

    return 0.0, 0.7, []
