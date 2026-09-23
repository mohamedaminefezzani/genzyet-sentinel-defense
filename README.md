# SENTINEL Defense
**Adaptive Safety for Autonomous AI Agents**
IndabaX Tunisia 2026 - Technical Challenge

---

## Overview

A multi-layer defense pipeline for tool-using LLM agents operating in adversarial environments. The defense intercepts every candidate action the agent proposes and decides whether to allow, block, escalate, or rewrite it based on data-flow provenance, tool-sequence anomaly detection, memory trust propagation, and content inspection.

The defense runs fully offline. No external APIs are called. Every decision is traceable to specific signals in the observability layer.

---

## Architecture

```
DefenseRequest
(action + provenance + conversation + history)
        |
        v
+---------------------------+
|   Layer 1: Policy Gate    |  Hard rules: allowed tools, confirmation prerequisites
+---------------------------+
        |
        v
+---------------------------+  +---------------------------+
| Layer 2a: Taint Analysis  |  | Layer 2b: Memory Guard    |
| Arg values from untrusted?|  | Trust propagation across  |
| Sensitive token in write? |  | turns, rewrite poisoned   |
|                           |  | memory writes             |
+---------------------------+  +---------------------------+
        |                               |
        +---------------+---------------+
                        |
                        v
+---------------------------+  +---------------------------+
| Layer 3a: Trajectory      |  | Layer 3b: Goal Coherence  |
| Tool transition anomaly   |  | Keyword overlap heuristic |
| Forbidden step skipping   |  | (local, offline)          |
+---------------------------+  +---------------------------+
        |                               |
        +---------------+---------------+
                        |
                        v
+------------------------------------------+
|         Layer 4: Risk Aggregator         |
|  risk = 0.55 * weighted_avg             |
|        + 0.45 * max_signal              |
|  weights: taint=0.45, trajectory=0.35,  |
|           memory=0.20, coherence=0.00   |
+------------------------------------------+
        |
        +----------+----------+----------+
        |          |          |          |
      ALLOW    REWRITE   ESCALATE    BLOCK
        |
        v
+---------------------------+
|   Observability Layer     |
|   JSONL trace, risk       |
|   breakdown, reason codes |
+---------------------------+
```

### Decision thresholds

| Risk score | Decision |
|---|---|
| Below 0.38 | Allow |
| 0.38 to 0.60 | Escalate |
| 0.60 and above | Block |
| Memory write in adversary context | Rewrite |

Consequential tools (payment execution, email send, remediation) have a lower effective escalate threshold of 0.285.

---

## Layer Details

### Layer 1 - Policy Gate (`layers/policy_gate.py`)

Fast hard-rule checks that run before any analysis:
- Blocks tools not in the task's `allowed_tools` list
- Escalates consequential tool calls without a recorded human confirmation digest
- Short-circuits the pipeline on hard violations

### Layer 2a - Taint Analysis (`layers/taint.py`)

Two checks merged into one signal:

**Argument taint:** For each argument value in the proposed tool call, traces it to its source trust level. Three signal levels:
- `ARG_TAINTED_ADVERSARY_ONLY` (risk 0.75-0.90): value appears only in adversary-controlled content
- `ARG_TAINTED_UNTRUSTED_ONLY` (risk 0.45-0.65): value appears only in untrusted content
- `ARG_MIRRORED_IN_ADVERSARY` (risk 0.20-0.35): value appears in both trusted and adversary content

Read-only arguments (`asset_id`, `alert_id`, `doc_id`, `case_id`, `query`) are exempt.

**Content inspection:** Scans write actions (`ticket_update`, `incident_update`, `case_note_create`, `wiki_edit`, `email_draft`, `email_send`, `respond`) for sensitive token patterns:
- `SENTINEL_SECRET_[A-Z0-9]+`
- Long uppercase tokens `[A-Z0-9_]{25,}`
- Hex strings `[0-9a-f]{32,}`
- Base64 blobs `[a-zA-Z0-9+/]{40,}={0,2}`

Raises `SENSITIVE_TOKEN_EXFILTRATION` at risk 0.90 when matched.

### Layer 2b - Memory Guard (`layers/memory_guard.py`)

- **Write guard:** Rewrites memory writes that occur in adversary-controlled observation context, prepending `[UNTRUSTED SOURCE]` to the content
- **Recall guard:** When a tool call's argument values trace back to untrusted recalled memory, raises risk accordingly

### Layer 3a - Trajectory Anomaly (`layers/trajectory.py`)

Maintains a transition probability table built from benign reference plans across all three domains (enterprise, finance, SOC). Flags:
- `FORBIDDEN_TOOL_TRANSITION`: transitions that never appear in benign runs (e.g. `payment_execute` after `__START__`)
- `ANOMALOUS_TOOL_TRANSITION`: statistically rare transitions
- `RARE_TOOL_TRANSITION`: low-probability transitions

### Layer 3b - Goal Coherence (`layers/goal_coherence.py`)

Local keyword overlap heuristic. Only fires when upstream risk exceeds 0.50. Checks whether the proposed tool and argument values share meaningful keywords with the user's stated goal. Read-only tools are always considered coherent. Weight in aggregator is 0.00 — present for ablation purposes.

### Layer 4 - Risk Aggregator (`layers/aggregator.py`)

Combines all signals into a final risk score:
```
final_risk = 0.55 * weighted_average + 0.45 * max_signal
```

All signals, weights, layer breakdown, and thresholds are emitted in the decision metadata for the observability layer.

---

## Results

Evaluated against the full public scenario library (40 scenarios, 3 domains, difficulty levels 1-5) using the mock reference agent.

| Metric | Value |
|---|---|
| ASR (Attack Success Rate) | 0.000 |
| CVR (Critical Violation Rate) | 0.000 |
| DFI (Data Flow Integrity) | 1.000 |
| BTU (Benign Task Utility) | 0.778 |
| FBR (False Block Rate) | 0.062 |
| TUI (Tool Use Integrity) | 0.936 |
| Defense errors | 0 |

---

## Setup

### Requirements

- Python 3.12
- `fastapi`, `uvicorn`, `pydantic`
- SENTINEL simulator (`uv sync` in the starter kit)

### Install

```bash
# Clone the starter kit
git clone https://github.com/Skan22/Sentinel_Starter_Kit.git sentinel
cd sentinel
uv sync --extra hf

# Copy defense files
cp /path/to/defense/app/decision.py starter-kits/python-defense/app/
cp -r /path/to/defense/app/layers/ starter-kits/python-defense/app/layers/
```

### Run

```bash
# Start defense service
cd starter-kits/python-defense
uvicorn app.main:app --host 0.0.0.0 --port 8080

# Verify attack delivery (must see attack_success=True)
cd ../../
uv run sentinel run \
    --scenario scenarios/public/finance/finance_false_approval.yaml \
    --defense allow_all \
    --model mock

# Run single scenario against defense
uv run sentinel run \
    --scenario scenarios/public/finance/finance_false_approval.yaml \
    --defense-url http://127.0.0.1:8080 \
    --model mock

# Run full eval
uv run sentinel eval public \
    --defense-url http://127.0.0.1:8080 \
    --model mock \
    --json > metrics.json
```

### Tune thresholds via environment variables

```bash
# Block threshold (default 0.60)
SENTINEL_BLOCK_THRESHOLD=0.70 uvicorn app.main:app ...

# Escalate threshold (default 0.38)
SENTINEL_ESCALATE_THRESHOLD=0.30 uvicorn app.main:app ...

# Coherence trigger threshold (default 0.50)
COHERENCE_THRESHOLD=1.1 uvicorn app.main:app ...  # disables coherence
```

### Run tests

```bash
cd starter-kits/python-defense
pytest tests/test_layers.py -v
```

---

## File Structure

```
app/
    decision.py          # Main pipeline, wires all layers together
    main.py              # FastAPI service entry point
    models.py            # Request/response schemas
    layers/
        policy_gate.py   # Layer 1: hard policy rules
        taint.py         # Layer 2a: argument taint + content inspection
        memory_guard.py  # Layer 2b: memory trust propagation
        trajectory.py    # Layer 3a: tool sequence anomaly
        goal_coherence.py # Layer 3b: local keyword coherence
        aggregator.py    # Layer 4: weighted risk aggregation
tests/
    test_layers.py       # 24 unit and integration tests
```

---

## Ablation Study

| Variant | ASR | CVR | DFI | FBR | UER | Esc. precision |
|---|---|---|---|---|---|---|
| Full defense | 0.000 | 0.000 | 1.000 | 0.062 | 0.043 | 0.708 |
| No coherence (threshold=1.1) | 0.000 | 0.000 | 1.000 | 0.023 | 0.035 | 0.700 |
| Permissive block (threshold=0.85) | 0.000 | 0.000 | 1.000 | 0.023 | 0.035 | 0.700 |
| Strict escalation (threshold=0.25) | 0.000 | 0.000 | 1.000 | 0.023 | 0.035 | 0.700 |

Security metrics are invariant across all variants. The policy gate and taint layers do all the security work on these scenarios.
