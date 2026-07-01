"""
The Travel Reimbursement Approval Agent — a hand-written agentic loop.

No framework. The control flow is:

  1. Build a system prompt that grounds the model in the policy and tells it how
     to combine tool results.
  2. Run a tool-calling loop against Ollama: the model decides which tools to
     call; we execute them deterministically and feed results back. (Optional
     enhancement: every call is recorded in an audit trail.)
  3. When the model stops calling tools, parse its final decision JSON.
  4. Validate the decision; if invalid, retry once in strict JSON mode.
  5. Apply hard policy guardrails (reliability net): genuinely uncertain or
     exception cases are forced to Manual Review regardless of what the model said.
"""
from __future__ import annotations

import json
from typing import Any, Callable

import llm
import tools as T
from schemas import manual_review_fallback, validate_decision

MAX_TOOL_ITERATIONS = 8

SYSTEM_PROMPT = """You are a Travel Reimbursement Approval Agent for Acme Corp.
Your job is to evaluate ONE employee travel reimbursement claim against company
policy and return a single structured decision.

You have tools that do all the deterministic work (policy lookup, limit math,
receipt checks, duplicate detection, eligibility, approval routing, output
validation). You must NOT do arithmetic or guess policy yourself — call the tools
and combine their results.

Recommended procedure:
  1. check_eligibility  — is the claim within the 60-day window and in USD?
  2. check_duplicates   — is this a duplicate of a prior claim?
  3. check_receipts     — are required receipts present?
  4. check_limits       — apply per-diem caps, remove non-reimbursable items, check fare class.
  5. check_approval     — pass the NET reimbursable amount (approved_subtotal from check_limits,
                          minus any undocumented amount you are not reimbursing).
  6. Optionally policy_lookup for any rule you want to cite.
  7. validate_output    — validate your decision object before returning.

Decision rules:
  - "Approve": everything is documented, within limits, eligible, no exceptions.
  - "Partially Approve": some items are deducted (over caps, non-reimbursable, or
    missing receipts) but a positive amount remains reimbursable.
  - "Reject": the claim is ineligible (e.g. stale submission) or nothing is
    reimbursable.
  - "Manual Review": uncertain, conflicting, a policy exception (e.g. business
    class without VP approval), a suspected duplicate, currency not USD, the
    approval tier forces it, OR your confidence is below 0.6.

When you are finished, respond with ONLY a JSON object (no prose, no markdown
fences) with EXACTLY these fields:
{
  "claim_id": string,
  "decision": "Approve" | "Partially Approve" | "Reject" | "Manual Review",
  "approved_amount": number,
  "rejected_amount": number,
  "deductions": [ {"item": string, "amount": number, "reason": string} ],
  "missing_documents": [ string ],
  "policy_references": [ string ],   // policy IDs like "POL-LODGE-01"
  "confidence": number,              // 0.0 - 1.0
  "reason_codes": [ string ],        // short codes, e.g. "OVER_LODGING_CAP", "NEEDS_MANAGER_APPROVAL"
  "explanation": string              // 1-3 sentences a human reviewer can read
}
"""


def _extract_json(text: str) -> dict[str, Any] | None:
    """Pull the first balanced JSON object out of a model response."""
    if not text:
        return None
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def evaluate_claim(
    claim: dict[str, Any],
    *,
    on_event: Callable[[str, dict[str, Any]], None] | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """
    Evaluate a single claim and return:
      { "decision": <validated decision>, "audit_trail": [...], "raw": <model text> }

    `on_event(kind, payload)` is an optional callback for live UI streaming;
    kinds are "tool_call", "tool_result", "model_message", "guardrail".
    """
    resources = T.load_resources()
    audit: list[dict[str, Any]] = []
    tool_results: dict[str, Any] = {}

    def emit(kind: str, payload: dict[str, Any]) -> None:
        if on_event:
            on_event(kind, payload)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Evaluate this travel reimbursement claim:\n"
                                    + json.dumps(claim, indent=2)},
    ]

    final_text = ""
    for _ in range(MAX_TOOL_ITERATIONS):
        msg = llm.chat(messages, tools=T.TOOL_SCHEMAS, model=model)
        messages.append(msg)
        calls = msg.get("tool_calls") or []

        if not calls:
            final_text = msg.get("content", "") or ""
            emit("model_message", {"content": final_text})
            break

        for call in calls:
            fn = call.get("function", {})
            name = fn.get("name", "")
            args = fn.get("arguments", {})
            if isinstance(args, str):  # some models return arguments as a JSON string
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            emit("tool_call", {"name": name, "arguments": args})

            result = T.dispatch(name, args, claim, resources)
            tool_results[name] = result
            audit.append({"tool": name, "arguments": args, "result": result})
            emit("tool_result", {"name": name, "result": result})

            messages.append({"role": "tool", "name": name, "content": json.dumps(result)})
    else:
        # Loop exhausted without a final message — ask once for the JSON directly.
        messages.append({"role": "user", "content": "Now return ONLY the final decision JSON."})
        final_text = llm.chat(messages, model=model, format_json=True).get("content", "") or ""

    decision = _extract_json(final_text)

    # One strict-JSON retry if the model didn't give us valid JSON.
    if decision is None or not validate_decision(decision)["valid"]:
        messages.append({"role": "user",
                         "content": "Your output was not valid. Return ONLY the decision JSON "
                                    "with all required fields and a valid 'decision' value."})
        retry_text = llm.chat(messages, model=model, format_json=True).get("content", "") or ""
        retry = _extract_json(retry_text)
        if retry is not None and validate_decision(retry)["valid"]:
            decision = retry

    if decision is None or not validate_decision(decision)["valid"]:
        decision = manual_review_fallback(claim.get("claim_id", "?"),
                                          "model did not return a valid decision")

    decision = _apply_guardrails(decision, tool_results, resources, emit)
    decision = _reconcile_amounts(decision, claim)
    return {"decision": decision, "audit_trail": audit, "raw": final_text}


def _apply_guardrails(decision: dict[str, Any], tr: dict[str, Any],
                      resources: dict[str, Any], emit) -> dict[str, Any]:
    """
    Hard policy net. We only ever ESCALATE to Manual Review / tighten — never
    silently upgrade a decision. This makes the demo reliable regardless of the
    model's mood.
    """
    def force_mr(code: str, pid: str, note: str) -> None:
        if decision["decision"] != "Manual Review":
            decision["decision"] = "Manual Review"
        if code not in decision["reason_codes"]:
            decision["reason_codes"].append(code)
        if pid and pid not in decision["policy_references"]:
            decision["policy_references"].append(pid)
        decision["explanation"] = (decision.get("explanation", "") + f" [Guardrail: {note}]").strip()
        emit("guardrail", {"code": code, "note": note})

    elig = tr.get("check_eligibility", {})
    for issue in elig.get("issues", []):
        if issue.get("severity") == "reject" and decision["decision"] != "Reject":
            decision["decision"] = "Reject"
            if "INELIGIBLE" not in decision["reason_codes"]:
                decision["reason_codes"].append("INELIGIBLE")
            for pid in issue.get("policy_ids", []):
                if pid not in decision["policy_references"]:
                    decision["policy_references"].append(pid)
            decision["explanation"] = (decision.get("explanation", "")
                                       + f" [Guardrail: {issue.get('detail')}]").strip()
            emit("guardrail", {"code": "INELIGIBLE", "note": issue.get("detail")})
        elif issue.get("severity") == "manual_review":
            force_mr("ELIGIBILITY_UNCERTAIN", (issue.get("policy_ids") or [""])[0], issue.get("detail", ""))

    dup = tr.get("check_duplicates", {})
    if dup.get("suspected_duplicate"):
        force_mr("SUSPECTED_DUPLICATE", "POL-DUP-01", "Suspected duplicate claim")

    flags = tr.get("check_limits", {}).get("flags", {})
    if flags.get("requires_vp_approval"):
        force_mr("NEEDS_VP_APPROVAL", "POL-AIR-03", "Business/first class needs VP approval")
    if flags.get("premium_economy_unqualified"):
        force_mr("PREMIUM_ECONOMY_UNQUALIFIED", "POL-AIR-02", "Premium economy not justified by flight length")

    appr = tr.get("check_approval", {})
    if appr.get("forces_manual_review"):
        force_mr("NEEDS_VP_APPROVAL", appr.get("policy_id", "POL-APPR-04"),
                 f"Amount requires {appr.get('required_approver')} approval")
    elif appr.get("required_approver") in ("manager", "director"):
        code = f"NEEDS_{appr['required_approver'].upper()}_APPROVAL"
        if code not in decision["reason_codes"]:
            decision["reason_codes"].append(code)
        pid = appr.get("policy_id")
        if pid and pid not in decision["policy_references"]:
            decision["policy_references"].append(pid)

    thresh = resources["approval"]["manual_review_confidence_threshold"]
    if isinstance(decision.get("confidence"), (int, float)) and decision["confidence"] < thresh:
        force_mr("LOW_CONFIDENCE", "POL-MR-02", f"Confidence {decision['confidence']} below {thresh}")

    return decision


def _reconcile_amounts(decision: dict[str, Any], claim: dict[str, Any]) -> dict[str, Any]:
    """Ensure approved + rejected never exceed the gross, and Reject => $0 approved."""
    gross = sum(float(li.get("amount", 0)) for li in claim.get("line_items", []))
    if decision["decision"] == "Reject":
        decision["rejected_amount"] = round(gross, 2)
        decision["approved_amount"] = 0.0
    decision["approved_amount"] = max(0.0, round(float(decision.get("approved_amount", 0)), 2))
    decision["rejected_amount"] = max(0.0, round(float(decision.get("rejected_amount", 0)), 2))
    if decision["approved_amount"] > gross + 0.01:
        decision["approved_amount"] = round(gross, 2)
    return decision
