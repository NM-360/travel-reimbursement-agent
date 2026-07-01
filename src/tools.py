"""
Tools for the Travel Reimbursement Approval Agent.

Each tool is a plain Python function that does ONE deterministic job (policy
lookup, limit math, receipt check, duplicate detection, eligibility, approval
routing, output validation). The LLM never does arithmetic itself — it decides
*which* tool to call and *combines* the structured results into a decision.

Two layers live here:
  1. The Python functions (the real logic).
  2. `TOOL_SCHEMAS` (OpenAI/Ollama-style JSON schemas) + `dispatch()` which maps
     an LLM tool call to the right function, injecting the current claim and the
     loaded policy resources so the model only has to pass simple arguments.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any

import llm
from schemas import validate_decision

# --------------------------------------------------------------------------- #
# Resource loading (grounding context)
# --------------------------------------------------------------------------- #
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_POLICY_DIR = os.path.join(_ROOT, "policies")
_DATA_DIR = os.path.join(_ROOT, "data")


def load_resources() -> dict[str, Any]:
    """Load all grounding files once and return them as a dict."""
    with open(os.path.join(_POLICY_DIR, "travel_policy.md"), encoding="utf-8") as f:
        policy_md = f.read()
    with open(os.path.join(_POLICY_DIR, "limits.json"), encoding="utf-8") as f:
        limits = json.load(f)
    with open(os.path.join(_POLICY_DIR, "approval_matrix.json"), encoding="utf-8") as f:
        approval = json.load(f)
    with open(os.path.join(_DATA_DIR, "claims_ledger.json"), encoding="utf-8") as f:
        ledger = json.load(f)
    return {
        "policy_md": policy_md,
        "policy_rules": _split_policy_rules(policy_md),
        "limits": limits,
        "approval": approval,
        "ledger": ledger,
    }


def _split_policy_rules(md: str) -> list[dict[str, str]]:
    """Break the markdown policy into individually-citable rule blocks."""
    rules = []
    for m in re.finditer(r"\*\*(POL-[A-Z]+-\d+)\*\*:?\s*(.+?)(?=\n- \*\*POL-|\n## |\Z)", md, re.S):
        rules.append({"policy_id": m.group(1), "text": " ".join(m.group(2).split())})
    return rules


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _round(x: float) -> float:
    return round(float(x) + 1e-9, 2)


def _contains_non_reimbursable(text: str, keywords: list[str]) -> str | None:
    low = (text or "").lower()
    for kw in keywords:
        if kw in low:
            return kw
    return None


# --------------------------------------------------------------------------- #
# TOOL 1: policy_lookup  (semantic retrieval via ChromaDB + nomic-embed-text)
# --------------------------------------------------------------------------- #
def _keyword_lookup(query: str, resources: dict[str, Any]) -> list[dict[str, Any]]:
    """Term-frequency fallback used when embeddings / the vector store fail."""
    terms = [t for t in re.findall(r"[a-z0-9]+", (query or "").lower()) if len(t) > 2]
    scored = []
    for rule in resources["policy_rules"]:
        hay = (rule["policy_id"] + " " + rule["text"]).lower()
        score = sum(hay.count(t) for t in terms)
        if score:
            scored.append((score, rule))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored[:5]]


def policy_lookup(query: str, resources: dict[str, Any], top_k: int = 5) -> dict[str, Any]:
    """
    Semantically retrieve the most relevant policy rules for a query from a
    persistent ChromaDB collection (embeddings: nomic-embed-text, cosine
    similarity). Falls back to keyword search if the vector store / embedding
    model is unavailable.
    """
    rules = resources["policy_rules"]
    try:
        import vectorstore  # local import so the rest of the app works without chromadb
        matches = vectorstore.query(rules, query, top_k=top_k)
        return {"query": query, "method": "chromadb:" + llm.OLLAMA_EMBED_MODEL,
                "matches": matches, "match_count": len(matches)}
    except Exception as exc:  # noqa: BLE001 — any store/emb( failure -> safe fallback
        matches = _keyword_lookup(query, resources)
        return {"query": query, "method": "keyword_fallback", "fallback_reason": str(exc),
                "matches": matches, "match_count": len(matches)}


# --------------------------------------------------------------------------- #
# TOOL 2: check_limits  (per-diem, caps, alcohol, fare class)
# --------------------------------------------------------------------------- #
def check_limits(claim: dict[str, Any], resources: dict[str, Any]) -> dict[str, Any]:
    limits = resources["limits"]
    intl = claim.get("trip_type", "domestic") == "international"
    region = "international" if intl else "domestic"
    meal_cap = limits["per_diem"]["meals"][region]
    lodge_cap = limits["per_diem"]["lodging_per_night"][region]
    ground_cap = limits["per_diem"]["ground_transport_per_day"][region]
    nonreimb = limits["non_reimbursable_keywords"]

    line_results: list[dict[str, Any]] = []
    flags = {"requires_vp_approval": False, "premium_economy_unqualified": False}
    meals_by_day: dict[str, float] = {}
    ground_by_day: dict[str, float] = {}

    for li in claim.get("line_items", []):
        amount = float(li.get("amount", 0))
        cat = li.get("category", "misc")
        desc = li.get("description", "")
        allowed, deducted, reasons, pids = amount, 0.0, [], []

        kw = _contains_non_reimbursable(desc, nonreimb)
        if kw:
            allowed, deducted = 0.0, amount
            reasons.append(f"Non-reimbursable item detected ('{kw}')")
            pids += ["POL-MEAL-02", "POL-GEN-04"]
        elif cat == "lodging":
            nights = int(li.get("nights", 1) or 1)
            cap = lodge_cap * nights
            if amount > cap:
                allowed, deducted = cap, amount - cap
                reasons.append(f"Over lodging cap ${lodge_cap}/night x {nights} = ${cap}")
                pids.append("POL-LODGE-01")
        elif cat == "airfare":
            fc = (li.get("fare_class") or "economy").lower()
            hours = float(li.get("flight_hours", 0) or 0)
            if fc in ("business", "first"):
                if not li.get("vp_approval", False):
                    flags["requires_vp_approval"] = True
                    reasons.append(f"{fc.title()} class requires VP approval")
                    pids.append("POL-AIR-03")
            elif fc == "premium_economy" and hours <= 6:
                flags["premium_economy_unqualified"] = True
                reasons.append("Premium economy only allowed for flights > 6h")
                pids.append("POL-AIR-02")
        elif cat == "meals":
            meals_by_day[li.get("date", "?")] = meals_by_day.get(li.get("date", "?"), 0) + amount
        elif cat == "ground_transport":
            ground_by_day[li.get("date", "?")] = ground_by_day.get(li.get("date", "?"), 0) + amount

        line_results.append({
            "category": cat, "description": desc, "amount": _round(amount),
            "allowed_amount": _round(allowed), "deducted_amount": _round(deducted),
            "reasons": reasons, "policy_ids": pids,
        })

    deductions: list[dict[str, Any]] = []
    for lr in line_results:
        if lr["deducted_amount"] > 0:
            deductions.append({
                "item": f"{lr['category']}: {lr['description']}",
                "amount": lr["deducted_amount"],
                "reason": "; ".join(lr["reasons"]),
                "policy_ids": lr["policy_ids"],
            })

    # Per-diem caps applied on the aggregated daily totals of clean lines.
    for day, total in meals_by_day.items():
        if total > meal_cap:
            deductions.append({
                "item": f"meals on {day}", "amount": _round(total - meal_cap),
                "reason": f"Over meal per-diem ${meal_cap}/day (claimed ${_round(total)})",
                "policy_ids": ["POL-MEAL-01"],
            })
    for day, total in ground_by_day.items():
        if total > ground_cap:
            deductions.append({
                "item": f"ground_transport on {day}", "amount": _round(total - ground_cap),
                "reason": f"Over ground transport cap ${ground_cap}/day (claimed ${_round(total)})",
                "policy_ids": ["POL-GROUND-01"],
            })

    gross = sum(float(li.get("amount", 0)) for li in claim.get("line_items", []))
    total_deducted = sum(d["amount"] for d in deductions)
    approved_subtotal = _round(gross - total_deducted)

    return {
        "trip_region": region,
        "gross_amount": _round(gross),
        "total_deductions": _round(total_deducted),
        "approved_subtotal": approved_subtotal,
        "deductions": deductions,
        "flags": flags,
        "line_results": line_results,
    }


# --------------------------------------------------------------------------- #
# TOOL 3: check_receipts
# --------------------------------------------------------------------------- #
def check_receipts(claim: dict[str, Any], resources: dict[str, Any]) -> dict[str, Any]:
    limits = resources["limits"]
    threshold = limits["receipt_required_above"]
    always = set(limits["always_require_receipt"])

    missing: list[dict[str, Any]] = []
    undocumented_amount = 0.0
    for li in claim.get("line_items", []):
        amount = float(li.get("amount", 0))
        cat = li.get("category", "misc")
        needs = (amount > threshold) or (cat in always)
        if needs and not li.get("attachment", False):
            missing.append({
                "item": f"{cat}: {li.get('description','')}",
                "amount": _round(amount),
                "reason": "Receipt required but attachment missing",
                "policy_ids": ["POL-DOC-02" if cat in always else "POL-DOC-01"],
            })
            undocumented_amount += amount

    gross = sum(float(li.get("amount", 0)) for li in claim.get("line_items", [])) or 1.0
    return {
        "missing_documents": missing,
        "undocumented_amount": _round(undocumented_amount),
        "undocumented_ratio": _round(undocumented_amount / gross),
        "complete": len(missing) == 0,
    }


# --------------------------------------------------------------------------- #
# TOOL 4: check_duplicates
# --------------------------------------------------------------------------- #
def check_duplicates(claim: dict[str, Any], resources: dict[str, Any]) -> dict[str, Any]:
    ledger = resources["ledger"]["entries"]
    emp = claim.get("employee_id")
    cross: list[dict[str, Any]] = []
    for li in claim.get("line_items", []):
        for e in ledger:
            if e.get("claim_id") == claim.get("claim_id"):
                continue  # don't match a claim against itself
            if (e.get("employee_id") == emp
                    and e.get("date") == li.get("date")
                    and abs(float(e.get("amount", -1)) - float(li.get("amount", 0))) < 0.01
                    and (e.get("vendor", "").lower() == (li.get("vendor", "") or "").lower())):
                cross.append({
                    "item": f"{li.get('category')}: {li.get('description','')}",
                    "amount": _round(float(li.get("amount", 0))),
                    "matches_prior_claim": e.get("claim_id"),
                    "policy_ids": ["POL-DUP-01"],
                })

    # In-claim double entries.
    seen: dict[tuple, int] = {}
    internal: list[dict[str, Any]] = []
    for li in claim.get("line_items", []):
        key = (li.get("date"), float(li.get("amount", 0)), (li.get("vendor", "") or "").lower(), li.get("category"))
        seen[key] = seen.get(key, 0) + 1
        if seen[key] == 2:
            internal.append({"item": f"{li.get('category')}: {li.get('description','')}",
                             "amount": _round(float(li.get("amount", 0))), "policy_ids": ["POL-DUP-02"]})

    return {
        "suspected_duplicate": bool(cross or internal),
        "cross_claim_matches": cross,
        "in_claim_duplicates": internal,
    }


# --------------------------------------------------------------------------- #
# TOOL 5: check_eligibility  (submission window, currency)
# --------------------------------------------------------------------------- #
def check_eligibility(claim: dict[str, Any], resources: dict[str, Any]) -> dict[str, Any]:
    limits = resources["limits"]
    issues: list[dict[str, Any]] = []

    try:
        trip_end = datetime.strptime(claim["trip_end"], "%Y-%m-%d").date()
        submitted = datetime.strptime(claim["submitted_date"], "%Y-%m-%d").date()
        age = (submitted - trip_end).days
        window = limits["submission_window_days"]
        if age > window:
            issues.append({
                "type": "stale_submission", "detail": f"Submitted {age} days after trip end (limit {window})",
                "policy_ids": ["POL-GEN-02"], "severity": "reject",
            })
    except (KeyError, ValueError) as exc:
        issues.append({"type": "bad_date", "detail": str(exc), "policy_ids": ["POL-GEN-02"], "severity": "manual_review"})
        age = None

    cur = (claim.get("currency") or "USD").upper()
    if cur != "USD":
        issues.append({
            "type": "non_usd_currency", "detail": f"Claim currency is {cur}, expected USD",
            "policy_ids": ["POL-GEN-03"], "severity": "manual_review",
        })

    return {"eligible": len(issues) == 0, "issues": issues, "claim_age_days": age}


# --------------------------------------------------------------------------- #
# TOOL 6: check_approval  (threshold routing)
# --------------------------------------------------------------------------- #
def check_approval(net_amount: float, resources: dict[str, Any]) -> dict[str, Any]:
    net_amount = float(net_amount)
    for tier in resources["approval"]["tiers"]:
        cap = tier["max_amount"]
        if cap is None or net_amount <= cap:
            return {
                "net_amount": _round(net_amount),
                "required_approver": tier["approver"],
                "forces_manual_review": tier["manual_review"],
                "policy_id": tier["policy_id"],
            }
    # Should be unreachable because the last tier has max_amount=null.
    return {"net_amount": _round(net_amount), "required_approver": "vp",
            "forces_manual_review": True, "policy_id": "POL-APPR-04"}


# --------------------------------------------------------------------------- #
# TOOL 7: validate_output
# --------------------------------------------------------------------------- #
def validate_output(decision: dict[str, Any], resources: dict[str, Any]) -> dict[str, Any]:
    return validate_decision(decision)


# --------------------------------------------------------------------------- #
# LLM-facing schemas + dispatcher
# --------------------------------------------------------------------------- #
# Tools that operate on "the current claim" take no claim arg from the model —
# the dispatcher injects it. This keeps arguments small and reliable for an 8B
# model.
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "policy_lookup",
            "description": "Search the travel policy for rules relevant to a topic "
                           "(e.g. 'business class airfare', 'meal per diem', 'receipts'). "
                           "Returns matching rule IDs and text to ground your reasoning.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Topic or keywords to look up"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_limits",
            "description": "Apply per-diem caps (meals, lodging, ground transport), remove "
                           "non-reimbursable items (alcohol, personal), and check airfare fare-class "
                           "rules for the current claim. Returns deductions, the approved subtotal, "
                           "and flags such as requires_vp_approval.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_receipts",
            "description": "Check the current claim for missing required receipts. Returns the list "
                           "of undocumented items, the undocumented amount, and its ratio of the total.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_duplicates",
            "description": "Detect whether the current claim duplicates a previously submitted claim "
                           "or contains internal double-entries.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_eligibility",
            "description": "Check basic eligibility of the current claim: submission within the 60-day "
                           "window and USD currency.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_approval",
            "description": "Given the NET reimbursable amount (after deductions), return the required "
                           "approver tier and whether it forces Manual Review.",
            "parameters": {
                "type": "object",
                "properties": {"net_amount": {"type": "number", "description": "Net reimbursable amount after deductions"}},
                "required": ["net_amount"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_output",
            "description": "Validate a final decision object against the required output schema before "
                           "returning it. Pass the full decision JSON.",
            "parameters": {
                "type": "object",
                "properties": {"decision": {"type": "object", "description": "The decision object to validate"}},
                "required": ["decision"],
            },
        },
    },
]

# Which functions need the current claim injected (vs. taking model args).
_CLAIM_TOOLS = {"check_limits", "check_receipts", "check_duplicates", "check_eligibility"}
_DISPATCH = {
    "policy_lookup": policy_lookup,
    "check_limits": check_limits,
    "check_receipts": check_receipts,
    "check_duplicates": check_duplicates,
    "check_eligibility": check_eligibility,
    "check_approval": check_approval,
    "validate_output": validate_output,
}


def dispatch(name: str, arguments: dict[str, Any], claim: dict[str, Any],
             resources: dict[str, Any]) -> dict[str, Any]:
    """Execute a tool call requested by the LLM and return its result dict."""
    arguments = arguments or {}
    fn = _DISPATCH.get(name)
    if fn is None:
        return {"error": f"unknown tool: {name}"}
    try:
        if name in _CLAIM_TOOLS:
            return fn(claim, resources)
        if name == "policy_lookup":
            return fn(arguments.get("query", ""), resources)
        if name == "check_approval":
            return fn(arguments.get("net_amount", 0), resources)
        if name == "validate_output":
            return fn(arguments.get("decision", {}), resources)
    except Exception as exc:  # noqa: BLE001 — tools must never crash the agent loop
        return {"error": f"tool '{name}' failed: {exc}"}
    return {"error": f"unhandled tool: {name}"}
