"""
Output schema + validator for the agent's final decision.

We use a plain hand-written validator (no pydantic, no framework) so the
dependency surface stays tiny and the rules are easy to read. The validator is
itself exposed to the LLM as the `validate_output` tool, and is also used as a
safety net by the agent before returning — if the model emits something invalid,
we fall back to Manual Review.
"""
from __future__ import annotations

from typing import Any

VALID_DECISIONS = {"Approve", "Partially Approve", "Reject", "Manual Review"}

# The canonical shape of a decision object.
REQUIRED_FIELDS = {
    "claim_id": str,
    "decision": str,
    "approved_amount": (int, float),
    "rejected_amount": (int, float),
    "deductions": list,
    "missing_documents": list,
    "policy_references": list,
    "confidence": (int, float),
    "reason_codes": list,
    "explanation": str,
}


def validate_decision(obj: Any) -> dict[str, Any]:
    """
    Validate a decision object. Returns {"valid": bool, "errors": [...]}.

    This is intentionally forgiving about extra keys but strict about the
    required ones, their types, and value ranges.
    """
    errors: list[str] = []

    if not isinstance(obj, dict):
        return {"valid": False, "errors": ["decision must be a JSON object"]}

    for field, ftype in REQUIRED_FIELDS.items():
        if field not in obj:
            errors.append(f"missing field: {field}")
            continue
        if not isinstance(obj[field], ftype):
            errors.append(f"field '{field}' has wrong type (got {type(obj[field]).__name__})")

    if isinstance(obj.get("decision"), str) and obj["decision"] not in VALID_DECISIONS:
        errors.append(f"decision must be one of {sorted(VALID_DECISIONS)}")

    conf = obj.get("confidence")
    if isinstance(conf, (int, float)) and not (0.0 <= conf <= 1.0):
        errors.append("confidence must be between 0 and 1")

    for amt in ("approved_amount", "rejected_amount"):
        v = obj.get(amt)
        if isinstance(v, (int, float)) and v < 0:
            errors.append(f"{amt} must be >= 0")

    return {"valid": len(errors) == 0, "errors": errors}


def manual_review_fallback(claim_id: str, reason: str) -> dict[str, Any]:
    """A guaranteed-valid Manual Review decision used when anything goes wrong."""
    return {
        "claim_id": claim_id,
        "decision": "Manual Review",
        "approved_amount": 0.0,
        "rejected_amount": 0.0,
        "deductions": [],
        "missing_documents": [],
        "policy_references": ["POL-MR-01"],
        "confidence": 0.0,
        "reason_codes": ["FALLBACK"],
        "explanation": f"Routed to Manual Review by safety fallback: {reason}",
    }
