"""
Lightweight evaluation script (no pytest required — runs as plain Python too).

Two kinds of checks:
  1. Deterministic tool tests — run WITHOUT the LLM, so they always pass offline
     and prove the policy engine is correct.
  2. End-to-end expectations — run only if Ollama is reachable; assert each sample
     claim lands on the expected decision.

Run:  python tests/test_agent.py        (or)   pytest tests/test_agent.py
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import agent  # noqa: E402
import llm  # noqa: E402
import tools as T  # noqa: E402

with open(os.path.join(ROOT, "data", "sample_claims.json"), encoding="utf-8") as f:
    CLAIMS = {c["claim_id"]: c for c in json.load(f)}
RES = T.load_resources()

EXPECTED = {
    "CLM-001": "Approve",
    "CLM-002": "Partially Approve",
    "CLM-003": "Reject",
    "CLM-004": "Manual Review",
    "CLM-005": "Manual Review",
}


# --------------------------- deterministic tool tests --------------------------- #
def test_limits_lodging_cap_and_alcohol():
    r = T.check_limits(CLAIMS["CLM-002"], RES)
    reasons = " ".join(d["reason"] for d in r["deductions"]).lower()
    assert any("lodging cap" in d["reason"].lower() for d in r["deductions"]), "lodging cap not applied"
    assert "non-reimbursable" in reasons, "alcohol not deducted"
    assert r["approved_subtotal"] < r["gross_amount"], "expected deductions"


def test_receipts_airfare_requires_receipt():
    r = T.check_receipts(CLAIMS["CLM-003"], RES)
    assert not r["complete"], "missing airfare receipt should be detected"
    assert r["undocumented_amount"] >= 280.0


def test_duplicate_detection():
    r = T.check_duplicates(CLAIMS["CLM-005"], RES)
    assert r["suspected_duplicate"], "CLM-005 should match ledger duplicate"


def test_eligibility_stale():
    r = T.check_eligibility(CLAIMS["CLM-003"], RES)
    assert not r["eligible"], "CLM-003 is past the 60-day window"


def test_business_class_flag():
    r = T.check_limits(CLAIMS["CLM-004"], RES)
    assert r["flags"]["requires_vp_approval"], "business class without VP approval should flag"


def test_approval_tiers():
    assert T.check_approval(400, RES)["required_approver"] == "auto"
    assert T.check_approval(1500, RES)["required_approver"] == "manager"
    assert T.check_approval(9000, RES)["forces_manual_review"] is True


# ------------------------------ end-to-end (LLM) ------------------------------ #
def run_e2e() -> int:
    failures = 0
    for cid, expected in EXPECTED.items():
        result = agent.evaluate_claim(CLAIMS[cid])
        got = result["decision"]["decision"]
        ok = got == expected
        failures += 0 if ok else 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {cid}: expected {expected!r}, got {got!r}")
    return failures


def main() -> int:
    print("Running deterministic tool tests...")
    det = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for fn in det:
        fn()
        print(f"  [PASS] {fn.__name__}")

    if llm.is_available():
        print("\nOllama reachable — running end-to-end decision tests...")
        fails = run_e2e()
        if fails:
            print(f"\n{fails} end-to-end expectation(s) failed.")
            return 1
        print("\nAll end-to-end expectations passed.")
    else:
        print("\nOllama not reachable — skipping end-to-end tests (tool tests passed).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
