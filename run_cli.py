#!/usr/bin/env python3
"""
CLI runner — evaluate one or all sample claims and write JSON outputs.

Usage:
  python run_cli.py                         # evaluate all sample claims
  python run_cli.py --claim CLM-002         # evaluate one claim
  python run_cli.py --file my_claims.json   # evaluate claims from a file (JSON list or single obj)
  python run_cli.py --no-save               # don't write to sample_outputs/

Outputs are written to sample_outputs/<claim_id>.json and a combined
sample_outputs/summary.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import agent  # noqa: E402
import llm  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
SAMPLES = os.path.join(ROOT, "data", "sample_claims.json")
OUT_DIR = os.path.join(ROOT, "sample_outputs")


def _load(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def _print_decision(d: dict) -> None:
    print(f"\n{'='*64}")
    print(f"Claim {d['claim_id']}  ->  {d['decision']}")
    print(f"{'-'*64}")
    print(f"  Approved : ${d['approved_amount']:.2f}   Rejected: ${d['rejected_amount']:.2f}")
    print(f"  Confidence: {d['confidence']}   Reason codes: {', '.join(d['reason_codes']) or '—'}")
    if d["deductions"]:
        print("  Deductions:")
        for ded in d["deductions"]:
            print(f"    - ${ded['amount']:.2f}  {ded['item']}  ({ded['reason']})")
    if d["missing_documents"]:
        print(f"  Missing docs: {', '.join(str(m) for m in d['missing_documents'])}")
    print(f"  Policy refs: {', '.join(d['policy_references']) or '—'}")
    print(f"  Explanation: {d['explanation']}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Travel Reimbursement Approval Agent — CLI")
    ap.add_argument("--claim", help="Evaluate only this claim_id from the sample file")
    ap.add_argument("--file", default=SAMPLES, help="Claims file (default: sample claims)")
    ap.add_argument("--no-save", action="store_true", help="Do not write output files")
    ap.add_argument("--model", help="Override Ollama model (default env OLLAMA_MODEL or qwen3:8b)")
    args = ap.parse_args()

    if not llm.is_available():
        print(f"ERROR: Ollama is not reachable at {llm.OLLAMA_HOST}.", file=sys.stderr)
        print("Start it with `ollama serve` and pull the model: `ollama pull qwen3:8b`.", file=sys.stderr)
        return 2

    claims = _load(args.file)
    if args.claim:
        claims = [c for c in claims if c.get("claim_id") == args.claim]
        if not claims:
            print(f"No claim with id {args.claim} in {args.file}", file=sys.stderr)
            return 1

    if not args.no_save:
        os.makedirs(OUT_DIR, exist_ok=True)

    summary = []
    for claim in claims:
        print(f"\nEvaluating {claim.get('claim_id')} ...", file=sys.stderr)
        result = agent.evaluate_claim(claim, model=args.model)
        decision = result["decision"]
        _print_decision(decision)
        summary.append(decision)
        if not args.no_save:
            with open(os.path.join(OUT_DIR, f"{decision['claim_id']}.json"), "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)

    if not args.no_save:
        with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved outputs to {OUT_DIR}/", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
