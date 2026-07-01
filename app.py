"""
Streamlit UI for the Travel Reimbursement Approval Agent.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import json
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import agent  # noqa: E402
import llm  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
SAMPLES = os.path.join(ROOT, "data", "sample_claims.json")

DECISION_STYLE = {
    "Approve": ("✅", "#1b7f37"),
    "Partially Approve": ("🟡", "#b8860b"),
    "Reject": ("❌", "#b00020"),
    "Manual Review": ("🔍", "#1f4e79"),
}

st.set_page_config(page_title="Travel Reimbursement Approval Agent", page_icon="🧾", layout="wide")


@st.cache_data
def load_samples() -> list[dict]:
    with open(SAMPLES, encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Sidebar — config & input
# --------------------------------------------------------------------------- #
st.sidebar.title("🧾 Reimbursement Agent")
st.sidebar.caption("Plain-Python agent · Ollama · qwen3:8b")

model = st.sidebar.text_input("Ollama model", value=os.environ.get("OLLAMA_MODEL", "qwen3:8b"))
st.sidebar.text_input("Ollama host", value=llm.OLLAMA_HOST, disabled=True)

if llm.is_available():
    st.sidebar.success("Ollama is reachable ✅")
else:
    st.sidebar.error(f"Ollama not reachable at {llm.OLLAMA_HOST}.\nRun `ollama serve` and "
                     "`ollama pull qwen3:8b`.")

st.sidebar.markdown("---")
mode = st.sidebar.radio("Claim input", ["Sample claim", "Paste JSON"])

samples = load_samples()
claim: dict | None = None

if mode == "Sample claim":
    ids = [c["claim_id"] for c in samples]
    chosen = st.sidebar.selectbox("Choose a claim", ids)
    claim = next(c for c in samples if c["claim_id"] == chosen)
else:
    pasted = st.sidebar.text_area("Paste a claim JSON object", height=240,
                                  value=json.dumps(samples[0], indent=2))
    try:
        claim = json.loads(pasted)
    except json.JSONDecodeError as exc:
        st.sidebar.error(f"Invalid JSON: {exc}")

run = st.sidebar.button("▶ Evaluate claim", type="primary", use_container_width=True,
                        disabled=claim is None)

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
st.title("Travel Reimbursement Approval Agent")
st.caption("Evaluates a claim against policy using tool-calling — returns a structured decision.")

if claim is not None:
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Claim")
        st.json(claim, expanded=False)

if run and claim is not None:
    if not llm.is_available():
        st.error("Cannot evaluate: Ollama is not reachable.")
        st.stop()

    events: list[tuple[str, dict]] = []
    status = st.status("Agent is reasoning and calling tools…", expanded=True)

    def on_event(kind: str, payload: dict) -> None:
        events.append((kind, payload))
        if kind == "tool_call":
            status.write(f"🔧 **{payload['name']}**  `{json.dumps(payload['arguments'])}`")
        elif kind == "tool_result":
            status.write(f"   ↳ result: `{json.dumps(payload['result'])[:300]}`")
        elif kind == "guardrail":
            status.write(f"🛡️ guardrail: **{payload['code']}** — {payload['note']}")

    try:
        result = agent.evaluate_claim(claim, on_event=on_event, model=model)
    except Exception as exc:  # noqa: BLE001
        status.update(label="Failed", state="error")
        st.exception(exc)
        st.stop()

    status.update(label="Done", state="complete")
    d = result["decision"]

    icon, color = DECISION_STYLE.get(d["decision"], ("•", "#333"))
    st.markdown(f"## {icon} <span style='color:{color}'>{d['decision']}</span>",
                unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Approved", f"${d['approved_amount']:.2f}")
    c2.metric("Rejected / Deducted", f"${d['rejected_amount']:.2f}")
    c3.metric("Confidence", f"{d['confidence']:.2f}")
    c4.metric("Reason codes", str(len(d["reason_codes"])))

    st.info(d["explanation"])

    cc1, cc2 = st.columns(2)
    with cc1:
        if d["deductions"]:
            st.subheader("Deductions")
            st.table([{"Item": x["item"], "Amount": f"${x['amount']:.2f}", "Reason": x["reason"]}
                      for x in d["deductions"]])
        if d["missing_documents"]:
            st.subheader("Missing documents")
            for m in d["missing_documents"]:
                st.write(f"- {m}")
    with cc2:
        st.subheader("Policy references")
        st.write(", ".join(d["policy_references"]) or "—")
        st.subheader("Reason codes")
        st.write(", ".join(d["reason_codes"]) or "—")

    with st.expander("🔎 Audit trail — tools called & results"):
        for step in result["audit_trail"]:
            st.markdown(f"**{step['tool']}**  ·  args: `{json.dumps(step['arguments'])}`")
            st.json(step["result"], expanded=False)

    with st.expander("🧾 Final decision JSON"):
        st.json(d)

    with st.expander("🤖 Raw model output"):
        st.code(result["raw"] or "(none)")
