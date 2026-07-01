# Travel Reimbursement Approval Agent

A working prototype of an AI-assisted agent that reviews employee travel
reimbursement claims against company policy, receipts, limits, and approval rules,
then returns a structured recommendation: **Approve**, **Partially Approve**,
**Reject**, or **Manual Review**.

Built with **plain Python** (no LangChain / LlamaIndex / CrewAI / framework), a
hand-written tool-calling agent loop, a local **Ollama** LLM (`qwen3:8b`) for
reasoning, **`nomic-embed-text`** embeddings stored in a persistent **ChromaDB**
vector store for semantic policy retrieval, and a **Streamlit** UI.

---

## 1. What it does (maps to the assignment)

| Requirement | How it's met |
|---|---|
| **Claim intake** | JSON claims via the Streamlit UI (sample picker or paste), or the CLI (`run_cli.py`, file/single claim). |
| **Context grounding** | Decisions are grounded in `policies/travel_policy.md` + machine-readable `limits.json` / `approval_matrix.json`. The `policy_lookup` tool does **semantic retrieval** from a persistent **ChromaDB** store (`nomic-embed-text` embeddings, cosine similarity), with a keyword fallback if the store/model is unavailable. |
| **Tool / function usage** | **7 tools** (well above the 2 minimum): `policy_lookup`, `check_limits`, `check_receipts`, `check_duplicates`, `check_eligibility`, `check_approval`, `validate_output`. |
| **GenAI / Agentic workflow** | The LLM decides *which* tools to call and *combines* their results; a hand-written loop (`src/agent.py`) executes the tools and feeds results back. |
| **Structured output** | Consistent JSON: `decision`, `approved_amount`, `rejected_amount`, `deductions`, `missing_documents`, `policy_references`, `confidence`, `reason_codes`, `explanation`. |
| **Manual review handling** | Uncertain / exception cases (duplicates, business class w/o VP approval, non-USD, low confidence, >$5k) are routed to Manual Review by hard guardrails. |
| **Audit trail** (optional) | Every tool call + result is recorded and shown in the UI and saved CLI output. |
| **Tests / eval** (optional) | `tests/test_agent.py` runs deterministic tool tests offline + end-to-end decision expectations when Ollama is up. |
| **Confidence / reason codes** (optional) | Both included in the output schema. |

---

## 2. Setup

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) installed and running

### Install
```bash
cd travel_reimbursement_agent
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# pull the models and make sure the server is running
ollama pull qwen3:8b          # reasoning + tool-calling LLM (~5 GB)
ollama pull nomic-embed-text  # embeddings for semantic policy lookup (~275 MB)
ollama serve                  # if it isn't already running
```

### Environment variables (optional)
| Var | Default | Meaning |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `qwen3:8b` | Reasoning / tool-calling model |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model for policy lookup |

---

## 3. How to run the demo

### Streamlit UI (recommended)
```bash
streamlit run app.py
```
Pick a sample claim (or paste your own JSON), click **Evaluate claim**, and watch
the agent call tools live, then read the structured decision + audit trail.

### CLI
```bash
python run_cli.py                 # evaluate all 5 sample claims, save to sample_outputs/
python run_cli.py --claim CLM-002 # one claim
python run_cli.py --file my.json  # your own claims (list or single object)
```

### Tests / evaluation
```bash
python tests/test_agent.py        # tool tests always run; e2e runs if Ollama is up
```

---

## 4. Architecture

```
            ┌──────────────┐     claim JSON      ┌─────────────────────────┐
 Streamlit  │   app.py     │ ──────────────────▶ │   agent.py (loop)       │
   / CLI    │  run_cli.py  │                     │  - system prompt        │
            └──────────────┘                     │  - tool-calling loop    │
                                                 │  - JSON validate+retry  │
                                                 │  - policy guardrails    │
                                                 └───────────┬─────────────┘
                                                  tool calls │  results
                                            ┌────────────────▼────────────────┐
                       llm.py (Ollama) ◀────┤  tools.py (deterministic engine) │
                                            │  + policies/*.md/json grounding  │
                                            └──────────────────────────────────┘
```

- **`src/llm.py`** — small Ollama REST client (`requests`): chat with tool calls + `embed()` for `nomic-embed-text`.
- **`src/vectorstore.py`** — persistent ChromaDB store; embeds & indexes the policy rules (content-hashed collection so edits auto-reindex).
- **`src/tools.py`** — the 7 tools (real, deterministic logic) + their JSON schemas
  + a dispatcher that injects the current claim so the model passes minimal args.
- **`src/agent.py`** — the agent loop, output validation/retry, and guardrails.
- **`src/schemas.py`** — output schema validator + a safe Manual-Review fallback.
- **`policies/`** — the grounding: a Markdown policy with stable rule IDs
  (`POL-LODGE-01`, …) plus machine-readable limits and the approval matrix.

### Key design choices & trade-offs
- **LLM orchestrates, tools compute.** The model never does arithmetic or invents
  policy — it chooses tools and composes their structured results. This keeps the
  numbers exact and the reasoning auditable.
- **Deterministic guardrails wrap the LLM.** After the model decides, hard rules
  can only *escalate* (e.g. force Manual Review for a duplicate or business-class
  claim) — never silently upgrade an Approve. This makes an 8B local model
  reliable enough for a demo.
- **Claim injected by the dispatcher.** Tools that act on "the current claim"
  take no claim argument from the model, so the small model only passes trivial
  args (`query`, `net_amount`). Far fewer malformed tool calls.
- **Validate-then-fallback.** Invalid model output triggers one strict-JSON retry,
  then a guaranteed-valid Manual Review fallback — the agent never crashes or
  returns malformed output.

---

## 5. Sample outputs

Generated decisions live in [`sample_outputs/`](sample_outputs/) (full audit
trail per claim + `summary.json`). Summary:

| Claim | Scenario | Decision | Approved | Why |
|---|---|---|---|---|
| CLM-001 | Clean domestic trip | **Approve** | $396 | Within all caps, receipts complete |
| CLM-002 | Over-cap lodging + alcohol + over-cap rides | **Partially Approve** | $550 | Deductions: lodging cap, wine, cocktails, ground cap |
| CLM-003 | Stale (114 days) + missing airfare receipt | **Reject** | $0 | Outside 60-day window (POL-GEN-02) |
| CLM-004 | Business class, no VP approval, >$5k | **Manual Review** | — | Needs VP approval (POL-AIR-03 / POL-APPR-04) |
| CLM-005 | Resubmission of a prior claim | **Manual Review** | — | Suspected duplicate (POL-DUP-01) |

UI screenshots of these runs (decision + audit trail) are in
[`screenshots/`](screenshots/).

---

## 6. Assumptions, simplifications & limitations

**Assumptions**
- All amounts are USD unless stated; non-USD claims go to Manual Review.
- A single itemized receipt is represented by an `attachment: true/false` flag.
- The "previously submitted claims" database is a small mock JSON ledger.
- Today's date is taken from the system clock for the 60-day window check.

**Simplifications**
- `policy_lookup` uses `nomic-embed-text` embeddings in a persistent **ChromaDB**
  store (cosine). The collection is content-hashed, so editing the policy file
  transparently rebuilds the index; the `.chroma/` dir holds the persisted DB.
- A line item containing an alcohol keyword is deducted in full (no per-item
  itemization to split out just the alcohol portion).
- The duplicate detector matches on (employee, date, amount, vendor) exactly.

**Known gaps / what I'd improve next**
- Add citation highlighting of the exact retrieved sentence, and tune retrieval
  (chunking, score thresholds) on a larger policy corpus.
- Per-line itemization (split alcohol from a mixed meal receipt; OCR receipts).
- Persist decisions to a real datastore and expose a FastAPI endpoint.
- Multi-currency support with a live FX tool.
- An eval harness with a larger labelled claim set and accuracy metrics.
- Expose the tools over **MCP** so other agents/clients can reuse them.

---

## 7. Project layout
```
travel_reimbursement_agent/
├── app.py                 # Streamlit UI
├── run_cli.py             # batch / single-claim CLI runner
├── requirements.txt
├── policies/
│   ├── travel_policy.md   # grounding policy (rule IDs POL-*)
│   ├── limits.json        # per-diem caps, receipt rules, fare rules
│   └── approval_matrix.json
├── data/
│   ├── sample_claims.json # 5 claims covering all 4 outcomes
│   └── claims_ledger.json # mock prior-claims DB for duplicate detection
├── src/
│   ├── llm.py             # plain Ollama client (chat + embeddings)
│   ├── vectorstore.py     # persistent ChromaDB store for policy retrieval
│   ├── tools.py           # 7 tools + schemas + dispatcher
│   ├── agent.py           # tool-calling loop + guardrails
│   └── schemas.py         # output validation + fallback
├── tests/test_agent.py    # tool tests + e2e decision expectations
├── sample_outputs/        # generated decisions (demo evidence)
├── screenshots/           # UI screenshots (demo evidence)
└── .chroma/               # persisted vector DB (auto-created, git-ignored)
```
