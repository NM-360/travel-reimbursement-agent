# Acme Corp — Travel & Expense Reimbursement Policy

_Version 2.1 — Effective 2026-01-01. Mock policy for demonstration only._

This document is the authoritative source of truth that the Travel Reimbursement
Approval Agent must ground its decisions in. Each rule has a stable ID (e.g.
`POL-MEAL-01`) so decisions can cite exactly which rule was applied.

---

## 1. General Eligibility (POL-GEN)

- **POL-GEN-01**: Only expenses incurred for legitimate, pre-approved business
  travel are reimbursable.
- **POL-GEN-02**: Claims must be submitted within **60 days** of the trip end
  date. Claims older than 60 days are rejected unless a manager exception note is
  attached (route to Manual Review).
- **POL-GEN-03**: All amounts are assumed to be in **USD**. Claims in other
  currencies must include a converted USD amount or are routed to Manual Review.
- **POL-GEN-04**: No personal, family, or entertainment expenses are reimbursable.

## 2. Receipts & Documentation (POL-DOC)

- **POL-DOC-01**: An itemized receipt is **required for any single expense over
  $25**. Expenses over $25 with `attachment: false` are not reimbursable and are
  deducted.
- **POL-DOC-02**: Airfare and lodging **always** require a receipt regardless of
  amount.
- **POL-DOC-03**: A claim missing one or more required receipts is **Partially
  Approved** (reimbursing only documented items) or routed to Manual Review if
  the missing amount exceeds 40% of the claim total.

## 3. Meals & Per Diem (POL-MEAL)

- **POL-MEAL-01**: Meals are capped at a **per-diem of $75/day (domestic)** and
  **$100/day (international)**. Amounts above the daily cap are deducted.
- **POL-MEAL-02**: **Alcohol is never reimbursable** and must be deducted in full.
- **POL-MEAL-03**: Meal expenses under $25 do not require a receipt.

## 4. Lodging (POL-LODGE)

- **POL-LODGE-01**: Lodging is capped at **$250/night (domestic)** and
  **$350/night (international)**. Amounts above the nightly cap are deducted.
- **POL-LODGE-02**: In-room entertainment, minibar, and spa charges are not
  reimbursable.

## 5. Airfare & Rail (POL-AIR)

- **POL-AIR-01**: **Economy class** is the default and is fully reimbursable.
- **POL-AIR-02**: **Premium economy** is allowed only for flights longer than
  **6 hours**.
- **POL-AIR-03**: **Business or first class requires written VP approval.**
  Without it, the claim is routed to Manual Review (not auto-rejected).

## 6. Ground Transportation (POL-GROUND)

- **POL-GROUND-01**: Taxi, rideshare, and public transit are capped at a
  combined **$50/day**. Amounts above the cap are deducted.
- **POL-GROUND-02**: Rental cars require a business justification note; without
  one, route to Manual Review.

## 7. Approval Thresholds (POL-APPR)

Approval authority is based on the **net reimbursable amount** (after deductions):

- **POL-APPR-01**: Up to **$500** — auto-approvable by the agent.
- **POL-APPR-02**: **$500.01 – $2,000** — requires Manager approval.
- **POL-APPR-03**: **$2,000.01 – $5,000** — requires Director approval.
- **POL-APPR-04**: Over **$5,000** — requires VP approval; always route to
  Manual Review.

Claims in the **Manager** or **Director** tier are still adjudicated normally
(Approve / Partially Approve / Reject); the agent records the required approver
in `reason_codes` so a human signs off, but this **does not** by itself force
Manual Review. Only the **VP tier (over $5,000)** forces a final decision of
**Manual Review**.

## 8. Duplicates & Fraud Signals (POL-DUP)

- **POL-DUP-01**: A claim that matches a previously submitted claim on
  (employee, date, amount, vendor) is a suspected duplicate and is routed to
  Manual Review.
- **POL-DUP-02**: Two line items within the same claim with identical
  (date, amount, vendor, category) are flagged as a possible double entry.

## 9. Ambiguity & Manual Review (POL-MR)

- **POL-MR-01**: When information is missing, conflicting, or a policy exception
  is invoked, the agent must route to **Manual Review** rather than guessing.
- **POL-MR-02**: Confidence below 0.6 must result in Manual Review.
