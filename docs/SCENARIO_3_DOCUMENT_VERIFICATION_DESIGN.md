# Scenario 3 — Detecting incorrect information in documents *(design notes only — not one of the two chosen scenarios)*

Thousands of emails, tickets and forms arrive; the goal is to flag documents that contain **incorrect information**.

**Key framing:** this is mostly **verification against trusted data**, not text classification. A form can be well
written and still carry a wrong bank account. So the solution combines extraction, rules, cross-checks, anomaly
detection and an LLM judge into a **risk score with reasons**, and sends high-risk or unclear documents to people.

## i. Model choices

| Stage | Choice | Why |
|---|---|---|
| Document type | Small text classifier (same cascade as Scenario 1) | Each type needs different checks |
| Field extraction | OCR/layout model for scans + LLM with a JSON schema for emails and free text | Turns documents into structured fields |
| Rules | Format checks (IBAN checksum, dates, VAT format), totals add up, required fields present | Cheap, explainable, catches many errors |
| Cross-checks | Compare fields with systems of record (vendor master, contracts, CRM): bank account changed? price outside contract? | The strongest signal for wrong or fraudulent data |
| Anomaly model | Isolation Forest or gradient boosting on structured features (amount vs history, new account, unusual sender) | Finds unusual patterns rules don't describe |
| Text consistency | LLM judge comparing claims inside the document and against reference documents (RAG) | Catches contradictions in free text |
| Final decision | Weighted score, later a supervised model trained on confirmed cases | Calibrated risk + list of reasons |

## ii. Handling ambiguous inputs

Three outcomes, never a forced yes/no: **ok · suspicious (with reasons) · needs review**.
Missing fields or low extraction confidence → review, not "fraud". Conflicting signals → review with both reasons
shown. Unknown document types → review and add to the taxonomy. Thresholds are chosen with the business from the
cost of a missed error versus the cost of a manual check.

## iii. Evaluation

Labels are rare and arrive late, so build a test set from past confirmed cases plus **synthetic injected errors**
(change an IBAN, alter a total). Use precision/recall on the "suspicious" class and **PR-AUC** — not accuracy (with
0.5 % positives, "everything is fine" is 99.5 % accurate). Report **recall at the team's daily review capacity**,
field-extraction accuracy, and in production the reviewer agreement rate and confirmed-issue rate in the flagged queue.

## iv. Scaling

```
sources → object storage + event → queue (Kafka / Service Bus)
  → stateless workers autoscaled on queue depth:
      classify → extract → rules → cross-check (cached master data) → anomaly → LLM judge (only if needed)
  → results table (risk, reasons, model versions) → review UI → confirmed labels back to training
```
Cheap steps first and the LLM only for the uncertain minority keep cost flat as volume grows. Processing is
idempotent (keyed by document hash), unreadable files go to a dead-letter queue, and access to documents follows
data-protection rules (GDPR).
