---
name: pending_docs_resolution
description: Pending Documents Resolution. Triggered when: The user asks about a pending-documents status, which documents are missing, or how long it takes to enter review or get paid after submitting them. Always verify the case facts first, then follow the skill steps to compose the reply.
---

# Pending Documents Resolution

## Trigger Conditions
- The user asks about a pending-documents status, which documents are missing, or how long it takes to enter review or get paid after submitting them.
- Route tags: pending_docs, why_pending, which_docs, timeline_scope

## Observe First
- get_claim_case
- get_uploaded_documents
- get_required_documents
- get_sla_rules

## Recommended Action Order
1. Read the case status first and confirm it is really pending_docs.
2. Compare required_documents with uploaded_documents and list what is missing.
3. Read the SLA; only give a time range, never promise an exact payment date.
4. Send a document request reminder when needed.

## Allowed Write Actions
- send_document_request

## Guardrails
- Never promise a payment date before reading the SLA.
- Never describe an uploaded document as missing.
- Never skip the missing-documents list and just tell the user to wait.

## Output Constraints
- The reply must include the current status, the missing documents, the time range, and the next step.
- The reply must be grounded in observed facts; never add terms, timelines, amounts, or case milestones that do not exist.
- If the user asks several questions at once, split them first, then cover each one.
