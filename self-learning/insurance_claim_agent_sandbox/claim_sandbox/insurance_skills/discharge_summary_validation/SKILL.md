---
name: discharge_summary_validation
description: Discharge Summary Validation. Triggered when: A hospitalization claim is missing the discharge summary, or the discharge summary lacks required information. Always verify the case facts first, then follow the skill steps to compose the reply.
---

# Discharge Summary Validation

## Trigger Conditions
- A hospitalization claim is missing the discharge summary, or the discharge summary lacks required information.
- Route tags: discharge_summary

## Observe First
- get_required_documents
- get_uploaded_documents
- search_claim_kb

## Recommended Action Order
1. Check whether the discharge summary has been provided.
2. If needed, explain that the discharge summary is used to establish the course of hospitalization and its link to coverage.
3. If it is missing, send a document request.

## Allowed Write Actions
- send_document_request

## Guardrails
- Never treat the discharge summary and the invoice as the same kind of document.

## Output Constraints
- The reply states what the discharge summary is for and what needs to be uploaded.
- The reply must be grounded in observed facts; never add terms, timelines, amounts, or case milestones that do not exist.
- If the user asks several questions at once, split them first, then cover each one.
