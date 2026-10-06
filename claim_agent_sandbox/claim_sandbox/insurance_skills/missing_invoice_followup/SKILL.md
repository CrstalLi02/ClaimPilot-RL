---
name: missing_invoice_followup
description: Missing Invoice Follow-up. Triggered when: The user has uploaded some hospitalization documents, but the original hospitalization invoice or e-invoice is missing. Always verify the case facts first, then follow the skill steps to compose the reply.
---

# Missing Invoice Follow-up

## Trigger Conditions
- The user has uploaded some hospitalization documents, but the original hospitalization invoice or e-invoice is missing.
- Route tags: pending_docs, invoice_missing

## Observe First
- get_uploaded_documents
- get_required_documents
- search_claim_kb

## Recommended Action Order
1. Check the uploaded documents.
2. Determine whether hospitalization invoice documents are missing.
3. Use the knowledge base to explain that missing invoices block the review.
4. Send a document request reminder and explain which invoice formats are accepted.

## Allowed Write Actions
- send_document_request

## Guardrails
- Never say that any invoice is acceptable.
- Never ask for a re-upload before checking the documents.

## Output Constraints
- The reply makes clear that only the invoice is missing, without extending the request to other documents.
- The reply must be grounded in observed facts; never add terms, timelines, amounts, or case milestones that do not exist.
- If the user asks several questions at once, split them first, then cover each one.
