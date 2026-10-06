---
name: hospitalization_material_check
description: Hospitalization Document Check. Triggered when: The case is a hospitalization claim and the user asks why it cannot pass review or whether the documents are complete. Always verify the case facts first, then follow the skill steps to compose the reply.
---

# Hospitalization Document Check

## Trigger Conditions
- The case is a hospitalization claim and the user asks why it cannot pass review or whether the documents are complete.
- Route tags: hospitalization, pending_docs

## Observe First
- get_claim_case
- get_required_documents
- get_uploaded_documents
- get_policy_summary

## Recommended Action Order
1. Confirm that this is a hospitalization insurance case.
2. List the key documents usually needed for hospitalization: discharge summary, invoices, and identity documents.
3. Based on the current case, determine which documents are complete and which are missing.
4. If anything is missing, send a document request.

## Allowed Write Actions
- send_document_request

## Guardrails
- Never treat outpatient documents as core hospitalization documents.
- Never output a generic checklist detached from the actual case.

## Output Constraints
- The reply lists both what is complete and what is still missing.
- The reply must be grounded in observed facts; never add terms, timelines, amounts, or case milestones that do not exist.
- If the user asks several questions at once, split them first, then cover each one.
