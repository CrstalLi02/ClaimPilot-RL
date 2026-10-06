---
name: identity_verification
description: Identity Document Verification. Triggered when: The user asks whether an ID document is missing, or whether an unclear ID image affects the review. Always verify the case facts first, then follow the skill steps to compose the reply.
---

# Identity Document Verification

## Trigger Conditions
- The user asks whether an ID document is missing, or whether an unclear ID image affects the review.
- Route tags: identity_doc

## Observe First
- get_uploaded_documents
- search_claim_kb

## Recommended Action Order
1. Read the uploaded documents.
2. Check whether both sides of the ID card are present.
3. If needed, cite the knowledge base to explain that an unclear image or a missing side blocks verification.
4. Send a document request reminder.

## Allowed Write Actions
- send_document_request

## Guardrails
- Never present an identity verification issue as a terms-based rejection.

## Output Constraints
- State the identity document requirement and the re-upload action clearly.
- The reply must be grounded in observed facts; never add terms, timelines, amounts, or case milestones that do not exist.
- If the user asks several questions at once, split them first, then cover each one.
