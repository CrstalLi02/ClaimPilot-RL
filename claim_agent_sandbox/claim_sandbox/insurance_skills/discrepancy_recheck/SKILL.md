---
name: discrepancy_recheck
description: Document Discrepancy Recheck. Triggered when: There are conflicts in invoice dates, hospitalization dates, amounts, or similar details. Always verify the case facts first, then follow the skill steps to compose the reply.
---

# Document Discrepancy Recheck

## Trigger Conditions
- There are conflicts in invoice dates, hospitalization dates, amounts, or similar details.
- Route tags: document_conflict

## Observe First
- get_claim_case
- get_claim_timeline
- get_uploaded_documents

## Recommended Action Order
1. Identify the conflict first.
2. Explain that the conflict comes from inconsistent documents.
3. Escalate to human review when needed.

## Allowed Write Actions
- append_case_note
- escalate_to_human_adjuster

## Guardrails
- Never give a payout decision before the conflict is resolved.

## Output Constraints
- The reply only explains the conflict and the review path.
- The reply must be grounded in observed facts; never add terms, timelines, amounts, or case milestones that do not exist.
- If the user asks several questions at once, split them first, then cover each one.
