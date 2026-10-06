---
name: manual_review_escalation
description: Manual Review Escalation. Triggered when: The case is in manual_review, or the user asks why it is stuck or when it will be closed. Always verify the case facts first, then follow the skill steps to compose the reply.
---

# Manual Review Escalation

## Trigger Conditions
- The case is in manual_review, or the user asks why it is stuck or when it will be closed.
- Route tags: manual_review, why_manual_review, next_step

## Observe First
- get_claim_case
- get_escalation_matrix
- get_claim_timeline

## Recommended Action Order
1. Confirm manual_review_required or risk_flags first.
2. Explain the direct reason the case entered manual review.
3. Read the escalation matrix.
4. Escalate, or confirm the case is already in the manual review queue.

## Allowed Write Actions
- escalate_to_human_adjuster
- append_case_note

## Guardrails
- Never promise a closing date before manual review is done.
- Never describe manual review as a system failure.

## Output Constraints
- The reply includes the current status, the trigger reason, the next step, and who takes over.
- The reply must be grounded in observed facts; never add terms, timelines, amounts, or case milestones that do not exist.
- If the user asks several questions at once, split them first, then cover each one.
