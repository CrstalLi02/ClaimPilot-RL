---
name: payment_status_trace
description: Payment Status Trace. Triggered when: The user asks why the money has not arrived after approval, or which stage the payment is at. Always verify the case facts first, then follow the skill steps to compose the reply.
---

# Payment Status Trace

## Trigger Conditions
- The user asks why the money has not arrived after approval, or which stage the payment is at.
- Route tags: payment_status

## Observe First
- get_claim_case
- get_claim_timeline
- get_sla_rules

## Recommended Action Order
1. Read the status and the timeline.
2. Determine whether it is stuck in review, payment, or manual review.
3. Create a callback task when needed.

## Allowed Write Actions
- append_case_note
- create_callback_task

## Guardrails
- Never treat approval as the same as the money having arrived.

## Output Constraints
- The reply states the current stage and what to watch for next.
- The reply must be grounded in observed facts; never add terms, timelines, amounts, or case milestones that do not exist.
- If the user asks several questions at once, split them first, then cover each one.
