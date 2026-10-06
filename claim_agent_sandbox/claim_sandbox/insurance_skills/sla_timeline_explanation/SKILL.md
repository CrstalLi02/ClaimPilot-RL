---
name: sla_timeline_explanation
description: SLA Timeline Explanation. Triggered when: The user presses on how long until payment, how long the review takes, or when the case will close. Always verify the case facts first, then follow the skill steps to compose the reply.
---

# SLA Timeline Explanation

## Trigger Conditions
- The user presses on how long until payment, how long the review takes, or when the case will close.
- Route tags: timeline_scope, timeline

## Observe First
- get_claim_case
- get_sla_rules
- get_claim_timeline

## Recommended Action Order
1. Read the SLA.
2. Distinguish the review time after documents are complete, the payment time after approval, and the manual review time.
3. Only give ranges, never specific dates.

## Allowed Write Actions
- append_case_note

## Guardrails
- Never output a specific payment date.
- Never confuse the review timeline with the payment timeline.

## Output Constraints
- The reply must be expressed as a range.
- The reply must be grounded in observed facts; never add terms, timelines, amounts, or case milestones that do not exist.
- If the user asks several questions at once, split them first, then cover each one.
