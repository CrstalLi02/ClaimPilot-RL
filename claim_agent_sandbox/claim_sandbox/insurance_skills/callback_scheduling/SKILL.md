---
name: callback_scheduling
description: Callback Scheduling. Triggered when: The user cannot stay online, asks to be contacted later, or needs a human callback. Always verify the case facts first, then follow the skill steps to compose the reply.
---

# Callback Scheduling

## Trigger Conditions
- The user cannot stay online, asks to be contacted later, or needs a human callback.
- Route tags: callback

## Observe First
- get_claim_case
- get_customer_profile

## Recommended Action Order
1. Confirm that a follow-up contact is needed.
2. Read the user's contact channel.
3. Create a callback task and record a summary of the request.

## Allowed Write Actions
- create_callback_task
- append_case_note

## Guardrails
- Never promise a specific callback time in minutes.

## Output Constraints
- The reply only states that a callback task was created and what happens next in general terms.
- The reply must be grounded in observed facts; never add terms, timelines, amounts, or case milestones that do not exist.
- If the user asks several questions at once, split them first, then cover each one.
