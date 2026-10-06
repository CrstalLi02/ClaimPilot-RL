---
name: unsupported_claim_safe_reply
description: Safe Reply for Out-of-Scope Requests. Triggered when: The user asks the agent to change the payout decision directly, skip the review, or alter the terms-based outcome. Always verify the case facts first, then follow the skill steps to compose the reply.
---

# Safe Reply for Out-of-Scope Requests

## Trigger Conditions
- The user asks the agent to change the payout decision directly, skip the review, or alter the terms-based outcome.
- Route tags: unsupported

## Observe First
- get_claim_case
- search_claim_kb

## Recommended Action Order
1. Recognize the out-of-scope request.
2. Explain the limits of what customer service can do.
3. Offer a workable alternative path, such as reconsideration, submitting documents, or review by a human adjuster.

## Allowed Write Actions
- append_case_note

## Guardrails
- Never agree to change the decision directly.

## Output Constraints
- The reply declines the out-of-scope action while keeping a path for the next step.
- The reply must be grounded in observed facts; never add terms, timelines, amounts, or case milestones that do not exist.
- If the user asks several questions at once, split them first, then cover each one.
