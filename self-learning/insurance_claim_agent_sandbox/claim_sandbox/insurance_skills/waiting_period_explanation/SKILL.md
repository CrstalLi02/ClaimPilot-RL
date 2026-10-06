---
name: waiting_period_explanation
description: Waiting Period Explanation. Triggered when: The user asks why the claim was rejected, and the reason involves the waiting period. Always verify the case facts first, then follow the skill steps to compose the reply.
---

# Waiting Period Explanation

## Trigger Conditions
- The user asks why the claim was rejected, and the reason involves the waiting period.
- Route tags: rejection_reason, waiting_period

## Observe First
- get_claim_case
- get_policy_terms
- get_claim_timeline

## Recommended Action Order
1. Read the case status and the rejection reason.
2. Read the waiting-period constraint in the policy terms.
3. Use the timeline to explain how the effective date relates to the date of the incident.
4. Give a factual explanation; do not bring up reconsideration first.

## Allowed Write Actions
- append_case_note

## Guardrails
- Never skip the timeline and simply say the claim is not covered.
- Never present the waiting period as a final exclusion without explaining the grounds.

## Output Constraints
- The reply covers the timeline, the policy terms, and the conclusion.
- The reply must be grounded in observed facts; never add terms, timelines, amounts, or case milestones that do not exist.
- If the user asks several questions at once, split them first, then cover each one.
