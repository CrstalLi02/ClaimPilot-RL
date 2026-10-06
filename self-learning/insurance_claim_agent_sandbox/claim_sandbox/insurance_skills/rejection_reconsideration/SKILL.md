---
name: rejection_reconsideration
description: Rejection Reconsideration. Triggered when: The user disagrees with a rejection and asks to appeal or request reconsideration. Always verify the case facts first, then follow the skill steps to compose the reply.
---

# Rejection Reconsideration

## Trigger Conditions
- The user disagrees with a rejection and asks to appeal or request reconsideration.
- Route tags: reconsideration_path, rejection_dispute

## Observe First
- get_claim_case
- get_policy_terms
- get_escalation_matrix

## Recommended Action Order
1. Confirm the current status is rejected.
2. Explain the grounds for the current decision.
3. Determine whether reconsideration can be filed.
4. Create a reconsideration ticket and record the new evidence that needs to be provided.

## Allowed Write Actions
- create_reconsideration_ticket
- append_case_note

## Guardrails
- Never discourage the user from requesting reconsideration.
- Never create a reconsideration ticket before reading the case status.

## Output Constraints
- The reply explains how to file for reconsideration, which documents to add, and what happens afterwards.
- The reply must be grounded in observed facts; never add terms, timelines, amounts, or case milestones that do not exist.
- If the user asks several questions at once, split them first, then cover each one.
