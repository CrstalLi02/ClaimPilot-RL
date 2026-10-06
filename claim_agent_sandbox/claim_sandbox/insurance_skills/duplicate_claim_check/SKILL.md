---
name: duplicate_claim_check
description: Duplicate Claim Check. Triggered when: The user says they already submitted, the system flags a duplicate, or the user worries about uploading twice. Always verify the case facts first, then follow the skill steps to compose the reply.
---

# Duplicate Claim Check

## Trigger Conditions
- The user says they already submitted, the system flags a duplicate, or the user worries about uploading twice.
- Route tags: duplicate_claim

## Observe First
- get_claim_case
- get_claim_timeline
- search_claim_kb

## Recommended Action Order
1. Review the case timeline.
2. Determine whether an identical claim is already being processed.
3. Explain the risk of duplicate submission and the correct next step.

## Allowed Write Actions
- append_case_note

## Guardrails
- Never accuse the user of duplicate claim fraud without evidence.

## Output Constraints
- The reply clearly identifies the existing claim and the recommended next step.
- The reply must be grounded in observed facts; never add terms, timelines, amounts, or case milestones that do not exist.
- If the user asks several questions at once, split them first, then cover each one.
