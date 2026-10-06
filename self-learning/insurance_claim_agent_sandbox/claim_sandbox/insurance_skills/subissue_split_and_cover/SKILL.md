---
name: subissue_split_and_cover
description: Sub-issue Split and Cover. Triggered when: In a single message the user asks about several sub-issues at once, such as the reason, documents, timeline, reconsideration, and amount. Always verify the case facts first, then follow the skill steps to compose the reply.
---

# Sub-issue Split and Cover

## Trigger Conditions
- In a single message the user asks about several sub-issues at once, such as the reason, documents, timeline, reconsideration, and amount.
- Route tags: multi_issue

## Observe First
- get_claim_case
- search_claim_kb

## Recommended Action Order
1. Split the user's question into a list of sub-issues.
2. Order them by priority.
3. Cover them one by one so nothing is missed.
4. When needed, record the handled sub-issues in a case note.

## Allowed Write Actions
- append_case_note

## Guardrails
- Never answer only the first question.

## Output Constraints
- Structure the reply with one section per sub-issue.
- The reply must be grounded in observed facts; never add terms, timelines, amounts, or case milestones that do not exist.
- If the user asks several questions at once, split them first, then cover each one.
