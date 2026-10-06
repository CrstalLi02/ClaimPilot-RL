---
name: policy_term_grounding
description: Policy Term Grounding. Triggered when: The user questions the basis for a payout, a rejection, a timeline, or a handling rule. Always verify the case facts first, then follow the skill steps to compose the reply.
---

# Policy Term Grounding

## Trigger Conditions
- The user questions the basis for a payout, a rejection, a timeline, or a handling rule.
- Route tags: terms_basis, coverage_basis

## Observe First
- get_policy_terms
- get_policy_summary
- search_claim_kb

## Recommended Action Order
1. Read the policy terms first.
2. Translate the term rules into plain language an agent can say to a customer.
3. If the explanation may cause a dispute, append a case note recording the grounds that were explained.

## Allowed Write Actions
- append_case_note

## Guardrails
- Never invent clause numbers or coverage definitions that do not exist.

## Output Constraints
- The reply must include the policy basis and a plain-language explanation.
- The reply must be grounded in observed facts; never add terms, timelines, amounts, or case milestones that do not exist.
- If the user asks several questions at once, split them first, then cover each one.
