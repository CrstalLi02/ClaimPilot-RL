---
name: deductible_explanation
description: Deductible Explanation. Triggered when: The user disputes the deductible, the payout ratio, or the reimbursement ratio. Always verify the case facts first, then follow the skill steps to compose the reply.
---

# Deductible Explanation

## Trigger Conditions
- The user disputes the deductible, the payout ratio, or the reimbursement ratio.
- Route tags: deductible_or_ratio

## Observe First
- get_payment_breakdown
- get_policy_terms

## Recommended Action Order
1. Read the payment breakdown and the policy terms.
2. Point out how the deductible or payout ratio reduces the amount.
3. Explain that this is set by the policy terms, not an arbitrary manual adjustment.

## Allowed Write Actions
- append_case_note

## Guardrails
- Never describe the deductible as a handling fee.

## Output Constraints
- State the deductible amount and the reason for the reduction explicitly.
- The reply must be grounded in observed facts; never add terms, timelines, amounts, or case milestones that do not exist.
- If the user asks several questions at once, split them first, then cover each one.
