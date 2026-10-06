---
name: partial_payment_breakdown
description: Partial Payment Breakdown. Triggered when: The user asks why only part of the claim was paid and wants to see how it was calculated. Always verify the case facts first, then follow the skill steps to compose the reply.
---

# Partial Payment Breakdown

## Trigger Conditions
- The user asks why only part of the claim was paid and wants to see how it was calculated.
- Route tags: payment_reason, payment_breakdown

## Observe First
- get_claim_case
- get_payment_breakdown
- get_policy_terms

## Recommended Action Order
1. Read the payout status and the payment breakdown.
2. Explain the eligible amount, the deductible, and any amount over the limit.
3. Map the result back to the gap the user mentioned between total expenses and the amount actually paid.

## Allowed Write Actions
- append_case_note

## Guardrails
- Never explain amounts before reading the breakdown.
- Never describe the reason for non-payment as a system failure.

## Output Constraints
- The reply includes the total amount, the eligible amount, and the deductible or over-limit reason.
- The reply must be grounded in observed facts; never add terms, timelines, amounts, or case milestones that do not exist.
- If the user asks several questions at once, split them first, then cover each one.
