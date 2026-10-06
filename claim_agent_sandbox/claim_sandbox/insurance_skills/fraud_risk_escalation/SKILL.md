---
name: fraud_risk_escalation
description: Risk Case Escalation. Triggered when: The case carries risk flags such as invoice conflicts, document conflicts, or suspected fraud. Always verify the case facts first, then follow the skill steps to compose the reply.
---

# Risk Case Escalation

## Trigger Conditions
- The case carries risk flags such as invoice conflicts, document conflicts, or suspected fraud.
- Route tags: risk_flags, discrepancy

## Observe First
- get_claim_case
- get_escalation_matrix
- search_claim_kb

## Recommended Action Order
1. Identify the risk_flags.
2. Consult the escalation matrix.
3. Explain that there is a document conflict that requires a human adjuster.
4. Write a case note.

## Allowed Write Actions
- escalate_to_human_adjuster
- append_case_note

## Guardrails
- Never tell the user outright that the case is fraud.
- Never continue self-service handling of a high-risk case on your own.

## Output Constraints
- The reply only mentions a document conflict or risk review, never a legal characterization.
- The reply must be grounded in observed facts; never add terms, timelines, amounts, or case milestones that do not exist.
- If the user asks several questions at once, split them first, then cover each one.
