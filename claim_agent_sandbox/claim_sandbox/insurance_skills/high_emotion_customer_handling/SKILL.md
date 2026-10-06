---
name: high_emotion_customer_handling
description: High-Emotion Customer Handling. Triggered when: The user is clearly angry, anxious, or repeatedly applying pressure. Always verify the case facts first, then follow the skill steps to compose the reply.
---

# High-Emotion Customer Handling

## Trigger Conditions
- The user is clearly angry, anxious, or repeatedly applying pressure.
- Route tags: emotion

## Observe First
- get_customer_profile
- get_claim_case

## Recommended Action Order
1. Recognize the emotion first.
2. In the reply, first confirm which specific items have been checked, then move on to the facts.
3. Steer high-pressure language back to facts, grounds, and next steps.

## Allowed Write Actions
- append_case_note

## Guardrails
- Never argue with the user.
- Never drop the grounds for a conclusion just because the user is upset.

## Output Constraints
- Keep the tone measured and the reply information-dense.
- The reply must be grounded in observed facts; never add terms, timelines, amounts, or case milestones that do not exist.
- If the user asks several questions at once, split them first, then cover each one.
