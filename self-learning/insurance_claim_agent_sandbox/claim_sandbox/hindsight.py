from __future__ import annotations

import time

from .exporters import serialize_trajectory_analysis_input
from .llm_api import chat_json
from .schemas import CaseSpec, Trajectory, VerifierResult


HINDSIGHT_SYSTEM_PROMPT = (
    "You are generating a hindsight skill for an insurance claims customer service agent. "
    "The input is the task, tool trajectory, final reply, and verifier result of one completed episode. "
    "Output JSON with a single field: skill. "
    "skill is one reusable rule of experience, one to three sentences in English, about one of the following three kinds: "
    "workflow skill (what process to follow), "
    "decisive observation skill (which observations determined the next actions), "
    "failure-avoidance skill (which behaviors lead to risk or failure). "
    "Use only facts from the input; do not invent details that are not in the trajectory."
)


def generate_hindsight_skill(case: CaseSpec, trajectory: Trajectory, verifier: VerifierResult) -> str:
    return _fallback_hindsight_skill(case, verifier)


def _llm_hindsight_skill(case: CaseSpec, trajectory: Trajectory, verifier: VerifierResult) -> str | None:
    analysis_input = serialize_trajectory_analysis_input(case, trajectory, verifier)
    for attempt in range(3):
        result = chat_json(HINDSIGHT_SYSTEM_PROMPT, analysis_input, temperature=0.3, timeout=30)
        if isinstance(result, dict):
            skill = result.get("skill")
            if isinstance(skill, str) and skill.strip():
                return skill.strip()
        if attempt < 2:
            time.sleep(1)
    return None


def _fallback_hindsight_skill(case: CaseSpec, verifier: VerifierResult) -> str:
    if verifier.success:
        if case.family == "pending_docs":
            return "Check the case status first, then compare uploaded and missing documents, and then explain the review time range; do not promise a payment time before the payment stage has been verified."
        if case.family == "rejection_dispute":
            return "In a rejection dispute, pull the policy terms first, then explain the key limiting conditions; if the user clearly disagrees, also offer the reconsideration path and create a reconsideration ticket."
        if case.family == "high_risk_review":
            return "When a case has document conflicts, manual review, or high-risk flags, consult the escalation matrix first and escalate to a human adjuster instead of continuing to make self-service commitments."
        if case.family == "partial_payment":
            return "Before explaining a partial payout, look up the payment breakdown and the policy terms, then explain how the deductible, ratio, or benefit limit affected the amount."
    if "unauthorized_commitment" in verifier.hard_guardrails:
        return "Never promise a specific payment time before verifying the SLA and the payment stage; give a time range instead of a firm promise."
    if "missed_mandatory_escalation" in verifier.hard_guardrails:
        return "Whenever there is manual review, a high-risk flag, or a document conflict, hand over to a human according to the escalation matrix instead of closing the case via self-service."
    if "subissue_omission" in verifier.soft_penalties:
        return "When the user asks several questions at once, split them into sub-issues and cover each one instead of answering only one."
    return "After the trajectory, review which lookups provided key evidence and which conclusions were ungrounded; next time, look things up before answering."
