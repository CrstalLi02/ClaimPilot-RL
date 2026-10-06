from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import List

from .llm_api import chat_json
from .schemas import AssistantReply, AuditLogEntry, CaseSpec, ReasoningStep, ToolObservation, Trajectory, VerifierResult


HARD_GUARDRAIL_PENALTIES = {
    "unauthorized_commitment": 1.0,
    "invalid_write": 1.0,
    "missed_mandatory_escalation": 1.0,
}

SOFT_PENALTIES = {
    "unnecessary_handoff": 0.3,
    "duplicate_side_effect": 0.2,
    "redundant_tool_loop": 0.2,
    "overlong_dialogue": 0.2,
}

# Matched against the lowercased reply text.
EXACT_PROMISE_PHRASES = [
    "guaranteed to arrive",
    "guarantee the payment",
    "guarantee you will be paid",
    "will definitely arrive",
    "will definitely be paid",
    "will definitely receive",
    "will certainly arrive",
    "will certainly be paid",
]

def verify_episode(case: CaseSpec, trajectory: Trajectory) -> VerifierResult:
    predicted_disposition = _infer_predicted_disposition(trajectory)
    hard_guardrails = _hard_guardrails(case, trajectory)
    soft_penalties = _soft_penalties(case, trajectory)

    rubric = {
        "outcome": _score_outcome(case, predicted_disposition, hard_guardrails),
        "policy": 0.0,
        "evidence": 0.0,
        "efficiency": _score_efficiency(trajectory),
        "communication": 0.0,
    }
    llm_rubric = _llm_rubric(case, trajectory, predicted_disposition) or {}
    for key in ["policy", "evidence", "communication"]:
        if key in llm_rubric and isinstance(llm_rubric[key], (int, float)):
            rubric[key] = round(float(llm_rubric[key]), 3)

    penalty_value = sum(HARD_GUARDRAIL_PENALTIES[x] for x in hard_guardrails) + sum(SOFT_PENALTIES[x] for x in soft_penalties)
    reward = 1.00 * rubric["outcome"] + 0.25 * rubric["policy"] + 0.15 * rubric["evidence"] + 0.10 * rubric["efficiency"] + 0.10 * rubric["communication"] - penalty_value
    reward = round(reward, 4)

    executed_success = [entry.tool_name for entry in trajectory.audit_log if entry.success]
    success = (
        predicted_disposition == case.ideal_disposition
        and all(action in executed_success for action in case.required_actions)
        and not hard_guardrails
    )

    reward_components = {
        "correct_resolution": round(1.0 if predicted_disposition == case.ideal_disposition and not hard_guardrails else 0.0, 4),
        "wrong_commitment": round(-1.0 if "unauthorized_commitment" in hard_guardrails else 0.0, 4),
        "unnecessary_handoff": round(-0.3 if "unnecessary_handoff" in soft_penalties else 0.0, 4),
        "few_turn_bonus": round(max(0.0, (6 - min(trajectory.turns, 6)) * 0.1), 4),
    }
    data_quality = _data_quality(case, trajectory, success, hard_guardrails, soft_penalties)

    return VerifierResult(
        case_id=case.case_id,
        expected_disposition=case.ideal_disposition,
        predicted_disposition=predicted_disposition,
        rubric=rubric,
        hard_guardrails=hard_guardrails,
        soft_penalties=soft_penalties,
        reward=reward,
        success=success,
        reward_components=reward_components,
        data_quality=data_quality,
        llm_rubric=llm_rubric or {},
    )


def _data_quality(case: CaseSpec, trajectory: Trajectory, success: bool, hard_guardrails: List[str], soft_penalties: List[str]):
    has_skills = bool(getattr(trajectory, "selected_skills", []))
    has_reasoning = bool(getattr(trajectory, "reasoning_steps", []))
    tool_steps = len([x for x in trajectory.audit_log if x.success])
    quality = 0.4
    if has_skills:
        quality += 0.2
    if has_reasoning:
        quality += 0.2
    if tool_steps >= 2:
        quality += 0.1
    if success:
        quality += 0.1
    quality -= 0.15 * len(hard_guardrails)
    quality -= 0.05 * len(soft_penalties)
    score = round(max(0.0, min(1.0, quality)), 3)
    label = "high" if score >= 0.8 else "medium" if score >= 0.55 else "low"
    return {
        "score": score,
        "label": label,
        "has_skills": has_skills,
        "has_reasoning_steps": has_reasoning,
        "tool_step_count": tool_steps,
    }


def _llm_rubric(case: CaseSpec, trajectory: Trajectory, predicted_disposition: str):
    reply = trajectory.assistant_reply.to_dict() if trajectory.assistant_reply else {}
    system_prompt = (
        "You are evaluating an insurance claims customer service trajectory. "
        "Output JSON with only the fields policy, evidence, communication, notes. "
        "policy, evidence, and communication range from 0 to 1, with up to 3 decimal places. "
        "Score only on the facts in the input; do not add invented facts. "
        "policy only evaluates coverage of required_actions, not tone, evidence, or the final outcome. 1.0 means all required_actions are covered; 0.7 means most are covered; 0.4 means only a few key actions are covered; 0.0 means almost none are covered. "
        "evidence only evaluates whether the action chain and the reply are grounded in facts: whether the key read tools were run, whether the final disposition matches the tool results, and whether the reply clearly departs from the tool results. 1.0 means a complete evidence chain; 0.7 means mostly grounded; 0.4 means weakly grounded; 0.0 means essentially ungrounded. "
        "communication only evaluates the reply text itself: whether it directly answers the user's current question, is clear, is safe, and avoids empty filler. 1.0 means clear and usable; 0.7 means mostly usable; 0.4 means vague or roundabout; 0.0 means essentially unusable. "
        "notes explains the deductions in one or two sentences."
    )
    payload = {
        "case_id": case.case_id,
        "family": case.family,
        "initial_user_message": case.initial_user_message,
        "ideal_disposition": case.ideal_disposition,
        "predicted_disposition": predicted_disposition,
        "subissues": case.subissues,
        "required_actions": case.required_actions,
        "claim_case": case.claim_case,
        "assistant_reply": reply,
        "tool_observations": [x.to_dict() for x in trajectory.tool_observations],
        "audit_log": [x.to_dict() for x in trajectory.audit_log],
    }
    return chat_json(system_prompt, str(payload), temperature=0.1)


def _infer_predicted_disposition(trajectory: Trajectory) -> str:
    if trajectory.assistant_reply:
        return trajectory.assistant_reply.disposition
    writes = [entry.tool_name for entry in trajectory.audit_log if entry.success]
    if "escalate_to_human_adjuster" in writes:
        return "escalate_human_adjuster"
    if "create_reconsideration_ticket" in writes:
        return "create_reconsideration_ticket"
    if "send_document_request" in writes:
        return "request_missing_documents"
    return "no_final_disposition"


def _hard_guardrails(case: CaseSpec, trajectory: Trajectory) -> List[str]:
    reply = trajectory.assistant_reply
    findings: List[str] = []
    if reply:
        text = reply.text.lower()
        if any(token in text for token in EXACT_PROMISE_PHRASES):
            if case.sla_rules.get("exact_promise_allowed") is False:
                findings.append("unauthorized_commitment")
    if any(not entry.success for entry in trajectory.audit_log):
        findings.append("invalid_write")
    if case.must_escalate and "escalate_to_human_adjuster" not in [entry.tool_name for entry in trajectory.audit_log if entry.success]:
        findings.append("missed_mandatory_escalation")
    return sorted(list(set(findings)))


def _soft_penalties(case: CaseSpec, trajectory: Trajectory) -> List[str]:
    findings: List[str] = []
    writes = [entry.tool_name for entry in trajectory.audit_log if entry.success]
    if case.safe_to_self_serve and "escalate_to_human_adjuster" in writes:
        findings.append("unnecessary_handoff")
    if any(entry.duplicate and entry.tool_name in {"send_document_request", "create_callback_task", "create_reconsideration_ticket", "escalate_to_human_adjuster"} for entry in trajectory.audit_log):
        findings.append("duplicate_side_effect")
    count = Counter(entry.tool_name for entry in trajectory.audit_log)
    if any(v > 2 for v in count.values()) or len(trajectory.audit_log) > len(case.required_actions) + 2:
        findings.append("redundant_tool_loop")
    if trajectory.turns > 7:
        findings.append("overlong_dialogue")
    return sorted(list(set(findings)))


def _score_outcome(case: CaseSpec, predicted_disposition: str, hard_guardrails: List[str]) -> float:
    if hard_guardrails:
        return -1.0
    if predicted_disposition == case.ideal_disposition:
        if predicted_disposition == "escalate_human_adjuster":
            return 0.6
        if predicted_disposition in {"create_reconsideration_ticket", "request_missing_documents"}:
            return 0.8
        return 1.0
    if predicted_disposition == "safe_clarification":
        return 0.2
    return -0.7


def _score_efficiency(trajectory: Trajectory) -> float:
    if trajectory.turns <= 5:
        return 1.0
    if trajectory.turns <= 6:
        return 0.7
    if trajectory.turns <= 7:
        return 0.4
    return 0.1


def _load_json(path: str):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl_record(path: str, index: int):
    target = max(0, int(index))
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            if line_no == target:
                return json.loads(line)
    raise IndexError(f"index out of range: {index}")


def _trajectory_from_dict(data: dict) -> Trajectory:
    assistant_reply = data.get("assistant_reply")
    return Trajectory(
        case_id=data["case_id"],
        user_message=data.get("user_message", ""),
        agent_name=data.get("agent_name", ""),
        tool_observations=[ToolObservation(**x) for x in data.get("tool_observations", [])],
        audit_log=[AuditLogEntry(**x) for x in data.get("audit_log", [])],
        assistant_reply=AssistantReply(**assistant_reply) if assistant_reply else None,
        selected_skills=data.get("selected_skills", []),
        selected_skill_cards=data.get("selected_skill_cards", []),
        skill_rationale=data.get("skill_rationale", {}),
        reasoning_steps=[ReasoningStep(**x) for x in data.get("reasoning_steps", [])],
        bucket=data.get("bucket", "train"),
        tags=data.get("tags", []),
        synthesis_meta=data.get("synthesis_meta", {}),
        turns=data.get("turns", 0),
    )


def _load_case_and_trajectory(case_file: str | None, trajectory_file: str | None, rollout_file: str | None, index: int):
    if rollout_file:
        record = _load_jsonl_record(rollout_file, index)
        if "case" in record and "trajectory" in record:
            case = CaseSpec.from_dict(record["case"])
            trajectory = _trajectory_from_dict(record["trajectory"])
            return case, trajectory
        raise ValueError(
            "This rollout record has no case and trajectory fields, so it cannot be passed to verify_episode directly. "
            "Use --case-file and --trajectory-file to pass a complete sample instead."
        )
    if not case_file or not trajectory_file:
        raise ValueError("Pass --case-file and --trajectory-file, or pass --rollout-file")
    case = CaseSpec.from_dict(_load_json(case_file))
    trajectory = _trajectory_from_dict(_load_json(trajectory_file))
    return case, trajectory


def main() -> None:
    parser = argparse.ArgumentParser(description="Score one insurance claim trajectory with verify_episode")
    parser.add_argument("--case-file", default="", help="A single case JSON file")
    parser.add_argument("--trajectory-file", default="", help="A single trajectory JSON file")
    parser.add_argument("--rollout-file", default="", help="Read one record from rl_rollouts.jsonl and take its case and trajectory")
    parser.add_argument("--index", type=int, default=0, help="Record index to use with --rollout-file")
    args = parser.parse_args()

    case, trajectory = _load_case_and_trajectory(args.case_file, args.trajectory_file, args.rollout_file, args.index)
    result = verify_episode(case, trajectory)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
