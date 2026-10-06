from __future__ import annotations

from typing import Any, Dict, List

from .prompt_format import build_turn_samples
from .schemas import CaseSpec, Trajectory, VerifierResult


def build_trajectory_record(
    case: CaseSpec,
    agent_name: str,
    trajectory: Trajectory,
    verifier: VerifierResult,
    hindsight_skill: str,
) -> Dict[str, Any]:
    return {
        "case_id": case.case_id,
        "family": case.family,
        "difficulty": case.difficulty,
        "agent": agent_name,
        "bucket": getattr(case, "bucket", "train"),
        "tags": getattr(case, "tags", []),
        "user_message": case.initial_user_message,
        "selected_skills": trajectory.selected_skills,
        "trajectory": trajectory.to_dict(),
        "verifier": verifier.to_dict(),
        "hindsight_skill": hindsight_skill,
    }


def build_reward_ledger_row(
    case: CaseSpec,
    agent_name: str,
    trajectory: Trajectory,
    verifier: VerifierResult,
) -> Dict[str, Any]:
    return {
        "case_id": case.case_id,
        "family": case.family,
        "difficulty": case.difficulty,
        "agent": agent_name,
        "expected_disposition": verifier.expected_disposition,
        "predicted_disposition": verifier.predicted_disposition,
        "selected_skills": trajectory.selected_skills,
        "hard_guardrails": verifier.hard_guardrails,
        "soft_penalties": verifier.soft_penalties,
        "reward_components": verifier.reward_components,
        "data_quality": verifier.data_quality,
        "reward": verifier.reward,
        "success": verifier.success,
    }


def build_hindsight_sft_record(
    case: CaseSpec,
    trajectory: Trajectory,
    verifier: VerifierResult,
    hindsight_skill: str,
) -> Dict[str, Any]:
    return {
        "id": case.case_id,
        "task": case.initial_user_message,
        "input": serialize_trajectory_analysis_input(case, trajectory, verifier),
        "target": hindsight_skill,
    }


def build_policy_sft_records(
    case: CaseSpec,
    trajectory: Trajectory,
    verifier: VerifierResult,
) -> List[Dict[str, Any]]:
    rows = []
    conversation_id = f"{case.case_id}__{trajectory.agent_name or 'unknown'}"
    for idx, sample in enumerate(build_turn_samples(case, trajectory), start=1):
        rows.append({
            "id": conversation_id,
            "turn_id": sample.get("meta", {}).get("turn_id", idx),
            "input": sample["input"],
            "target": sample["target"],
            "meta": {
                **sample.get("meta", {}),
                "family": case.family,
                "difficulty": case.difficulty,
                "success": verifier.success,
            },
        })
    return rows


def build_rl_rollout_record(
    case: CaseSpec,
    agent_name: str,
    trajectory: Trajectory,
    verifier: VerifierResult,
) -> Dict[str, Any]:
    policy_samples = build_turn_samples(case, trajectory)
    shaped_samples = []
    conversation_id = f"{case.case_id}__{agent_name or trajectory.agent_name or 'unknown'}"
    last_idx = max(0, len(policy_samples) - 1)
    for idx, sample in enumerate(policy_samples):
        shaped_samples.append({
            "turn_id": sample.get("meta", {}).get("turn_id", idx + 1),
            "input": sample["input"],
            "target": sample["target"],
            "rollout_output": sample["target"],
            "reward": verifier.reward if idx == last_idx else None,
        })
    return {
        "id": conversation_id,
        "agent": agent_name,
        "predicted_disposition": verifier.predicted_disposition,
        "expected_disposition": verifier.expected_disposition,
        "selected_skills": trajectory.selected_skills,
        "policy_samples": shaped_samples,
        "reward_components": verifier.reward_components,
        "data_quality": verifier.data_quality,
        "reward": verifier.reward,
        "success": verifier.success,
        "hard_guardrails": verifier.hard_guardrails,
        "soft_penalties": verifier.soft_penalties,
    }


def serialize_trajectory_analysis_input(
    case: CaseSpec,
    trajectory: Trajectory,
    verifier: VerifierResult,
) -> str:
    lines: List[str] = []
    lines.append(f"Task: {case.initial_user_message}")
    lines.append(f"Case ID: {case.case_id}")
    lines.append(f"Family: {case.family}")
    lines.append(f"Expected disposition: {verifier.expected_disposition}")
    lines.append(f"Predicted disposition: {verifier.predicted_disposition}")
    lines.append("Tool trajectory:")
    for idx, obs in enumerate(trajectory.tool_observations, start=1):
        lines.append(f"  {idx}. {obs.tool_name} -> success={obs.success} -> output={obs.output}")
    if trajectory.assistant_reply:
        lines.append("Assistant reply:")
        lines.append(trajectory.assistant_reply.text)
        lines.append(f"Addressed subissues: {trajectory.assistant_reply.addressed_subissues}")
        lines.append(f"Cited facts: {trajectory.assistant_reply.cited_facts}")
    lines.append(f"Verifier hard guardrails: {verifier.hard_guardrails}")
    lines.append(f"Verifier soft penalties: {verifier.soft_penalties}")
    lines.append(f"Reward: {verifier.reward}")
    return "\n".join(lines)
