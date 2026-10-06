from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from ..agents import build_agent
from ..exporters import build_hindsight_sft_record, build_reward_ledger_row, build_rl_rollout_record, build_trajectory_record
from ..hindsight import generate_hindsight_skill
from ..schemas import CaseSpec, Trajectory, VerifierResult
from ..verifier import verify_episode
from .env import ClaimSandboxEnv, run_episode


@dataclass
class OrchestratorRun:
    case: CaseSpec
    trajectory: Trajectory
    verifier: VerifierResult
    trajectory_record: Dict
    reward_ledger: Dict
    hindsight_sft: Dict
    rl_rollout: Dict


@dataclass
class BatchRunSummary:
    total_runs: int
    success_count: int
    avg_reward: float
    avg_data_quality: float
    by_agent: Dict[str, Dict]


class HarnessOrchestrator:
    def run_case(self, case: CaseSpec, agent_name: str) -> OrchestratorRun:
        agent = build_agent(agent_name)
        env = ClaimSandboxEnv(case)
        env.trajectory.agent_name = agent_name
        trajectory = run_episode(agent, env)
        verifier = verify_episode(case, trajectory)
        hindsight_skill = generate_hindsight_skill(case, trajectory, verifier)
        return OrchestratorRun(
            case=case,
            trajectory=trajectory,
            verifier=verifier,
            trajectory_record=build_trajectory_record(case, agent_name, trajectory, verifier, hindsight_skill),
            reward_ledger=build_reward_ledger_row(case, agent_name, trajectory, verifier),
            hindsight_sft=build_hindsight_sft_record(case, trajectory, verifier, hindsight_skill),
            rl_rollout=build_rl_rollout_record(case, agent_name, trajectory, verifier),
        )

    def score_rollout_record(self, trajectory_record: Dict) -> Dict:
        verifier = trajectory_record.get("verifier", {})
        return {
            "case_id": trajectory_record.get("case_id"),
            "agent": trajectory_record.get("agent"),
            "reward": verifier.get("reward"),
            "success": verifier.get("success"),
            "data_quality": verifier.get("data_quality", {}),
        }

    def run_batch(self, cases: Iterable[CaseSpec], agent_names: List[str], bucket: Optional[str] = None) -> List[OrchestratorRun]:
        outputs: List[OrchestratorRun] = []
        for case in cases:
            if bucket and getattr(case, "bucket", "train") != bucket:
                continue
            for agent_name in agent_names:
                outputs.append(self.run_case(case, agent_name))
        return outputs

    def summarize_runs(self, runs: List[OrchestratorRun]) -> BatchRunSummary:
        if not runs:
            return BatchRunSummary(total_runs=0, success_count=0, avg_reward=0.0, avg_data_quality=0.0, by_agent={})
        rewards = [float(run.reward_ledger["reward"]) for run in runs]
        qualities = [float(run.reward_ledger.get("data_quality", {}).get("score", 0.0)) for run in runs]
        by_agent: Dict[str, Dict] = {}
        for run in runs:
            agent = run.reward_ledger["agent"]
            item = by_agent.setdefault(agent, {"count": 0, "success": 0, "reward_sum": 0.0, "quality_sum": 0.0})
            item["count"] += 1
            item["success"] += 1 if run.reward_ledger["success"] else 0
            item["reward_sum"] += float(run.reward_ledger["reward"])
            item["quality_sum"] += float(run.reward_ledger.get("data_quality", {}).get("score", 0.0))
        for _, item in by_agent.items():
            item["avg_reward"] = round(item["reward_sum"] / item["count"], 4)
            item["avg_data_quality"] = round(item["quality_sum"] / item["count"], 4)
        return BatchRunSummary(
            total_runs=len(runs),
            success_count=sum(1 for run in runs if run.reward_ledger["success"]),
            avg_reward=round(sum(rewards) / len(rewards), 4),
            avg_data_quality=round(sum(qualities) / len(qualities), 4),
            by_agent=by_agent,
        )
