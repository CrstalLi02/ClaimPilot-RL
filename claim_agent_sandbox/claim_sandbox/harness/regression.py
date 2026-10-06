from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from ..case_loader import load_cases
from ..exporters import build_reward_ledger_row
from ..schemas import CaseSpec
from .serving import ServingHarnessSession, ServingRunResult


@dataclass
class RegressionSummary:
    total_cases: int
    success_count: int
    avg_reward: float
    by_case: List[Dict] = field(default_factory=list)
    by_bucket: Dict[str, Dict] = field(default_factory=dict)
    terminated_reason_counts: Dict[str, int] = field(default_factory=dict)
    hard_guardrail_counts: Dict[str, int] = field(default_factory=dict)
    soft_penalty_counts: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "total_cases": self.total_cases,
            "success_count": self.success_count,
            "avg_reward": self.avg_reward,
            "by_case": self.by_case,
            "by_bucket": self.by_bucket,
            "terminated_reason_counts": self.terminated_reason_counts,
            "hard_guardrail_counts": self.hard_guardrail_counts,
            "soft_penalty_counts": self.soft_penalty_counts,
        }


def run_regression_cases(cases: Sequence[CaseSpec], max_steps: int = 12, temperature: float = 0.2) -> Tuple[List[ServingRunResult], RegressionSummary]:
    runs: List[ServingRunResult] = []
    rows: List[Dict] = []
    bucket_rows: Dict[str, List[Dict]] = {}
    terminated_reason_counts: Counter = Counter()
    hard_guardrail_counts: Counter = Counter()
    soft_penalty_counts: Counter = Counter()
    for case in cases:
        session = ServingHarnessSession(case, temperature=temperature)
        session.start_user_turn(case.initial_user_message)
        result = session.run_until_respond(max_steps=max_steps)
        runs.append(result)
        terminated_reason_counts[result.terminated_reason] += 1
        hard_guardrail_counts.update(result.verifier.hard_guardrails)
        soft_penalty_counts.update(result.verifier.soft_penalties)
        rows.append({
            **build_reward_ledger_row(case, "serving_harness", result.trajectory, result.verifier),
            "bucket": getattr(case, "bucket", "train"),
            "terminated_reason": result.terminated_reason,
            "control_plane_stats": result.control_plane_stats,
        })
        bucket_rows.setdefault(getattr(case, "bucket", "train"), []).append(rows[-1])
    avg_reward = round(sum(float(x["reward"]) for x in rows) / len(rows), 4) if rows else 0.0
    by_bucket = {}
    for bucket, items in bucket_rows.items():
        by_bucket[bucket] = {
            "total_cases": len(items),
            "success_count": sum(1 for x in items if x["success"]),
            "avg_reward": round(sum(float(x["reward"]) for x in items) / len(items), 4) if items else 0.0,
        }
    summary = RegressionSummary(
        total_cases=len(rows),
        success_count=sum(1 for x in rows if x["success"]),
        avg_reward=avg_reward,
        by_case=rows,
        by_bucket=by_bucket,
        terminated_reason_counts=dict(terminated_reason_counts),
        hard_guardrail_counts=dict(hard_guardrail_counts),
        soft_penalty_counts=dict(soft_penalty_counts),
    )
    return runs, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed-case serving regression")
    parser.add_argument("--limit-cases", type=int, default=0)
    parser.add_argument("--outdir", default="")
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()

    cases = load_cases()
    if args.limit_cases and args.limit_cases > 0:
        cases = cases[: args.limit_cases]
    runs, summary = run_regression_cases(cases, max_steps=args.max_steps, temperature=args.temperature)

    if args.outdir:
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        with (outdir / "serving_regression_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary.to_dict(), f, ensure_ascii=False, indent=2)
        with (outdir / "serving_regression_runs.jsonl").open("w", encoding="utf-8") as f:
            for run in runs:
                f.write(json.dumps(run.to_dict(), ensure_ascii=False))
                f.write("\n")
        replay_dir = outdir / "replay_cases"
        replay_dir.mkdir(parents=True, exist_ok=True)
        for run in runs:
            target = replay_dir / f"{run.case.case_id}.json"
            target.write_text(json.dumps(run.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
