from __future__ import annotations

import argparse
import json
import json.encoder
import tempfile
from pathlib import Path
from typing import Dict, List

from claim_sandbox.case_loader import load_cases, load_cases_from_dir
from claim_sandbox.exporters import build_policy_sft_records
from claim_sandbox.orchestrator import HarnessOrchestrator
from claim_sandbox.synthesis import synthesize_cases


class FixedFloatEncoder(json.JSONEncoder):
    def iterencode(self, o, _one_shot=False):
        markers = {} if self.check_circular else None
        _encoder = json.encoder.encode_basestring_ascii if self.ensure_ascii else json.encoder.encode_basestring

        def floatstr(value, allow_nan=self.allow_nan, _inf=json.encoder.INFINITY, _neginf=-json.encoder.INFINITY):
            if value != value:
                text = "NaN"
            elif value == _inf:
                text = "Infinity"
            elif value == _neginf:
                text = "-Infinity"
            else:
                return format(value, ".4f")
            if not allow_nan:
                raise ValueError("Out of range float values are not JSON compliant")
            return text

        _iterencode = json.encoder._make_iterencode(
            markers,
            self.default,
            _encoder,
            self.indent,
            floatstr,
            self.key_separator,
            self.item_separator,
            self.sort_keys,
            self.skipkeys,
            _one_shot,
        )
        return _iterencode(o, 0)


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    encoder = FixedFloatEncoder(ensure_ascii=False)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(encoder.encode(row))
            f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build insurance claim sandbox datasets")
    parser.add_argument("--agents", nargs="+", default=["rule_based", "llm_skill", "risky_shortcut", "over_escalating"])
    parser.add_argument("--outdir", default="artifacts/final_dataset")
    parser.add_argument("--num-variants", type=int, default=1)
    parser.add_argument("--disable-llm-variants", action="store_true")
    parser.add_argument("--limit-cases", type=int, default=0)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rl_rows: List[Dict] = []
    policy_sft_rows: List[Dict] = []
    orchestrator = HarnessOrchestrator()

    base_cases = load_cases()
    if args.limit_cases and args.limit_cases > 0:
        base_cases = base_cases[: args.limit_cases]
    all_cases = list(base_cases)
    with tempfile.TemporaryDirectory(prefix="claim_variants_") as tmpdir:
        synthesize_cases(tmpdir, num_variants_per_case=args.num_variants, use_llm=not args.disable_llm_variants)
        all_cases.extend(load_cases_from_dir(tmpdir))

        for case in all_cases:
            for agent_name in args.agents:
                run = orchestrator.run_case(case, agent_name)
                rl_rows.append(run.rl_rollout)
                policy_sft_rows.extend(build_policy_sft_records(run.case, run.trajectory, run.verifier))

    write_jsonl(outdir / "policy_sft.jsonl", policy_sft_rows)
    write_jsonl(outdir / "rl_rollouts.jsonl", rl_rows)

    print(json.dumps({
        "outdir": str(outdir),
        "num_rows": {
            "policy_sft": len(policy_sft_rows),
            "rl_rollouts": len(rl_rows),
        }
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
