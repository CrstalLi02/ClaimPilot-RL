from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..case_loader import load_case
from ..llm_api import chat_json_detailed
from ..prompt_format import build_input_text
from ..schemas import AssistantReply, AuditLogEntry, CaseSpec, ReasoningStep, ToolObservation, Trajectory, VerifierResult
from ..skill_router import select_skills_for_case
from ..tools import execute_tool
from ..verifier import verify_episode


@dataclass
class ServingRunResult:
    case: CaseSpec
    trajectory: Trajectory
    verifier: VerifierResult
    prompt_trace: List[Dict[str, Any]] = field(default_factory=list)
    terminated_reason: str = "completed"
    world_state_delta: Dict[str, Any] = field(default_factory=dict)
    control_plane_stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case": self.case.__dict__,
            "trajectory": self.trajectory.to_dict(),
            "verifier": self.verifier.to_dict(),
            "prompt_trace": self.prompt_trace,
            "terminated_reason": self.terminated_reason,
            "world_state_delta": self.world_state_delta,
            "control_plane_stats": self.control_plane_stats,
        }


class ServingHarnessSession:
    def __init__(self, case: CaseSpec, temperature: float = 0.2):
        self.case = case
        self.temperature = temperature
        selected_cards, rationale = select_skills_for_case(case, case.initial_user_message)
        self.available_skill_cards = [x.to_dict() for x in selected_cards]
        self._skill_by_slug = {x["slug"]: x for x in self.available_skill_cards}
        self._allowed_actions = {
            slug: set(card.get("read_tools", []) + card.get("write_tools", []))
            for slug, card in self._skill_by_slug.items()
        }
        self.trajectory = Trajectory(
            case_id=case.case_id,
            user_message=case.initial_user_message,
            agent_name="serving_harness",
            selected_skills=[x["slug"] for x in self.available_skill_cards],
            selected_skill_cards=self.available_skill_cards,
            skill_rationale=rationale,
            bucket=getattr(case, "bucket", "train"),
            tags=getattr(case, "tags", []),
            synthesis_meta={**getattr(case, "synthesis_meta", {}), "runtime_mode": "serving_harness"},
        )
        self._writes: Dict[str, Any] = {}
        self._history_acc: List[str] = []
        self._loaded_skill_cards_acc: List[Dict[str, Any]] = []
        self._loaded_skill_slugs_acc = set()
        self._observation_acc: List[Dict[str, Any]] = []
        self._current_round_index = 0
        self._current_user_text: Optional[str] = None
        self._current_skill: Optional[str] = None
        self._phase = "idle"
        self._tool_cache: Dict[Tuple[str, str], Tuple[bool, Dict[str, Any], str]] = {}
        self._side_effect_tools = {"send_document_request", "append_case_note", "create_callback_task", "escalate_to_human_adjuster", "create_reconsideration_ticket", "create_claim_intake"}
        self._round_tool_budget = self._resolve_round_tool_budget()
        self._episode_tool_budget = self._resolve_episode_tool_budget()
        self._current_round_tool_calls = 0
        self._last_llm_error: str = ""
        self.prompt_trace: List[Dict[str, Any]] = []

    def start_user_turn(self, user_text: Optional[str] = None) -> None:
        self._current_round_index += 1
        self._current_user_text = (user_text or self.case.initial_user_message).strip()
        self._current_skill = None
        self._current_round_tool_calls = 0
        self._phase = "expect_skill"

    def build_current_input(self) -> str:
        if not self._current_user_text:
            raise ValueError("There is no pending user input")
        history_before = self._history_acc + [f"User: {self._current_user_text}"]
        return build_input_text(
            self.case,
            available_skill_cards=self.available_skill_cards,
            history_lines=history_before,
            loaded_skill_cards=self._loaded_skill_cards_acc,
            recent_observations=self._observation_acc,
        )

    def run_until_respond(self, max_steps: int = 12) -> ServingRunResult:
        if self._phase == "idle":
            self.start_user_turn(self.case.initial_user_message)
        terminated_reason = "completed"
        for _ in range(max_steps):
            input_text = self.build_current_input()
            raw, llm_error = self._sample_next_action(input_text)
            valid, normalized, error = self._validate_action(raw)
            self.prompt_trace.append({
                "round_index": self._current_round_index,
                "phase": self._phase,
                "input": input_text,
                "model_output": raw,
                "llm_error": llm_error,
                "valid": valid,
                "normalized_action": normalized if valid else None,
                "error": error,
            })
            if not valid:
                final_error = error or llm_error or "invalid_output"
                self._record_failsafe_reply(final_error)
                terminated_reason = final_error
                break
            if self._apply_action(normalized):
                break
        else:
            self._record_failsafe_reply("max_steps_exceeded")
            terminated_reason = "max_steps_exceeded"

        verifier = verify_episode(self.case, self.trajectory)
        return ServingRunResult(
            case=self.case,
            trajectory=self.trajectory,
            verifier=verifier,
            prompt_trace=self.prompt_trace,
            terminated_reason=terminated_reason,
            world_state_delta=self._world_state_delta(),
            control_plane_stats=self._control_plane_stats(),
        )

    def _world_state_delta(self) -> Dict[str, Any]:
        return json.loads(json.dumps(self._writes, ensure_ascii=False)) if self._writes else {}

    def _control_plane_stats(self) -> Dict[str, Any]:
        read_calls = 0
        write_calls = 0
        cache_hits = 0
        duplicate_write_blocks = 0
        blocked_responds = 0
        budget_blocks = 0
        for entry in self.trajectory.audit_log:
            if entry.tool_name in self._side_effect_tools:
                write_calls += 1
                if entry.message == "duplicate_side_effect_blocked":
                    duplicate_write_blocks += 1
            else:
                read_calls += 1
                if entry.message.startswith("cache_hit:"):
                    cache_hits += 1
        for item in self.prompt_trace:
            error = item.get("error") or ""
            if isinstance(error, str) and error.startswith("respond_blocked"):
                blocked_responds += 1
            if error in {"round_tool_budget_exceeded", "episode_tool_budget_exceeded"}:
                budget_blocks += 1
        return {
            "round_count": self._current_round_index,
            "turn_count": self.trajectory.turns,
            "read_calls": read_calls,
            "write_calls": write_calls,
            "cache_hits": cache_hits,
            "duplicate_write_blocks": duplicate_write_blocks,
            "loaded_skill_count": len(self._loaded_skill_slugs_acc),
            "round_tool_budget": self._round_tool_budget,
            "episode_tool_budget": self._episode_tool_budget,
            "blocked_responds": blocked_responds,
            "budget_blocks": budget_blocks,
            "required_action_count": len(self.case.required_actions),
            "executed_required_action_count": len([x for x in self.case.required_actions if x in set(self._executed_successful_tools())]),
        }

    def _resolve_round_tool_budget(self) -> int:
        return max(4, min(8, len(self.case.required_actions) + 1))

    def _resolve_episode_tool_budget(self) -> int:
        difficulty_budget = {"easy": 4, "medium": 6, "hard": 8}.get(self.case.difficulty, 6)
        return max(difficulty_budget, self._resolve_round_tool_budget())

    def _executed_successful_tools(self) -> List[str]:
        return [entry.tool_name for entry in self.trajectory.audit_log if entry.success]

    def _missing_required_actions(self) -> List[str]:
        executed = set(self._executed_successful_tools())
        return [action for action in self.case.required_actions if action not in executed]

    def _required_terminal_writes(self) -> List[str]:
        return [action for action in self.case.required_actions if action in self._side_effect_tools]

    def _sample_next_action(self, input_text: str) -> Tuple[Any, str]:
        last_error = ""
        for _ in range(4):
            result = chat_json_detailed("", input_text, temperature=self.temperature)
            raw = result.get("data")
            error = result.get("error", "")
            if isinstance(raw, dict):
                self._last_llm_error = ""
                return raw, ""
            last_error = error or "non_json_output"
        self._last_llm_error = last_error
        return None, last_error

    def _validate_action(self, payload: Any) -> Tuple[bool, Dict[str, Any], str]:
        if not isinstance(payload, dict):
            return False, {}, self._last_llm_error or "non_json_output"
        if "thought" in payload:
            return False, {}, "thought_not_allowed"
        action_type = payload.get("type")
        message = payload.get("message")
        if action_type not in {"load_skills", "load_action", "respond"}:
            return False, {}, "invalid_type"
        if not isinstance(message, str) or not message.strip():
            return False, {}, "missing_message"
        if action_type == "load_skills":
            if self._phase != "expect_skill":
                return False, {}, "unexpected_load_skills"
            skill_name = payload.get("skill_name")
            if skill_name not in self._skill_by_slug:
                return False, {}, "unknown_skill"
            return True, {"type": action_type, "skill_name": skill_name, "message": message.strip()}, ""
        if action_type == "load_action":
            if self._phase != "expect_action_or_respond" or not self._current_skill:
                return False, {}, "unexpected_load_action"
            if self._current_round_tool_calls >= self._round_tool_budget:
                return False, {}, "round_tool_budget_exceeded"
            if len(self.trajectory.audit_log) >= self._episode_tool_budget:
                return False, {}, "episode_tool_budget_exceeded"
            skill_name = payload.get("skill_name")
            if skill_name != self._current_skill:
                return False, {}, "skill_mismatch"
            action_name = payload.get("action_name")
            action_input = payload.get("action_input")
            if action_name not in self._allowed_actions.get(skill_name, set()):
                return False, {}, "invalid_action_for_skill"
            if not isinstance(action_input, dict):
                return False, {}, "invalid_action_input"
            return True, {
                "type": action_type,
                "skill_name": skill_name,
                "action_name": action_name,
                "action_input": action_input,
                "message": message.strip(),
            }, ""
        if self._phase != "expect_action_or_respond" or not self._current_skill:
            return False, {}, "unexpected_respond"
        skill_name = payload.get("skill_name")
        if skill_name != self._current_skill:
            return False, {}, "skill_mismatch"
        return True, {"type": action_type, "skill_name": skill_name, "message": message.strip()}, ""

    def _apply_action(self, action: Dict[str, Any]) -> bool:
        if action["type"] == "load_skills":
            self._current_skill = action["skill_name"]
            self._phase = "expect_action_or_respond"
            if self._current_skill not in self._loaded_skill_slugs_acc:
                self._loaded_skill_cards_acc.append(self._skill_by_slug[self._current_skill])
                self._loaded_skill_slugs_acc.add(self._current_skill)
            return False

        if action["type"] == "load_action":
            tool_name = action["action_name"]
            args = action["action_input"]
            key = (tool_name, json.dumps(args, ensure_ascii=False, sort_keys=True))
            prior = Counter((entry.tool_name, json.dumps(entry.args, ensure_ascii=False, sort_keys=True)) for entry in self.trajectory.audit_log)
            duplicate = prior[key] > 0
            cache_key = (tool_name, json.dumps(args, ensure_ascii=False, sort_keys=True))
            if tool_name not in self._side_effect_tools and cache_key in self._tool_cache:
                success, output, tool_message = self._tool_cache[cache_key]
                tool_message = f"cache_hit:{tool_message}"
            elif duplicate and tool_name in self._side_effect_tools:
                success, output, tool_message = False, {"error": "duplicate_side_effect_blocked"}, "duplicate_side_effect_blocked"
            else:
                success, output, tool_message = execute_tool(self.case, {"writes": self._writes}, tool_name, args)
                if tool_name not in self._side_effect_tools:
                    self._tool_cache[cache_key] = (success, output, tool_message)
            self.trajectory.turns += 1
            self.trajectory.tool_observations.append(ToolObservation(tool_name=tool_name, success=success, output=output))
            self.trajectory.audit_log.append(AuditLogEntry(tool_name=tool_name, args=args, success=success, duplicate=duplicate, message=tool_message))
            self.trajectory.reasoning_steps.append(
                ReasoningStep(
                    step_index=len(self.trajectory.reasoning_steps) + 1,
                    thought=action["message"],
                    action_type="tool",
                    round_index=self._current_round_index,
                    user_text=self._current_user_text,
                    selected_skill=self._current_skill,
                    tool_name=tool_name,
                    tool_args=args,
                    observation={"tool_name": tool_name, "success": success, "output": output},
                )
            )
            self._observation_acc.append({"tool_name": tool_name, "output": output})
            self._current_round_tool_calls += 1
            return False

        reply = self._build_reply_record(action["message"])
        self.trajectory.turns += 1
        self.trajectory.assistant_reply = reply
        self.trajectory.reasoning_steps.append(
            ReasoningStep(
                step_index=len(self.trajectory.reasoning_steps) + 1,
                thought=action["message"],
                action_type="reply",
                round_index=self._current_round_index,
                user_text=self._current_user_text,
                selected_skill=self._current_skill,
                response_text=reply.text,
            )
        )
        self._history_acc.extend([f"User: {self._current_user_text}", f"Agent: {reply.text}"])
        self._current_user_text = None
        self._current_skill = None
        self._phase = "idle"
        return True

    def _record_failsafe_reply(self, reason: str) -> None:
        reply_text = "The current information is not enough to give a final decision yet; the case status needs further checking or a handover to a human for follow-up."
        reply = self._build_reply_record(reply_text)
        self.trajectory.turns += 1
        self.trajectory.assistant_reply = reply
        self.trajectory.reasoning_steps.append(
            ReasoningStep(
                step_index=len(self.trajectory.reasoning_steps) + 1,
                thought=f"serving_harness_failsafe: {reason}",
                action_type="reply",
                round_index=max(1, self._current_round_index),
                user_text=self._current_user_text,
                selected_skill=self._current_skill,
                response_text=reply.text,
            )
        )
        if self._current_user_text:
            self._history_acc.extend([f"User: {self._current_user_text}", f"Agent: {reply.text}"])
        self._current_user_text = None
        self._current_skill = None
        self._phase = "idle"

    def _build_reply_record(self, message: str) -> AssistantReply:
        return AssistantReply(
            text=message,
            disposition=self._fallback_disposition(),
            addressed_subissues=[],
            cited_facts={},
            commitment_made=None,
            selected_skills=[x["slug"] for x in self._loaded_skill_cards_acc],
        )

    def _fallback_disposition(self) -> str:
        writes = [x.tool_name for x in self.trajectory.audit_log if x.success]
        if "create_reconsideration_ticket" in writes:
            return "create_reconsideration_ticket"
        if "send_document_request" in writes:
            return "request_missing_documents"
        if "escalate_to_human_adjuster" in writes:
            return "escalate_human_adjuster"
        if self.case.family == "partial_payment":
            return "explain_partial_payment"
        return "safe_clarification"



def main() -> None:
    parser = argparse.ArgumentParser(description="Run serving harness with the same step protocol as training data")
    parser.add_argument("--case-id", default="")
    parser.add_argument("--user-text", default="")
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()

    if not args.case_id:
        raise ValueError("--case-id is required")
    case = load_case(args.case_id)
    session = ServingHarnessSession(case, temperature=args.temperature)
    session.start_user_turn(args.user_text or case.initial_user_message)
    result = session.run_until_respond(max_steps=args.max_steps)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
