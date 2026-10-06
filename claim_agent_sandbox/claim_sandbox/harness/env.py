from __future__ import annotations

from collections import Counter
from typing import Any, Dict

from ..schemas import AgentAction, AuditLogEntry, CaseSpec, ReasoningStep, ToolObservation, Trajectory
from ..skill_router import select_skills_for_case
from ..tools import READ_TOOLS, WRITE_TOOLS, execute_tool


class ClaimSandboxEnv:
    def __init__(self, case: CaseSpec):
        self.case = case
        selected_skills, rationale = select_skills_for_case(case, case.initial_user_message)
        self.trajectory = Trajectory(
            case_id=case.case_id,
            user_message=case.initial_user_message,
            selected_skills=[x.slug for x in selected_skills],
            selected_skill_cards=[x.to_dict() for x in selected_skills],
            skill_rationale=rationale,
            bucket=getattr(case, "bucket", "train"),
            tags=getattr(case, "tags", []),
            synthesis_meta=getattr(case, "synthesis_meta", {}),
        )
        self._mutable_state: Dict[str, Any] = {"writes": {}}
        self._dialogue_history = [f"User: {case.initial_user_message}"]
        self._current_round_index = 1
        self._current_user_text = case.initial_user_message

    def snapshot(self) -> Dict[str, Any]:
        return {
            "user_message": self.case.initial_user_message,
            "current_user_text": self._current_user_text,
            "current_round_index": self._current_round_index,
            "dialogue_history": list(self._dialogue_history),
            "tool_observations": [x.to_dict() for x in self.trajectory.tool_observations],
            "audit_log": [x.to_dict() for x in self.trajectory.audit_log],
            "turns": self.trajectory.turns,
            "available_tools": sorted(list(READ_TOOLS | WRITE_TOOLS)),
            "selected_skills": self.trajectory.selected_skills,
            "skill_cards": selected_skill_cards(self.trajectory),
            "skill_rationale": self.trajectory.skill_rationale,
        }

    def adopt_action_context(self, action: AgentAction) -> None:
        if action.user_text:
            if action.round_index != self._current_round_index:
                self._current_round_index = action.round_index
                self._current_user_text = action.user_text
                if not self._dialogue_history or self._dialogue_history[-1] != f"User: {action.user_text}":
                    self._dialogue_history.append(f"User: {action.user_text}")
            else:
                self._current_user_text = action.user_text

    def run_tool(self, tool_name: str, args: Dict[str, Any]) -> None:
        key = (tool_name, tuple(sorted(args.items())))
        prior = Counter((entry.tool_name, tuple(sorted(entry.args.items()))) for entry in self.trajectory.audit_log)
        duplicate = prior[key] > 0
        success, output, message = execute_tool(self.case, self._mutable_state, tool_name, args)
        self.trajectory.turns += 1
        self.trajectory.tool_observations.append(ToolObservation(tool_name=tool_name, success=success, output=output))
        self.trajectory.audit_log.append(AuditLogEntry(tool_name=tool_name, args=args, success=success, duplicate=duplicate, message=message))

    def record_reasoning_step(self, action: AgentAction) -> None:
        observation = self.trajectory.tool_observations[-1].to_dict() if self.trajectory.tool_observations else {}
        response_text = action.reply.text if action.reply else None
        self.trajectory.reasoning_steps.append(
            ReasoningStep(
                step_index=len(self.trajectory.reasoning_steps) + 1,
                thought=action.reasoning,
                action_type=action.kind,
                round_index=action.round_index,
                user_text=action.user_text,
                selected_skill=action.selected_skill,
                tool_name=action.tool_name,
                tool_args=action.args,
                observation=observation if action.kind == "tool" else {},
                response_text=response_text,
            )
        )

    def record_reply(self, action: AgentAction) -> None:
        self.trajectory.turns += 1
        self.trajectory.assistant_reply = action.reply
        self.record_reasoning_step(action)
        if action.reply and action.reply.text:
            self._dialogue_history.append(f"Agent: {action.reply.text}")


def selected_skill_cards(trajectory: Trajectory):
    return trajectory.selected_skill_cards


def run_episode(agent: Any, env: ClaimSandboxEnv, max_steps: int = 10) -> Trajectory:
    agent.reset()
    for _ in range(max_steps):
        action: AgentAction = agent.act(env.snapshot())
        env.adopt_action_context(action)
        if action.kind == "tool":
            env.run_tool(action.tool_name or "", action.args)
            env.record_reasoning_step(action)
            continue
        if action.kind == "reply":
            env.record_reply(action)
            if getattr(agent, "has_pending_steps", lambda: False)():
                continue
            break
        raise ValueError(f"Unsupported action kind: {action.kind}")
    return env.trajectory
