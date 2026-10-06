from __future__ import annotations

from typing import Dict

from .llm_agents import LLMSkillAgent
from .schemas import AgentAction


class BaseAgent:
    def reset(self) -> None:
        return None

    def act(self, snapshot: Dict[str, object]) -> AgentAction:
        raise NotImplementedError


def build_agent(name: str) -> BaseAgent:
    if name in {"rule_based", "llm_skill", "risky_shortcut", "over_escalating"}:
        return LLMSkillAgent(profile_name=name)
    raise ValueError(f"Unknown agent: {name}")
