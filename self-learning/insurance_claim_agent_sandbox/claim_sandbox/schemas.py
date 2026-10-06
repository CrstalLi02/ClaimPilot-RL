from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CaseSpec:
    case_id: str
    family: str
    difficulty: str
    initial_user_message: str
    subissues: List[str]
    customer_profile: Dict[str, Any]
    policy_summary: Dict[str, Any]
    claim_case: Dict[str, Any]
    claim_timeline: List[Dict[str, Any]]
    required_documents: List[str]
    uploaded_documents: List[str]
    liability_rules: Dict[str, Any]
    payment_breakdown: Dict[str, Any]
    sla_rules: Dict[str, Any]
    escalation_matrix: Dict[str, Any]
    ideal_disposition: str
    required_slots: List[str]
    required_actions: List[str]
    must_escalate: bool
    safe_to_self_serve: bool
    hard_forbidden: List[str]
    bucket: str = "train"
    tags: List[str] = field(default_factory=list)
    user_variants: List[str] = field(default_factory=list)
    synthesis_meta: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CaseSpec":
        return cls(**data)


@dataclass
class ToolObservation:
    tool_name: str
    success: bool
    output: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SkillCard:
    slug: str
    title: str
    description: str
    body: str
    trigger_conditions: List[str] = field(default_factory=list)
    route_tags: List[str] = field(default_factory=list)
    read_tools: List[str] = field(default_factory=list)
    write_tools: List[str] = field(default_factory=list)
    guardrails: List[str] = field(default_factory=list)
    output_constraints: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReasoningStep:
    step_index: int
    thought: str
    action_type: str
    round_index: int = 1
    user_text: Optional[str] = None
    selected_skill: Optional[str] = None
    tool_name: Optional[str] = None
    tool_args: Dict[str, Any] = field(default_factory=dict)
    observation: Dict[str, Any] = field(default_factory=dict)
    response_text: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AuditLogEntry:
    tool_name: str
    args: Dict[str, Any]
    success: bool
    duplicate: bool = False
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AssistantReply:
    text: str
    disposition: str
    addressed_subissues: List[str]
    cited_facts: Dict[str, Any] = field(default_factory=dict)
    commitment_made: Optional[str] = None
    selected_skills: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentAction:
    kind: str
    round_index: int = 1
    user_text: Optional[str] = None
    tool_name: Optional[str] = None
    args: Dict[str, Any] = field(default_factory=dict)
    reply: Optional[AssistantReply] = None
    reasoning: str = ""
    selected_skill: Optional[str] = None


@dataclass
class Trajectory:
    case_id: str
    user_message: str
    agent_name: str = ""
    tool_observations: List[ToolObservation] = field(default_factory=list)
    audit_log: List[AuditLogEntry] = field(default_factory=list)
    assistant_reply: Optional[AssistantReply] = None
    selected_skills: List[str] = field(default_factory=list)
    selected_skill_cards: List[Dict[str, Any]] = field(default_factory=list)
    skill_rationale: Dict[str, str] = field(default_factory=dict)
    reasoning_steps: List[ReasoningStep] = field(default_factory=list)
    bucket: str = "train"
    tags: List[str] = field(default_factory=list)
    synthesis_meta: Dict[str, Any] = field(default_factory=dict)
    turns: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "user_message": self.user_message,
            "agent_name": self.agent_name,
            "tool_observations": [x.to_dict() for x in self.tool_observations],
            "audit_log": [x.to_dict() for x in self.audit_log],
            "assistant_reply": self.assistant_reply.to_dict() if self.assistant_reply else None,
            "selected_skills": self.selected_skills,
            "selected_skill_cards": self.selected_skill_cards,
            "skill_rationale": self.skill_rationale,
            "reasoning_steps": [x.to_dict() for x in self.reasoning_steps],
            "bucket": self.bucket,
            "tags": self.tags,
            "synthesis_meta": self.synthesis_meta,
            "turns": self.turns,
        }


@dataclass
class VerifierResult:
    case_id: str
    expected_disposition: str
    predicted_disposition: str
    rubric: Dict[str, float]
    hard_guardrails: List[str]
    soft_penalties: List[str]
    reward: float
    success: bool
    reward_components: Dict[str, float] = field(default_factory=dict)
    data_quality: Dict[str, Any] = field(default_factory=dict)
    llm_rubric: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
