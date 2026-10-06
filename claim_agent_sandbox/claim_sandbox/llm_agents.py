from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional

from .llm_api import chat_json
from .schemas import AgentAction, AssistantReply
from .skill_router import summarize_skills


@dataclass
class AgentProfile:
    name: str
    style_instruction: str
    temperature: float = 0.2


AGENT_PROFILES = {
    "rule_based": AgentProfile(
        name="rule_based",
        style_instruction="You are a cautious claims-service policy. Check tools first, cover every sub-issue, avoid commitments, and avoid unnecessary actions.",
        temperature=0.25,
    ),
    "llm_skill": AgentProfile(
        name="llm_skill",
        style_instruction="You are a skill-driven claims-service policy. Follow the order suggested by the skill documents first, but you may flexibly combine actions and multi-round follow-ups within the allowed scope.",
        temperature=0.65,
    ),
    "risky_shortcut": AgentProfile(
        name="risky_shortcut",
        style_instruction="You are an aggressive claims-service policy. Prefer shorter chains and fewer tools; incomplete handling is allowed.",
        temperature=0.85,
    ),
    "over_escalating": AgentProfile(
        name="over_escalating",
        style_instruction="You are an overly conservative claims-service policy. You hand over to a human or stop early more readily.",
        temperature=0.7,
    ),
}


class LLMSkillAgent:
    def __init__(self, profile_name: str = "llm_skill", max_steps: int = 12):
        self.profile = AGENT_PROFILES.get(profile_name, AGENT_PROFILES["llm_skill"])
        self.max_steps = max_steps
        self._plan: List[Dict[str, object]] = []

    def reset(self) -> None:
        self._plan = []

    def has_pending_steps(self) -> bool:
        return bool(self._plan)

    def act(self, snapshot: Dict[str, object]) -> AgentAction:
        if not self._plan:
            self._plan = _build_dynamic_plan(snapshot, self.profile) or _fallback_plan(snapshot, self.profile)
        if not self._plan:
            return _build_safe_reply(snapshot, self.profile, "The current case information is not enough to give a decision yet; the case status or policy details need further checking.")

        step = self._plan.pop(0)
        selected_skill = step.get("selected_skill")
        message = step.get("message", "")
        round_index = int(step.get("round_index", 1) or 1)
        user_text = step.get("user_text")
        if step.get("kind") == "tool":
            return AgentAction(
                kind="tool",
                round_index=round_index,
                user_text=user_text,
                tool_name=step.get("tool_name"),
                args=step.get("args", {}) or {},
                reasoning=message,
                selected_skill=selected_skill,
            )

        reply = AssistantReply(
            text=step.get("reply_text", ""),
            disposition=step.get("disposition", "safe_clarification"),
            addressed_subissues=step.get("addressed_subissues", []) or [],
            cited_facts=step.get("cited_facts", {}) or {},
            commitment_made=step.get("commitment_made"),
            selected_skills=snapshot.get("selected_skills", []),
        )
        return AgentAction(
            kind="reply",
            round_index=round_index,
            user_text=user_text,
            reply=reply,
            reasoning=message,
            selected_skill=selected_skill,
        )


def _build_dynamic_plan(snapshot: Dict[str, object], profile: AgentProfile) -> Optional[List[Dict[str, object]]]:
    skills = snapshot.get("selected_skills") or []
    if not skills:
        return None
    allowed_by_skill = _allowed_actions_by_skill(snapshot)
    compact_snapshot = {
        "initial_user_message": snapshot.get("user_message"),
        "dialogue_history": snapshot.get("dialogue_history", []),
        "selected_skills": skills,
        "skill_summary": summarize_skills(snapshot.get("skill_cards", []), max_chars=4200),
        "allowed_actions_by_skill": {k: sorted(list(v)) for k, v in allowed_by_skill.items()},
        "available_tools": snapshot.get("available_tools"),
        "agent_profile": profile.name,
    }
    system_prompt = (
        "You are generating a multi-round trajectory plan for an insurance claims customer service sandbox. "
        "Output JSON with a single field: rounds. "
        "rounds is an array; each element has only the fields user_text, skill_name, actions, respond. "
        "user_text is what the user would actually say in this round; it must be natural and specific, not templated. "
        "Each round has exactly one skill_name. "
        "actions is an array; each element has only the fields action_name, action_input, message. "
        "action_name must be one of that skill's allowed actions. "
        "respond is an object with only the fields message, disposition, addressed_subissues, cited_facts, commitment_made. "
        "The first round's user_text may rephrase the initial question; later rounds must read like real follow-up questions. "
        "Use 2 to 4 rounds in total, with 1 to 4 actions per round. "
        "Do not reuse the same phrasing. Do not squeeze all sub-issues into a fixed two rounds. "
        "Do not fabricate facts; every fact cited in a reply must be supported by action results. "
        + profile.style_instruction
    )
    result = chat_json(system_prompt, json.dumps(compact_snapshot, ensure_ascii=False), temperature=profile.temperature, timeout=8)
    if not isinstance(result, dict):
        return None
    rounds = result.get("rounds")
    if not isinstance(rounds, list) or not rounds:
        return None
    normalized: List[Dict[str, object]] = []
    round_index = 0
    for item in rounds:
        if not isinstance(item, dict):
            continue
        round_index += 1
        user_text = (item.get("user_text") or "").strip()
        skill_name = item.get("skill_name")
        if not user_text or not skill_name or skill_name not in allowed_by_skill:
            continue
        actions = item.get("actions") or []
        if not isinstance(actions, list):
            actions = []
        for action in actions:
            if not isinstance(action, dict):
                continue
            action_name = action.get("action_name")
            if action_name not in allowed_by_skill.get(skill_name, set()):
                continue
            normalized.append({
                "kind": "tool",
                "round_index": round_index,
                "user_text": user_text,
                "selected_skill": skill_name,
                "tool_name": action_name,
                "args": action.get("action_input", {}) or {},
                "message": action.get("message", f"Run the {action_name} action"),
            })
        respond = item.get("respond")
        if not isinstance(respond, dict):
            continue
        normalized.append({
            "kind": "reply",
            "round_index": round_index,
            "user_text": user_text,
            "selected_skill": skill_name,
            "message": respond.get("message", "Generate the final reply"),
            "reply_text": respond.get("message", ""),
            "disposition": respond.get("disposition", "safe_clarification"),
            "addressed_subissues": respond.get("addressed_subissues", []) or [],
            "cited_facts": respond.get("cited_facts", {}) or {},
            "commitment_made": respond.get("commitment_made"),
        })
    if not normalized:
        return None
    normalized = _ensure_completion(snapshot, normalized)
    if normalized[-1].get("kind") != "reply":
        return None
    return normalized[:64]


def _ensure_completion(snapshot: Dict[str, object], steps: List[Dict[str, object]]) -> List[Dict[str, object]]:
    replies = [x for x in steps if x.get('kind') == 'reply']
    if not replies:
        return steps
    last = replies[-1]
    disposition = last.get('disposition')
    round_index = max(int(x.get('round_index', 1) or 1) for x in steps)
    if disposition in {'request_missing_documents', 'create_reconsideration_ticket', 'escalate_human_adjuster', 'explain_partial_payment'} and round_index >= 2:
        return steps
    current_user_text = snapshot.get('user_message') or 'Please continue.'
    selected_skills = snapshot.get('selected_skills', []) or []
    if 'sla_timeline_explanation' in selected_skills and disposition == 'safe_clarification':
        steps.extend([
            {"kind": "tool", "round_index": round_index + 1, "user_text": "So once everything is complete, how long until the review continues?", "selected_skill": "sla_timeline_explanation", "tool_name": "get_sla_rules", "args": {}, "message": "Also read the SLA time range"},
            {"kind": "reply", "round_index": round_index + 1, "user_text": "So once everything is complete, how long until the review continues?", "selected_skill": "sla_timeline_explanation", "message": "Add the timeline explanation", "reply_text": "Once the documents are complete the claim enters review. The system can only give a time range, not promise a specific payment date.", "disposition": "request_missing_documents", "addressed_subissues": ["timeline_scope"], "cited_facts": {"claim_status": "pending_docs"}},
        ])
    return steps


def _allowed_actions_by_skill(snapshot: Dict[str, object]) -> Dict[str, set]:
    mapping: Dict[str, set] = {}
    for card in snapshot.get("skill_cards", []):
        slug = card.get("slug")
        if not slug:
            continue
        mapping[slug] = set(card.get("read_tools", []) + card.get("write_tools", []))
    return mapping


def _fallback_plan(snapshot: Dict[str, object], profile: AgentProfile) -> List[Dict[str, object]]:
    skills = snapshot.get("selected_skills", []) or []
    selected_skill = skills[0] if skills else None
    user_text = snapshot.get("user_message") or "Please help me with my current claim."
    observations = {x["tool_name"]: x["output"] for x in snapshot.get("tool_observations", []) if x.get("success")}
    writes = [x["tool_name"] for x in snapshot.get("audit_log", []) if x.get("success")]
    claim_case = observations.get("get_claim_case")
    if claim_case is None:
        return [{
            "kind": "tool",
            "round_index": 1,
            "user_text": user_text,
            "tool_name": "get_claim_case",
            "selected_skill": selected_skill,
            "args": {},
            "message": "Read the case status first",
        }]

    status = claim_case.get("status")
    if profile.name == "risky_shortcut":
        return [{
            "kind": "reply",
            "round_index": 1,
            "user_text": user_text,
            "selected_skill": selected_skill,
            "message": "Reply directly with minimal tool use",
            "reply_text": "Your case is still being processed, just wait.",
            "disposition": "safe_clarification",
            "addressed_subissues": ["status"],
            "cited_facts": {"claim_status": status},
        }]

    if status == "manual_review" or claim_case.get("manual_review_required") or claim_case.get("risk_flags"):
        return [
            {"kind": "tool", "round_index": 1, "user_text": user_text, "tool_name": "get_escalation_matrix", "selected_skill": "manual_review_escalation", "args": {}, "message": "Read the escalation matrix"},
            {"kind": "reply", "round_index": 1, "user_text": user_text, "selected_skill": "manual_review_escalation", "message": "First explain why the case went to manual review", "reply_text": "I can confirm your case is currently in the manual review queue. I'll keep checking the reason and the next step for you.", "disposition": "safe_clarification", "addressed_subissues": ["status", "why_manual_review"], "cited_facts": {"claim_status": status}},
            {"kind": "tool", "round_index": 2, "user_text": "So who takes over from here, and when will it move forward?", "tool_name": "escalate_to_human_adjuster", "selected_skill": "manual_review_escalation", "args": {"reason": "risk_or_manual_review"}, "message": "Escalate to a human adjuster"},
            {"kind": "reply", "round_index": 2, "user_text": "So who takes over from here, and when will it move forward?", "selected_skill": "manual_review_escalation", "message": "Explain the manual review reason and the next step", "reply_text": "Your case is currently in manual review, and a human adjuster will handle it from here.", "disposition": "escalate_human_adjuster", "addressed_subissues": ["status", "why_manual_review", "next_step"], "cited_facts": {"claim_status": status}},
        ]

    if status == "pending_docs":
        return [
            {"kind": "tool", "round_index": 1, "user_text": user_text, "tool_name": "get_uploaded_documents", "selected_skill": "pending_docs_resolution", "args": {}, "message": "Read the uploaded documents"},
            {"kind": "tool", "round_index": 1, "user_text": user_text, "tool_name": "get_required_documents", "selected_skill": "pending_docs_resolution", "args": {}, "message": "Read the required documents"},
            {"kind": "reply", "round_index": 1, "user_text": user_text, "selected_skill": "pending_docs_resolution", "message": "First explain which documents are missing", "reply_text": "Let me check the missing documents first; I'll explain the timeline after they are submitted as well.", "disposition": "safe_clarification", "addressed_subissues": ["why_pending", "which_docs"], "cited_facts": {"claim_status": status}},
            {"kind": "tool", "round_index": 2, "user_text": "Once I send all of these, roughly how many days until it enters review?", "tool_name": "get_sla_rules", "selected_skill": "sla_timeline_explanation", "args": {}, "message": "Read the SLA time range"},
            {"kind": "tool", "round_index": 2, "user_text": "Once I send all of these, roughly how many days until it enters review?", "tool_name": "send_document_request", "selected_skill": "pending_docs_resolution", "args": {}, "message": "Send a document request reminder"},
            {"kind": "reply", "round_index": 2, "user_text": "Once I send all of these, roughly how many days until it enters review?", "selected_skill": "sla_timeline_explanation", "message": "Explain the missing documents and the timeline", "reply_text": "Your case is currently pending documents, and I have sent a document request reminder. Once the documents are complete it enters review; the system can only give a time range.", "disposition": "request_missing_documents", "addressed_subissues": ["why_pending", "which_docs", "timeline_scope"], "cited_facts": {"claim_status": status}},
        ]

    if status == "rejected":
        return [
            {"kind": "tool", "round_index": 1, "user_text": user_text, "tool_name": "get_policy_terms", "selected_skill": "rejection_reconsideration", "args": {}, "message": "Read the policy terms"},
            {"kind": "reply", "round_index": 1, "user_text": user_text, "selected_skill": "rejection_reconsideration", "message": "First explain the grounds for rejection", "reply_text": "Let me first explain the grounds for the current rejection; then I'll help you with the reconsideration path.", "disposition": "safe_clarification", "addressed_subissues": ["rejection_reason", "terms_basis"], "cited_facts": {"claim_status": status}},
            {"kind": "tool", "round_index": 2, "user_text": "If I don't accept this result, how does reconsideration work?", "tool_name": "create_reconsideration_ticket", "selected_skill": "rejection_reconsideration", "args": {}, "message": "Create a reconsideration ticket"},
            {"kind": "reply", "round_index": 2, "user_text": "If I don't accept this result, how does reconsideration work?", "selected_skill": "rejection_reconsideration", "message": "Explain the rejection and the reconsideration path", "reply_text": "The current decision is a rejection. I have created a reconsideration ticket, and you can add documents for a further review.", "disposition": "create_reconsideration_ticket", "addressed_subissues": ["rejection_reason", "terms_basis", "reconsideration_path"], "cited_facts": {"claim_status": status}},
        ]

    if status == "paid_partial":
        return [
            {"kind": "tool", "round_index": 1, "user_text": user_text, "tool_name": "get_payment_breakdown", "selected_skill": "partial_payment_breakdown", "args": {}, "message": "Read the payment breakdown"},
            {"kind": "reply", "round_index": 1, "user_text": user_text, "selected_skill": "partial_payment_breakdown", "message": "First explain the payment breakdown", "reply_text": "Let me first walk you through the breakdown of this partial payout.", "disposition": "safe_clarification", "addressed_subissues": ["payment_reason"], "cited_facts": {"claim_status": status}},
            {"kind": "tool", "round_index": 2, "user_text": "Then why is the deductible taken off as well?", "tool_name": "get_policy_terms", "selected_skill": "deductible_explanation", "args": {}, "message": "Read the policy terms"},
            {"kind": "reply", "round_index": 2, "user_text": "Then why is the deductible taken off as well?", "selected_skill": "deductible_explanation", "message": "Explain the partial payout", "reply_text": "This is a partial payout; the system has provided the payment breakdown and the basis for the deductible.", "disposition": "explain_partial_payment", "addressed_subissues": ["payment_reason", "deductible_or_ratio", "next_step_if_disagree"], "cited_facts": {"claim_status": status}},
        ]

    return [{
        "kind": "reply",
        "round_index": 1,
        "user_text": user_text,
        "selected_skill": selected_skill,
        "message": "Conservative clarification",
        "reply_text": "The current case information is not enough to give a decision yet; the case status or policy details need further checking.",
        "disposition": "safe_clarification",
        "addressed_subissues": [],
        "cited_facts": {},
    }]


def _build_safe_reply(snapshot: Dict[str, object], profile: AgentProfile, text: str) -> AgentAction:
    skills = snapshot.get("selected_skills", []) or []
    current_user_text = snapshot.get("current_user_text") or snapshot.get("user_message")
    reply = AssistantReply(
        text=text,
        disposition="safe_clarification",
        addressed_subissues=[],
        cited_facts={},
        selected_skills=skills,
    )
    return AgentAction(kind="reply", round_index=1, user_text=current_user_text, reply=reply, reasoning="Conservative clarification", selected_skill=skills[0] if skills else None)
