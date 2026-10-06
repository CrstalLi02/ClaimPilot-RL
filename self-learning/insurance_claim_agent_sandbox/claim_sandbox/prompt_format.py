from __future__ import annotations

import json
from typing import Any, Dict, List

from .schemas import CaseSpec, Trajectory


SYSTEM_PROMPT = """You are an insurance claims customer service agent.
Each time, output only the next action, as exactly one JSON object.
Output nothing except that JSON object: no explanations, no Markdown, no code blocks.

Fixed flow: user question -> load_skills -> several load_action steps -> respond.
When a new round starts, the dialogue history must include the agent's previous reply and the user's new question.
Each step may be bound to only one skill.

Only three types are allowed: load_skills, load_action, respond.

Output schema:
1. load_skills outputs exactly these 3 fields: type, skill_name, message
   Example: {"type":"load_skills","skill_name":"pending_docs_resolution","message":"Given the user's question, pending_docs_resolution needs to be loaded first."}
2. load_action outputs exactly these 5 fields: type, skill_name, action_name, action_input, message
   Example: {"type":"load_action","skill_name":"pending_docs_resolution","action_name":"get_required_documents","action_input":{},"message":"Ran get_required_documents; the discharge summary and the original invoice are missing."}
3. respond outputs exactly these 3 fields: type, skill_name, message
   Example: {"type":"respond","skill_name":"pending_docs_resolution","message":"I checked your claim: it is currently pending documents, and the discharge summary and the original invoice are missing."}

Hard constraints:
- Never output a thought field
- Never output any field outside the schema
- message must be a string
- action_input must be a JSON object
- action_name must be one of the current skill's allowed actions
- Never fabricate policy terms, payout amounts, payment times, case statuses, or system processing results
- Never expose internal field names or internal tool names to the user

Make sure the output format is fully valid before thinking about business content. Invalid format counts as failure."""


def build_turn_samples(case: CaseSpec, trajectory: Trajectory) -> List[Dict[str, Any]]:
    rounds = _group_reasoning_rounds(trajectory)
    samples: List[Dict[str, Any]] = []
    history_acc: List[str] = []
    loaded_skill_cards_acc: List[Dict[str, Any]] = []
    loaded_skill_slugs_acc = set()
    observation_acc: List[Dict[str, Any]] = []
    turn_id = 0
    skill_cards = {card["slug"]: card for card in trajectory.selected_skill_cards}

    for round_spec in rounds:
        round_index = round_spec["round_index"]
        user_text = round_spec["user_text"]
        skill_name = round_spec["skill_name"]
        skill_card = skill_cards.get(skill_name)
        if not skill_card:
            continue
        history_before = history_acc + [f"User: {user_text}"]

        samples.append(
            {
                "input": build_input_text(
                    case,
                    available_skill_cards=trajectory.selected_skill_cards,
                    history_lines=history_before,
                    loaded_skill_cards=loaded_skill_cards_acc,
                    recent_observations=observation_acc,
                ),
                "target": json.dumps(
                    {
                        "type": "load_skills",
                        "skill_name": skill_name,
                        "message": _build_load_skill_message(user_text, skill_name, round_spec.get("load_skill_reasoning", "")),
                    },
                    ensure_ascii=False,
                ),
                "meta": {
                    "task_id": case.case_id,
                    "turn_id": turn_id + 1,
                    "round_index": round_index,
                    "step_kind": "load_skills",
                    "agent": _infer_agent_name(trajectory),
                    "loaded_skill": skill_name,
                },
            }
        )
        turn_id += 1

        if skill_name not in loaded_skill_slugs_acc:
            loaded_skill_cards_acc.append(skill_card)
            loaded_skill_slugs_acc.add(skill_name)

        step_index = 1
        for action in round_spec["actions"]:
            samples.append(
                {
                    "input": build_input_text(
                        case,
                        available_skill_cards=trajectory.selected_skill_cards,
                        history_lines=history_before,
                        loaded_skill_cards=loaded_skill_cards_acc,
                        recent_observations=observation_acc,
                    ),
                    "target": json.dumps(
                        {
                            "type": "load_action",
                            "skill_name": skill_name,
                            "action_name": action["action_name"],
                            "action_input": action["action_input"],
                            "message": _build_load_action_message(action["action_name"], action["observation"], action.get("reasoning", "")),
                        },
                        ensure_ascii=False,
                    ),
                    "meta": {
                        "task_id": case.case_id,
                        "turn_id": turn_id + 1,
                        "round_index": round_index,
                        "step_index": step_index,
                        "step_kind": "load_action",
                        "agent": _infer_agent_name(trajectory),
                        "loaded_skill": skill_name,
                    },
                }
            )
            turn_id += 1
            step_index += 1
            observation_acc.append({"tool_name": action["action_name"], "output": action["observation"]})

        reply_message = round_spec["reply_text"]
        samples.append(
            {
                "input": build_input_text(
                    case,
                    available_skill_cards=trajectory.selected_skill_cards,
                    history_lines=history_before,
                    loaded_skill_cards=loaded_skill_cards_acc,
                    recent_observations=observation_acc,
                ),
                "target": json.dumps(
                    {
                        "type": "respond",
                        "skill_name": skill_name,
                        "message": reply_message,
                    },
                    ensure_ascii=False,
                ),
                "meta": {
                    "task_id": case.case_id,
                    "turn_id": turn_id + 1,
                    "round_index": round_index,
                    "step_index": step_index,
                    "step_kind": "respond",
                    "agent": _infer_agent_name(trajectory),
                    "loaded_skill": skill_name,
                },
            }
        )
        turn_id += 1
        history_acc = history_before + [f"Agent: {reply_message}"]
    return samples


def build_input_text(
    case: CaseSpec,
    available_skill_cards: List[Dict[str, Any]],
    history_lines: List[str],
    loaded_skill_cards: List[Dict[str, Any]],
    recent_observations: List[Dict[str, Any]],
) -> str:
    lines: List[str] = []
    lines.append("SYSTEM:")
    lines.append(SYSTEM_PROMPT)
    lines.append("")
    lines.append("## Available skills")
    for card in available_skill_cards:
        lines.append(f"- {card['slug']}: {_compact_description(card)}")
    lines.append("")
    lines.append("## Signals")
    for key, value in _build_signals(case).items():
        lines.append(f"- {key}: {_render_value(value)}")
    lines.append("")
    lines.append("## Dialogue history")
    lines.extend(history_lines)
    lines.append("")
    lines.append("## Loaded skills")
    if loaded_skill_cards:
        for card in loaded_skill_cards:
            lines.append(f"- {card['slug']}")
    else:
        lines.append("[]")
    lines.append("")
    lines.append("## Loaded skill content")
    if loaded_skill_cards:
        first = True
        for card in loaded_skill_cards:
            if not first:
                lines.append("")
            lines.extend(_render_loaded_skill(card))
            first = False
    else:
        lines.append("None")
    lines.append("")
    lines.append("## Recent observations")
    if recent_observations:
        for obs in recent_observations:
            lines.append(f"- {obs['tool_name']} => {json.dumps(obs['output'], ensure_ascii=False)}")
    else:
        lines.append("None")
    return "\n".join(lines).strip()


def allowed_actions(skill_card: Dict[str, Any]) -> List[str]:
    return list(skill_card.get("read_tools", [])) + list(skill_card.get("write_tools", []))


def _group_reasoning_rounds(trajectory: Trajectory) -> List[Dict[str, Any]]:
    grouped: Dict[int, Dict[str, Any]] = {}
    for step in trajectory.reasoning_steps:
        round_index = int(getattr(step, "round_index", 1) or 1)
        user_text = getattr(step, "user_text", None) or trajectory.user_message
        skill_name = step.selected_skill or (trajectory.selected_skills[0] if trajectory.selected_skills else "")
        bucket = grouped.setdefault(round_index, {
            "round_index": round_index,
            "user_text": user_text,
            "skill_name": skill_name,
            "load_skill_reasoning": step.thought or "",
            "actions": [],
            "reply_text": "",
        })
        if step.action_type == "tool" and step.tool_name:
            bucket["actions"].append({
                "action_name": step.tool_name,
                "action_input": step.tool_args or {},
                "observation": (step.observation or {}).get("output", step.observation or {}),
                "reasoning": step.thought or "",
            })
        elif step.action_type == "reply" and step.response_text is not None:
            bucket["reply_text"] = step.response_text
    rounds = [grouped[k] for k in sorted(grouped)]
    return [item for item in rounds if item.get("skill_name") and item.get("reply_text")]


def _compact_description(card: Dict[str, Any]) -> str:
    route_tags = card.get("route_tags", [])
    if route_tags:
        return f"Handles tags: {', '.join(route_tags[:3])}"
    desc = card.get("description", "")
    return desc[:80]


def _build_signals(case: CaseSpec) -> Dict[str, Any]:
    return {
        "User age": case.customer_profile.get("age", "unknown"),
        "User gender": case.customer_profile.get("gender", "unknown"),
        "Insured product": case.policy_summary.get("product"),
        "Policy status": case.policy_summary.get("status", "active"),
        "Claim status": case.claim_case.get("status"),
        "User channel": case.customer_profile.get("channel"),
        "User emotion": case.customer_profile.get("emotion"),
        "Risk flags": case.claim_case.get("risk_flags", []),
        "Uploaded documents": case.uploaded_documents,
        "Required documents": case.required_documents,
        "Safe to self-serve": "yes" if case.safe_to_self_serve else "no",
        "Must escalate to human": "yes" if case.must_escalate else "no",
        "Current issue tags": case.subissues,
        "bucket": getattr(case, "bucket", "train"),
    }


def _render_loaded_skill(card: Dict[str, Any]) -> List[str]:
    lines = [f"[Skill: {card['slug']}]"]
    lines.append(f"- Goal: {_compact_description(card)}")
    actions = allowed_actions(card)
    if actions:
        lines.append(f"- Allowed actions: {', '.join(actions)}")
    guardrails = card.get("guardrails", [])
    if guardrails:
        lines.append(f"- Guardrails: {'; '.join(guardrails[:3])}")
    return lines


def _render_value(value: Any) -> str:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _infer_agent_name(trajectory: Trajectory) -> str:
    return getattr(trajectory, "agent_name", "unknown")


def _build_load_skill_message(user_text: str, skill_name: str, thought: str) -> str:
    thought = (thought or "").strip()
    if thought:
        return f"Based on the user's question, {skill_name} is needed to handle: \"{user_text}\" Reason: {thought}"
    return f"Based on the user's question, {skill_name} is needed to handle: \"{user_text}\""


def _build_load_action_message(action_name: str, observation: Dict[str, Any], thought: str) -> str:
    prefix = f"Ran the {action_name} action; the result is {json.dumps(observation, ensure_ascii=False)}"
    thought = (thought or "").strip()
    if thought:
        return f"{prefix}. Reason: {thought}"
    return prefix
