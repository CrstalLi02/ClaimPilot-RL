from __future__ import annotations

from typing import Dict, List, Sequence, Set, Tuple

from .schemas import CaseSpec, SkillCard
from .skill_loader import load_skill_cards


FAMILY_TO_SKILLS = {
    "pending_docs": [
        "pending_docs_resolution",
        "sla_timeline_explanation",
    ],
    "rejection_dispute": [
        "waiting_period_explanation",
        "policy_term_grounding",
        "rejection_reconsideration",
    ],
    "high_risk_review": [
        "manual_review_escalation",
    ],
    "partial_payment": [
        "partial_payment_breakdown",
        "deductible_explanation",
        "policy_term_grounding",
    ],
}


def select_skills_for_case(case: CaseSpec, user_message: str = "") -> Tuple[List[SkillCard], Dict[str, str]]:
    cards = {card.slug: card for card in load_skill_cards()}
    signal_tags = _collect_signal_tags(case, user_message)
    base_slugs = set(FAMILY_TO_SKILLS.get(case.family, []))
    ranked: List[Tuple[int, str, SkillCard]] = []
    rationale: Dict[str, str] = {}

    for slug, card in cards.items():
        score = 0
        reasons: List[str] = []
        if slug in base_slugs:
            score += 100
            reasons.append(f"family={case.family}")
        tag_hits = [tag for tag in card.route_tags if tag in signal_tags]
        if tag_hits:
            score += 20 + len(tag_hits)
            reasons.append("tags=" + ", ".join(tag_hits[:4]))
        if not reasons:
            reasons.append("full_pool_backup")
        rationale[slug] = " | ".join(reasons)
        ranked.append((score, slug, card))

    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [card for _, _, card in ranked], rationale


def summarize_skills(skills: Sequence[SkillCard], max_chars: int = 12000) -> str:
    chunks: List[str] = []
    used = 0
    for card in skills:
        if isinstance(card, dict):
            slug = card.get("slug", "")
            description = card.get("description", "")
            read_tools = card.get("read_tools", [])
            write_tools = card.get("write_tools", [])
            guardrails = card.get("guardrails", [])
        else:
            slug = card.slug
            description = card.description
            read_tools = card.read_tools
            write_tools = card.write_tools
            guardrails = card.guardrails
        snippet = (
            f"[skill:{slug}]\n"
            f"Description: {description}\n"
            f"Read tools: {', '.join(read_tools)}\n"
            f"Write tools: {', '.join(write_tools)}\n"
            f"Guardrails: {'; '.join(guardrails[:2])}\n"
        )
        if used + len(snippet) > max_chars:
            break
        chunks.append(snippet)
        used += len(snippet)
    return "\n".join(chunks)


def _collect_signal_tags(case: CaseSpec, user_message: str) -> Set[str]:
    tags: Set[str] = set()
    tags.add(case.family)
    tags.update(str(x) for x in getattr(case, "subissues", []) or [])
    tags.update(str(x) for x in getattr(case, "tags", []) or [])

    status = str(case.claim_case.get("status", "") or "")
    status_reason = str(case.claim_case.get("status_reason", "") or "")
    tags.add(status)

    if case.must_escalate:
        tags.update({"manual_review", "next_step"})
    if len(case.subissues) >= 2:
        tags.add("multi_issue")

    text_blob = " ".join(
        [
            user_message or case.initial_user_message,
            status,
            status_reason,
            " ".join(case.subissues),
            " ".join(case.tags),
            " ".join(case.required_documents),
            " ".join(case.uploaded_documents),
            str(case.customer_profile),
            str(case.policy_summary),
            str(case.claim_case),
            str(case.payment_breakdown),
        ]
    ).lower()

    if status == "pending_docs":
        tags.update({"pending_docs", "why_pending", "which_docs", "timeline_scope"})
    if status == "manual_review":
        tags.update({"manual_review", "why_manual_review", "next_step"})
    if status == "rejected":
        tags.update({"rejection_dispute", "rejection_reason", "reconsideration_path"})
    if status == "paid_partial":
        tags.update({"payment_reason", "payment_breakdown", "deductible_or_ratio"})
    if status in {"paid_partial", "approved", "paid"}:
        tags.add("payment_status")

    risk_flags = case.claim_case.get("risk_flags") or []
    if risk_flags:
        tags.update({"risk_flags", "discrepancy", "document_conflict", "manual_review"})

    uploaded_set = set(case.uploaded_documents)
    missing_docs = [doc for doc in case.required_documents if doc not in uploaded_set]
    missing_blob = " ".join(missing_docs).lower()
    merged_blob = text_blob + " " + missing_blob

    if _contains_any(merged_blob, ["hospital"]):
        tags.add("hospitalization")
    if _contains_any(merged_blob, ["id card", "identity", "id document"]):
        tags.add("identity_doc")
    if _contains_any(merged_blob, ["invoice", "receipt"]):
        tags.add("invoice_missing")
    if _contains_any(merged_blob, ["discharge summary", "discharge record", "discharge"]):
        tags.add("discharge_summary")

    if _contains_any(text_blob, ["callback", "call me back", "call back", "contact me later"]):
        tags.add("callback")
    if _contains_any(text_blob, ["furious", "complain", "in a hurry", "asap", "angry", "anxious", "emotion"]):
        tags.add("emotion")
    if _contains_any(text_blob, ["duplicate", "submitted twice", "uploaded twice", "already filed"]):
        tags.add("duplicate_claim")
    if _contains_any(text_blob, ["get paid", "money arrive", "not arrived", "not received the money", "payment status"]):
        tags.add("payment_status")
    if _contains_any(text_blob, ["waiting period"]):
        tags.add("waiting_period")
    if _contains_any(text_blob, ["clause", "terms", "coverage", "basis", "rules"]):
        tags.update({"terms_basis", "coverage_basis"})
    if _contains_any(text_blob, ["conflict", "inconsistent", "discrepancy"]):
        tags.update({"document_conflict", "discrepancy"})
    if _contains_any(text_blob, ["change the decision", "skip the review", "change the payout", "pay me right now", "you must pay", "unsupported"]):
        tags.add("unsupported")

    return {tag for tag in tags if tag}


def _contains_any(text: str, keywords: Sequence[str]) -> bool:
    return any(keyword in text for keyword in keywords)
