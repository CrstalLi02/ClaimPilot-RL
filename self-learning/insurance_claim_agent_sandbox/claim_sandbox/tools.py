from __future__ import annotations

from typing import Any, Dict, Tuple

from .schemas import CaseSpec


READ_TOOLS = {
    "get_customer_profile",
    "get_policy_summary",
    "get_policy_terms",
    "get_claim_case",
    "get_claim_timeline",
    "get_required_documents",
    "get_uploaded_documents",
    "get_liability_rules",
    "get_payment_breakdown",
    "get_sla_rules",
    "search_claim_kb",
    "get_escalation_matrix",
}

WRITE_TOOLS = {
    "create_claim_intake",
    "send_document_request",
    "append_case_note",
    "create_callback_task",
    "escalate_to_human_adjuster",
    "create_reconsideration_ticket",
}


def execute_tool(case: CaseSpec, mutable_state: Dict[str, Any], tool_name: str, args: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], str]:
    if tool_name in READ_TOOLS:
        return _execute_read(case, tool_name, args)
    if tool_name in WRITE_TOOLS:
        return _execute_write(case, mutable_state, tool_name, args)
    return False, {"error": "unknown_tool"}, "Unknown tool"


def _execute_read(case: CaseSpec, tool_name: str, args: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], str]:
    if tool_name == "get_customer_profile":
        return True, case.customer_profile, "customer_profile_loaded"
    if tool_name == "get_policy_summary":
        return True, case.policy_summary, "policy_summary_loaded"
    if tool_name == "get_policy_terms":
        return True, {"terms": case.liability_rules.get("terms", []), "coverage_decision": case.liability_rules.get("coverage_decision")}, "policy_terms_loaded"
    if tool_name == "get_claim_case":
        return True, case.claim_case, "claim_case_loaded"
    if tool_name == "get_claim_timeline":
        return True, {"timeline": case.claim_timeline}, "claim_timeline_loaded"
    if tool_name == "get_required_documents":
        return True, {"required_documents": case.required_documents}, "required_documents_loaded"
    if tool_name == "get_uploaded_documents":
        return True, {"uploaded_documents": case.uploaded_documents}, "uploaded_documents_loaded"
    if tool_name == "get_liability_rules":
        return True, case.liability_rules, "liability_rules_loaded"
    if tool_name == "get_payment_breakdown":
        return True, case.payment_breakdown, "payment_breakdown_loaded"
    if tool_name == "get_sla_rules":
        return True, case.sla_rules, "sla_rules_loaded"
    if tool_name == "search_claim_kb":
        query = args.get("query", "")
        snippets = [
            "Claims agents must not promise a payment date without grounds.",
            "For pending-documents cases, check the missing documents first, then explain the review time range.",
            "High-risk cases or cases with document conflicts must follow the escalation matrix.",
        ]
        return True, {"query": query, "snippets": snippets}, "kb_search_done"
    if tool_name == "get_escalation_matrix":
        return True, case.escalation_matrix, "escalation_matrix_loaded"
    return False, {"error": "unsupported_read_tool"}, "unsupported_read_tool"


def _execute_write(case: CaseSpec, mutable_state: Dict[str, Any], tool_name: str, args: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], str]:
    writes = mutable_state.setdefault("writes", {})
    if tool_name == "create_claim_intake":
        if case.claim_case.get("status") != "not_reported":
            return False, {"error": "invalid_state"}, "claim_intake_not_allowed"
        writes["claim_intake_created"] = True
        return True, {"created": True}, "claim_intake_created"
    if tool_name == "send_document_request":
        if case.claim_case.get("status") != "pending_docs":
            return False, {"error": "invalid_state"}, "document_request_not_allowed"
        writes["document_request_sent"] = True
        return True, {"sent": True}, "document_request_sent"
    if tool_name == "append_case_note":
        writes.setdefault("case_notes", []).append(args.get("note", ""))
        return True, {"note_appended": True}, "case_note_appended"
    if tool_name == "create_callback_task":
        writes["callback_task_created"] = True
        return True, {"callback_task_created": True}, "callback_task_created"
    if tool_name == "escalate_to_human_adjuster":
        writes["escalated_to"] = "human_adjuster"
        writes["escalation_reason"] = args.get("reason", "manual_review")
        return True, {"escalated_to": "human_adjuster"}, "human_adjuster_escalated"
    if tool_name == "create_reconsideration_ticket":
        if case.claim_case.get("status") != "rejected":
            return False, {"error": "invalid_state"}, "reconsideration_not_allowed"
        writes["reconsideration_created"] = True
        return True, {"reconsideration_created": True}, "reconsideration_created"
    return False, {"error": "unsupported_write_tool"}, "unsupported_write_tool"
