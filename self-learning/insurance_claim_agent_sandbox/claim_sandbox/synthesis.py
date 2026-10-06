from __future__ import annotations

import copy
import json
import random
import uuid
from pathlib import Path
from typing import List

from .case_loader import load_cases
from .llm_api import chat_json
from .schemas import CaseSpec


def synthesize_cases(output_dir: str, num_variants_per_case: int = 2, use_llm: bool = True) -> List[CaseSpec]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results: List[CaseSpec] = []
    for base_case in load_cases():
        for i in range(num_variants_per_case):
            variant = _make_variant(base_case, i, use_llm=use_llm)
            path = out / f"{variant.case_id}.json"
            path.write_text(json.dumps(variant.__dict__, ensure_ascii=False, indent=2), encoding="utf-8")
            results.append(variant)
    return results


def _make_variant(base_case: CaseSpec, index: int, use_llm: bool) -> CaseSpec:
    raw = copy.deepcopy(base_case.__dict__)
    raw["case_id"] = f"{base_case.case_id}_syn_{index+1}_{uuid.uuid4().hex[:6]}"
    raw["bucket"] = _pick_bucket(index)
    raw["tags"] = sorted(list(set(base_case.tags + [base_case.family, raw["bucket"], "synthetic"])))
    raw["user_variants"] = [base_case.initial_user_message]
    raw["synthesis_meta"] = {"source_case_id": base_case.case_id, "variant_index": index + 1}

    llm_variant = _llm_rewrite_case(base_case) if use_llm else None
    if llm_variant:
        raw["initial_user_message"] = llm_variant.get("initial_user_message", raw["initial_user_message"])
        raw["subissues"] = llm_variant.get("subissues", raw["subissues"])
        raw["claim_case"]["status_reason"] = llm_variant.get("status_reason", raw["claim_case"].get("status_reason"))
        raw["user_variants"].extend(llm_variant.get("user_variants", []))
    else:
        raw["initial_user_message"] = _fallback_rewrite(base_case.initial_user_message, index)

    raw["difficulty"] = random.choice([base_case.difficulty, "easy", "medium", "hard"])
    return CaseSpec.from_dict(raw)


def _pick_bucket(index: int) -> str:
    sequence = ["train", "train", "holdout", "redteam", "replay_pool"]
    return sequence[index % len(sequence)]


def _fallback_rewrite(text: str, index: int) -> str:
    suffix = [
        ". I'm getting anxious about this.",
        ". Please explain it clearly.",
        ". I want to know the next step as soon as possible.",
        ". Don't just give me a vague answer.",
    ][index % 4]
    return text.rstrip("?.!") + suffix


def _llm_rewrite_case(base_case: CaseSpec):
    system_prompt = (
        "You are generating synthetic case variants for an insurance claims customer service sandbox. "
        "Output JSON with the fields initial_user_message, subissues, status_reason, user_variants. All text must be in English. "
        "Keep the original task intent and the factual boundaries of the case; do not change ideal_disposition."
    )
    user_prompt = json.dumps(
        {
            "case_id": base_case.case_id,
            "family": base_case.family,
            "initial_user_message": base_case.initial_user_message,
            "subissues": base_case.subissues,
            "claim_status": base_case.claim_case.get("status"),
            "status_reason": base_case.claim_case.get("status_reason"),
            "ideal_disposition": base_case.ideal_disposition,
        },
        ensure_ascii=False,
    )
    return chat_json(system_prompt, user_prompt, temperature=0.6, timeout=8)
