from __future__ import annotations

import json
from pathlib import Path
from typing import List

from .schemas import CaseSpec


CASES_DIR = Path(__file__).resolve().parent / "cases"


def load_cases() -> List[CaseSpec]:
    return load_cases_from_dir(CASES_DIR)


def load_cases_from_dir(directory: Path | str) -> List[CaseSpec]:
    directory = Path(directory)
    cases: List[CaseSpec] = []
    for path in sorted(directory.glob("*.json")):
        cases.append(CaseSpec.from_dict(json.loads(path.read_text(encoding="utf-8"))))
    return cases


def load_case(case_id: str) -> CaseSpec:
    for case in load_cases():
        if case.case_id == case_id:
            return case
    raise FileNotFoundError(f"Case not found: {case_id}")
