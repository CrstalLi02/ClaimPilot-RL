from __future__ import annotations

from pathlib import Path
from typing import List

from .schemas import SkillCard


SKILLS_DIR = Path(__file__).resolve().parent / "insurance_skills"
ROUTE_TAGS_PREFIX = "Route tags:"


def load_skill_cards() -> List[SkillCard]:
    cards: List[SkillCard] = []
    for path in sorted(SKILLS_DIR.glob("*/SKILL.md")):
        cards.append(_parse_skill(path))
    return cards


def load_skill_card(slug: str) -> SkillCard:
    path = SKILLS_DIR / slug / "SKILL.md"
    if not path.exists():
        raise FileNotFoundError(f"Skill not found: {slug}")
    return _parse_skill(path)


def _parse_skill(path: Path) -> SkillCard:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    title = path.parent.name
    description = ""
    in_frontmatter = False
    frontmatter_done = False
    body_lines = []
    for idx, line in enumerate(lines):
        if idx == 0 and line.strip() == "---":
            in_frontmatter = True
            continue
        if in_frontmatter and line.strip() == "---":
            in_frontmatter = False
            frontmatter_done = True
            continue
        if in_frontmatter:
            if line.startswith("description:"):
                description = line.split(":", 1)[1].strip()
            elif line.startswith("name:"):
                title = line.split(":", 1)[1].strip() or title
            continue
        if frontmatter_done:
            body_lines.append(line)
    body = "\n".join(body_lines).strip()
    trigger_conditions = _extract_bullets(body, "## Trigger Conditions")
    read_tools = _extract_bullets(body, "## Observe First")
    write_tools = _extract_bullets(body, "## Allowed Write Actions")
    guardrails = _extract_bullets(body, "## Guardrails")
    output_constraints = _extract_bullets(body, "## Output Constraints")
    route_tags = []
    for item in trigger_conditions:
        if item.startswith(ROUTE_TAGS_PREFIX):
            route_tags.extend([x.strip() for x in item[len(ROUTE_TAGS_PREFIX):].split(",") if x.strip()])
    return SkillCard(
        slug=path.parent.name,
        title=title,
        description=description,
        body=body,
        trigger_conditions=trigger_conditions,
        route_tags=route_tags,
        read_tools=read_tools,
        write_tools=write_tools,
        guardrails=guardrails,
        output_constraints=output_constraints,
    )


def _extract_bullets(body: str, heading: str) -> List[str]:
    lines = body.splitlines()
    values: List[str] = []
    active = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("## "):
            active = stripped == heading
            continue
        if not active:
            continue
        if stripped.startswith("- "):
            values.append(stripped[2:].strip())
    return values
