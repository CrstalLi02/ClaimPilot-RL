Data Synthesis Pipeline

Training files live in `artifacts/final_dataset/`. Two artifacts are used:

| File                  | Purpose                                                  |
| --------------------- | -------------------------------------------------------- |
| `policy_sft.jsonl`  | SFT training samples, expanded per step                  |
| `rl_rollouts.jsonl` | RL training samples, one rollout per complete dialogue   |

### Protocol

Sample protocol:

- User question -> `load_skills` -> `load_action`  -> `respond`
- Only one skill may be loaded per round
- `load_action` may only call actions allowed by the current skill
- `thought` is not used
- Every step must carry a `message`
- The dialogue history in the next round's input strictly accumulates the user and agent turns of all earlier rounds

### SFT

A single record in `policy_sft.jsonl` looks like this:

```json
{
  "id": "claim_pending_docs_0142__llm_skill",
  "turn_id": 1,
  "input": "...",
  "target": "...",
  "meta": {
    "task_id": "claim_pending_docs_0142",
    "turn_id": 1,
    "round_index": 1,
    "step_kind": "load_skills",
    "agent": "llm_skill",
    "loaded_skill": "pending_docs_resolution",
    "family": "pending_docs",
    "difficulty": "medium",
    "success": true
  }
}
```

Fields:

- `id`: id of the whole dialogue, in the form `case_id__agent`
- `turn_id`: step number within the dialogue, starting at 1
- `input`: full input for the current step
- `target`: reference output for the current step, as a JSON string
- `meta.task_id`: case id
- `meta.turn_id`: same as the top-level `turn_id`
- `meta.round_index`: which user question/answer round this is
- `meta.step_kind`: step type, always one of `load_skills`, `load_action`, `respond`
- `meta.agent`: the agent that produced this trajectory
- `meta.loaded_skill`: the skill this step belongs to
- `meta.family`: case category
- `meta.difficulty`: case difficulty
- `meta.success`: whether the dialogue achieved its goal

Floating-point numbers in the jsonl files are written with 4 decimal places. SFT `meta` contains no reward fields.

`input` contains these blocks:

```text
SYSTEM:
<system prompt>

## Available skills
- <skill_slug>: <one-line summary>

## Signals
- User age: ...
- User gender: ...
- Insured product: ...
- Policy status: ...
- Claim status: ...
- User channel: ...
- User emotion: ...
- Risk flags: ...
- Uploaded documents: ...
- Required documents: ...
- Safe to self-serve: yes/no
- Must escalate to human: yes/no
- Current issue tags: ...
- bucket: ...

## Dialogue history
User: ...
Agent: ...
User: ...

## Loaded skills
- skill_a
- skill_b

## Loaded skill content
[Skill: skill_a]
- Goal: ...
- Allowed actions: ...
- Guardrails: ...

[Skill: skill_b]
- Goal: ...
- Allowed actions: ...
- Guardrails: ...

## Recent observations
- get_claim_case => {...}
- get_sla_rules => {...}
- create_reconsideration_ticket => {...}
```

Notes:

- `Dialogue history` accumulates strictly
- `Loaded skills` accumulates within the same `id`
- `Loaded skill content` accumulates within the same `id`
- `Recent observations` accumulates within the same `id`
- Each round's `User: ...` line comes directly from `user_text` in the rollout trajectory

`target` may only be one of three JSON strings.

`load_skills`:

```json
{
  "type": "load_skills",
  "skill_name": "pending_docs_resolution",
  "message": "Based on the user's question, pending_docs_resolution is needed to handle: \"Just tell me which documents are missing.\""
}
```

`load_action`:

```json
{
  "type": "load_action",
  "skill_name": "pending_docs_resolution",
  "action_name": "get_required_documents",
  "action_input": {},
  "message": "Ran the get_required_documents action; the result is {\"required_documents\":[\"ID card\",\"Medical record\",\"Invoice\"]}"
}
```

`respond`:

```json
{
  "type": "respond",
  "skill_name": "pending_docs_resolution",
  "message": "I checked your claim: it is currently pending documents, and the missing documents are the ID card, the medical record, and the invoice."
}
```

The `message` field:

- `load_skills.message`: explains why this skill is loaded for the current question
- `load_action.message`: explains which action was run and what it returned
- `respond.message`: what is actually said to the user

### RL

A single record in `rl_rollouts.jsonl` looks like this:

```json
{
  "id": "claim_pending_docs_0142__llm_skill",
  "agent": "llm_skill",
  "predicted_disposition": "request_missing_documents",
  "expected_disposition": "request_missing_documents",
  "selected_skills": ["pending_docs_resolution", "sla_timeline_explanation"],
  "policy_samples": [
    {
      "turn_id": 1,
      "input": "...",
      "target": "...",
      "rollout_output": "...",
      "reward": null
    },
    {
      "turn_id": 2,
      "input": "...",
      "target": "...",
      "rollout_output": "...",
      "reward": 1.1800
    }
  ],
  "reward_components": {
    "correct_resolution": 1.0000,
    "wrong_commitment": 0.0000,
    "unnecessary_handoff": 0.0000,
    "few_turn_bonus": 0.2000
  },
  "data_quality": {
    "score": 1.0,
    "label": "high",
    "has_skills": true,
    "has_reasoning_steps": true,
    "tool_step_count": 3
  },
  "reward": 1.1800,
  "success": true,
  "hard_guardrails": [],
  "soft_penalties": []
}
```

Fields:

- `id`: id of one rollout dialogue, in the form `case_id__agent`
- `agent`: the agent used for the rollout
- `predicted_disposition`: the final business disposition the verifier infers from the rollout trajectory
- `expected_disposition`: the expected disposition of the case
- `selected_skills`: the candidate skills for this dialogue, i.e. the available skill pool ranked by the skill router, not the list of skills that were actually loaded
- `policy_samples`: the step-level rollout sequence. Only the last step carries the final reward; intermediate steps have `reward` set to `null`. To see which skills were actually loaded, look at `target.skill_name` in each step, or at `reasoning_steps.selected_skill` in the trajectory
- `reward_components`: breakdown of the reward
- `data_quality`: data quality assessment
- `reward`: total reward
- `success`: whether the task was completed
- `hard_guardrails`: severe violations
- `soft_penalties`: soft violations

A single step sample in `policy_samples` looks like this:

```json
{
  "turn_id": 1,
  "input": "...",
  "target": "...",
  "rollout_output": "...",
  "reward": null
}
```

Notes:

- `turn_id`: step number within this rollout dialogue, starting at 1
- `input`: step context
- `target`: reference action
- `rollout_output`: the rollout output from expanding the offline trajectory; identical to `target`
- `reward`: terminal reward slot. `null` for intermediate steps; only the last step carries the final episode reward

`rollout_output` is aligned directly with `target`. The step outputs in the RL file come from expanding the same completed trajectory.

### Build command

Full build entry point: `build_datasets.py`

```bash
python3 build_datasets.py --outdir artifacts/final_dataset
```

### Build flow

`build_datasets.py` first reads the seed cases in `claim_sandbox/cases/`, generates temporary variants according to the configuration, and then hands each case to `HarnessOrchestrator.run_case(...)`. The environment maintains the dialogue history, skills, tool observations, and audit log, and the agent produces multi-round `user question -> load_skills -> load_action*N -> respond` trajectories. `prompt_format.py` expands each trajectory into step-level SFT data, and `exporters.py` and `verifier.py` produce `policy_sft.jsonl` and `rl_rollouts.jsonl`.

#### build_datasets.py

Entry function: `main()`

    base_cases = load_cases()
    if args.limit_cases and args.limit_cases > 0:
        base_cases = base_cases[: args.limit_cases]
    all_cases = list(base_cases)
    with tempfile.TemporaryDirectory(prefix="claim_variants_") as tmpdir:
        synthesize_cases(tmpdir, num_variants_per_case=args.num_variants, use_llm=not args.disable_llm_variants)
        all_cases.extend(load_cases_from_dir(tmpdir))

        for case in all_cases:
            for agent_name in args.agents:
                run = orchestrator.run_case(case, agent_name)
                rl_rows.append(run.rl_rollout)
                policy_sft_rows.extend(build_policy_sft_records(run.case, run.trajectory, run.verifier))

Order of execution: read cases, augment cases, run rollouts, export data. `--limit-cases` controls the number of seed cases, and `--num-variants` controls the number of temporary variants per seed case.

case_loader.py and synthesis.py

These two files handle the case set.

`case_loader.py`: reads JSON from a directory and converts it into `CaseSpec`.

    def load_cases() -> List[CaseSpec]:
        return load_cases_from_dir(CASES_DIR)

    def load_cases_from_dir(directory: Path | str) -> List[CaseSpec]:
        directory = Path(directory)
        cases: List[CaseSpec] = []
        for path in sorted(directory.glob("*.json")):
            cases.append(CaseSpec.from_dict(json.loads(path.read_text(encoding="utf-8"))))
        return cases

`synthesis.py` generates temporary variants. The entry point is `synthesize_cases(...)`, and the logic for a single variant is in `_make_variant(...)`.

    for base_case in load_cases():
        for i in range(num_variants_per_case):
            variant = _make_variant(base_case, i, use_llm=use_llm)
            path = out / f"{variant.case_id}.json"
            path.write_text(json.dumps(variant.__dict__, ensure_ascii=False, indent=2), encoding="utf-8")
            results.append(variant)

`_llm_rewrite_case(...)` calls the LLM to rewrite the first-round user question, `subissues`, `status_reason`, and `user_variants`, without changing the factual boundaries of the case or `ideal_disposition`.

    system_prompt = (
        "You are generating synthetic case variants for an insurance claims customer service sandbox. "
        "Output JSON with the fields initial_user_message, subissues, status_reason, user_variants. All text must be in English. "
        "Keep the original task intent and the factual boundaries of the case; do not change ideal_disposition."
    )
    return chat_json(system_prompt, user_prompt, temperature=0.6, timeout=8)

The output is still a `CaseSpec`. The orchestrator downstream does not distinguish seed cases from synthetic variants.

#### orchestrator.py

`build_datasets.py` uses `HarnessOrchestrator` from `claim_sandbox/orchestrator.py`.

    def run_case(self, case: CaseSpec, agent_name: str) -> OrchestratorRun:
        agent = build_agent(agent_name)
        env = ClaimSandboxEnv(case)
        env.trajectory.agent_name = agent_name
        trajectory = run_episode(agent, env)
        verifier = verify_episode(case, trajectory)
        hindsight_skill = generate_hindsight_skill(case, trajectory, verifier)
        return OrchestratorRun(
            case=case,
            trajectory=trajectory,
            verifier=verifier,
            trajectory_record=build_trajectory_record(case, agent_name, trajectory, verifier, hindsight_skill),
            reward_ledger=build_reward_ledger_row(case, agent_name, trajectory, verifier),
            hindsight_sft=build_hindsight_sft_record(case, trajectory, verifier, hindsight_skill),
            rl_rollout=build_rl_rollout_record(case, agent_name, trajectory, verifier),
        )

This layer wires everything together.

- `build_agent(...)` creates the agent profile
- `ClaimSandboxEnv(case)` initializes the environment
- `run_episode(...)` runs the full rollout
- `verify_episode(...)` computes the reward, the success flag, and the rubric
- `build_rl_rollout_record(...)` builds the RL sample

Four profiles run by default: `rule_based`, `llm_skill`, `risky_shortcut`, `over_escalating`. They produce trajectories of different styles and quality, and are the agents of the data synthesis stage.

#### environment.py, harness/env.py and skill_router.py

The environment state machine is implemented in `claim_sandbox/harness/env.py`.

`skill_router.py` puts all 20 skills into the candidate pool and ranks them using `family`, `subissues`, the case status, risk flags, missing documents, keywords in the user question, and each skill's own `route_tags`.

    def select_skills_for_case(case: CaseSpec, user_message: str = "") -> Tuple[List[SkillCard], Dict[str, str]]:
        cards = {card.slug: card for card in load_skill_cards()}
        signal_tags = _collect_signal_tags(case, user_message)
        base_slugs = set(FAMILY_TO_SKILLS.get(case.family, []))
        ranked: List[Tuple[int, str, SkillCard]] = []

        for slug, card in cards.items():
            score = 0
            if slug in base_slugs:
                score += 100
            tag_hits = [tag for tag in card.route_tags if tag in signal_tags]
            if tag_hits:
                score += 20 + len(tag_hits)
            ranked.append((score, slug, card))

        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [card for _, _, card in ranked], rationale

`_collect_signal_tags(...)` extracts tags from `family`, `subissues`, `claim_case.status`, `risk_flags`, uploaded/missing documents, and the user's wording, and matches them against each skill's `route_tags`. Skills near the top are more relevant; skills further down stay in the candidate pool.

`ClaimSandboxEnv.__init__(...)` writes these skills into the trajectory and initializes the dialogue history. The code is in `claim_sandbox/harness/env.py`.

    selected_skills, rationale = select_skills_for_case(case, case.initial_user_message)
    self.trajectory = Trajectory(
        case_id=case.case_id,
        user_message=case.initial_user_message,
        selected_skills=[x.slug for x in selected_skills],
        selected_skill_cards=[x.to_dict() for x in selected_skills],
        skill_rationale=rationale,
        ...
    )
    self._dialogue_history = [f"User: {case.initial_user_message}"]
    self._current_round_index = 1
    self._current_user_text = case.initial_user_message

The environment's main loop is `run_episode(...)`, also in `claim_sandbox/harness/env.py`.

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
        return env.trajectory

A `reply` does not end the whole episode by itself; execution continues while the agent still has pending steps. `adopt_action_context(...)` writes the user's new follow-up question into `_dialogue_history` when the round changes, and `record_reply(...)` writes the agent's reply after each response. Later rounds use the full accumulated history as input.

    def adopt_action_context(self, action: AgentAction) -> None:
        if action.user_text:
            if action.round_index != self._current_round_index:
                self._current_round_index = action.round_index
                self._current_user_text = action.user_text
                if not self._dialogue_history or self._dialogue_history[-1] != f"User: {action.user_text}":
                    self._dialogue_history.append(f"User: {action.user_text}")

#### llm_agents.py

Key class: `LLMSkillAgent`

Key method: `act(...)`

    def act(self, snapshot: Dict[str, object]) -> AgentAction:
        if not self._plan:
            self._plan = _build_dynamic_plan(snapshot, self.profile) or _fallback_plan(snapshot, self.profile)
        if not self._plan:
            return _build_safe_reply(...)

Key methods: `act(...)` and `_build_dynamic_plan(...)`.

When `self._plan` is empty, `LLMSkillAgent.act(...)` first calls `_build_dynamic_plan(...)` to generate a complete plan; if the LLM returns nothing usable, it falls back to `_fallback_plan(...)`. Each later call of `act(...)` by the environment takes one step from the existing plan, wraps it as an `AgentAction`, and hands it to the environment.

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
            return AgentAction(...)
        return AgentAction(kind="reply", ...)

Responsibilities: make sure a plan exists, translate the dict steps in the plan into a uniform `AgentAction`, and carry `message`, `selected_skill`, `round_index`, and `user_text` along with it. The plan holds the intermediate representation; the environment consumes `AgentAction`.

`_build_dynamic_plan(...)` handles multi-round planning. It extracts the minimal planning context from the snapshot, sends it to the LLM, and asks the model to return all `rounds` at once. A 3-round dialogue is generated here in one go.

    skills = snapshot.get("selected_skills") or []
    if not skills:
        return None
    allowed_by_skill = _allowed_actions_by_skill(snapshot)
    compact_snapshot = {
        "initial_user_message": snapshot.get("user_message"),
        "dialogue_history": snapshot.get("dialogue_history", []),
        "selected_skills": skills,
        "skill_summary": summarize_skills(snapshot.get("skill_cards", []), max_chars=12000),
        "allowed_actions_by_skill": {k: sorted(list(v)) for k, v in allowed_by_skill.items()},
        "available_tools": snapshot.get("available_tools"),
        "agent_profile": profile.name,
    }
    result = chat_json(system_prompt, json.dumps(compact_snapshot, ensure_ascii=False), temperature=profile.temperature, timeout=8)

The input includes:

- `dialogue_history`: the full accumulated history, not just the latest user question
- `selected_skills`: the candidate skills for the current case
- `skill_summary`: a summary of each skill's description, read tools, write tools, and guardrails
- `allowed_actions_by_skill`: which actions each skill may call
- `agent_profile`: which agent style this rollout uses

The model generates a multi-round dialogue plan within a constrained action space. `system_prompt` specifies that the only returned field is `rounds`, that each round has only `user_text`, `skill_name`, `actions`, `respond`, and that each round is bound to exactly one `skill_name`.

After the model returns, the function flattens `rounds` into a sequence of steps and filters for validity along the way.

    for item in rounds:
        ...
        user_text = (item.get("user_text") or "").strip()
        skill_name = item.get("skill_name")
        if not user_text or not skill_name or skill_name not in allowed_by_skill:
            continue
        ...
        for action in actions:
            ...
            action_name = action.get("action_name")
            if action_name not in allowed_by_skill.get(skill_name, set()):
                continue
            normalized.append({
                "kind": "tool",
                "round_index": round_index,
                "user_text": user_text,
                "selected_skill": skill_name,
                ...
            })
        respond = item.get("respond")
        if not isinstance(respond, dict):
            continue
        normalized.append({
            "kind": "reply",
            "round_index": round_index,
            "user_text": user_text,
            "selected_skill": skill_name,
            ...
        })

What it does: checks that `skill_name` is among the candidate skills, checks that `action_name` belongs to that skill's allowed actions, expands each round's actions and its single respond into an executable step list, and fills in `round_index`, `user_text`, and `selected_skill` for every step.

`_ensure_completion(...)` enforces how an episode ends. If the model finishes with only a vague clarification while the case could still reach a disposition with the selected skills, it appends a round with a timeline explanation or a document request, so the episode ends in a verifiable disposition.

#### tools.py

The tools in this sandbox are examples used to record trajectories, observations, and side effects. A tool layer in a real application usually includes the following kinds:

- Database reads: look up user information, policy summaries, claim status, review timelines, uploaded documents, payment records, and past tickets
- Retrieval tools: search the claims knowledge base, policy term explanations, FAQs, regulatory or internal company rules, and web search for public information
- Database writes: create document requests, create callback tasks, write case notes, file reconsideration tickets, update case tags, and record why a human took over
- Workflow / approval tools: trigger manual review, submit to the adjuster queue, assign to a specific reviewer, and move the case to the next stage
- Notification tools: send SMS, in-app messages, and emails, and push document reminders and progress updates
- Document tools: read image metadata, check whether the ID card, invoices, and discharge summary are complete, fetch OCR results, and compare document fields for conflicts
- Payment and settlement tools: check payout status, look up failure reasons, check bank receipts, retry payments, and sync settlement system status

The tool layer turns each `load_action` into a concrete observation. No real business systems are connected here; state is simulated from the case content.

    def execute_tool(case: CaseSpec, mutable_state: Dict[str, Any], tool_name: str, args: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], str]:
        if tool_name in READ_TOOLS:
            return _execute_read(case, tool_name, args)
        if tool_name in WRITE_TOOLS:
            return _execute_write(case, mutable_state, tool_name, args)
        return False, {"error": "unknown_tool"}, "Unknown tool"

Read tools take fields directly from `CaseSpec`. Write tools modify `mutable_state["writes"]` so that the verifier can later judge side effects.

    if tool_name == "send_document_request":
        if case.claim_case.get("status") != "pending_docs":
            return False, {"error": "invalid_state"}, "document_request_not_allowed"
        writes["document_request_sent"] = True
        return True, {"sent": True}, "document_request_sent"

#### prompt_format.py

Main entry point: `build_turn_samples(...)`. It expands a complete trajectory into step-level training samples.

    for round_spec in rounds:
        ...
        samples.append({
            "input": build_input_text(...),
            "target": json.dumps({
                "type": "load_skills",
                "skill_name": skill_name,
                "message": ...,
            }, ensure_ascii=False),
            "meta": {...},
        })
        ...
        for action in round_spec["actions"]:
            samples.append({
                "input": build_input_text(...),
                "target": json.dumps({
                    "type": "load_action",
                    "skill_name": skill_name,
                    "action_name": action["action_name"],
                    "action_input": action["action_input"],
                    "message": ...,
                }, ensure_ascii=False),
                "meta": {...},
            })
        ...
        samples.append({
            "input": build_input_text(...),
            "target": json.dumps({
                "type": "respond",
                "skill_name": skill_name,
                "message": reply_message,
            }, ensure_ascii=False),
            "meta": {...},
        })

Corresponding protocol:

- The input side accumulates `System prompt / Available skills / Signals / Dialogue history / Loaded skills / Loaded skill content / Recent observations`
- The output side only allows `load_skills / load_action / respond`
- Each step is bound to exactly one `skill_name`
- Dialogue history, loaded skills, and recent observations keep accumulating within the same `id`

`build_input_text(...)` builds the input text. `_group_reasoning_rounds(...)` organizes the rollout trajectory by round into a `user -> skill -> actions -> reply` structure.

#### exporters.py

`exporters.py` writes trajectories into the final data format.

The SFT export entry point is `build_policy_sft_records(...)`.

    for idx, sample in enumerate(build_turn_samples(case, trajectory), start=1):
        rows.append({
            "id": conversation_id,
            "turn_id": sample.get("meta", {}).get("turn_id", idx),
            "input": sample["input"],
            "target": sample["target"],
            "meta": {
                **sample.get("meta", {}),
                "family": case.family,
                "difficulty": case.difficulty,
                "success": verifier.success,
            },
        })

The RL export entry point is `build_rl_rollout_record(...)`.

    for idx, sample in enumerate(policy_samples):
        shaped_samples.append({
            "turn_id": sample.get("meta", {}).get("turn_id", idx + 1),
            "input": sample["input"],
            "target": sample["target"],
            "rollout_output": sample["target"],
            "reward": verifier.reward if idx == last_idx else None,
        })

The RL data here is episode-level: one `id` corresponds to one complete rollout. Intermediate steps have no reward; only the last step carries the final reward. See the "Reward" section below for how the reward is computed.

## Reward

The reward is computed in `verify_episode(...)` in `claim_sandbox/verifier.py`. It first extracts the final disposition, action coverage, hard guardrails, and soft penalties from the complete rollout, then combines them into the final episode-level reward.

### Overview

The reward has three parts:

1. The base terminal reward, `outcome`
2. Process quality rewards: `policy`, `evidence`, `efficiency`, `communication`
3. Penalties: hard guardrails and soft penalties

`reward_components` records four items of interest: correct resolution, wrong commitment, unnecessary handoff, and finishing in few turns. This field is used for the reward ledger breakdown and manual inspection. The final `reward` follows the overall formula below:

```text
R_raw = 1.00 × r_outcome + 0.25 × r_policy + 0.15 × r_evidence + 0.10 × r_efficiency + 0.10 × r_communication - penalty_value
```

In the current code:

    predicted_disposition = _infer_predicted_disposition(trajectory)
    hard_guardrails = _hard_guardrails(case, trajectory)
    soft_penalties = _soft_penalties(case, trajectory)
    rubric = {
        "outcome": _score_outcome(case, predicted_disposition, hard_guardrails),
        "policy": 0.0,
        "evidence": 0.0,
        "efficiency": _score_efficiency(trajectory),
        "communication": 0.0,
    }
    llm_rubric = _llm_rubric(case, trajectory, predicted_disposition) or {}
    for key in ["policy", "evidence", "communication"]:
        if key in llm_rubric and isinstance(llm_rubric[key], (int, float)):
            rubric[key] = round(float(llm_rubric[key]), 3)
    penalty_value = sum(HARD_GUARDRAIL_PENALTIES[x] for x in hard_guardrails) + sum(SOFT_PENALTIES[x] for x in soft_penalties)
    reward = 1.00 * rubric["outcome"] + 0.25 * rubric["policy"] + 0.15 * rubric["evidence"] + 0.10 * rubric["efficiency"] + 0.10 * rubric["communication"] - penalty_value
    reward = round(reward, 4)

`policy`, `evidence`, and `communication` come directly from the LLM rubric. In `claim_sandbox/verifier.py`, `verify_episode(...)` calls `_llm_rubric(case, trajectory, predicted_disposition)` and writes these three scores. `outcome`, `efficiency`, hard guardrails, and soft penalties are rule-based.

### Reward details

#### outcome

`outcome` is computed by `_score_outcome(...)`. It first checks whether any `hard_guardrails` were triggered; if so, the score is `-1.0`. Without hard guardrails, it compares `predicted_disposition` with `ideal_disposition`:

- Correct self-service resolution: `1.0`
- Correct intermediate action, such as a document request or reconsideration: `0.8`
- Correct escalation to a human: `0.6`
- Safe clarification: `0.2`
- Wrong disposition direction: `-0.7`

#### policy

`policy` is scored by `_llm_rubric(...)` and measures coverage of `required_actions`.

#### evidence

`evidence` is scored by `_llm_rubric(...)` and measures whether the action chain and the reply are grounded in facts: whether the key read tools were run, whether the final disposition matches the tool results, and whether the reply clearly departs from the tool results.

#### efficiency

`efficiency` is computed by `_score_efficiency(...)` and only looks at `trajectory.turns`:

- `<= 5`: `1.0`
- `<= 6`: `0.7`
- `<= 7`: `0.4`
- `> 7`: `0.1`

This score reflects the dialogue length and the length of the action chain.

#### communication

`communication` is scored by `_llm_rubric(...)` and only evaluates whether the reply text itself is direct, clear, safe, and usable.

`_llm_rubric(...)` lives in `claim_sandbox/verifier.py`. Its inputs are `required_actions`, `subissues`, `assistant_reply`, `tool_observations`, and `audit_log`; its outputs are `policy`, `evidence`, `communication`, and `notes`.

#### hard guardrails

`hard_guardrails` is computed by `_hard_guardrails(...)` and checks:

- Unauthorized promises about payment or outcomes (phrases listed in `EXACT_PROMISE_PHRASES`)
- Invalid write operations
- A required escalation to a human that did not happen

#### soft penalties

`soft_penalties` is computed by `_soft_penalties(...)` and checks:

- Handing over to a human when the case could be self-served
- Duplicate side-effect writes
- Tool loops or too many actions
- Overly long dialogues

## Harness

The runtime harness uses the same input and output protocol as training, with the control logic placed in a runtime control plane.

### Runtime protocol

At runtime, the `input` sent to the model at each step reuses `build_input_text(...)`, with the same block structure as SFT.

- User question -> `load_skills` -> `load_action` -> `respond`
- Each step allows only one `skill_name`
- `load_action` may only call actions allowed by the current skill
- Output must be a strict JSON object
- The `thought` field is forbidden
- Every step must carry a `message`

Input blocks:

- `SYSTEM`
- `Available skills`
- `Signals`
- `Dialogue history`
- `Loaded skills`
- `Loaded skill content`
- `Recent observations`

The model may only return one of three JSON formats:

    {"type": "load_skills", "skill_name": "pending_docs_resolution", "message": "Based on the user's question, pending_docs_resolution is needed to handle: \"Just tell me which documents are missing.\""}

    {"type": "load_action", "skill_name": "pending_docs_resolution", "action_name": "get_required_documents", "action_input": {}, "message": "Ran the get_required_documents action; the result is {\"required_documents\":[\"ID card\",\"Medical record\",\"Invoice\"]}"}

    {"type": "respond", "skill_name": "pending_docs_resolution", "message": "I checked your claim: it is currently pending documents, and the missing documents are the ID card, the medical record, and the invoice."}

### ServingHarnessSession

`ServingHarnessSession` prepares the candidate skills, maintains session state, calls the model step by step, validates model output, runs tools, feeds observations back, and calls `verify_episode(...)` at the end.

Initialization starts with skill routing, writing the candidate skills, skill cards, and route rationale into the runtime trajectory.

    selected_cards, rationale = select_skills_for_case(case, case.initial_user_message)
    self.available_skill_cards = [x.to_dict() for x in selected_cards]
    self._skill_by_slug = {x["slug"]: x for x in self.available_skill_cards}
    self._allowed_actions = {
        slug: set(card.get("read_tools", []) + card.get("write_tools", []))
        for slug, card in self._skill_by_slug.items()
    }

Session state fields initialized:

- `_history_acc`: accumulated dialogue history
- `_loaded_skill_cards_acc`: accumulated loaded skill content
- `_observation_acc`: accumulated tool observations
- `_current_skill`: the skill in use for the current round
- `_phase`: the current phase, always one of `expect_skill`, `expect_action_or_respond`, `idle`
- `_tool_cache`: read tool cache
- `_writes`: world state produced by write tools
- `_round_tool_budget`: per-round tool budget
- `_episode_tool_budget`: tool budget for the whole dialogue

Input is built by `build_current_input()`, which calls `build_input_text(...)` directly.

    history_before = self._history_acc + [f"User: {self._current_user_text}"]
    return build_input_text(
        self.case,
        available_skill_cards=self.available_skill_cards,
        history_lines=history_before,
        loaded_skill_cards=self._loaded_skill_cards_acc,
        recent_observations=self._observation_acc,
    )

The main loop is `run_until_respond(...)`. Each iteration builds the input, calls the model, validates the output, and then runs the action or reply. Runtime sampling goes through `_sample_next_action(...)`, which calls `chat_json_detailed(...)` with up to 4 retries per step.

    input_text = self.build_current_input()
    raw, llm_error = self._sample_next_action(input_text)
    valid, normalized, error = self._validate_action(raw)
    if self._apply_action(normalized):
        break

`_validate_action(...)` checks:

- Whether the output is a JSON object
- Whether it includes `thought`
- Whether `type` is one of `load_skills`, `load_action`, `respond`
- Whether `message` is present
- Whether `skill_name` is among the candidate skills
- Whether `action_name` belongs to the current skill's allowed actions
- Whether `action_input` is a JSON object
- Whether `respond` is aligned with the current skill
- Whether the per-round or per-dialogue tool budget is exceeded

The runtime control plane adds these constraints:

- Tool budgets: `_round_tool_budget` caps actions per round, and `_episode_tool_budget` caps actions for the whole dialogue
- Read cache: the same read tool with the same arguments hits `_tool_cache`, and the audit log `message` gets a `cache_hit:` prefix
- Duplicate side-effect blocking: repeating the same write tool with the same arguments returns `duplicate_side_effect_blocked`

`respond` has no semantic gate before replying; it only checks the protocol fields, the current phase, and `skill_name` consistency.

`_apply_action(...)` runs actions. `load_action` calls `execute_tool(...)` and writes the result into:

- `trajectory.tool_observations`
- `trajectory.audit_log`
- `trajectory.reasoning_steps`
- `_observation_acc`

`respond` turns `message` into an `AssistantReply`. At runtime the fields that matter are `text`, `disposition`, and `selected_skills`. `disposition` is inferred from the write actions that have succeeded so far, such as a reconsideration ticket, a document request, or an escalation to a human. The `AssistantReply` struct also keeps `addressed_subissues`, `cited_facts`, and `commitment_made`; on the serving side they default to an empty list, an empty object, and `None`.

If the model does not return valid JSON, or the current step fails `_validate_action(...)`, the runtime harness triggers a failsafe reply and records a `terminated_reason`. Possible values of `terminated_reason`:

- `request_failed`
- `client_timeout`
- `empty_response`
- `invalid_json_text`
- `non_object_json`
- `non_json_output`
- `gateway_non_json`
- `api_error`
- `invalid_response_shape`
- `client_exception`
- `thought_not_allowed`
- `invalid_type`
- `missing_message`
- `unknown_skill`
- `unexpected_load_skills`
- `unexpected_load_action`
- `unexpected_respond`
- `invalid_action_for_skill`
- `skill_mismatch`
- `invalid_action_input`
- `round_tool_budget_exceeded`
- `episode_tool_budget_exceeded`
- `max_steps_exceeded`

Besides `trajectory` and `verifier`, `ServingRunResult` contains two kinds of runtime information:

- `world_state_delta`: the final changes write tools made to the sandbox world state
- `control_plane_stats`: runtime control plane statistics, including `round_count`, `turn_count`, `read_calls`, `write_calls`, `cache_hits`, `duplicate_write_blocks`, `loaded_skill_count`, `round_tool_budget`, `episode_tool_budget`, `blocked_responds`, `budget_blocks`, `required_action_count`, `executed_required_action_count`

### regression.py

`regression.py` runs the serving harness on a fixed case set and writes three kinds of files: `serving_regression_summary.json`, `serving_regression_runs.jsonl`, and `replay_cases/<case_id>.json`.

The entry function is `run_regression_cases(...)`.

    for case in cases:
        session = ServingHarnessSession(case, temperature=temperature)
        session.start_user_turn(case.initial_user_message)
        result = session.run_until_respond(max_steps=max_steps)
        runs.append(result)

After each case, `trajectory` and `verifier` are saved as regression results. Summary fields:

- `total_cases`
- `success_count`
- `avg_reward`
- `by_case`
- `by_bucket`
- `terminated_reason_counts`
- `hard_guardrail_counts`
- `soft_penalty_counts`

Each record in `by_case` keeps:

- `case_id`
- `family`
- `difficulty`
- `agent`
- `expected_disposition`
- `predicted_disposition`
- `selected_skills`
- `hard_guardrails`
- `soft_penalties`
- `reward_components`
- `data_quality`
- `reward`
- `success`
- `bucket`
- `terminated_reason`
- `control_plane_stats`

`by_bucket` aggregates statistics by each case's `bucket` field. The current case set mainly uses `train`, `holdout`, `redteam`, and `replay_pool`.

### Run command

Run a single case:

    python3 -m claim_sandbox.harness.serving --case-id claim_rejection_dispute_0202 --max-steps 8

### Coverage

The harness covers:

- Input construction identical to the training protocol
- Step-level JSON output validation
- Skill / action whitelist validation
- Tool execution and observation feedback
- Dialogue history accumulation
- Runtime failsafe reply
- Tool budgets
- Read cache
- Duplicate side-effect write blocking
- Terminal verifier scoring
- Fixed-case regression
