# SEED Algorithm Technical Note

## Abstract

This document updates the description of the SEED algorithm in this repository
based on the paper `ICLR_27_SEED (3).pdf`. In the paper, SEED stands for
**SElf-Evolving On-Policy Distillation**. Its core goal in long-horizon agentic
RL is to turn hindsight information, which only becomes visible once a full
trajectory has finished, into a token-level training signal, without relying on
extra skill prompts, retrieval stores, or external analyzers at inference time.

SEED has two stages:

1. **Hindsight Skill SFT**: Collect ordinary agent trajectories, use an external
   analyzer to generate an episode-level hindsight skill for each complete
   trajectory, then run SFT on the current backbone so that it learns to "read a
   complete trajectory and summarize a skill".
2. **Self-Evolving OPD**: During RL, a frozen snapshot of the current policy both
   samples on-policy trajectories and serves as the synchronized analyzer that
   summarizes them. The policy being trained then re-scores the same sampled
   action tokens under the plain context and the skill-augmented context, builds
   a gated OPD loss from the skill-induced log-prob shift, and optimizes it
   jointly with the GRPO loss.

The most important conclusion for inference is: **SEED's skills are used only as
privileged supervision during training; only the ordinary policy is deployed at
inference time**.

## 1. Core Idea of the Paper

SEED is designed around three requirements of long-horizon agentic RL:

| Requirement | SEED mechanism |
| --- | --- |
| on-policy | Trajectories are sampled by the current policy snapshot, and skills are produced by analyzing them with the same snapshot |
| dense | The same action token is re-scored under the plain context and the skill context, producing a token-level OPD signal |
| self-evolving | After each update, both the actor and the analyzer are refreshed in sync with the latest checkpoint |

Intuitively, the episode reward only tells us whether a trajectory ultimately
succeeded or failed, whereas the hindsight skill from a complete trajectory can
point out "the successful workflow", "the key observations", and "rules for
avoiding failure". SEED does not use these skills as inference prompts. Instead,
it distills the effect a skill has on the current policy's action probabilities
into the model parameters.

## 2. Current Code Paths

This repository keeps the SFT + self-evolving OPD main path used in the paper:

| Path | Typical script | Description |
| --- | --- | --- |
| Paper main path | `examples/seed_trainer/run_*_sft*.sh` | Initializes the policy and the `policy_vllm` analyzer from the hindsight-skill SFT checkpoint and enables the gated OPD loss via `opd_loss_coef` |

The launch script names no longer include the historical field
`episode_no_skill_loss`. That field meant the extra skill-generation LM auxiliary
loss is disabled by default; it did not mean the paper-style OPD loss was turned
off. OPD is still enabled through `actor_rollout_ref.actor.opd_loss_coef=0.01`.

## 3. Stage 1: Hindsight Skill SFT

In the paper, 180 SFT tasks are selected per benchmark and 8 rollouts are
sampled per task, for a total of 1,440 complete trajectories:

$$
B_j = \{\tau_{j,k}\}_{k=1}^{K_0}, \quad K_0=8.
$$

Each trajectory contains the task description, observations, actions, rewards,
and the final outcome. The external analyzer generates an episode-level skill
for each complete trajectory:

$$
s_\tau = A_{\text{ext}}(\tau).
$$

For successful trajectories, the skill should summarize a reusable workflow; for
failed trajectories, it should summarize an avoidance rule. The paper's prompt
requires valid JSON output:

```json
{
  "episode_summary": "string",
  "episode_skill": "string"
}
```

Samples that pass format validation form:

$$
D_{\text{sft}} = \{(x_\tau, s_\tau): v_\tau=1\},
$$

where \(x_\tau\) is the serialized trajectory-analysis input. The SFT objective is
the standard negative log-likelihood:

$$
L_{\text{sft}}(\theta)
= -\mathbb{E}_{(x_\tau,s_\tau)\sim D_{\text{sft}}}
\sum_\ell \log \pi_\theta(s_{\tau,\ell}\mid x_\tau,s_{\tau,<\ell}).
$$

The SFT checkpoint initializes both the subsequent RL actor and the subsequent
synchronized trajectory analyzer.

Corresponding scripts:

```bash
# ALFWorld
bash scripts/sft/alfworld/prepare_data.sh
bash scripts/sft/alfworld/train_sft.sh

# WebShop
bash scripts/sft/webshop/prepare_data.sh
bash scripts/sft/webshop/train_sft.sh

# Search-based QA
bash scripts/sft/search/prepare_data.sh
bash scripts/sft/search/train_sft.sh

# EZPoints
bash scripts/sft/ezpoints/prepare_data.sh
bash scripts/sft/ezpoints/train_sft.sh

# Sokoban
bash scripts/sft/sokoban/prepare_data.sh
bash scripts/sft/sokoban/train_sft.sh
```

## 4. Stage 2: Self-Evolving OPD

At the start of the \(k\)-th policy update, SEED freezes the current policy as
\(\pi_{\theta_{\text{old}}}\). This snapshot plays two roles:

1. actor: samples on-policy trajectories in the environment;
2. analyzer: reads complete trajectories and generates hindsight skills.

For task \(q\), \(N\) trajectories are sampled:

$$
G_q = \{\tau_q^{(1)},\ldots,\tau_q^{(N)}\},\quad
\tau_q^{(n)} \sim \pi_{\theta_{\text{old}}}(\cdot\mid q).
$$

Both the paper and the main scripts use a rollout group size of \(N=8\). The
synchronized analyzer generates:

$$
s_q^{(n)} = A_{\theta_{\text{old}}}(x_{\tau_q^{(n)}}).
$$

This forms the self-evolving loop: as the policy gets stronger, the distribution
of sampled trajectories shifts, and the analysis ability of the same checkpoint
also changes with training. Hindsight supervision therefore does not stay tied to
an old policy or a static skill library.

## 5. Skill-Augmented Re-Scoring

Let \(h_{q,n,t}\) denote the plain interaction history at step \(t\) of the
\(n\)-th trajectory, and \(a_{q,n,t}\) the already-sampled action token sequence.
SEED does not resample actions; instead, it inserts the episode skill into the
context:

$$
\tilde{h}_{q,n,t}=H(h_{q,n,t},s_q^{(n)}).
$$

The current policy under training, \(\pi_\theta\), computes two log-probs for the
same sampled action tokens:

$$
\ell^{\text{skill}}_{q,n,t,\ell}
=\log\pi_\theta(a_{q,n,t,\ell}\mid \tilde{h}_{q,n,t},a_{q,n,t,<\ell}),
$$

$$
\ell^\theta_{q,n,t,\ell}
=\log\pi_\theta(a_{q,n,t,\ell}\mid h_{q,n,t},a_{q,n,t,<\ell}).
$$

The two branches share the same model parameters, but the teacher branch sees the
hindsight skill while the student branch sees only the plain context. Gradients
flow only through the plain student branch.

The skill-induced log-prob shift is defined as:

$$
\Delta_{q,n,t,\ell}
=\operatorname{sg}\left[
\ell^{\text{skill}}_{q,n,t,\ell}
-\ell^\theta_{q,n,t,\ell}
\right],
$$

where \(\operatorname{sg}\) denotes stop-gradient. A sigmoid gate then controls
the strength of OPD:

$$
g_{q,n,t,\ell}=\sigma(\beta_{\text{opd}}\Delta_{q,n,t,\ell}).
$$

The paper's default is \(\beta_{\text{opd}}=5.0\).

The OPD loss is:

$$
L_{\text{opd}}(\theta)
=
\mathbb{E}_{q,n,t,\ell}
\left[
m_{q,n,t,\ell}\,
g_{q,n,t,\ell}\,
\left(
\operatorname{sg}[\ell^{\text{skill}}_{q,n,t,\ell}]
-\ell^\theta_{q,n,t,\ell}
\right)
\right].
$$

Because both the gate and the teacher log-prob are detached, this objective is
equivalent to a gate-weighted NLL: tokens supported by the skill receive stronger
distillation, while the influence of tokens the skill does not support is
attenuated.

Implementation entry points:

- `compute_opd_loss` in
  [`verl/trainer/ppo/core_algos.py`](../../verl/trainer/ppo/core_algos.py)
- the actor update in
  [`verl/workers/actor/dp_actor.py`](../../verl/workers/actor/dp_actor.py)
- SEED analysis and teacher/OPD signal construction in
  [`verl/trainer/ppo/ray_trainer.py`](../../verl/trainer/ppo/ray_trainer.py)

## 6. Joint Objective with GRPO

SEED keeps the group-relative RL objective. For the task group \(G_q\), compute
the mean and standard deviation of trajectory outcomes:

$$
\mu_q=\frac{1}{N}\sum_{n=1}^N R(\tau_q^{(n)}),\quad
\sigma_q=
\sqrt{
\frac{1}{N}\sum_{n=1}^N
\left(R(\tau_q^{(n)})-\mu_q\right)^2
}.
$$

The trajectory-level advantage is:

$$
A^{\text{rl}}_{q,n}
=
\frac{R(\tau_q^{(n)})-\mu_q}{\sigma_q+\epsilon}.
$$

This advantage is broadcast to the valid action tokens of the trajectory. The
PPO/GRPO ratio is:

$$
\rho_{q,n,t,\ell}(\theta)
=
\exp\left(
\ell^\theta_{q,n,t,\ell}
-\ell^{\text{old}}_{q,n,t,\ell}
\right).
$$

Final objective:

$$
L_{\text{SEED}}(\theta)
=
L_{\text{rl}}(\theta)
+
\lambda_{\text{opd}} L_{\text{opd}}(\theta).
$$

The paper's defaults are \(\lambda_{\text{opd}}=0.01\) and a KL coefficient of
\(0.01\).

## 7. Inference

At inference time, SEED uses only the plain interaction history:

$$
a_t \sim \pi_\theta(\cdot\mid h_t).
$$

It does not need:

- a trajectory analyzer;
- a skill bank;
- skill retrieval;
- an extra skill prompt;
- privileged context.

This is the key difference between SEED and methods such as Skill-Prompt and
Skill-GRPO*: those still depend on skill context at evaluation time, whereas SEED
internalizes the behavioral effect of skills into the parameters.

## 8. Main Experimental Results

The table below summarizes the aggregate GRPO vs. SEED comparison from Table 1
of the paper. Each cell is `GRPO -> SEED (+gain)`.

| Backbone | ALFWorld Avg. | Search-QA Avg. | WebShop Score | WebShop Succ. |
| --- | ---: | ---: | ---: | ---: |
| Qwen2.5-3B-Instruct | 75.0 -> 91.8 (+16.8) | 36.4 -> 45.7 (+9.3) | 79.8 -> 88.5 (+8.7) | 63.3 -> 78.9 (+15.6) |
| Qwen2.5-7B-Instruct | 81.2 -> 96.1 (+14.9) | 42.0 -> 48.6 (+6.6) | 80.9 -> 89.7 (+8.8) | 72.6 -> 78.1 (+5.5) |
| Qwen3-1.7B-Instruct | 46.1 -> 92.0 (+45.9) | 40.8 -> 42.2 (+1.4) | 67.3 -> 87.1 (+19.8) | 38.3 -> 77.3 (+39.0) |

The paper also reports:

- Across the three backbones, SEED improves over GRPO by +14.9 to +45.9 on
  ALFWorld, +1.4 to +9.3 on Search-QA, +8.7 to +19.8 on WebShop Score, and
  +5.5 to +39.0 on WebShop Success.
- Skill-Prompt and Skill-GRPO* insert skills at evaluation time, yet SEED, which
  uses no skill prompt at inference, is still stronger on the vast majority of
  aggregate metrics.
- On ALFWorld, SEED with Qwen2.5-3B reaches 91.8, higher than SDAR's 84.4 and
  GRPO+OPSD's 81.2.

## 9. Sample Efficiency, Generalization, and Ablations

### Sample Efficiency

Table 6 of the paper shows that SEED outperforms GRPO at every training-data
fraction:

| Benchmark | Data | GRPO | SEED | Gain |
| --- | ---: | ---: | ---: | ---: |
| ALFWorld | 20% | 27.3 | 40.7 | +13.4 |
| ALFWorld | 40% | 42.2 | 58.9 | +16.7 |
| ALFWorld | 60% | 56.3 | 80.7 | +24.4 |
| ALFWorld | 80% | 58.6 | 88.8 | +30.2 |
| ALFWorld | 100% | 75.0 | 91.8 | +16.8 |
| WebShop | 20% | 31.3 | 37.5 | +6.2 |
| WebShop | 40% | 45.3 | 53.1 | +7.8 |
| WebShop | 60% | 57.0 | 62.5 | +5.5 |
| WebShop | 80% | 63.6 | 75.0 | +11.4 |
| WebShop | 100% | 63.3 | 78.9 | +15.6 |

Key takeaway: on ALFWorld, SEED reaches 80.7 with 60% of the data, already above
full-data GRPO at 75.0.

### ALFWorld Unseen Generalization

From Table 7 of the paper, Qwen2.5-3B on the ALFWorld unseen split:

| Method | Pick | Look | Clean | Heat | Cool | Pick2 | Avg. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GRPO | 73.9 | 60.0 | 82.4 | 59.3 | 72.7 | 76.9 | 70.9 |
| SEED | 90.4 | 78.3 | 79.5 | 94.3 | 86.2 | 88.2 | 86.2 |
| Gain | +16.5 | +18.3 | -2.9 | +35.0 | +13.5 | +11.3 | +15.3 |

SEED outperforms GRPO on 5 of the 6 unseen task families, with an average gain of
15.3.

### Ablations

ALFWorld ablations from Table 2 of the paper:

| Variant | Avg. | Drop |
| --- | ---: | ---: |
| SEED | 91.8 | 0.0 |
| w/o Hindsight Skill SFT | 86.0 | -5.8 |
| w/o Self-Evolving OPD | 87.0 | -4.8 |
| w/o On-Policy Skill | 84.4 | -7.4 |

All three components matter: SFT initialization, continued OPD distillation, and
generating on-policy skills from the current policy's own trajectories.

### Extension to Visual Agents

Table 8 of the paper reports results on Qwen2.5-VL-3B-Instruct:

| Method | Sokoban 6x6 | EZPoints |
| --- | ---: | ---: |
| GRPO | 67.1 | 86.9 |
| SEED | 82.0 | 100.0 |
| Gain | +14.9 | +13.1 |

## 10. Key Configuration

Key settings in the paper and in the current paper-style scripts:

| Setting | Value |
| --- | --- |
| SFT tasks | 180 per benchmark |
| SFT rollouts per task | 8 |
| SFT trajectories | 1,440 |
| SFT epochs | 3 |
| RL updates | 150 in paper; current scripts commonly set `trainer.total_epochs=160` |
| Rollout group size | `env.rollout.n=8` |
| OPD coefficient | `actor_rollout_ref.actor.opd_loss_coef=0.01` |
| OPD gate sharpness | `actor_rollout_ref.actor.opd_gate_beta=5.0` |
| KL coefficient | `actor_rollout_ref.actor.kl_loss_coef=0.01` |
| Learning rate | `actor_rollout_ref.actor.optim.lr=1e-6` |
| Analyzer backend | `algorithm.seed.analysis_backend=policy_vllm` for paper-style SEED |
| Skill mode | `algorithm.seed.skill_mode=episode_only` for paper-style episode-skill OPD |

## 11. Implementation Map

| Function | Main implementation location |
| --- | --- |
| SFT data construction: ALFWorld | [`scripts/sft/alfworld`](../../scripts/sft/alfworld) |
| SFT data construction: WebShop | [`scripts/sft/webshop`](../../scripts/sft/webshop) |
| SFT data construction: Search-QA | [`scripts/sft/search`](../../scripts/sft/search) |
| SFT data construction: EZPoints | [`scripts/sft/ezpoints`](../../scripts/sft/ezpoints) |
| SFT data construction: Sokoban | [`scripts/sft/sokoban`](../../scripts/sft/sokoban) |
| SFT training | [`verl/trainer/fsdp_sft_trainer.py`](../../verl/trainer/fsdp_sft_trainer.py) |
| Multi-step rollout | [`agent_system/multi_turn_rollout/rollout_loop.py`](../../agent_system/multi_turn_rollout/rollout_loop.py) |
| SEED trainer integration | [`verl/trainer/ppo/ray_trainer.py`](../../verl/trainer/ppo/ray_trainer.py) |
| Trajectory-analysis prompt and JSON parsing | [`seed/analysis.py`](../../seed/analysis.py) |
| Skill injection into observations | [`seed/prompting.py`](../../seed/prompting.py) |
| gated OPD loss | [`verl/trainer/ppo/core_algos.py`](../../verl/trainer/ppo/core_algos.py) |
| actor update | [`verl/workers/actor/dp_actor.py`](../../verl/workers/actor/dp_actor.py) |
| SEED configuration | [`verl/trainer/config/ppo_trainer.yaml`](../../verl/trainer/config/ppo_trainer.yaml) |

## 12. Relationship to the Legacy Teacher-Advantage Path

The current code still supports an earlier path: it computes the difference
between the enhanced-prompt log-prob and the ordinary-prompt log-prob, and adds
this delta to the PPO advantage with a weight:

$$
A^{\text{SEED}}
=
A^{\text{ep}}
+w_{\text{ep}}\Delta^{\text{ep}}
+w_{\text{step}}\Delta^{\text{step}}.
$$

This path is controlled by `episode_skill_teacher_advantage_w` and
`step_skill_teacher_advantage_w`. When
`actor_rollout_ref.actor.opd_loss_coef > 0`, the code prefers the paper-style
auxiliary OPD loss, and these teacher-advantage weights are ignored or set to 0.

Therefore:

- To reproduce the paper's main line, use the SFT checkpoint + the `policy_vllm`
  analyzer + `opd_loss_coef=0.01`;
- For direct ablations or quick experiments, override the analyzer, loss, and
  teacher-advantage settings on the corresponding `run_*_sft*.sh` via environment
  variables or Hydra arguments. Separate legacy launch scripts are no longer
  maintained.

## 13. Conclusion

The paper's version of SEED can be summarized as:

$$
\text{SEED}
=
\text{GRPO outcome optimization}
+
\text{self-evolving hindsight-skill OPD}.
$$

It is not about "adding one more skill prompt" at inference time, nor about
retrieving from a static skill library. During training, the current policy
generates hindsight skills from its own complete trajectories, and the effect of
those skills on action probabilities is distilled back into the ordinary policy.
Decision-making and trajectory-analysis ability thus co-evolve during training,
and the result is an agent policy that needs no extra context at inference time.
