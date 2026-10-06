verl x Ascend
===================================


We have added support for Huawei Ascend devices to verl.

Hardware Support
-----------------------------------

Atlas 200T A2 Box16

Atlas 800T A2


Installation
-----------------------------------

Base Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+-----------+-------------+
| software  | version     |
+-----------+-------------+
| Python    | == 3.10     |
+-----------+-------------+
| CANN      | == 8.1.RC1  |
+-----------+-------------+
| torch     | == 2.5.1    |
+-----------+-------------+
| torch_npu | == 2.5.1.RC1|
+-----------+-------------+


vllm & vllm-ascend
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To use vllm with verl, build and install vllm and vllm-ascend with the commands below. Note that the installation steps differ by machine type.

.. code-block:: bash
    
    # vllm
    git clone -b v0.7.3 --depth 1 https://github.com/vllm-project/vllm.git
    cd vllm
    pip install -r requirements-build.txt

    # for Atlas 200T A2 Box16
    VLLM_TARGET_DEVICE=empty pip install -e . --extra-index https://download.pytorch.org/whl/cpu/
    
    # for Atlas 800T A2
    VLLM_TARGET_DEVICE=empty pip install -e .

.. code-block:: bash
    
    # vllm-ascend
    git clone -b v0.7.3 --depth 1 https://github.com/vllm-project/vllm-ascend.git
    cd vllm-ascend
    export COMPILE_CUSTOM_KERNELS=1
    python setup.py install

Installing verl
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    git clone https://github.com/volcengine/verl.git
    cd verl
    pip install -r requirements-npu.txt
    pip install -e .

Notes on Third-Party Libraries
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+--------------+---------------+
| software     | description   |
+--------------+---------------+
| transformers | >= v4.52.0    |
+--------------+---------------+
| flash_attn   | not supported |
+--------------+---------------+
| liger-kernel | not supported |
+--------------+---------------+

1. Enabling --flash_attention_2 through transformers is supported; transformers must be version 4.52.0 or later.
2. Enabling flash attention acceleration through flash_attn is not supported.
3. Enabling liger-kernel is not supported.


Quick Start
-----------------------------------
Before real use, we recommend running a trial GRPO training of Qwen2.5-0.5B to verify that the environment and installation are correct.

1. Download the dataset and preprocess it into parquet format so that it contains the fields required to compute RL rewards

.. code-block:: bash

    python3 examples/data_preprocess/gsm8k.py --local_dir ~/data/gsm8k

2. Run training

.. code-block:: bash

    set -x

    export VLLM_ATTENTION_BACKEND=XFORMERS

    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        data.train_files=$HOME/data/gsm8k/train.parquet \
        data.val_files=$HOME/data/gsm8k/test.parquet \
        data.train_batch_size=128 \
        data.max_prompt_length=512 \
        data.max_response_length=128 \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
        actor_rollout_ref.actor.optim.lr=5e-7 \
        actor_rollout_ref.model.use_remove_padding=False \
        actor_rollout_ref.actor.entropy_coeff=0.001 \
        actor_rollout_ref.actor.ppo_mini_batch_size=64 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=20 \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.001 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=40 \
        actor_rollout_ref.rollout.enable_chunked_prefill=False \
        actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
        actor_rollout_ref.rollout.n=5 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=40 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        algorithm.kl_ctrl.kl_coef=0.001 \
        trainer.critic_warmup=0 \
        trainer.logger=['console'] \
        trainer.project_name='verl_grpo_example_gsm8k' \
        trainer.experiment_name='qwen2_7b_function_rm' \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=1 \
        trainer.save_freq=-1 \
        trainer.test_freq=5 \
        trainer.total_epochs=1 \
        trainer.device=npu $@


Current Support
-----------------------------------

+-----------+----------------------+-------------+-------------------+----------------------+
| algorithm |         model        | rewards mae |  throughput ratio |        hardware      |
+-----------+----------------------+-------------+-------------------+----------------------+
|   GRPO    | Qwen2.5-7B-instruct  |    0.38%    |        0.588      |  Atlas 200T A2 Box16 |
+-----------+----------------------+-------------+-------------------+----------------------+
|   GRPO    | Qwen2.5-32B-instruct |    0.30%    |        0.685      |  Atlas 200T A2 Box16 |
+-----------+----------------------+-------------+-------------------+----------------------+

GRPO training of Qwen2.5 is currently supported. GRPO training of Qwen2.5-VL will be supported once vllm-ascend fixes the following issues:

1. `issues#809 <https://github.com/vllm-project/vllm-ascend/issues/809>`_

2. `issues#825 <https://github.com/vllm-project/vllm-ascend/issues/825>`_


Accuracy Comparison
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For SFT-style algorithms, we expect the mean absolute error of the loss between Huawei Ascend devices and A100 under the same configuration to be <= 2%. The calculation is shown in the figure below. For more information, see the `accuracy calculation guide <https://www.hiascend.com/document/detail/zh/Pytorch/600/ptmoddevg/trainingmigrguide/LMaccuracy_0001.html>`_.

.. image:: https://github.com/eric-haibin-lin/verl-community/blob/main/docs/loss_comparison.png?raw=true
   :alt: loss_comparison

Empirically, for RL algorithms such as GRPO, we expect the mean absolute error of the rewards between Huawei Ascend devices and A100 under the same configuration to be <= 4%, calculated as in the figure above.


Throughput Comparison
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For both the Ascend NPU and the A100, average "perf/throughput" over the first 4 steps in the logs; throughput ratio = NPU average / A100 average. 



Roadmap
-----------------------------------

See the `roadmap <https://github.com/volcengine/verl/discussions/900>`_ for the support status of more features.



Disclaimer
-----------------------------------
The Ascend support code in verl is provided as reference examples only. For commercial use, please contact us through official channels. Thank you.