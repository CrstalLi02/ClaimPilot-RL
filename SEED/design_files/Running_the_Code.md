# SEED ALFWorld Training Guide

## 1. Environment Setup

Create a virtual environment and install the dependencies:

```bash
# Create a conda environment (recommended)
conda create -n seed python=3.12 -y
conda activate seed

# Install dependencies
cd SEED
pip install -e .
```

Download the ALFWorld data (about 4 GB):

```bash
alfworld-download -f
```

The data is downloaded to `~/.alfworld/`.

---


## 2. Model Download

### Option A: Use OPID-ALFWorld-1.7B (recommended)

Download the model provided by the authors, which has **already been trained on ALFWorld data**. It comes from another project by the same authors and can already handle ALFWorld tasks, so its baseline rollouts succeed more often and data construction is more efficient:

```bash
cd models

huggingface-cli download Jinyang23/OPID-ALFWorld-1.7B \
    --local-dir OPID-ALFWorld-1.7B \
    --local-dir-use-symlinks False
```
### Option B: Use the original Qwen3-1.7B

If you want to train from scratch, you can use the base model instead:

```bash
cd models

huggingface-cli download Qwen/Qwen3-1.7B \
    --local-dir Qwen3-1.7B \
    --local-dir-use-symlinks False
```

## 3. SFT Training

### 3.1 Data Preparation

```bash
cd SEED

# Set the policy model path
export MODEL_PATH=models/OPID-ALFWorld-1.7B
export OUTPUT_DIR=SEED/outputs/alfworld_data

# Set the ALFWorld data path
export ALFWORLD_DATA=alfworld_data

# Configure the OpenAI API used for skill generation
export OPENAI_BASE_URL=your-base-url
export OPENAI_API_KEY=your-api-key
export OPENAI_MODEL=your-model


bash scripts/sft/alfworld/prepare_data.sh
```

### 3.2 SFT Training

```bash
cd SEED

export MODELS_ROOT=models

export DATA_DIR=SEED/outputs/alfworld_data
export EXPORT_MODEL_DIR=models/Qwen3-1.7B-alfworld-sft

bash scripts/sft/alfworld/train_sft.sh
```

## 4. RL Training

```bash
cd SEED

export HF_MODEL_PATH=models/Qwen3-1.7B-alfworld-sft
export MODEL_PATH=models/Qwen3-1.7B-alfworld-sft
export DEFAULT_LOCAL_DIR=SEED/outputs/rl/alfworld_qwen_self

# Optional: if installing flash-attn fails, use the SDPA attention backend instead (avoids installing flash-attn)
export VLLM_ATTENTION_BACKEND=TORCH_SDPA

bash examples/seed_trainer/run_alfworld_sft_qwen_self.sh
```

## 5. Model Merging

```bash
cd SEED

python3 scripts/model_merger.py merge \
    --backend fsdp \
    --local_dir outputs/rl/alfworld_qwen_self/global_step_160/actor \
    --target_dir models/alfworld_qwen_rl_final
```

## 6. Evaluation

### 6.1 Evaluating the SFT Model

```bash
cd SEED
# Replace with your actual model path
export MODEL_PATH=models/Qwen3-1.7B-alfworld-sft
export MODEL_NAME=Qwen3-1.7B-alfworld-sft

bash examples/prompt_agent/run_local_vllm_alfworld.sh
```

Example evaluation results:

| Metric | Example value |
|------|--------|
| `overall_success` | 0.408 (40.8%) |
| `category_macro_success` | 0.386 (38.6%) |
| `Pick` | 0.732 (73.2%) |
| `Look` | 0.377 (37.7%) |
| `Clean` | 0.543 (54.3%) |
| `Heat` | 0.348 (34.8%) |
| `Cool` | 0.173 (17.3%) |
| `Pick2` | 0.145 (14.5%) |

### 6.2 Evaluating the RL Model

```bash
cd SEED
# Replace with your actual model path
export MODEL_PATH=models/alfworld_qwen_rl_final
export MODEL_NAME=alfworld_qwen_rl_final

bash examples/prompt_agent/run_local_vllm_alfworld.sh
```


