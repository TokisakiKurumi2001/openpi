## Instruction

## Clone this repo

```bash
git clone --recurse-submodules https://github.com/TokisakiKurumi2001/openpi.git

# Or if you already cloned the repo:
git submodule update --init --recursive
```

## Setup environment

The author use `uv`, you can setup following [this instruction](https://docs.astral.sh/uv/getting-started/installation/)

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

## Monitoring using wandb

You can create an [WandB account](https://wandb.ai/home), and then copy the API key

```bash
wandb login # then paste your API key
```

## Training

### Task 1: transfer cube

In the file `src/openpi/training/config.py`, line 913-926, you can see the training config

```python
TrainConfig(
    name="pi0_aloha_sim",
    model=pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
    data=LeRobotAlohaDataConfig(
        repo_id="lerobot/aloha_sim_transfer_cube_human",
        default_prompt="Pick up the cube with the right arm and transfer it to the left arm.",
        use_delta_joint_actions=False,
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
    freeze_filter=pi0_config.Pi0Config(
        paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
    ).get_freeze_filter(),
    num_train_steps=20_000,
),
```

First calculate the norm first

```bash
uv run scripts/compute_norm_stats.py --config-name pi0_aloha_sim
```

Now we train

```bash
CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_aloha_sim --exp-name=my_experiment --overwrite
```

### Task 2: insertion

In the file `src/openpi/training/config.py`, line 938-957, you can see the training config

```python
TrainConfig(
    name="pi0_aloha_sim_insert",
    # Here is an example of loading a pi0 model for LoRA fine-tuning.
    model=pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
    data=LeRobotAlohaDataConfig(
        repo_id="lerobot/aloha_sim_insertion_human",
        default_prompt="Insert the peg into the socket.",
        use_delta_joint_actions=False,
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
    num_train_steps=20_000,
    # The freeze filter defines which parameters should be frozen during training.
    # We have a convenience function in the model config that returns the default freeze filter
    # for the given model config for LoRA finetuning. Just make sure it matches the model config
    # you chose above.
    freeze_filter=pi0_config.Pi0Config(
        paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
    ).get_freeze_filter(),
    # Turn off EMA for LoRA finetuning.
    ema_decay=None,
),
```

First calculate the norm first

```bash
uv run scripts/compute_norm_stats.py --config-name pi0_aloha_sim_insert
```

Now we train

```bash
CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_aloha_sim_insert --exp-name=my_experiment --overwrite
```

## Evaluating

### Task 1: transfer cube
Serve the model

```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_aloha_sim --policy.dir=checkpoints/pi0_aloha_sim/my_experiment/19999
```

Open a new terminal and run

```
MUJOCO_GL=egl python examples/aloha_sim/main_transfer_cube.py
```

### Task 2: insertion

Serve the model

```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_aloha_sim_insert --policy.dir=checkpoints/pi0_aloha_sim_insert/my_experiment/19999
```

Open a new terminal and run

```
MUJOCO_GL=egl python examples/aloha_sim/main_insert.py
```

### Compare against baseline

```bash
cd assets
mkdir -p assets/pi0_aloha_base/lerobot/aloha_sim_transfer_cube_human
cp assets/pi0_aloha_sim/lerobot/aloha_sim_transfer_cube_human/norm_stats.json assets/pi0_aloha_base/lerobot/aloha_sim_transfer_cube_human/
```

Serve model
```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_aloha_base --policy.dir=gs://openpi-assets/checkpoints/pi0_base
```

Open a new terminal and run

```
MUJOCO_GL=egl python examples/aloha_sim/main_transfer_cube.py
```