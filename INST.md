```bash

# norm stuffs
uv run scripts/compute_norm_stats.py --config-name pi0_aloha_base

# train stuffs
CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_aloha_sim --exp-name=my_experiment --overwrite

# serving stuffs
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_aloha_sim --policy.dir=checkpoints/pi0_aloha_sim/my_experiment/19999

uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_aloha_base --policy.dir=gs://openpi-assets/checkpoints/pi0_base

uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_aloha_sim_insert --policy.dir=checkpoints/pi0_aloha_sim_insert/big_experiment/19999
```