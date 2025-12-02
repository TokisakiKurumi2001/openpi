#!/bin/bash
export MUJOCO_GL=egl

for i in {0..49}; do
    python examples/aloha_sim/main_transfer_cube.py --args.out_dir exp_data/aloha_sim_base/videos --args.seed "$i"
done