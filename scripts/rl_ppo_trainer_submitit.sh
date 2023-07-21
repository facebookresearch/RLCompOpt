
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

outdir=outputs_rl/cg_online2

# submit experiments to Slurm
python -m rlcompopt.train --config-path cl/conf/rl_online --config-name train_attn \
    hydra.run.dir=$outdir \
    dataset.train=$outdir/summary.db \
    submitit.log_dir=./log_dir