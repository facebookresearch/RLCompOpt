
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

outdir=outputs_rl/autophase

# run locally
python -m torch.distributed.launch --nproc_per_node=1 --use_env --master_port=54597 \
    rlcompopt/train.py --config-path cl/conf/rl_online --config-name train_autophase \
    hydra.run.dir=$outdir \
    dataset.train=$outdir/summary.db \
    dataset.num_generators=20