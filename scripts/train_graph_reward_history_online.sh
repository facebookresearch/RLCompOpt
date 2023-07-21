
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

outdir=outputs_rl/cg_online

# run locally
python -m torch.distributed.launch --nproc_per_node=1 --use_env --master_port=54567 \
    rlcompopt/train.py --config-path cl/conf/rl_online --config-name train_gcn \
    hydra.run.dir=$outdir \
    dataset.train=$outdir/summary.db \
    dataset.num_generators=5