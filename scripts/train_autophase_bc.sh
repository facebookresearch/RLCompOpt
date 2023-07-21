
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

make install

for i in 0 1 2
do
python -m torch.distributed.launch --nproc_per_node=1 --use_env --master_port=54567 \
    rlcompopt/train.py --config-name autophase_bc \
    seed=$i
done