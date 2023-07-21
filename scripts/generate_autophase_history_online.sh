
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#!/bin/bash -x

outdir=outputs_rl/autophase

# run locally
python -m rlcompopt.cl.generate --config-path conf/rl_online --config-name generate_autophase \
    hydra.run.dir=$outdir \
    outdir=$outdir \
    n_model_workers=1 \
    nproc=20