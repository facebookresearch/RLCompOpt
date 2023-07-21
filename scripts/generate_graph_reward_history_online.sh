
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#!/bin/bash -x

outdir=outputs_rl/cg_online

# run locally
python -m rlcompopt.cl.generate --config-path conf/rl_online --config-name generate_online \
    hydra.run.dir=$outdir \
    outdir=$outdir \
    nproc=5