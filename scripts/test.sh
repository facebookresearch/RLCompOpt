
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

export NUM_CPU=14
export NUM_GPU=1

FILES="outputs/*"
for f in $FILES
do
  echo "Processing $f "
  CUDA_VISIBLE_DEVICES=0 python rlcompopt/eval_local.py --args_path "$f/args.pkl"
done