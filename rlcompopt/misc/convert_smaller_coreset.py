
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from typing import List, Tuple
from rlcompopt.pipeline.lib.types import TrajectoryDataset, ActionSequences, ActionSequence


def convert(old_traj_path: str, new_traj_path: str, n: int, sorted_coreset: List[Tuple[int]]):
    assert isinstance(sorted_coreset, list)
    old_traj = TrajectoryDataset.load(old_traj_path)

    using_seqs = sorted_coreset[:n]
    ac = [ActionSequence(actions=aa) for aa in using_seqs]
    old_acs = old_traj.action_sequences
    acs = ActionSequences(
        name=old_acs.name + f"_using_best_{n}",
        actionseqs=ac,
        train_dataset_name=old_acs.train_dataset_name,
    )

    idx = [old_acs.actionseqs.index(aa) for aa in ac]

    def extract_ir(irs):
        return [irs[ii] for ii in idx]

    old_samples = old_traj.samples
    new_samples = []
    for sample in old_samples:
        sample.all_ir_searches = extract_ir(sample.all_ir_searches)
        assert len(sample.all_ir_searches) == n
        new_samples.append(sample)
    old_traj.samples = new_samples
    old_traj.action_sequences = acs

    old_traj.save(new_traj_path)


def main():
    parser = argparse.ArgumentParser(description="Convert a TrajectoryDataset to use a smaller coreset.")
    parser.add_argument("--old_traj_path", type=str, help="The path to the old TrajectoryDataset file")
    parser.add_argument("--new_traj_path", type=str, help="The path to the new TrajectoryDataset file")
    parser.add_argument("--n", type=int, help="The number of action sequences to use in the new TrajectoryDataset")
    args = parser.parse_args()
    with open("rlcompopt/pipeline/lib/coreset_sorted.txt", "rt") as f:
        lines = f.read().splitlines()
    action_seqs = [eval(line) for line in lines]
    convert(args.old_traj_path, args.new_traj_path, args.n, action_seqs)


if __name__ == "__main__":
    main()
