
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import pickle
from rlcompopt.train import testing


def main():
    """
    Perform local testing with the saved arg.pkl file
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--args_path", type=str, help="The path to the args.pkl file")
    args = parser.parse_args()
    with open(args.args_path, "rb") as f:
        args_ = pickle.load(f)

    testing(args_, locally=True)


if __name__ == "__main__":
    main()
