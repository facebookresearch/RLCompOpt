
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from glob import glob
import hydra
import os

from rlcompopt.env_wrapper.wrapper_offline import merge_dbs

import logging
log = logging.getLogger(__file__)


@hydra.main(config_path="./conf", config_name="merge.yaml")
def main(args):
    """Merge input databases into a single output database."""
    inputs_list = []
    output = args.output
    tables = args.tables
    for name in args.inputs.split(","):
        if not name.startswith("/") and not name.startswith("."):
            name = os.path.join(args.root, name)
        if not name.endswith(".db"):
            name = os.path.join(name + args.suffix, "summary.db")
        inputs_list.extend(list(glob(name)))

    if not output.startswith("/") and not output.startswith("."):
        output = os.path.join(args.root, output)

    logging.info(f"Input lists: {inputs_list}")
    logging.info(f"Output: {output}")

    tables = tables.split(",") if tables is not None else [] 
    merge_dbs(inputs_list, output, tables=tables)


if __name__ == "__main__":
    main()