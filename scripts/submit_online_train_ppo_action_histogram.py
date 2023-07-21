
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import datetime
import argparse
import subprocess
from typing import Dict

FS_LIMIT = 200  # file system path length limit

GEN = "python -m rlcompopt.cl.generate --config-path conf/rl_online --config-name generate_online "
type2_train = "python -m rlcompopt.train --config-path cl/conf/rl_online --config-name train_gnn_type2 "
TRAIN = {
    "gcn": "python -m rlcompopt.train --config-path cl/conf/rl_online --config-name train_gcn ",
    "gat": type2_train,
    "gcn_real": type2_train,
    "gin": type2_train,
    "attn": "python -m rlcompopt.train --config-path cl/conf/rl_online --config-name train_attn "
}


def make_str(s: str):
    assert not s.startswith("'")
    assert not s.endswith("'")
    return "'" + s + "'"


def fill_param(cmd: str, params: Dict = None, exclude={}):
    assert isinstance(cmd, str)
    exp_name = ""
    if params is not None:
        assert isinstance(params, dict)
        names = {k: v for k, v in params.items() if k not in exclude.keys()}
        exp_name = "_".join(f"{k.split('.')[-1]}_{v}" for k, v in names.items())
        params = " ".join(f"{k}={v}" for k, v in params.items())
        if not cmd.endswith(" "):
            cmd = cmd + " "
        cmd = cmd + params
    return cmd, exp_name


def submit(exp_name: str, dry_run=False):
    for gen_config, train_config in exp[exp_name]:
        gen_config = gen_config.copy()
        train_config = train_config.copy()
        gen_config.update(gen_config_common)
        train_config.update(train_config_common)
        gen_cmd, gen_name = fill_param(GEN, gen_config)
        train_cmd, train_name = fill_param(TRAIN[exp_name], train_config)

        now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        outdir = "_".join((now, train_name, gen_name))
        outdir = log_dir + outdir
        if len(outdir) > FS_LIMIT:
            print("Truncated outdir")
            outdir = outdir[:FS_LIMIT]  # file system limit: 256

        gen_basic = {
            "hydra.run.dir": make_str(outdir),
            "outdir": make_str(outdir),
        }

        train_basic = {
            "hydra.run.dir": make_str(outdir),
            "dataset.train": make_str(os.path.join(outdir, "summary.db")),
            # "model_db_path": make_str(os.path.join(outdir, "model.db")),
        }

        gen_cmd, _ = fill_param(gen_cmd, gen_basic)
        train_cmd, _ = fill_param(train_cmd, train_basic)

        if dry_run or local_run:
            print(f"{outdir=}")
            print(f"{gen_cmd=}")
            print(f"{train_cmd=}")
        if not local_run:
            print("====================")
            os.system(gen_cmd)
            print("\n")
            os.system(train_cmd)
        else:
            print("Running locally")
            subprocess.Popen(gen_cmd.split())
            subprocess.Popen(train_cmd.split())


# GCN
gcn_exp = []
gen_config = {}
train_config = {
    "model.gnn_type": "GatedGraphConv",
}
gcn_exp.append((gen_config, train_config.copy()))

# GAT
gat_exp = []
gen_config = {}
train_config = {
    "model.gnn_type": "GAT",
    "model.entropy_factor": 0.0006,
    "optim.weight_decay": 0,
}
gat_exp.append((gen_config, train_config.copy()))

# GIN
gin_exp = []
gen_config = {}
train_config = {
    "model.gnn_type": "GIN",
    "model.entropy_factor": 0.003,
    "optim.lr": 1e-5,
    "optim.weight_decay": 0,
}
gin_exp.append((gen_config, train_config.copy()))

# GCN real
gcn_real_exp = []
gen_config = {}
train_config = {
    "model.gnn_type": "GCN",
}
gcn_real_exp.append((gen_config, train_config.copy()))


# EdgeAttn
attn_exp = []
gen_config = {}
train_config = {
    "model.gnn_type": "EdgeAttn",
}

attn_exp.append((gen_config.copy(), train_config.copy()))

exp = {
    "gcn": gcn_exp,
    "gat": gat_exp,
    "gin": gin_exp,
    "gcn_real": gcn_real_exp,
    "attn": attn_exp
}


def submit_all(dry_run=True):

    submit("gcn", dry_run=dry_run)
    submit("gcn_real", dry_run=dry_run)
    submit("gat", dry_run=dry_run)
    submit("gin", dry_run=dry_run)
    submit("attn", dry_run=dry_run)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run RL-PPO experiments locally or on Slurm.')
    parser.add_argument(
        "--submitit_log_dir", default="null", type=str, 
        help="If null, run experiments locally, otherwise, submit them to Slurm via submitit.")
    parser.add_argument(
        "--nproc_per_node", default=80, type=int, 
        help="This is for setting how many processes to use when experiments are run locally."
             "For experiments on Slurm, the number is determined by 'submitit' entries in config file.")
    parser.add_argument(
        "--num_seeds", default=3, type=int,
        help="number of seeds (runs) to repeat experiments")
    parser.add_argument("--dry_run", action="store_true", help="whether it is a dry run.")
    parser.add_argument(
        "--log_dir", default="outputs_rl/", type=str,
        help="log dir to save checkpoints and testing results")
    parser.add_argument(
        "--slurm_partition", default="", type=str,
        help="slurm partition to use")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    nproc_per_node = args.nproc_per_node
    submitit_log_dir = args.submitit_log_dir
    partition = args.slurm_partition
    log_dir = args.log_dir

    local_run = submitit_log_dir == "null"

    gen_config_common0 = {
        "nproc": nproc_per_node,
        "submitit.log_dir": submitit_log_dir,
        "submitit.partition": partition,
    }
    train_config_common0 = {
        "dataset.num_generators": nproc_per_node,
        "submitit.log_dir": submitit_log_dir,
        "submitit.partition": partition,
    }
    for seed in range(args.num_seeds):
        gen_config_common = {"seed": seed, **gen_config_common0}
        train_config_common = {"seed": seed, **train_config_common0}
        submit_all(args.dry_run)
