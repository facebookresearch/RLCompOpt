
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gym
import random
import numpy as np
import torch

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def run_agent(job_id: int, benchmark: str, seed: int, args, random_walk=False, check_flag_action_had_no_effect=False):
    """Get some statistics of a single benchmark."""

    # Send random seed
    set_all_seeds(11)  # random seed is not working, trajectory is different across runs

    with gym.make("llvm-autophase-ic-v0", benchmark=benchmark) as env:
        env.reset()

        features = ["Programl", "IrInstructionCountOz", "IrInstructionCount"]
        obs_space = [ env.observation.spaces[feature_name] for feature_name in features ]
        observations, rewards, done, info = env.step(action=[], observation_spaces=obs_space)
        programl = observations[0]  # networkX graph
        graph_info = {"benchmark": benchmark, "#nodes": programl.number_of_nodes(), "#edges": programl.number_of_edges()}
        # print(benchmark, programl.info())
        if random_walk:
            # debug how the graph statistics change during compiler optimization
            i = 0
            print(graph_info, observations[1:])
            n_nodes = [programl.number_of_nodes()]
            while not done:
                observations, rewards, done, info = env.step(action=env.action_space.sample(), observation_spaces=obs_space)
                if not info['action_had_no_effect']:
                    programl = observations[0]
                    graph_info = [programl.number_of_nodes(), programl.number_of_edges(), observations[1:]]
                    print(graph_info)
                    n_nodes.append(programl.number_of_nodes())
                    if n_nodes[-1] > n_nodes[-2]:
                        print(env.commandline())
                    i += 1
                    if i > 100:
                        break
        if check_flag_action_had_no_effect:
            # debug the info['action_had_no_effect'] flag
            i = 0
            print(graph_info, observations[1:])
            n_nodes = [(programl.number_of_nodes(), programl.number_of_edges())]
            while not done:
                observations, rewards, done, info = env.step(action=env.action_space.sample(), observation_spaces=obs_space)
                programl = observations[0]
                n_nodes.append((programl.number_of_nodes(), programl.number_of_edges()))
                print(n_nodes[-1], info['action_had_no_effect'])
                if info['action_had_no_effect']:
                    if n_nodes[-1] != n_nodes[-2]:
                        print(f"***********************\n{n_nodes[-1]} != {n_nodes[-2]}")

    return graph_info


def get_stat_packed_args(job):
    return run_agent(*job)