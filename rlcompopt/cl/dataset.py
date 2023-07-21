
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import collections
import logging
import pickle
import random
import sqlite3
import zlib
import os
import time
from collections import OrderedDict
from collections.abc import MutableMapping

import dgl
import torch
import torch.nn.functional as F
from dgl.data import DGLDataset
from torch_geometric.data import Batch, Data

import rlcompopt.utils as utils
from rlcompopt.cl.data_socket import DataBuffer, DataClient
from rlcompopt.cl.faster_balanced_sampler_stream import SocketDataPacking
from rlcompopt.env_wrapper.pyg_utils import dgl2pyg, get_blk_idx
from rlcompopt.pipeline.lib.types import TrajectoryDataset
from rlcompopt.env_wrapper.pyg_utils import remove_type_nodes


log = logging.getLogger(__file__)


class CompilerGymDatasetBase:
    def __init__(
        self,
        filepath,
        max_len_nodes=50000,
        autophase_max_bin=10,
        input_key="dgl_graph",
        output_key="cumulative_reward",
        pre_load=True,
        load_next_state=False,
        remove_large_graph=False,
        max_nodes=2**20,
        featurized_dataset=None,
        mode="dgl",
        ssl=False,
        load_balance=False,
        load_cumulative_reward2=False,
        divided_by_this_ir=False,
        load_subgraph_feature=False,
        subgraph_feature="function",
        use_autophase=False,
        use_history=False,
        use_cl=False,
        queue_size=1000,  # for new contrastive SSL
        min_queue_size=50,
        graph_feat_dim=64,
        send_data_via_socket=False,
        socket_db=None,
        num_workers=64,
        num_servers=1,
        bin_size=126,
        full_rate=0.9,
        num_max_bin=10,
        q_learning=False,
        cache_data=False,
        real_q_learning=False,
        add_block_idx=False,
        random_mixup=0,
        seq_classification=False,  # path to seq file
        dense_seq_cls=False,  # path to db of all_benchmark to all_seq reward
        pydantic_dataset_path=None,
        dense_cls_metric="oz",
        action_histogram_steps=0,  # 0 mean not using it
        num_actions=124,
        remove_type_graph=False,
    ):
        super().__init__()
        self.filepath = filepath
        self.num_workers = num_workers
        self.max_len_nodes = max_len_nodes

        self.graph_key = input_key
        self.output_key = output_key
        self.pre_load = pre_load
        self.load_next_state = load_next_state
        self.remove_large_graph = remove_large_graph
        self.max_nodes = max_nodes

        self.autophase_max_bin = autophase_max_bin
        self.featurized_dataset = featurized_dataset
        self.mode = mode
        self.ssl = ssl
        self.load_balance = load_balance
        self.load_cumulative_reward2 = load_cumulative_reward2
        self.divided_by_this_ir = divided_by_this_ir
        self.load_subgraph_feature = load_subgraph_feature
        self.subgraph_feature = subgraph_feature
        self.use_autophase = use_autophase
        self.use_history = use_history
        self.use_cl = use_cl
        self.queue_size = queue_size
        self.min_queue_size = min_queue_size
        self.graph_feat_dim = graph_feat_dim
        self.send_data_via_socket = send_data_via_socket
        self.socket_db = socket_db
        self.num_servers = num_servers
        self.bin_size = bin_size
        self.full_rate = full_rate
        self.num_max_bin = num_max_bin
        self.q_learning = q_learning
        self.real_q_learning = real_q_learning
        self.cache_data = cache_data
        self.add_block_idx = add_block_idx
        self.random_mixup = random_mixup
        self.seq_classification = seq_classification
        self.dense_seq_cls = dense_seq_cls
        self.pydantic_dataset_path = pydantic_dataset_path
        self.dense_cls_metric = dense_cls_metric
        self.action_histogram_steps = action_histogram_steps
        self.num_actions = num_actions
        self.remove_type_graph = remove_type_graph
        assert dense_cls_metric in ["oz", "o0"]
        if action_histogram_steps:
            assert self.use_history

        self.convert_graph = dgl2pyg if self.mode == "pyg" else lambda x: x
        if utils.is_main_process():
            log.info(f"using filepath: {self.filepath}")

    def _get_columns(self, table_name):
        # Get meta data for each table. Here we just use column names
        columns = list(self.cursor.execute(f"pragma table_info({table_name});"))
        return ["rowid"] + [name for idx, name, tp, _, _, _ in columns]

    def process(self):
        # Create connection.
        self.connection = sqlite3.connect(self.filepath, timeout=1200)

        # get counts.
        self.cursor = self.connection.cursor()

        # mapping from tablename to column names.
        self.cols = {
            table: self._get_columns(table) for table in ("States", "Transitions")
        }

        nodes_func = (
            lambda x: x.num_nodes if self.mode == "pyg" else lambda x: x.num_nodes()
        )

        if True:

            all_states_ = []
            original_state_id = set()
            before_n_graph = 0
            state2ir = {}
            if self.ssl:
                idx = torch.ones(117 + 1, dtype=torch.long) * -1  # FIXME
            # iterate the rows to avoid OOM
            for rec in self.cursor.execute(f"select rowid, * from States;"):
                before_n_graph += 1
                state = self._add_col_name(rec, "States")
                state2ir[state["state_id"]] = state["IrInstructionCount"]
                graph = self.convert_graph(
                    pickle.loads(zlib.decompress(state["dgl_graph"]))
                )
                if self.ssl:
                    fine_node_type = graph["x"]
                    coarse_node_type = graph["type"]
                    idx[fine_node_type] = coarse_node_type
                state["num_nodes"] = nodes_func(graph)
                original_state_id.add(state["state_id"])
                if self.remove_large_graph:
                    if state["num_nodes"] <= self.max_nodes:
                        all_states_.append(state)
                else:
                    all_states_.append(state)
                if self.pre_load:
                    state["dgl_graph"] = graph
                else:
                    del state["dgl_graph"]  # avoid OOM
            self.state_rowid = [
                {
                    "state_id": s["state_id"],
                    "rowid": s["rowid"],
                    "num_nodes": s["num_nodes"],
                }
                for s in all_states_
            ]
            if self.ssl:
                self.table_idx = idx

            after_n_graph = len(all_states_)
            avg_nodes = sum(g["num_nodes"] for g in all_states_) / len(all_states_)
            if utils.is_main_process():
                log.info(
                    f"using {after_n_graph}/{before_n_graph} "
                    f"({after_n_graph/before_n_graph :.2%}) graphs, avg_nodes: {avg_nodes}"
                )

            if self.ssl:
                # only load states
                if self.pre_load:
                    self.data = [{"dgl_graph": v["dgl_graph"]} for v in all_states_]
                else:
                    self.data = [
                        {"dgl_graph_num_nodes": v["num_nodes"]}
                        for v in self.state_rowid
                    ]  # make __len__ and the size_func api in BalancedBatchSampler consistent
                # just set the below to some possibly incorrect numbers
                self.autophase_bounds = 64
                self.autophase_dim = 64
                return

            state_mapping = {rec["state_id"]: rec for rec in all_states_}
            all_transitions = [
                self._add_col_name(rec, "Transitions")
                for rec in self.cursor.execute(f"select rowid, * from Transitions;")
            ]
            starting_states = set(
                v["state_id"]
                for v in all_transitions
                if v["state_id"] in original_state_id
                and v["next_state_id"] in original_state_id
            )
            unique_state_action = set(
                (v["state_id"], v["action_value"]) for v in all_transitions
            )
            if utils.is_main_process():
                log.info(
                    f"there are {len(unique_state_action)}/{len(all_transitions)} "
                    f"({len(unique_state_action)/len(all_transitions):.2%}) unique state-action pairs"
                )

            # merge
            all_transitions_ = []
            for v in all_transitions:
                result = state_mapping.get(v["state_id"], None)
                if result is None:
                    continue
                if self.pre_load:
                    v.update(result)
                else:
                    v["state_rowid"] = result["rowid"]
                    v["dgl_graph_num_nodes"] = result["num_nodes"]
                if self.load_next_state:
                    result = state_mapping.get(v["next_state_id"], None)
                    if result is not None:
                        if self.pre_load:
                            v["next_state_graph"] = result["dgl_graph"]
                        else:
                            v["next_state_rowid"] = result["rowid"]
                    else:
                        continue
                v["non_terminal"] = 0.0
                if v["next_state_id"] in starting_states:
                    v["non_terminal"] = 1.0

                # fix rewards
                traj_id = v["traj_id"]
                start_state_id = traj_id[:-5]
                if self.divided_by_this_ir:
                    ir0 = v["ir_instruction_count"] + v["ir_instruction_count_reward"]
                else:
                    ir0 = state2ir[start_state_id]
                v["cumulative_reward"] = (
                    v["cumulative_reward2"] / ir0
                )  # for monte-carlo q learning
                v["ir_instruction_count_oz_reward"] = (
                    v["ir_instruction_count_reward"] / ir0
                )  # for TD learning, should set load_cumulative_reward2=False

                all_transitions_.append(v)
            assert (
                not self.load_cumulative_reward2
            ), "this is only for whole graph reward prediction, and cannot be used as we have divided rewards by ir0"

            self.data = all_transitions_

            before_n = len(all_transitions)
            after_n = len(all_transitions_)
            if utils.is_main_process():
                log.info(
                    f"using {after_n}/{before_n} ({after_n/before_n :.2%}) transitions"
                )

        self.stats_reward = None
        # self._check()

        if self.featurized_dataset is None:
            self._create_autophase_feature_space()
        else:
            self.autophase_bounds = self.featurized_dataset.autophase_bounds
            self.autophase_dim = self.featurized_dataset.autophase_dim

    def _check(self):
        # Check whether there exists same (s, a) pair but different rewards.
        stats = collections.defaultdict(list)

        for d in self.data:
            key = d["autophase"] + "-" + d["action_value"]
            stats[key].append(d["cumulative_reward"])

        # check any states with >= 2 entries.
        err = 0
        n_sample = 0
        for key, stat in stats.items():
            n_sample += len(stat)
            if len(stat) >= 2:
                a = torch.FloatTensor(stat)
                err += (a - a.mean()).pow(2).sum().item()
        if utils.is_main_process():
            log.info(f"Intrinsic err: {err / len(self.data)} [{n_sample}]")

        self.stats_reward = stats

    def _create_autophase_feature_space(self):
        stats_per_axis = None
        for d in self.data:
            values = [k for k in d["autophase"].split(" ")]
            if stats_per_axis is None:
                stats_per_axis = [collections.Counter() for _ in range(len(values))]

            for v, c in zip(values, stats_per_axis):
                c[int(v)] += 1

        # Make dictionary
        partition_bounds = []
        for c in stats_per_axis:
            # if there are too many items, group them into max_bin.
            if len(c) > self.autophase_max_bin:
                # Order w.r.t keys.
                entries = sorted(c.items(), key=lambda x: x[0])
                # Do partition.
                cum_cnt = 0
                last_cum_cnt = 0
                bin_bounds = [entries[0][0]]
                for k, cnt in entries:
                    if (
                        cum_cnt
                        >= last_cum_cnt + len(self.data) // self.autophase_max_bin
                    ):
                        bin_bounds.append(k)
                        last_cum_cnt = cum_cnt
                    cum_cnt += cnt
            else:
                bin_bounds = sorted(c.keys())

            partition_bounds.append(bin_bounds)

        self.autophase_bounds = partition_bounds
        self.autophase_dim = sum([len(b) + 1 for b in self.autophase_bounds])
        if utils.is_main_process():
            log.info(f"Autophase_bounds: {self.autophase_bounds}")
            log.info(f"Autophase_dim: {self.autophase_dim}")

    def _cheat_pred_from_keys(self, keys):
        if self.stats_reward is not None:
            return torch.FloatTensor(
                [
                    sum(self.stats_reward[key]) / len(self.stats_reward[key])
                    for key in keys
                ]
            )
        else:
            return None

    def _load_one(self, table_name, key, value):
        all_ret = self.cursor.execute(
            f'select * from {table_name} where {key} = "{value}";'
        )
        ret = list(all_ret)[0]
        return ret

    def _add_col_name(self, rec, table_name):
        # Convert record (as tuple) to be a dictionary with proper column names.
        return {name: v for name, v in zip(self.cols[table_name], rec)}

    def get_subgraph(self, graph):
        if self.load_subgraph_feature:
            feat = graph[self.subgraph_feature]
            max_idx = feat.max()
            graph["num_subgraphs"] = max_idx.item() + 1

    def __getitem__(self, i):
        ret = self.data[i]
        if self.pre_load:
            self.get_subgraph(ret["dgl_graph"])
            if self.load_next_state:
                self.get_subgraph(ret["next_state_graph"])
            return ret
        if self.ssl:
            return {
                "dgl_graph": self.load_state_with_rowid(self.state_rowid[i]["rowid"])
            }
        ret = dict(
            **ret, dgl_graph=self.load_state_with_rowid(ret["state_rowid"])
        )  # create a new ret to avoid loading the graph into the ret in data
        self.get_subgraph(ret["dgl_graph"])
        if self.load_next_state:
            ret["next_state_graph"] = self.load_state_with_rowid(
                ret["next_state_rowid"]
            )
            self.get_subgraph(ret["next_state_graph"])
        return ret

    def load_state_with_rowid(self, rowid):
        all_ret = self.cursor.execute(
            f"select dgl_graph from States where rowid = {rowid};"
        )  # using rowid is much faster than offset
        ret = self.convert_graph(pickle.loads(zlib.decompress(list(all_ret)[0][0])))
        return ret

    def get_translation_matrix(self, vocab_size):
        # map vocab type to coarse node type (e.g., "instruction", "variable")
        idx = self.table_idx
        mask = ~(idx == -1)
        if utils.is_main_process():
            after_n = mask.float().sum().item()
            log.info(f"set {after_n/vocab_size :.2%} vocab for type loss")
        num_types = idx.max().item() + 1  # starts from 0
        table = torch.zeros(vocab_size, num_types)
        tmp = table[mask].scatter_(1, idx[mask].unsqueeze(1), 1)
        table[mask] = tmp
        if utils.is_main_process():
            log.info(f"type table: {table}")
        return table

    def __len__(self):
        return len(self.data)

    def remove_graphs(self, samples):
        if self.mode == "dgl":
            n_nodes = [sample[self.graph_key].number_of_nodes() for sample in samples]
        elif self.mode == "pyg":
            n_nodes = [sample[self.graph_key].num_nodes for sample in samples]
        total = sum(n_nodes)
        th = self.max_nodes
        if total > th:
            remain = total
            p = torch.randperm(len(n_nodes)).tolist()
            for i, x in enumerate(p):
                remain -= n_nodes[x]
                if remain <= th:
                    p = p[: i + 1]
                    break
            if len(p) == len(n_nodes):
                n_nodes = torch.tensor(n_nodes)
                smallest_idx = torch.sort(n_nodes, descending=False)[1].tolist()[0]
                samples = [samples[smallest_idx]]
            else:
                samples = [x for i, x in enumerate(samples) if i not in p]
        return samples

    def prepare_subgraph(self, graphs):
        subgraph_count = 0
        for i in range(len(graphs)):
            pre_num_subgraphs = (
                torch.ones(graphs[i].num_nodes, 1, dtype=torch.long) * subgraph_count
            )
            graphs[i]["subgraph_idx"] = (
                torch.ones(graphs[i]["num_subgraphs"], dtype=torch.long) * i
            )
            graphs[i]["subgraph_node_idx"] = (
                pre_num_subgraphs + graphs[i][self.subgraph_feature]
            )
            subgraph_count += graphs[i]["num_subgraphs"]
            feat, count = torch.unique(
                graphs[i][self.subgraph_feature].flatten(),
                sorted=True,
                return_counts=True,
            )
            graphs[i]["subgraph_num_nodes"] = count / graphs[i].num_nodes
        return graphs

    def collate_fn(self, samples):
        samples = [sample for sample in samples if sample is not None]
        if (
            self.remove_large_graph and not self.load_balance
        ):  # load_balance is handled in sampler
            # without load balance, hard remove graphs on the fly
            samples = self.remove_graphs(samples)

        graphs = [sample[self.graph_key] for sample in samples]
        if self.load_subgraph_feature:
            graphs = self.prepare_subgraph(graphs)

        # Takes a list of graphs and makes it into one big graph that dgl operates on
        ret = None
        if samples:
            if self.mode == "dgl":
                graph = dgl.batch(graphs)
            elif self.mode == "pyg":
                graph = Batch.from_data_list(graphs)
            else:
                raise NotImplementedError
            actions = torch.LongTensor(
                [int(sample["action_value"]) for sample in samples]
            )
            autophase = torch.cat(
                [self.process_reward("autophase", sample) for sample in samples]
            )
            reward = torch.cat(
                [self.process_reward(self.output_key, sample) for sample in samples]
            )
            if self.load_cumulative_reward2:
                instruction_count_reward = torch.tensor(
                    [sample["cumulative_reward2"] for sample in samples]
                )
            else:
                instruction_count_reward = torch.tensor(
                    [sample["ir_instruction_count_oz_reward"] for sample in samples]
                )
            non_terminal = torch.tensor([sample["non_terminal"] for sample in samples])
            pred_rewards = self._cheat_pred_from_keys(
                [d["autophase"] + "-" + d["action_value"] for d in samples]
            )
            ret = (
                graph,
                actions,
                autophase,
                reward,
                instruction_count_reward,
                non_terminal,
                pred_rewards,
            )
            if self.load_next_state:
                next_state_graph = [sample["next_state_graph"] for sample in samples]
                if self.load_subgraph_feature:
                    next_state_graph = self.prepare_subgraph(next_state_graph)
                if self.mode == "dgl":
                    next_state_graph = dgl.batch(next_state_graph)
                elif self.mode == "pyg":
                    next_state_graph = Batch.from_data_list(next_state_graph)
                ret += (next_state_graph,)
            else:
                ret += (None,)
        return ret

    def collate_fn_ssl(self, samples):
        ret = None
        if samples:
            if (
                self.remove_large_graph and not self.load_balance
            ):  # load_balance is handled in sampler
                # without load balance, hard remove graphs on the fly
                samples = self.remove_graphs(samples)
            if self.mode == "dgl":
                graph = dgl.batch([sample["dgl_graph"] for sample in samples])
            elif self.mode == "pyg":
                graph = Batch.from_data_list(
                    [sample["dgl_graph"] for sample in samples]
                )
            else:
                raise NotImplementedError
            ret = graph
        return ret

    def _autophase2vec_0(self, reward):
        f = torch.zeros(self.autophase_dim)
        n = 0
        for i, (b, x) in enumerate(zip(self.autophase_bounds, reward.split(" "))):
            idx = len(b)
            x = int(x)
            for j, th in enumerate(b):
                if x <= th:
                    idx = j
                    break
            f[n + idx] = 1.0
            n += len(b) + 1
        assert (
            n == self.autophase_dim
        ), f"Autophase dimension mismatch! n = {n}, autophase_dim = {self.autophase_dim}"
        return f

    def _autophase2vec(self, raw_state):
        """A simpler version for encoding autophase feature"""
        if isinstance(raw_state, str):
            raw_state = [int(x) for x in raw_state.split(" ")]
        autophase = torch.tensor(raw_state, dtype=torch.float)
        # if self.normalize_autophase:
        # normalize autophase instruction counts
        total_instructions = raw_state[51]
        autophase = autophase / total_instructions
        return autophase

    def process_reward(self, key, sample):
        reward = sample[key]
        if key in ["instcount"]:  # "autophase"]:
            return torch.Tensor([int(x) for x in reward.split()]).unsqueeze(0)
        elif key == "autophase":
            return self._autophase2vec(reward).unsqueeze(0)
        return torch.Tensor([reward])


class CompilerGymDataset(CompilerGymDatasetBase, DGLDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, mode="dgl")
        DGLDataset.__init__(self, name="CompilerGym")


class CompilerGymDatasetPyG(CompilerGymDatasetBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, mode="pyg")
        self.process()

    def __iter__(self):
        return iter(self.data)


class CompilerGymDatasetOnline(CompilerGymDatasetPyG):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.pydantic_dataset_path is not None:
            return  # handled by CompilerGymDatasetPydantic 

        self.autophase_bounds = [
            [0, 1, 3, 7, 14, 28],
            [0, 1, 3, 6, 10, 18, 39, 59, 152],
            [0, 11, 16, 27, 45, 81, 133, 200, 295, 562],
            [0, 7, 12, 19, 31, 57, 94, 147, 230, 485],
            [0, 2, 4, 8, 16, 25, 41, 58, 106, 189],
            [0, 9, 14, 21, 33, 60, 101, 159, 243, 464],
            [0, 4, 7, 10, 19, 36, 55, 80, 122, 331],
            [0, 2, 3, 5, 10, 17, 32, 63],
            [0, 2, 4, 7, 14, 24, 45, 61, 109],
            [0, 6, 9, 16, 31, 49, 94, 123, 214, 430],
            [0, 1, 3, 5, 9, 13, 19, 32, 95],
            [0, 1, 3, 6, 9, 16, 39, 58, 113],
            [0, 1, 3, 7, 32],
            [1, 14, 21, 32, 54, 94, 156, 252, 377, 694],
            [0, 1, 3, 6, 10, 18, 42, 62, 110],
            [0, 15, 22, 34, 61, 104, 179, 265, 375, 713],
            [0, 1, 4, 6, 10, 16, 29, 68],
            [0, 1, 4, 11, 26, 65, 119, 374],
            [0, 22, 32, 49, 95, 156, 274, 389, 569, 1104],
            [1, 21, 32, 50, 100, 196, 279, 488, 806, 1666],
            [0, 7, 15, 25, 43, 81, 152, 275, 488, 1064],
            [0, 11, 19, 30, 55, 101, 161, 345, 661, 1749],
            [0, 8, 15, 24, 54, 93, 157, 223, 359, 790],
            [0, 9, 14, 21, 33, 60, 101, 159, 243, 464],
            [0, 3, 6, 11, 22, 46, 75, 128, 190, 340],
            [0, 1, 4, 16],
            [0, 1, 4, 7, 13, 24, 46, 75, 158, 446],
            [0, 1, 2, 4, 9, 20, 61, 128, 478],
            [0, 1, 3, 8, 18, 32, 67, 141],
            [0, 1, 2, 4, 8, 14, 27, 36, 55, 111],
            [0, 17, 25, 37, 62, 105, 180, 277, 423, 831],
            [0, 2, 5, 11, 22, 42, 69, 118, 227, 643],
            [0, 15, 22, 34, 61, 104, 179, 265, 375, 713],
            [0, 5, 11, 14, 21, 31, 67, 130, 204, 576],
            [0, 7, 13, 21, 46, 83, 134, 250, 387, 711],
            [0, 6, 9, 16, 30, 47, 89, 128, 186, 388],
            [0, 1, 4, 13, 34],
            [0, 9, 17, 31, 60, 95, 171, 280, 484, 848],
            [0, 1, 3, 7, 19, 48],
            [0, 1, 2, 5, 9, 19, 40],
            [0, 1, 3, 7, 13, 27, 66, 99, 176, 482],
            [1, 2, 4, 9, 16, 29, 52, 149],
            [0, 1, 2, 4, 6, 9, 14, 27, 41],
            [0, 1, 3, 7, 13, 29, 71],
            [0, 1, 3, 7, 13, 34],
            [0, 3, 8, 19, 31, 55, 100, 201, 300, 670],
            [0, 1, 3, 6, 12, 19, 36, 82],
            [0, 1, 3, 6, 9, 16, 33],
            [0, 1, 3, 7, 36],
            [0, 1, 3, 7, 12, 20, 33, 70],
            [1, 18, 26, 39, 69, 119, 204, 307, 448, 875],
            [8, 87, 132, 210, 415, 787, 1370, 1904, 2621, 4964],
            [6, 36, 54, 96, 164, 270, 490, 895, 1379, 2787],
            [1, 4, 6, 11, 15, 22, 39, 70, 110, 241],
            [0, 1, 6, 13, 23, 43, 107, 184, 296, 637],
            [0, 22, 36, 64, 119, 197, 331, 553, 887, 1814],
        ]
        self.autophase_dim = 553
        if self.seq_classification and os.path.isfile(str(self.seq_classification)):
            with open(self.seq_classification, "rt") as f:
                lines = f.read().splitlines()
            self.action_seq = [" ".join(line.strip("()").split(", ")) for line in lines]
        if self.dense_seq_cls:
            assert self.seq_classification, "should set self.seq_classification"
            num_seqs = len(self.action_seq)
            con = sqlite3.connect(self.dense_seq_cls)
            cur = con.cursor()
            self.benchmark2seqrew = {
                # item[0]: torch.tensor(pickle.loads(item[3]), dtype=torch.float) / item[2]  # (O0 - ir) / O0
                item[0]: (torch.tensor(pickle.loads(item[3]), dtype=torch.float) - item[2] + item[1]) / item[1]
                for item in cur.execute("select * from Val")
            }
            self.benchmark2seqrew = {k : v for k, v in self.benchmark2seqrew.items() if v.numel() == num_seqs}
            print(f"Len of benchmark2seqrew: {len(self.benchmark2seqrew)}")
            self.benchmark2seqrew["default"] = torch.zeros_like(next(iter(self.benchmark2seqrew.values())))

        # for recording training speed
        self.iter_count = 0
        self.read_rows = 0
        self.traj_buffer = DictQueue(maxlen=1000)
        if self.q_learning:
            self.fields = "state_id, action_value, reward, next_state_id, traj_id, traj_step, advantage, logp, time_stamp"
            self.state_fields = "benchmark_uri, graph, autophase, IrInstructionCount"
        else:
            self.fields = "state_id, action_value, cumulative_reward, next_state_id, traj_id, traj_step, advantage, logp, time_stamp"
            self.state_fields = "benchmark_uri, graph, autophase"
        if self.real_q_learning:
            self.state_fields = "benchmark_uri, graph, autophase, IrInstructionCount"
        if self.cache_data:
            self.item_cache = {}

    def process(self):
        # Create connection.
        self.connection = sqlite3.connect(self.filepath, timeout=1200)
        self.cursor = self.connection.cursor()
        self.cursor2 = self.connection.cursor()
        self.data = []

    def collate_fn(self, samples):
        samples = [sample for sample in samples if sample is not None]
        assert len(samples) > 0
        rewards = [sample[3] for sample in samples]

        if self.use_autophase:
            autophase = torch.stack([sample[1] for sample in samples])
            graphs = None
        else:
            graphs = [sample[0] for sample in samples]
            # if self.load_subgraph_feature:
            #     graphs = self.prepare_subgraph(graphs)
            if self.remove_type_graph:
                graphs = [remove_type_nodes(graph_) for graph_ in graphs]

            if self.random_mixup:
                final_graphs = [sample[11] for sample in samples]
                random.shuffle(final_graphs)
                gs = []
                for i, (gt, fgt) in enumerate(zip(graphs, final_graphs)):
                    g_, ir0 = gt
                    fg, ir1 = fgt
                    if random.random() < self.random_mixup:
                        b_ = Batch.from_data_list([g_, fg])
                        bd = b_.to_dict()
                        gs.append(Data.from_dict({k: bd[k] for k in g_.to_dict()}))
                        rewards[i] = rewards[i] * ir0 / (ir0 + ir1)
                    else:
                        gs.append(g_)
                graphs = gs
            if self.add_block_idx:
                for g_ in graphs:
                    get_blk_idx(g_)
                n_nodes = 0
                for g_ in graphs:
                    g_['ordered_instr_idx'] = g_['ordered_instr_idx0'] + n_nodes
                    n_nodes += g_['x'].shape[0]
            graphs = Batch.from_data_list(graphs)
            autophase = None
        actions = [sample[2] for sample in samples]

        if samples[0][6] is not None:
            advantage = torch.tensor(
                [sample[6] for sample in samples], dtype=torch.float
            )
            logp = torch.tensor([sample[7] for sample in samples], dtype=torch.float)
        else:
            advantage, logp = None, None

        actions_ = torch.tensor(actions, dtype=torch.long)
        if self.dense_seq_cls:
            rewards_ = torch.stack(rewards)
        else:
            rewards_ = torch.tensor(rewards, dtype=torch.float)

        # record the progress of the trainer
        # self.iter_count += 1
        # self.read_rows += len(samples)
        # if not self.send_data_via_socket:
        #     trainer_speed = float(self.read_rows) / self.iter_count * utils.get_world_size()
        #     rowid = max(sample[-1] for sample in samples)
        #     self.cursor2.execute(
        #         "INSERT INTO TrainerProgress VALUES (?, ?)", (rowid, trainer_speed)
        #     )
        #     self.connection.commit()

        if self.use_history:
            max_seq_len = max(sample[5].shape[0] for sample in samples)
            if self.action_histogram_steps:
                # use action histogram and normalize it
                action_history = torch.stack([sample[4] for sample in samples]) / self.action_histogram_steps
                reward_history = None
            else:
                action_history = torch.stack(
                    [
                        F.pad(
                            sample[4], (0, max_seq_len - sample[4].shape[0]), "constant", 0
                        )
                        for sample in samples
                    ]
                )
                reward_history = torch.stack(
                    [
                        F.pad(
                            sample[5], (0, max_seq_len - sample[5].shape[0]), "constant", 0
                        )
                        for sample in samples
                    ]
                )
            padding_mask = torch.ones(len(samples), max_seq_len, dtype=torch.bool)
            for i in range(len(samples)):
                padding_mask[i, : samples[i][4].shape[0]] = 0
            max_seq_len += 1
            seq_pos = (
                torch.tensor(
                    [sample[8].shape[0] for sample in samples], dtype=torch.long
                )
                - 1
            )
            reward_sign_hist = torch.stack(
                [
                    F.pad(
                        sample[8], (0, max_seq_len - sample[8].shape[0]), "constant", 2
                    )
                    for sample in samples
                ]
            )
        else:
            action_history, reward_history, padding_mask, seq_pos, reward_sign_hist = (
                None,
            ) * 5

        if self.use_cl:
            cl_data = [sample[9] for sample in samples]
        else:
            cl_data = None
        avg_time = sum(sample[10] for sample in samples) / len(samples)
        return (
            graphs,
            actions_,
            autophase,
            rewards_,
            advantage,
            logp,
            reward_sign_hist,
            seq_pos,
            action_history,
            reward_history,
            padding_mask,
            cl_data,
            avg_time,
        )

    def __getitem__(self, i):
        if self.cache_data:
            ret = self.item_cache.get(i, None)
            if ret is not None:
                return ret + (i,)
        ret = self.load_data_with_rowid(i)
        if self.cache_data:
            self.item_cache[i] = ret
        return ret + (i,)

    def load_data_with_rowid(self, rowid):
        all_ret = self.cursor.execute(
            f"select {self.fields} from Transitions where rowid = ?", (rowid,)
        )  # using rowid is much faster than offset
        (
            state_id,
            action,
            reward_,
            next_state_id,
            traj_id,
            traj_step,
            advantage,
            logp,
            time_stamp,
        ) = next(all_ret)
        all_ret = self.cursor.execute(
            f"select {self.state_fields} from States where state_id = ?", (state_id,)
        )
        if self.q_learning or self.real_q_learning:
            benchmark_uri, graph, autophase, IrInstructionCount = next(all_ret)
            reward_ /= IrInstructionCount
        else:
            benchmark_uri, graph, autophase = next(all_ret)
        graph = pickle.loads(zlib.decompress(graph))
        final_graph = None
        if self.dense_seq_cls:
            reward_ = self.benchmark2seqrew.get(benchmark_uri, self.benchmark2seqrew["default"])
            if graph is not None:
                graph["ir_oz"] = self.benchmark2oz.get(benchmark_uri, self.benchmark2oz["default"])
        if self.seq_classification:
            assert self.dense_seq_cls
            traj_ret = self.cursor.execute(
                f"select actions from Trajectories where traj_id = ?",
                (traj_id,),
            ).fetchone()
            # action_seq = traj_ret[0]
            # assert action_seq in self.action_seq, f"{action_seq} not in {self.action_seq}"
            # action = self.action_seq.index(action_seq)
            action = reward_.max(dim=0)[1].item()
        if self.pydantic_dataset_path is not None:
            action = benchmark_uri  # handled by CompilerGymDatasetPydantic 
        if self.random_mixup:
            traj_ret = self.cursor.execute(
                f"select state_ids from Trajectories where traj_id = ?",
                (traj_id,),
            ).fetchone()
            final_state_id = traj_ret[0].split(" ")[-1]
            ret = self.cursor.execute(
                f"select graph, IrInstructionCount from States where state_id = ?", (final_state_id,)
            ).fetchone()
            final_graph = pickle.loads(zlib.decompress(ret[0]))
            IrInstructionCount1 = ret[1]
            assert self.q_learning or self.real_q_learning
            final_graph = (final_graph, IrInstructionCount1)
            graph = (graph, IrInstructionCount)
        if self.use_autophase:
            # autophase = torch.tensor([int(x) for x in autophase.split()])
            autophase = self._autophase2vec(autophase)
        if self.use_history or self.use_cl:
            # load traj
            if traj_id not in self.traj_buffer.keys():
                traj_ret = self.cursor.execute(
                    f"select state_ids, actions, rewards, graph_repr from Trajectories where traj_id = ?",
                    (traj_id,),
                )
                state_ids, actions, rewards, graph_repr = next(traj_ret)
                if self.use_cl:
                    state_ids = state_ids.split(" ")
                    graph_repr = pickle.loads(graph_repr)
                    state2repr, reprs = self.get_unique_states(state_ids, graph_repr)
                else:
                    state2repr, reprs = None, None
                actions = torch.tensor(
                    [int(a) for a in actions.split(" ")], dtype=torch.long
                )
                rewards = torch.tensor(pickle.loads(rewards), dtype=torch.float)
                reward_sign = torch.sign(rewards).to(torch.long) + 2
                reward_sign = torch.cat([torch.zeros(1, dtype=torch.long), reward_sign])
                self.traj_buffer[traj_id] = (
                    actions,
                    rewards,
                    reward_sign,
                    state2repr,
                    reprs,
                )
            else:
                actions, rewards, reward_sign, state2repr, reprs = self.traj_buffer[
                    traj_id
                ]
            act_history = actions[:traj_step]
            if self.action_histogram_steps:
                histogram = torch.zeros(self.num_actions, dtype=torch.float)
                for a in act_history.tolist():
                    histogram[a] += 1
                act_history = histogram
            rew_history = rewards[:traj_step]
            reward_sign_hist = reward_sign[: traj_step + 1]
            if self.use_cl:
                curr_i = state2repr[state_id]
                next_i = state2repr[next_state_id]
                traj_packed = (traj_id, curr_i, next_i, reprs)
            else:
                traj_packed = None
        else:
            act_history, rew_history, reward_sign_hist, traj_packed = (None,) * 4
        return (
            graph,
            autophase,
            action,
            reward_,
            act_history,
            rew_history,
            advantage,
            logp,
            reward_sign_hist,
            traj_packed,
            time_stamp,
            final_graph,
        )

    def __len__(self):
        return 0

    @staticmethod
    def get_unique_states(state_ids, state_reprs):
        state2repr = {}
        i = 0
        reprs = []
        for sid, sr in zip(state_ids, state_reprs):
            if sid not in state2repr.keys():
                state2repr[sid] = i
                reprs.append(sr)
                i += 1
        reprs = torch.stack(reprs)
        return state2repr, reprs


class CompilerGymDatasetPydantic(CompilerGymDatasetOnline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if os.path.getsize(self.pydantic_dataset_path) / (1024 ** 2) < 200:
            dataset = TrajectoryDataset.load(self.pydantic_dataset_path)
            self.benchmark2seqidx = {item.benchmark : item.best_actionseq_idx for item in dataset.samples}

            if self.dense_cls_metric == "oz":
                self.benchmark2seqrew = {
                    item.benchmark : (item.ir_compiler - torch.tensor(item.all_ir_searches, dtype=torch.float)) / item.ir_compiler
                    for item in dataset.samples
                }
                self.benchmark2oz = {item.benchmark : torch.tensor([item.ir_compiler], dtype=torch.float) for item in dataset.samples}
            elif self.dense_cls_metric == "o0":
                self.benchmark2seqrew = {
                    item.benchmark : (item.ir_original - torch.tensor(item.all_ir_searches, dtype=torch.float)) / item.ir_original
                    for item in dataset.samples
                }
                self.benchmark2oz = {item.benchmark : torch.tensor([item.ir_original], dtype=torch.float) for item in dataset.samples}
            else:
                raise ValueError
        else:
            pkl_file = self.pydantic_dataset_path.replace(".json", "_seqidx.pkl")
            with open(pkl_file, "rb") as f:
                self.benchmark2seqidx = pickle.load(f)
            pkl_file = self.pydantic_dataset_path.replace(".json", "_seqrew.pkl")
            with open(pkl_file, "rb") as f:
                benchmark2seqrew = pickle.load(f)
            if self.dense_cls_metric == "oz":
                self.benchmark2seqrew = {
                    k: (v[0] - v[2]) / v[0] for k, v in benchmark2seqrew.items()
                }
                self.benchmark2oz = {k: torch.tensor([v[0]], dtype=torch.float) for k, v in benchmark2seqrew.items()}
            elif self.dense_cls_metric == "o0":
                self.benchmark2seqrew = {
                    k: (v[1] - v[2]) / v[1] for k, v in benchmark2seqrew.items()
                }
                self.benchmark2oz = {k: torch.tensor([v[1]], dtype=torch.float) for k, v in benchmark2seqrew.items()}
            else:
                raise ValueError
        print(f"Len of benchmark2seqrew: {len(self.benchmark2seqrew)}")
        self.benchmark2seqrew["default"] = torch.zeros_like(next(iter(self.benchmark2seqrew.values())))
        self.benchmark2oz["default"] = torch.tensor([1], dtype=torch.float)
        self.random_mixup = 0  # for this class, mixup is performed in GNN
        self.dense_seq_cls = True

        # for recording training speed
        self.iter_count = 0
        self.read_rows = 0
        self.traj_buffer = DictQueue(maxlen=1000)
        if self.q_learning:
            self.fields = "state_id, action_value, reward, next_state_id, traj_id, traj_step, advantage, logp, time_stamp"
            self.state_fields = "benchmark_uri, graph, autophase, IrInstructionCount"
        else:
            self.fields = "state_id, action_value, cumulative_reward, next_state_id, traj_id, traj_step, advantage, logp, time_stamp"
            self.state_fields = "benchmark_uri, graph, autophase"
        if self.real_q_learning:
            self.state_fields = "benchmark_uri, graph, autophase, IrInstructionCount"
        if self.cache_data:
            self.item_cache = {}

    def load_data_with_rowid(self, rowid):
        # patch the action issue: query the best_actionseq_idx
        ret = super().load_data_with_rowid(rowid)
        ret = list(ret)
        benchmark_uri = ret[2]  # the action field is filled with benchmark_uri
        ret[2] = self.benchmark2seqidx[benchmark_uri]
        return tuple(ret)


class CompilerGymDatasetOnlineSocket(
    torch.utils.data.IterableDataset, CompilerGymDatasetOnline
):
    """The dataset that uses socket for receiving data."""

    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)

        if self.q_learning:
            # output reward
            self.fields = [1, 2, 3, 5, 7, 8, 9, 10, 11]
            self.state_fields = "graph, autophase, IrInstructionCount"
        else:
            # output cumulative_reward
            self.fields = [1, 2, 4, 5, 7, 8, 9, 10, 11]
            self.state_fields = "graph, autophase"

        # signal file for ending training; created by date generation process
        self.sig_file = self.socket_db.replace("socket.db", "term.signal")

    def process(self):
        pass

    def _init_data_tunnel(self):
        self.data_buffer = DataBuffer()
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            client_rank = worker_info.id + utils.get_rank() * worker_info.num_workers
            num_clients = utils.get_world_size() * worker_info.num_workers
        else:
            client_rank = utils.get_rank()
            num_clients = utils.get_world_size()
        self.data_tunnel = DataClient(
            self.socket_db, client_rank, self.num_servers, num_clients
        )
        self.data_packer = SocketDataPacking(
            self.bin_size, self.full_rate, self.num_max_bin
        )

    def __add__(self, other):
        raise NotImplementedError

    def __iter__(self):
        self._init_data_tunnel()
        while True:
            data = self.data_tunnel.serve_data()
            while data is not None:
                self.data_buffer.store(data)
                data = self.data_tunnel.serve_data()
            # check for termination
            if self.term_signal():
                yield None
                continue

            idx_and_data_size = self.data_buffer.pop()
            if idx_and_data_size is not None:  # there are data in data_buffer
                this_idx = self.data_packer.pack(*idx_and_data_size)
                if this_idx is not None:
                    samples = self.get_data(this_idx)
                    # samples may be empty bause old cache is discarded
                    if len(samples) == 0 or self.term_signal():
                        time.sleep(0.1)
                        continue
                    batch = self.collate_fn(samples)
                    yield batch

    def term_signal(self):
        return os.path.exists(self.sig_file)

    def get_data(self, transition_ids):
        transitions = self.data_buffer.get_data("Transitions", transition_ids)
        transitions = [tr for tr in transitions if tr is not None]
        samples = [self.parse_data(transition) for transition in transitions]
        samples = [sa for sa in samples if sa is not None]
        return samples

    def parse_data(self, transition):
        (
            state_id,
            action,
            reward_,
            next_state_id,
            traj_id,
            traj_step,
            advantage,
            logp,
            time_stamp,
        ) = [transition[i] for i in self.fields]
        state_ = self.data_buffer.get_data("States", [state_id])[0]
        if state_ is None:
            return None
        if self.q_learning:
            graph, autophase, IrInstructionCount = state_[2:5]
            reward_ /= IrInstructionCount
        else:
            graph, autophase = state_[2:4]
        if self.use_autophase:
            autophase = self._autophase2vec(autophase)
        if self.use_history or self.use_cl:
            # load traj
            if traj_id not in self.traj_buffer.keys():
                traj_ = self.data_buffer.get_data("Trajectories", [traj_id])[0]
                if traj_ is None:
                    return None
                state_ids, actions, rewards, graph_repr = traj_[2:]
                if self.use_cl:
                    state2repr, reprs = self.get_unique_states(state_ids, graph_repr)
                else:
                    state2repr, reprs = None, None
                actions = torch.tensor(actions, dtype=torch.long)
                rewards = torch.tensor(rewards, dtype=torch.float)
                reward_sign = torch.sign(rewards).to(torch.long) + 2
                reward_sign = torch.cat([torch.zeros(1, dtype=torch.long), reward_sign])
                self.traj_buffer[traj_id] = (
                    actions,
                    rewards,
                    reward_sign,
                    state2repr,
                    reprs,
                )
            else:
                actions, rewards, reward_sign, state2repr, reprs = self.traj_buffer[
                    traj_id
                ]
            act_history = actions[:traj_step]
            if self.action_histogram_steps:
                histogram = torch.zeros(self.num_actions, dtype=torch.float)
                for a in act_history.tolist():
                    histogram[a] += 1
                act_history = histogram
            rew_history = rewards[:traj_step]
            reward_sign_hist = reward_sign[: traj_step + 1]
            if self.use_cl:
                curr_i = state2repr[state_id]
                next_i = state2repr[next_state_id]
                traj_packed = (traj_id, curr_i, next_i, reprs)
            else:
                traj_packed = None
        else:
            act_history, rew_history, reward_sign_hist, traj_packed = (None,) * 4
        return (
            graph,
            autophase,
            action,
            reward_,
            act_history,
            rew_history,
            advantage,
            logp,
            reward_sign_hist,
            traj_packed,
            time_stamp,
        )


def dummy_collate(x):
    return x


class DummyBatchSampler:
    def __iter__(self):
        yield [0]


class CompilerGymDatasetOnlinePreload(CompilerGymDatasetOnline):
    """The data are preloaded in batch sampler and sent to get_item"""

    def __getitem__(self, i):
        rowid, state_id, action, cumulative_reward, num_nodes, graph, autophase = i
        graph = pickle.loads(zlib.decompress(graph))
        if self.use_autophase:
            autophase = self._autophase2vec(autophase)
        return graph, autophase, action, cumulative_reward, rowid


class DictQueue(MutableMapping):
    def __init__(self, maxlen, *args, **kwargs):
        self.maxlen = maxlen
        self.store = OrderedDict()
        self.update(dict(*args, **kwargs))  # use the free update to set keys

    def __getitem__(self, key):
        return self.store[self._keytransform(key)]

    def __setitem__(self, key, value):
        self.store[self._keytransform(key)] = value
        if len(self) > self.maxlen:
            first_key = next(iter(self.store.keys()))
            self.store.pop(first_key)

    def __delitem__(self, key):
        del self.store[self._keytransform(key)]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def _keytransform(self, key):
        return key
