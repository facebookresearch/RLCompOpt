
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from pathlib import Path

from functools import cached_property
import pydantic
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

from pydantic import BaseModel

from scipy.stats.mstats import gmean
import numpy as np


########################################################################
# trainer / inference skeleton example
# these will be run at different times, so do not share any state between them
# the only inputs should be file names and potentially command line args to select
# which ablations we are running e.g.:
# (baseline GNN / new GNN) or (BC / BC + label smoothing)

def trainer(train_dataset_path : str, val_dataset_path : str):
    train_dataset = TrajectoryDataset.load(train_dataset_path)
    val_dataset = TrajectoryDataset.load(val_dataset_path)

    # train loop goes here. in the end it should save its full state into a
    # checkpoint or something (feel free to use anything, but it should be self
    # contained in one file)
    return 'ckpt_path'


def runner(test_dataset_path : str, ckpt_path : str):
    test_dataset = BenchmarkDataset.load(test_dataset_path)
    # load model from ckpt_path...
    samples = []

    for benchmark in test_dataset.benchmarks:
        # rollout policy on this benchmark
        ir_counts = []
        for step in range(200):
            # this would be the ir count at the current step
            ir_count = 123456
            ir_counts.append(ir_count)
        samples.append(PolicyRolloutSample(benchmark=benchmark, ir_search_trajector=ir_counts))

    output = PolicyRolloutDataset(
        benchmark_dataset=test_dataset,
        policy_name='awesome-policy',
        samples=samples)

    save_path = 'somewhere_good.json'
    output.save(save_path)
    return save_path
        

########################################################################
# helper functions

def pydantic_save(obj, path):
    with open(path, 'w') as fh:
        fh.write(obj.json())

def pydantic_load(cls, path):
    return cls.parse_file(path)


########################################################################
# BenchmarkDataset
from collections import defaultdict

class BenchmarkDataset(BaseModel):
    name: str
    split: str
    benchmarks: List[str]

    @property
    def fullname(self):
        return f'{self.name}-{self.split}'
    
    def save(self, path):
        pydantic_save(self, path)

    @classmethod
    def load(cls, path):
        return pydantic_load(cls, path)
    
    def sample_by_dataset(self, max_per_dataset : int) -> List[str]:
        dataset_to_bm = defaultdict(int)
        output = []
        for bm in self.benchmarks:
            _, dataset, _ = parse_benchmark(bm)
            if dataset_to_bm[dataset] >= max_per_dataset:
                continue
            dataset_to_bm[dataset] += 1
            output.append(bm)
        return output
    
    def __repr__(self):
        return f'BenchmarkDataset(name={self.name},split={self.split},benchmarks={len(self.benchmarks)}'
    
    def __str__(self):
        return repr(self)
    

def parse_benchmark(bm : str):
    typ, _, rest = bm.partition('://')
    dataset, _, name = rest.partition('/')
    return typ, dataset, name
    

    

########################################################################
# ActionSequences
#
# We can turn this into a tree, but for now we do not use it this way.

class ActionSequence(BaseModel):
    # I like using a tuple here because tuples are immutable in python
    actions: Tuple[int, ...]

class ActionSequences(BaseModel):
    name: str
    actionseqs: List[ActionSequence]
    train_dataset_name: str
    
    @property
    def fullname(self):
        return f'{self.name}-{self.train_dataset_name}'

    def save(self, path):
        pydantic_save(self, path)

    @classmethod
    def load(cls, path):
        return pydantic_load(cls, path)
    
    def __str__(self):
        return repr(self)
    
    def __repr__(self):
        return f'ActionSequences(name={self.name}, actionseqs={len(self.actionseqs)}, train_dataset_name={self.train_dataset_name})'

########################################################################
# TrajectoryDataset
#
# We embed the action sequence into the trajectory dataset instead of storing it seperately
# We also keep a copy of the original benchmark dataset used to generate the trajectories
# this wastes a little space, but makes the pipeline plumbing easier
class TrajectorySample(BaseModel):
    benchmark: str

    # the number of ir instruction count before any optimization
    ir_original: int

    # the number of ir instruction after running -Oz
    ir_compiler: int

    # the number of ir instruction after running the action sequence
    # NOTE: this will be best seen during running the whole action sequence
    # NOTE: the length of this list is the same as the number of action sequences
    all_ir_searches: List[int]

    # this is the canonically best action sequence,
    # it may tie with others, but our sorting order makes this unique
    best_actionseq_idx: int

    # These are the normalized autophase features for the intial state of the benchmark
    autophase: List[float]

    # Here are a few example helper functions, no need to use, but good for self documentation
    @property
    def best_ir_search(self):
        best_ir_search = min(self.all_ir_searches)
        assert best_ir_search == self.all_ir_searches[self.best_actionseq_idx]
        return best_ir_search
    
    def oracle_improvement_over_o0(self):
        return (self.ir_original - self.best_ir_search) / self.ir_original

    def oracle_improvement_over_oz(self):
        # Seems to be the common metric from other compiler pass ordering papers
        return (self.ir_compiler - self.best_ir_search) / self.ir_compiler

    def oracle_instruction_count_reduction(self):
        # Used in the compilergym paper (in combination with geometric mean)
        return self.ir_compiler / self.best_ir_search
    


class TrajectoryDataset(BaseModel):
    benchmark_dataset: BenchmarkDataset
    action_sequences: ActionSequences
    samples: List[TrajectorySample]

    @property
    def fullname(self):
        return f'{self.benchmark_dataset.fullname}-{self.action_sequences.fullname}'

    @property
    def name(self):
        return f'{self.benchmark_dataset.fullname}'

    def save(self, path):
        pydantic_save(self, path)

    @classmethod
    def load(cls, path):
        return pydantic_load(cls, path)
    
    def __str__(self):
        return repr(self)
    
    def __repr__(self):
        return f'TrajectoryDataset(benchmark_dataset={self.benchmark_dataset}, action_sequences={self.action_sequences}, samples={len(self.samples)}'

    def cg_dataset_map(self):
        mapping = defaultdict(list)
        for s in self.samples:
            _, dataset, _ = parse_benchmark(s.benchmark)
            mapping[dataset].append(s)
        return mapping

    def oracle_metrics(self):
        print("ORACLE metrics for:", self)

        print('\nMean of instruction count reduction relative to -Oz (Autophase paper metric)')
        mean_improvements = []
        mean_improvement_map = {}
        for cg_dataset, samples in self.cg_dataset_map().items():
            improvements = []
            for sample in samples:
                improvements.append(sample.oracle_improvement_over_oz())
            mean_improvement = np.mean(improvements)
            mean_improvement_map[cg_dataset] = mean_improvement
            mean_improvements.append(mean_improvement)
        mean_mean_improvement = np.mean(mean_improvements)
        mean_improvement_map['_all_'] = mean_mean_improvement

        for dataset, metric in sorted(mean_improvement_map.items()):
            print(f"  {dataset:20s}: {1+metric:9.3f} {metric:9.1%}")

        print('\nGeometric mean of instruction count reduction ratio (CompilerGym paper metric)')
        mean_improvements = []
        mean_improvement_map = {}
        for cg_dataset, samples in self.cg_dataset_map().items():
            improvements = []
            for sample in samples:
                improvements.append(sample.oracle_instruction_count_reduction())
            mean_improvement = gmean(improvements)
            mean_improvement_map[cg_dataset] = mean_improvement
            mean_improvements.append(mean_improvement)
        mean_mean_improvement = gmean(mean_improvements)
        mean_improvement_map['_all_'] = mean_mean_improvement

        for dataset, metric in sorted(mean_improvement_map.items()):
            print(f"  {dataset:20s}: {metric:9.3f} {metric - 1:9.1%}")

        print('\nInterquantile mean of instruction count reduction relative to -Oz (more robust to outliers)')
        mean_improvements = []
        mean_improvement_map = {}
        for cg_dataset, samples in self.cg_dataset_map().items():
            improvements = []
            for sample in samples:
                improvements.append(sample.oracle_improvement_over_oz())
            mean_improvement = iqm(improvements)
            mean_improvement_map[cg_dataset] = mean_improvement
            mean_improvements.append(mean_improvement)
        mean_mean_improvement = np.mean(mean_improvements)
        mean_improvement_map['_all_'] = mean_mean_improvement
        for dataset, metric in sorted(mean_improvement_map.items()):
            print(f"  {dataset:20s}: {1+metric:9.3f} {metric:9.1%}")
    
########################################################################
# PolicyDataset
#
class PolicyRolloutSample(BaseModel):
    benchmark: str

    # the number of ir instruction count before any optimization
    ir_original: int

    # the number of ir instruction after running -Oz
    ir_compiler: int

    # the trajectroy of the ir count when rolling out the policy, during eval we 
    # will truncate this to max number of steps allowed (e.g. 45) and then take
    # the minimum to get the final `ir_search` value for downstream analysis
    ir_search_trajectory: List[int]

    def best_ir_search(self, max_steps):
        return min(self.ir_search_trajectory[:max_steps+1])

    def improvement_over_oz(self, max_steps):
        return (self.ir_compiler - self.best_ir_search(max_steps)) / self.ir_compiler
    
    def instruction_count_reduction(self, max_steps):
        return self.ir_compiler / self.best_ir_search(max_steps)
    
    def best_ir_reduction(self, max_steps):
        return (self.ir_compiler - self.best_ir_search(max_steps))

class PolicyRolloutDataset(BaseModel):
    benchmark_dataset: BenchmarkDataset
    policy_name: str
    samples: List[PolicyRolloutSample]

    @property
    def fullname(self):
        return f'{self.benchmark_dataset.fullname}-{self.action_sequences.name}-{self.policy_name}'

    def save(self, path):
        pydantic_save(self, path)

    @classmethod
    def load(cls, path):
        return pydantic_load(cls, path)
    
    def cg_dataset_map(self):
        mapping = defaultdict(list)
        for s in self.samples:
            _, dataset, _ = parse_benchmark(s.benchmark)
            mapping[dataset].append(s)
        return mapping
    
    def get_metric(self, max_steps):
        mapping = self.cg_dataset_map()
        metrics = {
            k : [s.improvement_over_oz(max_steps) for s in v] for k, v in mapping.items()
        }
        metrics = {
            k : sum(v) / len(v) for k, v in metrics.items()
        }
        return metrics

    def __str__(self):
        return repr(self)
    
    def __repr__(self) -> str:
        return f'PolicyRolloutDataset(benchmark_dataset={self.benchmark_dataset},policy_name={self.policy_name},samples={len(self.samples)}'
    
    def eval_metrics(self, max_steps=45, return_oz_metric=False):
        logstr = ""
        logstr += ('\nMean of instruction count reduction relative to -Oz (Autophase paper metric)\n')
        mean_improvements = []
        mean_improvement_map = {}
        total_improvements = []
        total_improvement_map = {}
        for cg_dataset, samples in self.cg_dataset_map().items():
            improvements = []
            total_imp = 0
            total_ir_compiler = 0
            for sample in samples:
                best_ir_search = sample.best_ir_search(max_steps)
                if best_ir_search == 0:
                    print(f"Warning: found best_ir_search == 0 for {sample.benchmark}")
                assert sample.ir_original != 0
                if best_ir_search > sample.ir_original:
                    logstr += ('warning, fix trajectory to inlucde original sample at the beginning\n')
                    best_ir_search = sample.ir_original
                improvements.append(sample.improvement_over_oz(max_steps))
                total_imp += sample.best_ir_reduction(max_steps)
                total_ir_compiler += sample.ir_compiler
            total_metric = total_imp / max(1, total_ir_compiler)
            total_improvements.append(total_metric)
            total_improvement_map[cg_dataset] = total_metric
            mean_improvement = np.mean(improvements)
            mean_improvement_map[cg_dataset] = mean_improvement
            mean_improvements.append(mean_improvement)
        mean_mean_improvement = np.mean(mean_improvements)
        mean_improvement_map['_all_'] = mean_mean_improvement
        for dataset, metric in sorted(mean_improvement_map.items()):
            logstr += (f"  {dataset:20s}: {1+metric:9.3f} {metric:9.1%}\n")
        mean_total_imp = np.mean(total_improvements)
        total_improvement_map['_all_'] =mean_total_imp
        logstr += ('\nMean of per-dataset total instruction count reduction relative to -Oz\n')
        for dataset, metric in sorted(total_improvement_map.items()):
            logstr += (f"  {dataset:20s}: {1+metric:9.3f} {metric:9.1%}\n")
        if return_oz_metric:
            return mean_mean_improvement

        logstr += ('\nGeometric mean of instruction count reduction ratio (CompilerGym paper metric)\n')
        mean_improvements = []
        mean_improvement_map = {}
        for cg_dataset, samples in self.cg_dataset_map().items():
            improvements = []
            for sample in samples:
                improvements.append(sample.instruction_count_reduction(max_steps))
            mean_improvement = gmean(improvements)
            mean_improvement_map[cg_dataset] = mean_improvement
            mean_improvements.append(mean_improvement)
        mean_mean_improvement = gmean(mean_improvements)
        mean_improvement_map['_all_'] = mean_mean_improvement
        for dataset, metric in sorted(mean_improvement_map.items()):
            logstr += (f"  {dataset:20s}: {metric:9.3f} {metric - 1:9.1%}\n")

        logstr += ('\nInterquantile mean of instruction count reduction relative to -Oz (more robust to outliers)\n')
        mean_improvements = []
        mean_improvement_map = {}
        for cg_dataset, samples in self.cg_dataset_map().items():
            improvements = []
            for sample in samples:
                improvements.append(sample.improvement_over_oz(max_steps))
            mean_improvement = iqm(improvements)
            mean_improvement_map[cg_dataset] = mean_improvement
            mean_improvements.append(mean_improvement)
        mean_mean_improvement = np.mean(mean_improvements)
        mean_improvement_map['_all_'] = mean_mean_improvement
        for dataset, metric in sorted(mean_improvement_map.items()):
            logstr += (f"  {dataset:20s}: {1+metric:9.3f} {metric:9.1%}\n")
        return logstr

########################################################################
# Action Tree

@dataclass
class Node:
    uid: int
    value: str = ""
    is_terminal: bool = False
    children: dict = field(default_factory=dict)

    @property
    def is_root(self):
        return self.value == ""


class PrefixTree:
    def __init__(self):
        self._uid = 0
        self.root = Node(self._get_uid())
        self.n_nodes = 1

    def _get_uid(self):
        uid = self._uid
        self._uid += 1
        return uid

    def add(self, xs):
        node = self.root
        n_new_nodes = 0
        for x in xs:
            if x not in node.children:
                node.children[x] = Node(self._get_uid(), value=x)
                n_new_nodes += 1
                self.n_nodes += 1
            node = node.children[x]
        node.is_terminal = True
        return n_new_nodes

    def make_dot(self):
        output = ["digraph prefix_tree {"]
        nodes = [self.root]
        while nodes:
            next_nodes = []
            for node in nodes:
                if node.is_root:
                    label = "(root)"
                else:
                    label = node.value
                if node.is_terminal:
                    shape = "box"
                else:
                    shape = "circle"
                output.append(f'node{node.uid} [label="{label}" shape="{shape}"];')
                for child_node in node.children.values():
                    output.append(f"node{node.uid} -> node{child_node.uid};")
                    next_nodes.append(child_node)
            nodes = next_nodes
        #    node0 -> node1;
        output.append("}")
        return "\n".join(output)
    
    def __repr__(self):
        return f'PrefixTree(n_nodes={self.n_nodes})'

########################################################################
# helper functions

def iqm(xs):
    xs = np.array(xs)
    assert len(xs.shape) == 1
    assert xs.shape[0] >= 4, 'not enough elements for IQM calc'

    xs = np.sort(xs)
    n = len(xs)
    a, b = n // 4, n * 3 // 4
    xs = xs[a:b]
    return np.mean(xs)
