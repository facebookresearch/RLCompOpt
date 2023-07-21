
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph


def dgl2pyg(dgl_graph):
    u, v = dgl_graph.edges()
    edge_index = torch.cat([u.unsqueeze(0), v.unsqueeze(0)], dim=0)
    text_idx = dgl_graph.ndata.pop('text_idx')  # dgl_graph.ndata is like a dict
    # other_node_feat = dict(**dgl_graph.ndata)
    # edge_feat = dict(**dgl_graph.edata)
    other_node_feat = {k: v.view(v.shape[0], -1) for k, v in dgl_graph.ndata.items()}
    edge_feat = {k: v.view(v.shape[0], -1) for k, v in dgl_graph.edata.items()}

    # node feature `x` should have the shape [num_nodes, ...], so unsqueeze
    data = Data(x=text_idx.unsqueeze(1), edge_index=edge_index, **other_node_feat, **edge_feat)
    # get_blk_idx(data)
    return data


def remove_type_nodes(pyg_graph):
    mask = pyg_graph['type'].flatten() != 3  # 3 denotes the type nodes
    return remove_nodes(pyg_graph, mask)


def remove_nodes(pyg_graph, subset, edge_attr=('flow', 'position'), node_attr=('x', 'type', 'function', 'block')):
    assert isinstance(pyg_graph, Data)
    edge_index = pyg_graph.edge_index
    edge_index, _, edge_mask = subgraph(subset, edge_index, relabel_nodes=True, num_nodes=pyg_graph.num_nodes, return_edge_mask=True)
    pyg_graph.edge_index = edge_index

    for attr in node_attr:
        pyg_graph[attr] = pyg_graph[attr][subset]
    for attr in edge_attr:
        pyg_graph[attr] = pyg_graph[attr][edge_mask]
    return pyg_graph


def remove_edges(pyg_graph, edge_ids_to_remove=None, edge_ids_to_keep=None,  edge_attr=()):
    # edge_index has shape [2, num_edges], edges features have shape [num_edges, dim_feat]
    if edge_ids_to_keep is not None:
        pyg_graph.edge_index = pyg_graph.edge_index[:, edge_ids_to_keep]
        for attr in edge_attr:
            pyg_graph[attr] = pyg_graph[attr][edge_ids_to_keep]
        if pyg_graph.edge_attr is not None:
            pyg_graph.edge_attr = pyg_graph.edge_attr[edge_ids_to_keep]
    else:
        raise NotImplementedError
    return pyg_graph


def get_blk_idx(graph):
    """
    This function adds the position of each intruction node 
    within a basic block to the graph node attribute.
    After that, we have function idx, block idx, position in block.
    This can be used in attention layer to put some info to the attn.
    Args:
        graph (PyG graph): the graph to convert
    Outputs:
        in-place modification to the graph by adding an extra block_pos attribute, and
        the instruction nodes idx ordered by blocks (excluding blocks with single instruciton)
    """
    if graph.get("ordered_instr_idx", None) is not None:
        return
    flow_mask = graph['flow'].flatten() == 0
    instr_edges = graph.edge_index.T[flow_mask]
    num_block = graph['block'].max().item() + 1
    blk = graph['block'].flatten().tolist()
    block_edges = [[] for _ in range(num_block)]
    for ie in instr_edges:
        b0 = blk[ie[0]]
        b1 = blk[ie[1]]
        block_edges[b0].append(ie.tolist())
        if b0 != b1:
            block_edges[b1].append(ie.tolist())
    nodes = []
    pos = []
    func = graph['function'].flatten().tolist()
    for i, be in enumerate(block_edges):
        if len(be) == 0:
            continue
        thic = func[be[0][0]]
        for e in be:
            assert func[e[0]] == thic and func[e[1]] == thic
        ordered_nodes = order_block(be, blk, i)
        nodes.append(torch.tensor(ordered_nodes, dtype=torch.long))
        pos.append(torch.arange(len(ordered_nodes)))

    # instr_mask = graph['type'].flatten() == 0
    # blk_pos = torch.zeros(instr_mask.shape[0], dtype=torch.long)
    if nodes:
        nodes_ = torch.cat(nodes)  # excluding the instructions in blocks with single instruciton
        pos_ = torch.cat(pos)
        # blk_pos[nodes_] = pos_
        graph['ordered_instr_idx0'] = nodes_
        graph['ordered_instr_idx'] = nodes_  # hack to make batching work
        graph['blk_pos'] = pos_
    else:
        graph['ordered_instr_idx'] = torch.tensor([], dtype=torch.long)
        graph['ordered_instr_idx0'] = graph['ordered_instr_idx']
        graph['blk_pos'] = torch.tensor([], dtype=torch.long)
    # graph['blk_pos'] = blk_pos


def order_block(blk_edges, idx2block, this_blk):
    """
    Given a list of edges (2-tuple), 
    find the order of the nodes in the control flow.
    Args:
        blk_edges: a list of edges (2-tuple)
        idx2block (list): given the node idx, get its block idx
        this_blk (int): the block idx for this run
    """
    assert isinstance(blk_edges, list)
    assert isinstance(idx2block, list)
    assert isinstance(this_blk, int)
    starts = set(e[0] for e in blk_edges)
    ends = set(e[1] for e in blk_edges)
    end_not_in_start = ends - starts  # could have more than 2: multiple ends in branching; could be empty
    start_not_in_end = starts - ends  # could have more than 2: multiple branches into this block; could be empty

    starters = []
    start_of_function = False
    for s in start_not_in_end:
        if idx2block[s] == this_blk:
            # this is the start of a function
            starter = s
            start_of_function = True
            break
        starters.extend([e[1] for e in blk_edges if e[0] == s])
    if len(starters) > 0:
        assert not start_of_function
        starters = set(starters)
        assert len(starters) == 1, f"{starters=}"
        starter = list(starters)[0]
    
    # at the end of the block, there is a branch going to the predecessor of the starter of the block
    if not start_not_in_end:
        for e in blk_edges:
            if idx2block[e[0]] != this_blk and idx2block[e[1]] == this_blk:
                starter = e[1]
                break
    for e in blk_edges:
        if e[1] == starter and idx2block[e[0]] != this_blk:
            # make sure the end not going to the starter
            end_not_in_start.add(e[0])

    ordered_nodes = [starter]
    while True:
        end_node = ordered_nodes[-1]
        if end_node in end_not_in_start:
            # "ret" could be in the same block,
            # in this case not to remove the last added node
            if idx2block[end_node] != this_blk:
                ordered_nodes = ordered_nodes[:-1]
            break
        end_flag = False
        for e in blk_edges:
            if e[0] == end_node:
                if e[1] in ordered_nodes:
                    # the end goes to the starter
                    assert e[1] == starter
                    assert idx2block[e[0]] == this_blk
                    end_flag = True
                    break
                ordered_nodes.append(e[1])
                break
        if end_flag:
            break
    return ordered_nodes
