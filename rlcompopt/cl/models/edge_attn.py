
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch_scatter
from torch import Tensor, LongTensor


def indexing(x: Tensor, idx: LongTensor):
    assert idx.ndim == 1
    return torch.index_select(x, 0, idx)


class EdgeEncoding(nn.Module):
    r"""
    Edge encoding that encodes edge types, edge positions,
    and block differences (whether in the same basic block, 
    positional differences in the basic block).
    """

    def __init__(
        self,
        edge_dim: int,
        num_edge_types: int = 4,
        max_edge_positions: int = 32,
        max_blk_diff: int = 32,
    ):
        super().__init__()
        self.edge_dim = edge_dim
        self.max_edge_positions = max_edge_positions - 1
        self.max_blk_diff = max_blk_diff
        self.edge_type_enc = nn.Embedding(num_edge_types, edge_dim)
        self.edge_pos_enc = nn.Embedding(max_edge_positions, edge_dim)
        self.blk_diff = nn.Embedding(3, edge_dim)  # before/same/after block
        # self.blk_enc = nn.Embedding(max_blk_diff * 2 + 1, edge_dim)

    def get_relative_pos(self, idx0, idx1, max_diff):
        diff = idx0 - idx1
        diff.clamp_(-max_diff, max_diff).add_(max_diff)
        return diff.detach()

    def get_sign(self, idx0, idx1):
        diff = idx0 - idx1
        sign = diff.sign().long() + 1
        return sign.detach()

    def forward(self, edge_types, edge_pos, block_idx, block_pos=None):
        assert edge_types.ndim == edge_pos.ndim == 1
        assert block_idx.ndim == 2 and block_idx.shape[0] == 2
        if block_pos is not None:
            assert block_pos.ndim == 2 and block_pos.shape[0] == 2

        type_emb = self.edge_type_enc(edge_types)

        edge_pos = edge_pos.clone()
        edge_pos.clamp_(0, self.max_edge_positions)
        pos_emb = self.edge_pos_enc(edge_pos)

        block_diff = self.get_sign(block_idx[0], block_idx[1])
        block_d = self.blk_diff(block_diff)
        # same_blk = block_diff == 1

        # blk_pos_diff = self.get_relative_pos(block_pos[0], block_pos[1], self.max_blk_diff)
        # blk_pos_diff = blk_pos_diff[same_blk]  # only encode if in the same block
        # blk_pos_emb = self.blk_enc(blk_pos_diff)

        edge_emb = type_emb + pos_emb + block_d
        # edge_emb = edge_emb.clone()
        # edge_emb[same_blk] += blk_pos_emb
        return edge_emb


class EdgeAttn(nn.Module):
    r"""
    A graph neural network with node-edge-node basic computation blocks.
    """

    def __init__(
        self,
        out_channels: int,
        edge_dim: int,
        bias: bool = True,
        num_heads: int = 1,
        zero_edge_emb: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.out_channels = out_channels
        self.edge_dim = edge_dim
        concat_dim = 2 * out_channels + edge_dim
        out_dim = concat_dim + num_heads * 2
        self.num_heads = num_heads
        self.head_dim = out_channels // num_heads
        self.zero_edge_emb = zero_edge_emb
        assert self.head_dim * num_heads == out_channels

        self.attn = Mlp(concat_dim, concat_dim, out_dim, bias=bias)
        # self.attn = nn.Linear(concat_dim, out_dim, bias=bias)
        self.node_mlp = Mlp(out_channels, out_channels)
        self.edge_mlp = Mlp(edge_dim, edge_dim)
        self.node_norm0 = nn.LayerNorm(out_channels)
        self.edge_norm0 = nn.LayerNorm(edge_dim)
        self.node_norm1 = nn.LayerNorm(out_channels)
        self.edge_norm1 = nn.LayerNorm(edge_dim)

        self.reset_parameters()

    def reset_parameters(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(
        self, x: Tensor, edge_index: LongTensor, edge_attr: Tensor = None, **kwargs
    ) -> Tensor:
        r"""
        Before sending a graph to this module, the type nodes/edges should be processed separately,
        and the call edges should be removed.
        """
        shortcut = x
        x = self.node_norm0(x)
        if self.zero_edge_emb:
            edge_attr = edge_attr * 0
        shortcut_e = edge_attr
        edge_attr = self.edge_norm0(edge_attr)
        # TODO: add self loop?
        src_idx = edge_index[0]
        tgt_idx = edge_index[1]
        src = indexing(x, src_idx)
        tgt = indexing(x, tgt_idx)

        node_pair_edge = torch.cat([src, tgt, edge_attr], dim=1)
        node_pair_edge = self.attn(node_pair_edge)
        raw_attn0, raw_attn1, node_s, node_t, edges = torch.split(
            node_pair_edge, 
            [self.num_heads, self.num_heads, self.out_channels, self.out_channels, self.edge_dim],
            dim=1
        )
        raw_attn = torch.cat([raw_attn0, raw_attn1], dim=0)  # [2 * num_edges, num_heads]
        e_idx = torch.cat([src_idx, tgt_idx], dim=0)
        nodes = torch.cat([node_s, node_t], dim=0).view(-1, self.num_heads, self.head_dim)  # [2 * num_edges, num_heads, head_dim]

        attn = torch_scatter.scatter_softmax(raw_attn, e_idx, dim=0)  # [2 * num_edges, num_heads]
        nodes = nodes * attn.unsqueeze(-1)
        nodes = nodes.view(-1, self.out_channels)  # [2 * num_edges, out_channels]

        new_nodes = torch_scatter.scatter_add(nodes, e_idx, dim=0, dim_size=x.shape[0])  # [num_nodes, out_channels]
        # TODO: residual connection?

        new_nodes = self.node_mlp(self.node_norm1(new_nodes))
        new_edges = self.edge_mlp(self.edge_norm1(edges))

        new_nodes = new_nodes + shortcut
        new_edges = new_edges + shortcut_e
        if self.zero_edge_emb:
            new_edges = new_edges * 0

        return new_nodes, new_edges


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        bias=True,
    ):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = None
        if out_features is not None:
            self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        if self.fc2 is not None:
            x = self.fc2(x)
        return x
