
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Borrowed code from https://github.com/snap-stanford/ogb/blob/master/examples/graphproppred/code2/conv.py

import torch
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch_geometric.utils import degree


### GIN convolution along the graph structure
class GINConv(MessagePassing):
    def __init__(self, in_channels=None, edge_dim=None, aggr=None, **kwargs):
        '''
        emb_dim (int): node embedding dimensionality
        '''
        emb_dim = in_channels

        super(GINConv, self).__init__(aggr = aggr)

        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

        self.edge_dim = edge_dim
        if self.edge_dim is not None:
            self.edge_encoder = torch.nn.Linear(edge_dim, emb_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = None
        if self.edge_dim is not None:
            edge_embedding = self.edge_encoder(edge_attr)
        out = self.mlp((1 + self.eps) *x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))

        return out

    def message(self, x_j, edge_attr):
        if edge_attr is not None:
            x_j = x_j + edge_attr
        return F.relu(x_j)

    def update(self, aggr_out):
        return aggr_out


### GCN convolution along the graph structure
class GCNConv(MessagePassing):
    def __init__(self, in_channels=None, edge_dim=None, aggr=None, **kwargs):
        super(GCNConv, self).__init__(aggr=aggr)
        emb_dim = in_channels

        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)

        self.edge_dim = edge_dim
        if self.edge_dim is not None:
            self.edge_encoder = torch.nn.Linear(edge_dim, emb_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        edge_embedding = None
        if self.edge_dim is not None:
            edge_embedding = self.edge_encoder(edge_attr)

        row, col = edge_index

        #edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        deg = degree(row, x.size(0), dtype = x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, edge_attr = edge_embedding, norm=norm) + F.relu(x + self.root_emb.weight) * 1./deg.view(-1,1)

    def message(self, x_j, edge_attr, norm):
        if edge_attr is not None:
            x_j = x_j + edge_attr
        return norm.view(-1, 1) * F.relu(x_j)

    def update(self, aggr_out):
        return aggr_out

