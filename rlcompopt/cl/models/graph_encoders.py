
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch_scatter
from torch import Tensor
from torch.nn import Parameter as Param
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import uniform
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import add_self_loops
from torch_sparse import SparseTensor, matmul


class GatedGraphConv(MessagePassing):
    r"""The gated graph convolution operator from the `"Gated Graph Sequence
    Neural Networks" <https://arxiv.org/abs/1511.05493>`_ paper

    .. math::
        \mathbf{h}_i^{(0)} &= \mathbf{x}_i \, \Vert \, \mathbf{0}

        \mathbf{m}_i^{(l+1)} &= \sum_{j \in \mathcal{N}(i)} e_{j,i} \cdot
        \mathbf{\Theta} \cdot \mathbf{h}_j^{(l)}

        \mathbf{h}_i^{(l+1)} &= \textrm{GRU} (\mathbf{m}_i^{(l+1)},
        \mathbf{h}_i^{(l)})

    up to representation :math:`\mathbf{h}_i^{(L)}`.
    The number of input channels of :math:`\mathbf{x}_i` needs to be less or
    equal than :obj:`out_channels`.
    :math:`e_{j,i}` denotes the edge weight from source node :obj:`j` to target
    node :obj:`i` (default: :obj:`1`)

    Args:
        out_channels (int): Size of each output sample.
        num_layers (int): The sequence length :math:`L`.
        aggr (string, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"add"`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})`,
          edge indices :math:`(2, |\mathcal{E}|)`
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})`

    """

    def __init__(
        self,
        out_channels: int,
        edge_dim: int = 0,
        num_layers: int = 2,
        aggr: str = "add",
        bias: bool = True,
        use_reversed_edge: bool = True,
        num_heads: int = 1,
        **kwargs,
    ):
        super().__init__(aggr=aggr, **kwargs)

        self.out_channels = out_channels
        self.num_layers = num_layers
        self.use_reversed_edge = use_reversed_edge

        multiplier = 2 if self.use_reversed_edge else 1
        self.weight = Param(Tensor(num_layers, out_channels, out_channels * multiplier))
        self.rnn = torch.nn.GRUCell(out_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.out_channels, self.weight)
        self.rnn.reset_parameters()

    def forward(
        self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None, flow=None, **kwargs
    ) -> Tensor:
        """"""
        if x.size(-1) > self.out_channels:
            raise ValueError(
                "The number of input channels is not allowed to "
                "be larger than the number of output channels"
            )

        if x.size(-1) < self.out_channels:
            zero = x.new_zeros(x.size(0), self.out_channels - x.size(-1))
            x = torch.cat([x, zero], dim=1)

        if self.use_reversed_edge:
            assert edge_index.shape[0] == 2
            if self.use_reversed_edge == 1:
                ctrl_flow = flow.flatten() == 0
            elif self.use_reversed_edge == 2:
                ctrl_flow = flow.flatten() <= 1
            back_edge_index = edge_index.T[ctrl_flow].T
            reversed_edge_index = torch.flip(back_edge_index, dims=(0,))

        for i in range(self.num_layers):
            m = torch.matmul(x, self.weight[i])
            # propagate_type: (x: Tensor, edge_weight: OptTensor)
            if self.use_reversed_edge:
                if edge_weight is None:
                    edge_weight = (edge_weight,) * 2
                m0, m1 = torch.chunk(m, 2, dim=-1)
                m0 = self.propagate(edge_index, x=m0, edge_weight=edge_weight[0], size=None)
                m1 = self.propagate(reversed_edge_index, x=m1, edge_weight=edge_weight[1], size=None)
                if self.aggr == "add":
                    m = m0 + m1
                elif self.aggr == "mean":
                    m = (m0 + m1) / 2
                else:
                    raise NotImplementedError
            else:
                m = self.propagate(edge_index, x=m, edge_weight=edge_weight, size=None)
            x = self.rnn(m, x)

        return x

    def message(self, x_j: Tensor, edge_weight: OptTensor):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.out_channels}, "
            f"num_layers={self.num_layers})"
        )


class GatedGraphConvAttn(GatedGraphConv):
    """Gated graph convolution with attention as the edge weights."""

    def __init__(
        self,
        out_channels: int,
        num_layers: int,
        aggr: str = "add",
        bias: bool = True,
        use_reversed_edge: bool = True,
        num_heads: int = 1,
        **kwargs,
    ):
        MessagePassing.__init__(self, aggr=aggr, **kwargs)

        self.out_channels = out_channels
        self.num_layers = num_layers
        self.use_reversed_edge = use_reversed_edge
        head_dim = out_channels
        self.scale = head_dim**-0.5

        self.multiplier = 3
        self.weight = Param(
            Tensor(num_layers, out_channels, out_channels * self.multiplier)
        )
        self.back_edge_enc = Param(
            torch.tensor(out_channels, dtype=torch.float)
        )
        self.rnn = torch.nn.GRUCell(out_channels, out_channels, bias=bias)
        self.layernorm = torch.nn.ModuleList(
            [
                torch.nn.LayerNorm(normalized_shape=out_channels)
                for _ in range(num_layers)
            ]
        )

        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.out_channels, self.weight)
        uniform(self.out_channels, self.back_edge_enc)
        self.rnn.reset_parameters()
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
        self,
        x: Tensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
        key_edge_mask=None,
        **kwargs
    ) -> Tensor:
        """"""
        if x.size(-1) > self.out_channels:
            raise ValueError(
                "The number of input channels is not allowed to "
                "be larger than the number of output channels"
            )

        if x.size(-1) < self.out_channels:
            zero = x.new_zeros(x.size(0), self.out_channels - x.size(-1))
            x = torch.cat([x, zero], dim=1)

        bz = x.shape[0]
        assert edge_index.shape[0] == 2
        reversed_edge_index = torch.flip(edge_index, dims=(0,))
        assert key_edge_mask.ndim == 1
        key_reversed_edge_index = reversed_edge_index.T[key_edge_mask].T

        # add self loops
        edge_index = add_self_loops(edge_index)[0]

        new_edge_index = torch.cat([edge_index, key_reversed_edge_index], dim=1)

        for i in range(self.num_layers):
            x = self.layernorm[i](x)
            m = torch.matmul(x, self.weight[i])

            m = m.reshape(bz, self.multiplier, self.out_channels)
            m = m.transpose(0, 1)
            q, k, v = m[0], m[1], m[2]  # make torchscript happy (cannot use tensor as tuple)

            attn = self.get_attn_weight(q, k, edge_index, key_reversed_edge_index)
            raise NotImplementedError
            new_v = self.propagate(new_edge_index, x=v, edge_weight=attn, size=None)

            x = self.rnn(new_v, x)

        return x


class GraphAttn(GatedGraphConvAttn):
    """Pure graph attention layers."""

    def __init__(
        self,
        out_channels: int,
        edge_dim: int = 0,
        num_layers: int = 2,
        aggr: str = "add",
        bias: bool = True,
        use_reversed_edge: bool = True,
        num_heads: int = 1,
        max_blk_diff: int = 16,
        max_pos_diff: int = 16,
        **kwargs,
    ):
        MessagePassing.__init__(self, aggr=aggr, **kwargs)
        if num_layers != 1:
            print(f"Warning: ignoring num_layers != 1")

        self.out_channels = out_channels
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.use_reversed_edge = use_reversed_edge
        head_dim = out_channels // num_heads
        self.scale = head_dim ** -0.5
        self.head_dim = head_dim
        self.max_b_diff = max_blk_diff
        self.max_pos_diff = max_pos_diff

        self.multiplier = 3
        self.qkv = nn.Linear(
            out_channels, 
            out_channels * self.multiplier, 
            bias=bias
        )
        self.back_edge_enc = Param(
            torch.zeros(num_heads, head_dim, dtype=torch.float)
        )
        self.func_enc = Param(
            torch.zeros(3, num_heads, 1, dtype=torch.float)  # same/different function; use -1/0/+1 for encoding convenience 
        )
        self.blk_enc = Param(
            torch.zeros(max_blk_diff * 2 + 1, num_heads, head_dim, dtype=torch.float)
        )
        self.pos_enc = Param(
            torch.zeros(max_pos_diff * 2 + 1, num_heads, head_dim, dtype=torch.float)
        )
        self.layernorm = torch.nn.LayerNorm(normalized_shape=out_channels)
        self.layernorm2 = torch.nn.LayerNorm(normalized_shape=out_channels)
        self.proj = nn.Linear(out_channels, out_channels)
        self.mlp = Mlp(out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.back_edge_enc, std=0.02)
        nn.init.trunc_normal_(self.func_enc, std=0.02)
        nn.init.trunc_normal_(self.blk_enc, std=0.02)
        nn.init.trunc_normal_(self.pos_enc, std=0.02)
        self.apply(self._init_weights)

    def edge_attn(self, query, key, edge_index, node2func, node2block, back_edge: bool = False):
        q_idx = edge_index[1, :].flatten()
        k_idx = edge_index[0, :].flatten()
        q = query[q_idx] * self.scale
        k = key[k_idx]
        if back_edge:
            k = k + self.back_edge_enc
        blk_enc = self.encode_pos(node2block[q_idx], node2block[k_idx], self.blk_enc, self.max_b_diff, self.head_dim)
        pos_enc = self.encode_pos(q_idx, k_idx, self.pos_enc, self.max_pos_diff, self.head_dim)
        k = k + blk_enc + pos_enc
        attn = (q * k).sum(-1)  # [num_q, num_heads]
        func_bias = self.encode_pos(node2func[q_idx], node2func[k_idx], self.func_enc, 1, 1)
        attn = attn + func_bias.squeeze(-1)
        return q_idx, attn

    def encode_pos(self, idx0, idx1, encoding_table, max_diff, last_dim):
        diff = idx0 - idx1
        diff.clamp_(-max_diff, max_diff).add_(max_diff)  # [num_q]
        diff = diff.detach()
        diff = diff.unsqueeze(1).unsqueeze(1).expand(-1, self.num_heads, last_dim)
        encoding = torch.gather(encoding_table, 0, diff)  # [num_q, num_heads, head_dim]
        return encoding

    def get_attn_weight(self, query, key, edge_index, edge_index1, node2func, node2block):
        """
        Compute the attention from query to key and normalize then over the keys.
        Per PyG conventions, the first row is the source indices, the second row
        is the target indices, and the message is passed from the source to the
        target (i.e., aggregated in the target nodes). So in the attention, the
        target is the query.
        Args:
            query (tensor): [n, dim]
            key (tensor): [n, dim]
            edge_index (tensor): [2, n_edges]
            edge_index1 (tensor): for back edges [2, n_edges]
            node2func (tensor): map node idx to function idx, [num_nodes]
            node2block (tensor): map node idx to block idx, [num_nodes]
        Outpus:
            attn (tensor): [n_edges]
        """
        q_idx0, attn0 = self.edge_attn(query, key, edge_index, node2func, node2block)

        # for reversed edges
        q_idx1, attn1 = self.edge_attn(query, key, edge_index1, node2func, node2block, back_edge=True)

        # combine the attention of original and reversed edges
        q_idx = torch.cat([q_idx0, q_idx1])
        attn = torch.cat([attn0, attn1])

        max_ = torch_scatter.scatter_max(attn.detach(), q_idx, dim=0)[0]  # max_values, max_idx
        max_ = max_.detach()  # [idx_size, num_heads]
        # use the softmax subtraction trick to avoid exponential explosion
        attn = attn - max_[q_idx]
        attn = attn.exp()
        dnt = torch_scatter.scatter_add(attn, q_idx, dim=0)
        attn = attn / dnt[q_idx]  # softmax
        return attn.T  # [num_heads, num_q]

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
        key_edge_mask=None,
        node2func=None, 
        node2block=None,
        **kwargs,
    ) -> Tensor:
        """"""
        if x.size(-1) > self.out_channels:
            raise ValueError(
                "The number of input channels is not allowed to "
                "be larger than the number of output channels"
            )

        if x.size(-1) < self.out_channels:
            zero = x.new_zeros(x.size(0), self.out_channels - x.size(-1))
            x = torch.cat([x, zero], dim=1)

        bz = x.shape[0]
        assert edge_index.shape[0] == 2
        reversed_edge_index = torch.flip(edge_index, dims=(0,))
        assert key_edge_mask.ndim == 1
        key_reversed_edge_index = reversed_edge_index.T[key_edge_mask].T

        # add self loops
        edge_index = add_self_loops(edge_index)[0]

        new_edge_index = torch.cat([edge_index, key_reversed_edge_index], dim=1)

        x0 = x
        x = self.layernorm(x)
        m = self.qkv(x)

        m = m.reshape(bz, self.multiplier, self.num_heads, -1)
        m = m.transpose(0, 1)
        q, k, v = m[0], m[1], m[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = self.get_attn_weight(q, k, edge_index, key_reversed_edge_index, node2func, node2block)

        new_out = []
        v = v.transpose(0, 1)  # [num_heads, bz, dim]
        for i in range(self.num_heads):
            new_v = self.propagate(new_edge_index, x=v[i], edge_weight=attn[i], size=None)
            new_out.append(new_v)
        new_v = torch.cat(new_out, dim=1)
        new_v = self.proj(new_v)

        out = x0 + new_v

        x_out = out + self.mlp(self.layernorm2(out))

        return x_out

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.out_channels}, "
            f"num_heads={self.num_heads})"
        )


class GraphAttnWin(GraphAttn):
    """
    Graph attention with extra local window attention layers
    """
    def __init__(
        self, 
        out_channels: int, 
        edge_dim: int = 0,
        num_layers: int = 2, 
        aggr: str = "add", 
        bias: bool = True, 
        use_reversed_edge: bool = True, 
        num_heads: int = 1,
        window_attn_width: int = 32, 
        max_blk_diff: int = 32,
        max_blk_pos_diff: int = 64,
        shift_win=False,
        **kwargs
    ):
        super().__init__(out_channels, num_layers, aggr, bias, use_reversed_edge, num_heads, max_blk_diff // 2, max_blk_pos_diff // 2, **kwargs)
        self.window_attn_width = window_attn_width
        self.shift_win = shift_win
        self.max_blk_diff = max_blk_diff
        self.max_blk_pos_diff = max_blk_pos_diff
        self.block_enc = Param(torch.zeros(num_heads, 2 * max_blk_diff + 1))
        self.block_pos_enc = Param(torch.zeros(num_heads, 2 * max_blk_pos_diff + 1))
        self.shift_size = window_attn_width // 2

        nn.init.trunc_normal_(self.block_enc, std=0.02)
        nn.init.trunc_normal_(self.block_pos_enc, std=0.02)

        self.attn = WindowAttention(out_channels, num_heads, bias)
        self.attn.apply(self._init_weights)

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
        key_edge_mask=None,
        node2func=None, 
        node2block=None,
        pos=None,
        ordered_instr_idx=None,
        **kwargs
    ) -> Tensor:
        x = super().forward(x, edge_index, edge_weight, key_edge_mask, node2func, node2block)
        instr_nodes = torch.index_select(x, 0, ordered_instr_idx)

        # pad to enable window division
        pad_len = self.window_attn_width - instr_nodes.shape[0] % self.window_attn_width
        if pad_len == self.window_attn_width:
            pad_len = 0
        if pad_len != 0:
            pad = torch.zeros(pad_len, x.shape[1], dtype=x.dtype, device=x.device)
            instr_nodes = torch.cat([instr_nodes, pad], dim=0)
            pos_pad = torch.zeros(pad_len, pos.shape[1], dtype=pos.dtype, device=pos.device)
            pos_pad[:, 0].fill_(-1)  # make a dummy batch idx to block info from padding
            pos = torch.cat([pos, pos_pad], dim=0)

        # window division
        if self.shift_win:
            instr_nodes = torch.roll(instr_nodes, shifts=-self.shift_size, dims=0)
        bz = instr_nodes.shape[0] // self.window_attn_width
        win = instr_nodes.reshape(bz, self.window_attn_width, -1)
        pos_win = pos.reshape(bz, self.window_attn_width, -1)

        # mask attn between token from different graphs/functions
        rel_pos = pos_win.unsqueeze(2) - pos_win.unsqueeze(1)  # [bz, win, win, 4]
        rel_pos = rel_pos.permute(3, 0, 1, 2)  # [4, bz, win, win]
        attn_mask_ = (rel_pos[0] != 0) | (rel_pos[1] != 0)  # True-element should be masked. [bz, win, win]
        attn_mask = torch.zeros(attn_mask_.shape, dtype=x.dtype, device=x.device)
        attn_mask.masked_fill_(attn_mask_, -100)

        # get relative positional encoding (as attention bias)
        # the positional encoding table is limited, so the positions
        # outside the table would be clamped
        blk_idx = rel_pos[2].flatten()
        blk_pos = rel_pos[3].flatten()
        blk_idx.clamp_(-self.max_blk_diff, self.max_blk_diff).add_(self.max_blk_diff)
        blk_pos.clamp_(-self.max_blk_pos_diff, self.max_blk_pos_diff).add_(self.max_blk_pos_diff)
        attn_bias = []
        for h in range(self.num_heads):
            head_bias = torch.gather(self.block_enc[h], 0, blk_idx) + torch.gather(self.block_pos_enc[h], 0, blk_pos)
            attn_bias.append(head_bias)
        attn_bias = torch.stack(attn_bias)  # [num_heads, -1]
        attn_bias = attn_bias.reshape(self.num_heads, bz, self.window_attn_width, -1).transpose(0, 1)

        new_instr = self.attn(win, attn_bias, attn_mask)  # [bz, win, dim]
        new_instr = new_instr.reshape(bz * self.window_attn_width, -1)

        # reverse window shift
        if self.shift_win:
            new_instr = torch.roll(new_instr, shifts=self.shift_size, dims=0)

        if pad_len != 0:
            new_instr = new_instr[:-pad_len]
    
        x = x.clone()
        # don't use add_; add_ is not in-place for x
        # x[ordered_instr_idx] += new_instr
        x.index_add_(0, ordered_instr_idx, new_instr)

        return x

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.out_channels}, "
            f"num_heads={self.num_heads}, shift_win={self.shift_win})"
        )


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    """

    def __init__(self, dim, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, attn_bias=None, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # [bz, num_heads, N, N]

        if attn_bias is not None:
            attn = attn + attn_bias

        if mask is not None:
            attn = attn + mask.unsqueeze(1)
        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SimpleMessagePassing(MessagePassing):

    def __init__(
        self,
        out_channels: int,
        aggr: str = "add",
        bias: bool = True,
        act: bool = True,
    ):
        super().__init__(aggr=aggr)

        self.out_channels = out_channels
        self.fc = nn.Linear(out_channels, out_channels, bias=bias)
        self.act = nn.ReLU() if act else None

    def forward(
        self, x: Tensor, edge_index: Tensor,
    ) -> Tensor:
        m = self.propagate(edge_index, x=x, edge_weight=None, size=None)
        new_x = self.fc(m)
        if self.act:
            new_x = self.act(new_x)
        return new_x

    def message(self, x_j: Tensor, edge_weight: OptTensor):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)
