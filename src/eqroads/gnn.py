"""Message-passing GNN that decides which gap-bridging candidates are real road links.

The graph mixes detected road segments and candidate links. Nodes and edges both carry
hidden states, and each layer runs the classic three steps of message passing:

    1. message   m_ij = phi_m(h_i, h_j, e_ij)          (computed on every directed edge)
    2. aggregate M_i  = mean_j m_ij
    3. update    h_i <- h_i + phi_h(h_i, M_i),   e_ij <- e_ij + phi_e(h_i, h_j, e_ij)

This is the learned counterpart of belief propagation: in an LDPC decoder the update rules
are fixed and derived from probabilities; here they are learned from data. Edge states play
the role of the per-edge messages of BP, which lets competing candidates at the same dead end
inhibit each other (a dead end usually continues in one direction only).

All inputs are E(2)-invariant (see graph.py), hence so are the predictions.
"""
from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import MessagePassing


def mlp(cin: int, cout: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(cin, hidden), nn.SiLU(), nn.Linear(hidden, cout))


class EdgeNodeLayer(MessagePassing):
    def __init__(self, dim: int):
        super().__init__(aggr="mean")
        self.phi_m = mlp(3 * dim, dim, dim)
        self.phi_h = mlp(2 * dim, dim, dim)
        self.phi_e = mlp(3 * dim, dim, dim)
        self.norm_h = nn.LayerNorm(dim)
        self.norm_e = nn.LayerNorm(dim)

    def forward(self, h, edge_index, e):
        agg = self.propagate(edge_index, h=h, e=e)
        h_new = self.norm_h(h + self.phi_h(torch.cat([h, agg], dim=-1)))
        src, dst = edge_index
        e_new = self.norm_e(e + self.phi_e(torch.cat([h[src], h[dst], e], dim=-1)))
        return h_new, e_new

    def message(self, h_i, h_j, e):
        return self.phi_m(torch.cat([h_i, h_j, e], dim=-1))


class GapGNN(nn.Module):
    def __init__(self, node_in: int, edge_in: int, dim: int = 32, layers: int = 3):
        super().__init__()
        self.node_enc = mlp(node_in, dim, dim)
        self.edge_enc = mlp(edge_in, dim, dim)
        self.layers = nn.ModuleList(EdgeNodeLayer(dim) for _ in range(layers))
        self.head = mlp(3 * dim, 1, dim)

    def forward(self, x, edge_index, edge_attr):
        """edge_index / edge_attr list every undirected edge twice (both directions).

        Returns one logit per directed edge; average the two directions per undirected edge.
        """
        h, e = self.node_enc(x), self.edge_enc(edge_attr)
        for layer in self.layers:
            h, e = layer(h, edge_index, e)
        src, dst = edge_index
        # Symmetric in (src, dst) so both directions give the same score.
        return self.head(torch.cat([h[src] + h[dst], (h[src] - h[dst]).abs(), e], dim=-1)).squeeze(-1)


def to_tensors(nodes, x, edges, edge_attr):
    """Directed edge_index (both directions) from the undirected arrays of graph.build_features."""
    index = {n: i for i, n in enumerate(nodes)}
    pairs = torch.tensor([[index[u], index[v]] for u, v in edges], dtype=torch.long).reshape(-1, 2)
    edge_index = torch.cat([pairs.t(), pairs.flip(1).t()], dim=1)
    ea = torch.as_tensor(edge_attr, dtype=torch.float32)
    return torch.as_tensor(x, dtype=torch.float32), edge_index, torch.cat([ea, ea], dim=0)


def undirected_logits(logits: torch.Tensor) -> torch.Tensor:
    n = logits.shape[0] // 2
    return 0.5 * (logits[:n] + logits[n:])
