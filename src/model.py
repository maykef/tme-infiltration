"""Shared model definitions for pretraining (Stage 4) and fine-tuning (Stage 5).

The transferable component across the two different marker panels is the GATv2 message-passing
*trunk*, which operates entirely in hidden dimension. Each dataset has its own input `stem`
(Linear) mapping its marker vector -> hidden, so a 33-marker Jackson encoder and a 35-marker
Moldoveanu encoder can share trunk weights. `trunk_state_dict()` / `load_trunk()` move just the
trunk between them.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class GNNEncoder(nn.Module):
    def __init__(self, in_dim, hidden=128, heads=4, layers=3, dropout=0.1):
        super().__init__()
        assert hidden % heads == 0
        self.stem = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList()
        for _ in range(layers):
            self.convs.append(
                GATv2Conv(hidden, hidden // heads, heads=heads, concat=True, dropout=dropout)
            )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.dropout = dropout

    def forward(self, x, edge_index):
        h = F.elu(self.stem(x))
        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, edge_index)
            h = norm(F.elu(h_new) + h)          # residual + layernorm
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def trunk_state_dict(self):
        """Weights that transfer across panels: the GATv2 convs + their norms (not the stem)."""
        return {k: v for k, v in self.state_dict().items()
                if k.startswith("convs.") or k.startswith("norms.")}

    def load_trunk(self, trunk_sd, freeze=False):
        missing = self.load_state_dict(trunk_sd, strict=False)
        if freeze:
            for name, p in self.named_parameters():
                if name.startswith("convs.") or name.startswith("norms."):
                    p.requires_grad_(False)
        return missing


class MaskedMarkerModel(nn.Module):
    """Stage 4: encoder + linear head reconstructing all markers. Input = [masked_x, mask_flag]."""
    def __init__(self, n_markers, hidden=128, heads=4, layers=3, dropout=0.1):
        super().__init__()
        self.n_markers = n_markers
        self.encoder = GNNEncoder(2 * n_markers, hidden, heads, layers, dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ELU(), nn.Linear(hidden, n_markers)
        )

    def forward(self, masked_x, mask_flag, edge_index):
        inp = torch.cat([masked_x, mask_flag], dim=1)
        h = self.encoder(inp, edge_index)
        return self.head(h)


class SlideClassifier(nn.Module):
    """Stage 5: encoder + attention pooling -> slide embedding -> MLP -> binary logit."""
    def __init__(self, n_markers, hidden=128, heads=4, layers=3, dropout=0.2):
        super().__init__()
        self.encoder = GNNEncoder(n_markers, hidden, heads, layers, dropout)
        # GlobalAttention (a.k.a. AttentionalAggregation): gate scores per node
        gate_nn = nn.Sequential(nn.Linear(hidden, hidden), nn.ELU(), nn.Linear(hidden, 1))
        try:
            from torch_geometric.nn import GlobalAttention
            self.pool = GlobalAttention(gate_nn)
        except Exception:
            from torch_geometric.nn.aggr import AttentionalAggregation
            self.pool = AttentionalAggregation(gate_nn)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ELU(), nn.Dropout(dropout), nn.Linear(hidden, 1)
        )

    def node_embeddings(self, x, edge_index):
        return self.encoder(x, edge_index)

    def forward(self, x, edge_index, batch):
        h = self.encoder(x, edge_index)
        pooled = self.pool(h, batch)
        return self.head(pooled).squeeze(-1)      # (num_graphs,) logits
