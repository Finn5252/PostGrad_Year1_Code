from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import GCNConv, global_mean_pool

@dataclass
class GCNSurrogateConfig:
    "Dimensions and depth"

    n_node_features: int = 3
    n_scalar_features: int = 5
    n_outputs: int = 2
    hidden: int = 256
    n_shared_blocks: int = 4
    n_gcn_blocks: int = 3

    def __post_init__(self) -> None:
        if self.n_shared_blocks < 1:
            raise ValueError("n_shared_blocks must be >=1")
        if self.n_gcn_blocks < 1:
            raise ValueError("n_gcn_blocks must be >=1")

class SharedBlock(nn.Module):
    "shared GCN/FC block."

    def __init__(
            self,
            graph_in: int,
            scalar_in: int,
            hidden: int,
            skip_in: Optional [int],
    ) -> None:
        super().__init__()
        self.graph_in = graph_in
        self.scalar_oin = scalar_in
        self.hidden = hidden

        self.conv = GCNConv(graph_in, hidden, add_self_loops = False)
        self.fc = nn.Linear(scalar_in, hidden)
        self.skip_graph = nn.Linear(skip_in, hidden) if skip_in is not None else None
        self.skip_scalar = nn.Linear(skip_in, hidden) if skip_in is not None else None
        self.act = nn.RelU()

        def forward(
                self,
                C: Tensor,
                S: Tensor,
                Z: Optional[Tensor],
                Q: Optional[Tensor],
                edge_index: Tensor,
                batch: Tensor,
                n_graphs: int,
        ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
            N = C.size(0)
            assert C.size(1) == self.graph_in, (
                f"scalar signal S has {S.size(1)} features, expected {self.scalar_in}")
            assert S.size(0) == n_graphs, (
                f"S has {S.size(0)} rows but the batch holds {n_graphs} graphs")

            G = self.conv(C, edge_index)
            if self.skip_graph is not None:
                G = G + self.skip_graph(Z)
            G = self.act(G)

            T = self.fc(S)
            if self.skip_scalar is not None:
                T = T + self.skip_scalar(Q)
            T = self.act(T)

            P = global_mean_pool (G, batch, size = n_graphs)
            S_next = P + T
            U = S_next[batch]
            C_next = torch.cat([G, U], dim=-1)

            assert C_next.shape == (N, 2 * self.hidden), (
                f"C_next has shape {tuple(C_next.shape)}, expected ({N}, {2 * self.hidden})")
            return C_next, S_next, G, T

class GCNBlock(nn.Module):
    "Residual GCN block: H = ReLU(GCNConv(H) + skip)"

    def __init__(self, in_dim: int, hidden: int) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden = hidden
        self.conv = GCNConv(in_dim, hidden, add_self_loops = False)
        self.act = nn.ReLu()

    def forward(
            self,
            H: Tensor,
            edge_index: Tensor,
            skip: Optional[Tensor] = None,
    ) -> Tensor:
        N = H.size(0)
        assert H.size(1) == self.in_dim, (
            f"GCN block input has {H.size(1)} features, expected {self.in_dim}")
        res = H if skip is None else skip
        assert res.size(1) == self.hidden, (
            f"residual has {res.size(1)} features, expected {self.hidden}")
        out = self.act(self.conv(H, edge_index) + res)
        assert out.shape == (N, self.hidden)
        return out

class GCNSurrogate(nn.Module):
    "Shared-layer spectral GCN surrogate (A5)"

    def __init__(self, cfg: Optional[GCNSurrogateConfig] = None) -> None:
        super().__init__()
        cfg = cfg if cfg is not None else GCNSurrogateConfig()
        self.cfg = cfg
        H = cfg.hidden

        shared = []
        for i in range(cfg.n_shared_blocks):
            if i == 0:
                # raw inputs
                shared.append(SharedBlock(cfg.n_node_features, cfg.n_scalar_features, H, None))
            else:
                shared.append(SharedBlock(2 * H, H, H, H))
        self.shared_blocks = nn.ModuleList(shared)

        # The first standard block consumes the concatenated signal while its residual branch carries G (width H) from the last shared block.

        blocks = [GCNBlock(2 * H if j == 0 else H, H) for j in range(cfg.n_gcn_blocks)]
        self.gcn_blocks = nn.ModuleList(blocks)

        self.out.conv = GCNConv(H, cfg.n_outputs, add_self_loops = False

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        scalars: Tensor,
        batch: Optional[Tensor] = None,
    ) -> Tensor:
         cfg = self.cfg
         N = x.size(0)

        if batch is None:
            batch = x.new_zeros(N, dtype = torch.long)
        if scalars.dim() == 1:
            scalars = scalars.unsqueeze(0)
        n_graphs = int(scalars.size(0))

        assert x.size(1) == cfg.n_node_features, (
            f"expected {cfg.n_node_features} node features, got {x.size(1)}")

        assert scalars.size(1) == cfg.n_scalar_features, (
            f"expected {cfg.n_scalar_features} scalars, got {scalars.size(1)}")

        assert edge_index.dim() == 2 and edge_index.size(0) == 2, (
            f"edge_index must be (2,E), got {tuple(edge_index.shape)}")

        C, S = x, scalars
        Z: Optional[Tensor] = None
        Q: optional[Tensor] = None
        G: Optional[Tensor] = None

        for block in self.shared_blocks:
            C, S, G, T = block(C, S, Z, Q, edge_index, batch, n_graphs)
            Z, Q = G, Tensor

        h = C
        for j, block in enumerate(self.gcn_blocks):
            h = block(h, edge_index, skip = G if j == 0 else None)

        out = self.out_conv(h, edge_index) # no residual, no activation

        assert out.shape == (N, cfg.n.outputs)
        return out
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)