# libraries

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import GATConv, GATv2Conv, GCNConv, global_mean_pool

CONV_TYPES = ("gcn", "gat", "gatv2")

@dataclass
class GCNSurrogateConfig:
    "Dimensions and depth"
    n_node_features: int = 3 # x, y, cell volume
    n_scalar_features: int = 5 # m, p, t, AoA, V_in
    n_outputs: int = 2 # static pressure, velocity magnitude
    hidden: int = 256 # neurons per layer
    n_shared_blocks: int = 4 # number of shared layer blocks
    n_gcn_blocks: int = 3 # number of gcn blocks with the exception of the final output convolution
    conv: str = "gcn" # "gcn", "gat" or "gatv2"
    heads: int = 4 # attention heads

    def __post_init__(self) -> None:
        if self.n_shared_blocks < 1:
            raise ValueError("n_shared_blocks must be >=1")
        if self.n_gcn_blocks < 1:
            raise ValueError("n_gcn_blocks must be >=1")
        if self.conv not in CONV_TYPES:
            raise ValueError(f"conv must be one of {CONV_TYPES}, got {self.conv!r}")
        if self.conv in ("gat", "gatv2") and self.hidden % self.heads != 0:
            raise ValueError(f"hidden ({self.hidden}) must be divisible by heads ({self.heads})")

def make_conv(in_dim: int, out_dim: int, cfg: GCNSurrogateConfig, last: bool = False) -> nn.Module:
    "A graph convolution of the configured type, out_dim wide either way"
    if cfg.conv == "gcn":
        return GCNConv(in_dim, out_dim, add_self_loops = False)
    # extra holds keyword arguments for one attention type only, e.g. share_weights for gatv2. None are set at present
    if cfg.conv == "gat":
        layer, extra = GATConv, {}
    else:
        layer, extra = GATv2Conv, {}
    if last:
        # heads averaged, so the width stays n_outputs
        return layer(in_dim, out_dim, heads = cfg.heads, concat = False, add_self_loops = False, **extra)
    # each head gives out_dim / heads features, concatenated back to out_dim
    return layer(in_dim, out_dim // cfg.heads, heads = cfg.heads, add_self_loops = False, **extra)

class SharedBlock(nn.Module):
    """shared GCN/FC block:
    Two pathways side by side, with one carrying a row per mesh cell and the other
    carrying a row of case (boundary/scalar) parameters. The mesh is averaged down into the
    case pathway, then the merged result is copied back out to every cell.
    """

    def __init__(
            self,
            graph_in: int,
            scalar_in: int,
            hidden: int,
            skip_in: Optional[int],
            cfg: GCNSurrogateConfig,
    ) -> None:
        super().__init__()
        self.graph_in = graph_in
        self.scalar_in = scalar_in
        self.hidden = hidden

        self.conv = make_conv(graph_in, hidden, cfg)
        self.fc = nn.Linear(scalar_in, hidden)

        # skip_in is None only for block 1, which has no preceding block. Everywhere else the shortcut passes through a learned layer
        self.skip_graph = nn.Linear(skip_in, hidden) if skip_in is not None else None
        self.skip_scalar = nn.Linear(skip_in, hidden) if skip_in is not None else None
        self.act = nn.ReLU()

    def forward(
        self,
        C: Tensor, # graph signal in,  one row per cell
        S: Tensor, # scalar signal in, one row per case
        Z: Optional[Tensor], # graph shortcut from the previous block
        Q: Optional[Tensor], # scalar shortcut from the previous block
        edge_index: Tensor,
        batch: Tensor, # which case each cell belongs to
        n_graphs: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        N = C.size(0)
        # the graph width changes through the stack (node features, then 2 * hidden), so the shapes are checked explicitly
        assert C.size(1) == self.graph_in, (
            f"graph signal C has {C.size(1)} features, expected {self.graph_in}"
        )
        assert S.size(1) == self.scalar_in, (
            f"scalar signal S has {S.size(1)} features, expected {self.scalar_in}"
        )
        assert S.size(0) == n_graphs, (
            f"S has {S.size(0)} rows but the batch holds {n_graphs} graphs"
        )

        # Mesh pathway. The convolution averages each cell with its neighbours, the shortcut restores the detail that averaging smooths out
        G = self.conv(C, edge_index)
        if self.skip_graph is not None:
            G = G + self.skip_graph(Z)
        G = self.act(G)

        # Case pathway
        T = self.fc(S)
        if self.skip_scalar is not None:
            T = T + self.skip_scalar(Q)
        T = self.act(T)

        # Downward: pool the mesh to one row per case, the same shape as T
        P = global_mean_pool(G, batch, size = n_graphs)
        # S_next carries both the case parameters and the pooled mesh
        S_next = P + T
        # Upward: copy each case's row to every one of its cells.
        U = S_next[batch]
        # concatenated so the next layer can weigh the spatial part and the case part separately. The width doubles to 2 * hidden
        C_next = torch.cat([G, U], dim=-1)

        assert C_next.shape == (N, 2 * self.hidden), (
            f"C_next has shape {tuple(C_next.shape)}, expected ({N}, {2 * self.hidden})"
        )
        # C_next and S_next are the next block's inputs, G and T its shortcuts
        return C_next, S_next, G, T

class GCNBlock(nn.Module):
    "Residual graph convolution block"

    def __init__(self, in_dim: int, hidden: int, cfg: GCNSurrogateConfig) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden = hidden
        self.conv = make_conv(in_dim, hidden, cfg)
        self.act = nn.ReLU()

    def forward(
            self,
            H: Tensor,
            edge_index: Tensor,
            skip: Optional[Tensor] = None,
    ) -> Tensor:
        N = H.size(0)
        assert H.size(1) == self.in_dim, (
            f"GCN block input has {H.size(1)} features, expected {self.in_dim}"
        )
        # Blocks after the first use their own input as the residual
        res = H if skip is None else skip
        assert res.size(1) == self.hidden, (
            f"residual has {res.size(1)} features, expected {self.hidden}"
        )
        out = self.act(self.conv(H, edge_index) + res)
        assert out.shape == (N, self.hidden)
        return out

class GCNSurrogate(nn.Module):
    "Shared-layer GCN surrogate: shared blocks, then residual blocks, then an output convolution"

    def __init__(self, cfg: Optional[GCNSurrogateConfig] = None) -> None:
        super().__init__()
        cfg = cfg if cfg is not None else GCNSurrogateConfig()
        self.cfg = cfg
        H = cfg.hidden

        shared = []
        for i in range(cfg.n_shared_blocks):
            if i == 0:
                # Raw inputs, no shortcut since there is no preceding block.
                shared.append(SharedBlock(cfg.n_node_features, cfg.n_scalar_features, H, None, cfg))
            else:
                # 2 * H in, because the previous block concatenated the broadcast case signal onto its mesh signal.
                shared.append(SharedBlock(2 * H, H, H, H, cfg))
        self.shared_blocks = nn.ModuleList(shared)

        # The first standard block also uses the concatenated signal, so it takes 2 * H; the rest are H wide.
        blocks = [GCNBlock(2 * H if j == 0 else H, H, cfg) for j in range(cfg.n_gcn_blocks)]
        self.gcn_blocks = nn.ModuleList(blocks)

        # output convolution
        self.out_conv = make_conv(H, cfg.n_outputs, cfg, last = True)

    def forward(
        self,
        x: Tensor, # (N, n_node_features)
        edge_index: Tensor, # (2, E) KNN connectivity
        scalars: Tensor, # (B, n_scalar_features) one row per case
        batch: Optional[Tensor] = None,
    ) -> Tensor:
        cfg = self.cfg
        N = x.size(0)

        # A single graph: every cell belongs to case 0. new_zeros keeps it on the same device as x.
        if batch is None:
            batch = x.new_zeros(N, dtype = torch.long)
        if scalars.dim() == 1:
            scalars = scalars.unsqueeze(0)

        n_graphs = int(scalars.size(0))

        assert x.size(1) == cfg.n_node_features, (
            f"expected {cfg.n_node_features} node features, got {x.size(1)}"
        )

        assert scalars.size(1) == cfg.n_scalar_features, (
            f"expected {cfg.n_scalar_features} scalars, got {scalars.size(1)}"
        )

        assert edge_index.dim() == 2 and edge_index.size(0) == 2, (
            f"edge_index must be (2,E), got {tuple(edge_index.shape)}"
        )

        C, S = x, scalars
        Z: Optional[Tensor] = None      # no shortcut into block 1
        Q: Optional[Tensor] = None
        G: Optional[Tensor] = None

        for block in self.shared_blocks:
            C, S, G, T = block(C, S, Z, Q, edge_index, batch, n_graphs)
            # This block's outputs become the next block's shortcuts
            Z, Q = G, T

        # G is the last shared block's pre-concatenation embedding, hidden wide, so it serves as the first residual
        h = C
        for j, block in enumerate(self.gcn_blocks):
            h = block(h, edge_index, skip = G if j == 0 else None)

        out = self.out_conv(h, edge_index) # no residual, no activation

        assert out.shape == (N, cfg.n_outputs)
        return out

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
