"""BiLSTM on angle features; ST-GCN on normalized joint coordinates."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from exercise_cls.config import NUM_JOINTS, STGCN_EDGES
from exercise_cls.geometry import angle_feature_dim


def build_normalized_adj(num_nodes: int, edges: tuple[tuple[int, int], ...]) -> np.ndarray:
    a = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i, j in edges:
        a[i, j] = 1.0
        a[j, i] = 1.0
    np.fill_diagonal(a, 1.0)
    d = np.sum(a, axis=1)
    d_inv_sqrt = np.power(d, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat = np.diag(d_inv_sqrt)
    return d_mat @ a @ d_mat


class AngleBiLSTM(nn.Module):
    def __init__(self, num_classes: int = 2, hidden: int = 128, input_dim: int | None = None):
        super().__init__()
        d = input_dim or angle_feature_dim()
        self.lstm = nn.LSTM(d, hidden, batch_first=True, bidirectional=True, num_layers=2, dropout=0.2)
        self.head = nn.Sequential(nn.Linear(hidden * 2, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        o, _ = self.lstm(x)
        return self.head(o[:, -1, :])


class STGCNBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, adj: torch.Tensor, temporal_kernel: int = 9):
        super().__init__()
        self.register_buffer("adj", adj)
        self.conv_s = nn.Conv2d(in_c, out_c, kernel_size=(1, 1))
        pad = temporal_kernel // 2
        self.conv_t = nn.Conv2d(out_c, out_c, kernel_size=(temporal_kernel, 1), padding=(pad, 0))
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, T, V)
        n, c, t, v = x.shape
        x_s = self.conv_s(x)
        # graph conv: aggregate neighbors
        x_agg = torch.einsum("nctv,vw->nctw", x_s, self.adj)
        x_agg = self.conv_t(x_agg)
        return self.act(self.bn(x_agg))


class STGCNClassifier(nn.Module):
    """Lightweight ST-GCN-style network for (N, T, V, C_in) with C_in=3."""

    def __init__(self, num_classes: int = 2, channels: tuple[int, ...] = (64, 128, 256)):
        super().__init__()
        adj = torch.from_numpy(build_normalized_adj(NUM_JOINTS, STGCN_EDGES))
        self.adj = adj
        blocks = []
        c_in = 3
        for c_out in channels:
            blocks.append(STGCNBlock(c_in, c_out, adj, temporal_kernel=9))
            c_in = c_out
        self.blocks = nn.ModuleList(blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, T, V, C) -> (N, C, T, V)
        x = x.permute(0, 3, 1, 2)
        for b in self.blocks:
            x = b(x)
        # global average over time and joints
        n, c, t, v = x.shape
        x = x.mean(dim=(2, 3))
        return self.fc(x)
