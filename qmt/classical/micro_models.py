"""Micro-student models: 12 -> h -> 5 with 41-95 parameters.

Linear(12, h) + Tanh + Linear(h, 5)
No BatchNorm, no large embeddings, no hidden tricks.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class MicroStudent(nn.Module):
    """12 -> h -> task_dim micro-student.

    Parameter counts for task_dim=5:
        h=2:  12*2+2 + 2*5+5  = 41
        h=3:  12*3+3 + 3*5+5  = 59
        h=4:  12*4+4 + 4*5+5  = 77
        h=5:  12*5+5 + 5*5+5  = 95
    """

    def __init__(self, input_dim: int = 12, hidden: int = 4, task_dim: int = 5,
                 aux_dim: int = 0):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(input_dim, hidden), nn.Tanh())
        self.task_head = nn.Linear(hidden, task_dim)
        self.aux_head = nn.Linear(hidden, aux_dim) if aux_dim > 0 else None

    def forward(self, x):
        z = self.backbone(x)
        task = self.task_head(z)
        aux = self.aux_head(z) if self.aux_head is not None else None
        return task, aux


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def param_count(input_dim: int, hidden: int, task_dim: int, aux_dim: int = 0) -> int:
    """Compute parameter count without building the model."""
    p = input_dim * hidden + hidden  # backbone
    p += hidden * task_dim + task_dim  # task head
    if aux_dim > 0:
        p += hidden * aux_dim + aux_dim  # aux head
    return p
