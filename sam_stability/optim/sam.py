"""SAM optimizer wrapper with optional training metrics."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from torch import nn


@dataclass
class SamMetrics:
    loss_ascent: Dict[float, List] = field(default_factory=dict)
    loss_descent: Dict[float, List] = field(default_factory=dict)
    grad_cosine: Dict[float, List] = field(default_factory=dict)

    def record_ascent_loss(self, rho: float, loss: torch.Tensor) -> None:
        self.loss_ascent.setdefault(rho, []).append(loss)

    def record_descent_loss(self, rho: float, loss: torch.Tensor) -> None:
        self.loss_descent.setdefault(rho, []).append(loss)

    def record_grad_cosine(self, rho: float, cosine: torch.Tensor) -> None:
        self.grad_cosine.setdefault(rho, []).append(cosine)


class SAM:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        params,
        rho: float = 0.5,
        metrics: Optional[SamMetrics] = None,
    ):
        self.optimizer = optimizer
        self.model = list(params)
        self.rho = rho
        self.state = defaultdict(dict)
        self.metrics = metrics
        self.ascents: Optional[torch.Tensor] = None

    @torch.no_grad()
    def ascent_step(self) -> None:
        grads_all, grads = [], []
        for p in self.model:
            if p.grad is None:
                continue
            grads_all.append(p.grad.detach().clone())
            grads.append(torch.norm(p.grad, p=2))
        grad_norm = torch.norm(torch.stack(grads), p=2) + 1e-16
        self.ascents = torch.cat(grads_all).flatten()

        for p in self.model:
            if p.grad is None:
                continue
            eps = self.state[p].get("eps")
            if eps is None:
                eps = torch.clone(p).detach()
                self.state[p]["eps"] = eps
            eps[...] = p.grad[...]
            eps.mul_(self.rho / grad_norm)
            p.add_(eps)
        self.optimizer.zero_grad()

    @torch.no_grad()
    def descent_step(self) -> None:
        grads_all = []
        for p in self.model:
            if p.grad is None:
                continue
            grads_all.append(p.grad.detach().clone())
            p.sub_(self.state[p]["eps"])

        if self.metrics is not None and self.ascents is not None:
            descents = torch.cat(grads_all).flatten()
            cosine = nn.CosineSimilarity(dim=0, eps=1e-30)(descents, self.ascents)
            self.metrics.record_grad_cosine(self.rho, cosine)

        self.optimizer.step()
        self.optimizer.zero_grad()
