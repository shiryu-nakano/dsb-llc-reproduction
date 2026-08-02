"""
SGLDの各drawで w_t の w* (チェックポイントからロードした初期値) からの
距離 ||w_t - w*||_2 を記録するコールバック。
MalaAcceptanceRate (devinterp/slt/mala.py) と同じ SamplerCallback インターフェースに準拠。
"""
from typing import Union

import numpy as np
import torch
import torch.nn as nn

from devinterp.slt.callback import SamplerCallback


class WeightDistanceTracker(SamplerCallback):
    def __init__(
        self,
        num_chains: int,
        num_draws: int,
        initial_params,
        device: Union[torch.device, str] = "cpu",
    ):
        self.num_chains = num_chains
        self.num_draws = num_draws
        self.device = device
        self.initial_params = [p.to(device) for p in initial_params]
        self.distance = torch.zeros(
            (num_chains, num_draws), dtype=torch.float32
        ).to(device)

    def __call__(
        self, chain: int, draw: int, model: nn.Module, loss: float, optimizer, **kwargs
    ):
        self.update(chain, draw, model, loss, optimizer)

    def update(self, chain: int, draw: int, model: nn.Module, loss: float, optimizer):
        current_params = [
            p.detach() for p in model.parameters() if p.requires_grad
        ]
        total_sq = 0.0
        for cp, ip in zip(current_params, self.initial_params):
            total_sq += torch.sum((cp - ip) ** 2).item()
        self.distance[chain, draw] = total_sq ** 0.5

    def get_results(self):
        d = self.distance.cpu().numpy()
        return {
            "weight_distance/trace": d,
            "weight_distance/mean": float(np.mean(d)),
            "weight_distance/final": float(np.mean(d[:, -1])),
        }
