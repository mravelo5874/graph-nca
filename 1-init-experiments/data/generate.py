import os
import torch
import numpy as np
from typing import Tuple

# * coords.shape: [x, 2] for 2D or [x, 3] in 3D (default to 3D)
# * edges.shape: [2, x]

def generate_line_graph(n_points: int, length: float=1.0) -> Tuple[torch.Tensor, torch.LongTensor]:
    coords = torch.linspace(0, length, steps=n_points).unsqueeze(1).repeat(1, 3)
    dist = ((coords.unsqueeze(1) - coords.unsqueeze(0)) ** 2).sum(dim=-1).fill_diagonal_(torch.inf)
    edges = torch.argwhere(dist <= dist.min() + 0.001).T
    return coords, edges.long()

def generate_square_plane_graph(n_points: int, length: float=1.0) -> Tuple[torch.Tensor, torch.LongTensor]:
    values = torch.linspace(0, length, steps=n_points)
    x, y = torch.meshgrid(values, values, indexing='ij')
    z = torch.zeros_like(x)
    coords = torch.stack([x, y, z]).reshape(3, -1).permute(1, 0)
    dist = ((coords.unsqueeze(1) - coords.unsqueeze(0)) ** 2).sum(dim=-1).fill_diagonal_(torch.inf)
    edges = torch.argwhere(dist <= dist.min() + 0.001).T
    return coords, edges.long()

def generate_bunny_graph() -> Tuple[torch.Tensor, torch.LongTensor]:
    path = os.path.realpath(__file__).replace('\\generate.py', '')
    coords = torch.tensor(np.load(f'{path}/bunny/bunny-coords.npy'), dtype=torch.float32)
    edges = torch.tensor(np.load(f'{path}/bunny/bunny-edges.npy'), dtype=torch.int64)
    return coords, edges

def generate_cube_graph():
    raise NotImplementedError

def generate_sphere_graph():
    raise NotImplementedError