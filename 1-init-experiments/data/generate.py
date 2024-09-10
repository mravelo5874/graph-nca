import torch

def generate_line_graph(n_points: int):
    coord = torch.linspace(0, 1, steps=n_points).unsqueeze(1).repeat(1, 2)
    coord = (coord - coord.mean(0)) / coord.std(0)
    dist = ((coord.unsqueeze(1) - coord.unsqueeze(0)) ** 2).sum(dim=-1).fill_diagonal_(torch.inf)
    edge_index = torch.argwhere(dist <= dist.min() + 0.001).T
    return coord, edge_index

def generate_plane_graph():
    raise NotImplementedError

def generate_cube_graph(length: int, radius: float):
    values = torch.linspace(0, 1, steps=length)
    coord = torch.stack(torch.meshgrid(values, values, values, indexing='xy')).reshape(3, -1).T
    dist = torch.norm(coord - coord.unsqueeze(1), dim=-1) < radius
    dist.fill_diagonal_(0)
    edge_index = dist.nonzero().T
    return coord, edge_index

def generate_sphere_graph():
    raise NotImplementedError