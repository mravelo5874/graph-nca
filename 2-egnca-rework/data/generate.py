import os
import torch
import numpy as np
from typing import Tuple

# * coords.shape: [x, 2] for 2D or [x, 3] in 3D (default to 3D)
# * edges.shape: [2, x]

def generate_line_graph(
    n_points: int, 
    length: float=1.0,
) -> Tuple[torch.Tensor, torch.LongTensor]:
    assert n_points > 1
    assert length > 0
    coords = torch.linspace(-length/2, length/2, steps=n_points).unsqueeze(1).repeat(1, 3)
    dist = ((coords.unsqueeze(1) - coords.unsqueeze(0)) ** 2).sum(dim=-1).fill_diagonal_(torch.inf)
    edges = torch.argwhere(dist <= dist.min() + 0.001).T
    return coords, edges.long()

def generate_square_plane_graph(
    n_points: int, 
    length: float=1.0,
) -> Tuple[torch.Tensor, torch.LongTensor]:
    assert n_points > 1
    assert length > 0
    values = torch.linspace(-length/2, length/2, steps=n_points)
    x, y = torch.meshgrid(values, values, indexing='ij')
    z = torch.zeros_like(x)
    coords = torch.stack([x, y, z]).reshape(3, -1).permute(1, 0)
    dist = ((coords.unsqueeze(1) - coords.unsqueeze(0)) ** 2).sum(dim=-1).fill_diagonal_(torch.inf)
    edges = torch.argwhere(dist <= dist.min() + 0.001).T
    return coords, edges.long()

def generate_cube_graph(
    n_points: int, 
    length: float=1.0,
) -> Tuple[torch.Tensor, torch.LongTensor]:
    assert n_points > 1
    assert length > 0
    plane_coords, _ = generate_square_plane_graph(n_points, length)
    delta = torch.zeros_like(plane_coords)
    delta[:, 2] = length/(n_points-1)
    planes = []
    for i in range(n_points):
        planes.append(plane_coords.clone().add(delta*i))
    coords = torch.cat(planes, dim=0)
    dist = ((coords.unsqueeze(1) - coords.unsqueeze(0)) ** 2).sum(dim=-1).fill_diagonal_(torch.inf)
    edges = torch.argwhere(dist <= dist.min() + 0.001).T
    return coords, edges.long()

def generate_geodesic_polyhedron_graph(
    subdivisions: int,
    normalize: bool = False
):
    assert subdivisions >= 0
    import itertools
    
    # Create an icosahedron
    phi = (1 + np.sqrt(5)) / 2   # golden ratio
    vertices = np.array([
        [-1,  phi, 0],
        [ 1,  phi, 0],
        [-1, -phi, 0],
        [ 1, -phi, 0],
        [0, -1,  phi],
        [0,  1,  phi],
        [0, -1, -phi],
        [0,  1, -phi],
        [ phi, 0, -1],
        [ phi, 0, 1],
        [-phi, 0, -1],
        [-phi, 0, 1],
    ])

    vertices /= np.linalg.norm(vertices[0])  # Normalize to unit sphere

    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ])

    # Subdivide the faces
    for _ in range(subdivisions):
        new_faces = []
        midpoints = {}  # Store the midpoints of the edges

        for face in faces:
            v0, v1, v2 = vertices[face]
            a = tuple(np.round((v0 + v1) / 2, 8))
            b = tuple(np.round((v1 + v2) / 2, 8))
            c = tuple(np.round((v2 + v0) / 2, 8))

            for midpoint in (a, b, c):
                if midpoint not in midpoints:
                    midpoints[midpoint] = len(vertices)
                    vertices = np.vstack([vertices, midpoint])

            a_index, b_index, c_index = midpoints[a], midpoints[b], midpoints[c]
            new_faces.extend([
                [face[0], a_index, c_index],
                [face[1], b_index, a_index],
                [face[2], c_index, b_index],
                [a_index, b_index, c_index],
            ])

        faces = np.array(new_faces)

    # Generate edges
    edges = set()
    for face in faces:
        for i, j in itertools.combinations(face, 2):
            edges.add(tuple(sorted([i, j])))
        
    if normalize:
        vertices /= np.linalg.norm(vertices, axis=1, keepdims=True)
    vertices = torch.tensor(vertices)
    edges = torch.tensor(list(edges)).permute([1, 0]).long()
    
    return vertices, edges

def retrieve_bunny_graph() -> Tuple[torch.Tensor, torch.LongTensor]:
    path = os.path.realpath(__file__).replace('\\generate.py', '')
    coords = torch.tensor(np.load(f'{path}/bunny/bunny-coords.npy'), dtype=torch.float32)
    coords = (coords - coords.mean(0)) / coords.std(0)
    edges = torch.tensor(np.load(f'{path}/bunny/bunny-edges.npy'), dtype=torch.int64)
    return coords, edges